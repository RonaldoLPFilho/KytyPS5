#pragma once
// desport: KYTY_GPU_PROFILE (diagnostic): GPU time per translated shader (draw / dispatch), sampled.
//
// KYTY_GPU_PROFILE=1 hooks the entries every Vulkan draw and dispatch of ours ends in - the recording worker's replay,
// the draw and descriptor packets (native XPR, table path), the direct calls: LocalVulkanRecording::HookDispatch() -
// plus command-buffer begin/end, pipeline binds and dynamic rendering. Once every KYTY_GPU_PROFILE_EVERY flips
// (default 30) a window opens until the next flip: every command buffer of ours whose begin is replayed inside it (pools
// of a queue family with timestamps) gets a GPU timestamp at its begin and end, and every draw and dispatch recorded into
// it one right before and one right after (both "after all previous commands": the draw's time is what it adds past
// the work before it, so consecutive draws never count twice). A reader thread takes the results without waiting once
// the window's command buffers ended (host query reset; a pair not available 3 s after the window closed is "lost"),
// aggregates per key - kind (draw/draw_indexed/draw_indirect/draw_indexed_indirect/mesh/dispatch/dispatch_indirect),
// pipeline kind (gfx-normal/gfx-native/gfx-table/cs-normal/cs-table/unk), vs and ps hash (cs hash), render target size
// (dynamic rendering area) or group counts - and every KYTY_GPU_PROFILE_SECONDS (default 10) writes the top
// KYTY_GPU_PROFILE_TOP (default 25) as "DES_GPU_PROF rank=..." lines and one "DES_GPU_PROF total ..." line:
//   total: frames (sampled windows), cbs, calls, attributed_ms (the sum of every draw/dispatch's time), busy_ms (the
//   union of the command buffers' [begin, end] on the GPU clock), span_ms (first begin to last end per window),
//   frame_*_ms (per sampled window), coverage_pct (attributed / busy), dropped (no free query pair), lost, orphans.
//   rank: calls, calls_per_frame, gpu_ms, ms_per_frame, pct (of busy), avg_us, max_us, target=<w>x<h> (draws) or
//   groups=<x>x<y>x<z> (dispatches; "indirect"), wave (cs: 32/64, "64e" = wave64 emulated on 32 lanes), verts (draws:
//   mean vertices x instances per call).
// Streamline's and frame generation's own command buffers are never touched (they do not go through these entries).
// The timestamps serialize the sampled window's draws: its frame runs slower than the others (compare with
// KYTY_GPU_TIMESTAMPS); off (the default): nothing is hooked.

#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace Desport {
void     EmitLine(const std::string& text);
uint64_t ElapsedMilliseconds();
} // namespace Desport
namespace LocalPlatform {
void SetThreadName(const char* name);
} // namespace LocalPlatform

namespace GpuProfile {

enum Op : uint8_t {
	OpDraw,
	OpDrawIndexed,
	OpDrawIndirect,
	OpDrawIndexedIndirect,
	OpMesh,
	OpDispatch,
	OpDispatchIndirect,
	OpCommandBuffer,
	OpCount
};
enum Pipe : uint8_t { PipeUnknown, PipeGfxNormal, PipeGfxNative, PipeGfxTable, PipeCsNormal, PipeCsTable, PipeCount };

inline const char* OpName(uint8_t op) {
	static constexpr const char* names[] = {"draw",     "draw_indexed", "draw_indirect",     "draw_indexed_indirect",
	                                        "mesh",     "dispatch",     "dispatch_indirect", "cb"};
	return op < OpCount ? names[op] : "?";
}
inline const char* PipeName(uint8_t pipe) {
	static constexpr const char* names[] = {"unk", "gfx-normal", "gfx-native", "gfx-table", "cs-normal", "cs-table"};
	return pipe < PipeCount ? names[pipe] : "?";
}

struct PipeInfo {
	uint64_t vs       = 0; // graphics: the vertex shader's hash
	uint64_t main     = 0; // graphics: the pixel shader's (0: none); compute: the compute shader's; unknown: the handle
	uint8_t  pipe     = PipeUnknown;
	uint8_t  wave     = 0; // compute: 32 or 64
	uint8_t  emulated = 0; // compute: a wave64 program run on 32-lane subgroups
};

inline bool Requested() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GPU_PROFILE");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

// Pipelines noted when made (shaders.cpp), whatever the profiler's state: a bind looks them up.
struct Pipelines {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, PipeInfo> infos;
};
inline Pipelines g_pipelines;
inline uint64_t  HandleKey(VkPipeline pipeline) {
    return (uint64_t)(pipeline); // NOLINT (a pointer or an integer, by platform)
}
inline void NotePipeline(VkPipeline pipeline, uint8_t pipe, uint64_t vs, uint64_t main, uint32_t wave = 0,
                         bool emulated = false) {
	if (!Requested() || pipeline == VK_NULL_HANDLE) return;
	PipeInfo info;
	info.vs       = vs;
	info.main     = main;
	info.pipe     = pipe;
	info.wave     = static_cast<uint8_t>(wave);
	info.emulated = emulated ? 1 : 0;
	std::lock_guard lock(g_pipelines.mutex);
	g_pipelines.infos[HandleKey(pipeline)] = info;
}
inline PipeInfo LookupPipeline(VkPipeline pipeline) {
	{
		std::lock_guard lock(g_pipelines.mutex);
		const auto      found = g_pipelines.infos.find(HandleKey(pipeline));
		if (found != g_pipelines.infos.end()) return found->second;
	}
	PipeInfo info;
	info.main = HandleKey(pipeline);
	return info;
}

constexpr uint32_t Pairs      = 32768; // query pairs (65536 timestamps)
constexpr uint32_t BatchSlots = 16;

struct Rec {
	uint32_t pair     = 0;
	uint8_t  op       = 0;
	uint8_t  pipe     = 0;
	uint8_t  wave     = 0;
	uint8_t  emulated = 0;
	uint64_t vs = 0, main = 0;
	uint32_t a = 0, b = 0, c = 0; // target width/height, or group counts
	uint64_t work = 0;            // draws: vertices x instances
};

struct Batch {
	std::mutex            mutex;
	std::vector<Rec>      recs;
	std::atomic<uint32_t> gen {0};
	std::atomic<uint32_t> begun {0}, ended {0};
};

struct CbState {
	Batch*   batch      = nullptr;
	uint32_t gen        = 0;
	uint32_t cb_pair    = UINT32_MAX;
	uint32_t width      = 0, height = 0;
	bool     no_ts      = false; // inside a multiview or secondary-contents rendering scope
	bool     timestamps = true;  // its pool's queue family has timestamps
	PipeInfo gfx, cs;
};

struct Real {
	PFN_vkCmdBindPipeline         bind                  = nullptr;
	PFN_vkCmdDraw                 draw                  = nullptr;
	PFN_vkCmdDrawIndexed          draw_indexed          = nullptr;
	PFN_vkCmdDrawIndirect         draw_indirect         = nullptr;
	PFN_vkCmdDrawIndexedIndirect  draw_indexed_indirect = nullptr;
	PFN_vkCmdDrawMeshTasksEXT     mesh                  = nullptr;
	PFN_vkCmdDispatch             dispatch              = nullptr;
	PFN_vkCmdDispatchIndirect     dispatch_indirect     = nullptr;
	PFN_vkBeginCommandBuffer      begin                 = nullptr;
	PFN_vkEndCommandBuffer        end                   = nullptr;
	PFN_vkCmdBeginRendering       begin_rendering       = nullptr;
	PFN_vkCmdEndRendering         end_rendering         = nullptr;
	PFN_vkCreateCommandPool       create_pool           = nullptr;
	PFN_vkAllocateCommandBuffers  allocate              = nullptr;
	PFN_vkCmdWriteTimestamp       timestamp             = nullptr;
	PFN_vkGetQueryPoolResults     results               = nullptr;
	PFN_vkResetQueryPool          reset                 = nullptr;
};

struct Key {
	uint8_t  op = 0, pipe = 0, wave = 0, emulated = 0;
	uint64_t vs = 0, main = 0;
	uint32_t a = 0, b = 0, c = 0;
	bool     operator==(const Key& other) const {
        return op == other.op && pipe == other.pipe && wave == other.wave && emulated == other.emulated &&
               vs == other.vs && main == other.main && a == other.a && b == other.b && c == other.c;
	}
};
struct KeyHash {
	size_t operator()(const Key& k) const {
		uint64_t h = k.main * 0x9e3779b97f4a7c15ull ^ (k.vs + 0x632be59bd9b4e019ull);
		h ^= (uint64_t {k.a} << 32u | k.b) * 0xbf58476d1ce4e5b9ull;
		h ^= (uint64_t {k.c} << 32u | uint64_t {k.op} << 24u | uint64_t {k.pipe} << 16u | uint64_t {k.wave} << 8u |
		      k.emulated) *
		     0x94d049bb133111ebull;
		return static_cast<size_t>(h ^ (h >> 31u));
	}
};
struct Agg {
	uint64_t calls = 0, ns = 0, max_ns = 0, work = 0;
};
struct Totals {
	uint64_t frames = 0, cbs = 0, calls = 0, attributed_ns = 0, draw_ns = 0, dispatch_ns = 0, busy_ns = 0, cb_sum_ns = 0,
	         span_ns = 0, lost = 0;
};

struct State {
	VkDevice                                          device = VK_NULL_HANDLE;
	VkQueryPool                                       pool   = VK_NULL_HANDLE;
	Real                                              real;
	double                                            period = 1.0;
	uint64_t                                          mask   = ~uint64_t {0};
	uint32_t                                          every = 30, seconds = 10, top = 25;
	const std::atomic<uint64_t>*                      flips = nullptr;
	std::vector<uint32_t>                             family_bits;
	uint32_t                                          graphics_family = 0;
	std::atomic<uint64_t>                             next_pair {0};
	std::unique_ptr<std::atomic<uint8_t>[]>           pair_busy;
	std::atomic<uint64_t>                             dropped {0};
	std::array<Batch, BatchSlots>                     batches;
	std::atomic<Batch*>                               open {nullptr};
	std::mutex                                        cb_mutex;
	std::unordered_map<VkCommandBuffer, CbState>      cbs;
	std::unordered_map<uint64_t, uint32_t>            pool_family; // under cb_mutex
	std::mutex                                        orphan_mutex;
	std::vector<std::pair<uint32_t, uint64_t>>        orphans; // (pair, ms): reset once surely done or never run
	std::atomic<uint64_t>                             orphaned {0};
	std::atomic<bool>                                 stopped {false};
	std::jthread                                      reader;
};
inline std::atomic<State*> g_state {nullptr};

inline CbState& StateOf(State& s, VkCommandBuffer command) {
	// (A buffer is recorded by one thread at a time; the map's nodes stay where they are.)
	thread_local VkCommandBuffer t_command = VK_NULL_HANDLE;
	thread_local CbState*        t_cb      = nullptr;
	if (command != t_command || t_cb == nullptr) {
		std::lock_guard lock(s.cb_mutex);
		t_cb      = &s.cbs[command];
		t_command = command;
	}
	return *t_cb;
}
inline bool Live(const CbState& cb) {
	return cb.batch != nullptr && cb.batch->gen.load(std::memory_order_relaxed) == cb.gen;
}
inline uint32_t AllocPair(State& s) {
	const uint64_t seq      = s.next_pair.fetch_add(1, std::memory_order_relaxed);
	const auto     pair     = static_cast<uint32_t>(seq % Pairs);
	uint8_t        expected = 0;
	if (!s.pair_busy[pair].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
		s.dropped.fetch_add(1, std::memory_order_relaxed);
		return UINT32_MAX;
	}
	return pair;
}
inline void Orphan(State& s, uint32_t pair) {
	s.orphaned.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(s.orphan_mutex);
	s.orphans.emplace_back(pair, Desport::ElapsedMilliseconds());
}
inline void Append(State& s, const CbState& cb, const Rec& rec) {
	Batch&          batch = *cb.batch;
	std::lock_guard lock(batch.mutex);
	if (batch.gen.load(std::memory_order_relaxed) != cb.gen) {
		Orphan(s, rec.pair); // (the window was taken already)
		return;
	}
	batch.recs.push_back(rec);
}
inline void Stamp(State& s, VkCommandBuffer command, uint32_t query) {
	s.real.timestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, query);
}

template <typename Call>
inline void Measure(VkCommandBuffer command, uint8_t op, uint32_t a, uint32_t b, uint32_t c, uint64_t work, Call&& call) {
	State&   s  = *g_state.load(std::memory_order_acquire);
	CbState& cb = StateOf(s, command);
	if (!Live(cb) || cb.no_ts) {
		call();
		return;
	}
	const uint32_t pair = AllocPair(s);
	if (pair == UINT32_MAX) {
		call();
		return;
	}
	Stamp(s, command, pair * 2u);
	call();
	Stamp(s, command, pair * 2u + 1u);
	const bool      compute = op == OpDispatch || op == OpDispatchIndirect;
	const PipeInfo& info    = compute ? cb.cs : cb.gfx;
	Rec             rec;
	rec.pair     = pair;
	rec.op       = op;
	rec.pipe     = info.pipe;
	rec.wave     = info.wave;
	rec.emulated = info.emulated;
	rec.vs       = compute ? 0 : info.vs;
	rec.main     = info.main;
	rec.a        = compute ? a : cb.width;
	rec.b        = compute ? b : cb.height;
	rec.c        = compute ? c : 0;
	rec.work     = work;
	Append(s, cb, rec);
}

inline VKAPI_ATTR void VKAPI_CALL HookDraw(VkCommandBuffer command, uint32_t vertices, uint32_t instances,
                                          uint32_t first_vertex, uint32_t first_instance) {
	Measure(command, OpDraw, 0, 0, 0, uint64_t {vertices} * instances,
	        [&] { g_state.load(std::memory_order_relaxed)->real.draw(command, vertices, instances, first_vertex, first_instance); });
}
inline VKAPI_ATTR void VKAPI_CALL HookDrawIndexed(VkCommandBuffer command, uint32_t indices, uint32_t instances,
                                                 uint32_t first_index, int32_t vertex_offset, uint32_t first_instance) {
	Measure(command, OpDrawIndexed, 0, 0, 0, uint64_t {indices} * instances, [&] {
		g_state.load(std::memory_order_relaxed)
		    ->real.draw_indexed(command, indices, instances, first_index, vertex_offset, first_instance);
	});
}
inline VKAPI_ATTR void VKAPI_CALL HookDrawIndirect(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset,
                                                  uint32_t count, uint32_t stride) {
	Measure(command, OpDrawIndirect, 0, 0, 0, 0,
	        [&] { g_state.load(std::memory_order_relaxed)->real.draw_indirect(command, buffer, offset, count, stride); });
}
inline VKAPI_ATTR void VKAPI_CALL HookDrawIndexedIndirect(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset,
                                                         uint32_t count, uint32_t stride) {
	Measure(command, OpDrawIndexedIndirect, 0, 0, 0, 0, [&] {
		g_state.load(std::memory_order_relaxed)->real.draw_indexed_indirect(command, buffer, offset, count, stride);
	});
}
inline VKAPI_ATTR void VKAPI_CALL HookMesh(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z) {
	Measure(command, OpMesh, 0, 0, 0, uint64_t {x} * y * z,
	        [&] { g_state.load(std::memory_order_relaxed)->real.mesh(command, x, y, z); });
}
inline VKAPI_ATTR void VKAPI_CALL HookDispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z) {
	Measure(command, OpDispatch, x, y, z, uint64_t {x} * y * z,
	        [&] { g_state.load(std::memory_order_relaxed)->real.dispatch(command, x, y, z); });
}
inline VKAPI_ATTR void VKAPI_CALL HookDispatchIndirect(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset) {
	Measure(command, OpDispatchIndirect, 0, 0, 0, 0,
	        [&] { g_state.load(std::memory_order_relaxed)->real.dispatch_indirect(command, buffer, offset); });
}

inline VKAPI_ATTR void VKAPI_CALL HookBind(VkCommandBuffer command, VkPipelineBindPoint point, VkPipeline pipeline) {
	State& s = *g_state.load(std::memory_order_acquire);
	s.real.bind(command, point, pipeline);
	CbState& cb = StateOf(s, command);
	if (cb.batch == nullptr) return;
	const PipeInfo info = LookupPipeline(pipeline);
	if (point == VK_PIPELINE_BIND_POINT_COMPUTE) cb.cs = info;
	else if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) cb.gfx = info;
}

inline VKAPI_ATTR void VKAPI_CALL HookBeginRendering(VkCommandBuffer command, const VkRenderingInfo* info) {
	State& s = *g_state.load(std::memory_order_acquire);
	s.real.begin_rendering(command, info);
	CbState& cb = StateOf(s, command);
	if (cb.batch == nullptr || info == nullptr) return;
	cb.width  = info->renderArea.extent.width;
	cb.height = info->renderArea.extent.height;
	// (A multiview scope writes one timestamp per view; a secondary-contents one takes no timestamp at all.)
	cb.no_ts = info->viewMask != 0 || (info->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT) != 0;
}
inline VKAPI_ATTR void VKAPI_CALL HookEndRendering(VkCommandBuffer command) {
	State& s = *g_state.load(std::memory_order_acquire);
	s.real.end_rendering(command);
	CbState& cb = StateOf(s, command);
	cb.no_ts    = false;
}


inline VKAPI_ATTR VkResult VKAPI_CALL HookBegin(VkCommandBuffer command, const VkCommandBufferBeginInfo* info) {
	State&         s      = *g_state.load(std::memory_order_acquire);
	const VkResult result = s.real.begin(command, info);
	CbState&       cb     = StateOf(s, command);
	cb.batch              = nullptr;
	cb.cb_pair            = UINT32_MAX;
	cb.width              = 0;
	cb.height             = 0;
	cb.no_ts              = false;
	cb.gfx                = {};
	cb.cs                 = {};
	if (result != VK_SUCCESS || !cb.timestamps || s.stopped.load(std::memory_order_relaxed)) return result;
	// (A secondary buffer continuing a render pass: not ours to time.)
	if (info != nullptr && (info->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT) != 0) return result;
	Batch* batch = s.open.load(std::memory_order_acquire);
	if (batch == nullptr) return result;
	cb.gen   = batch->gen.load(std::memory_order_acquire);
	cb.batch = batch;
	batch->begun.fetch_add(1, std::memory_order_acq_rel);
	cb.cb_pair = AllocPair(s);
	if (cb.cb_pair != UINT32_MAX) Stamp(s, command, cb.cb_pair * 2u);
	return result;
}

inline VKAPI_ATTR VkResult VKAPI_CALL HookEnd(VkCommandBuffer command) {
	State&   s  = *g_state.load(std::memory_order_acquire);
	CbState& cb = StateOf(s, command);
	if (cb.batch != nullptr) {
		const bool live = Live(cb);
		if (cb.cb_pair != UINT32_MAX) {
			Stamp(s, command, cb.cb_pair * 2u + 1u);
			if (live) {
				Rec rec;
				rec.pair = cb.cb_pair;
				rec.op   = OpCommandBuffer;
				Append(s, cb, rec);
			} else {
				Orphan(s, cb.cb_pair);
			}
		}
		if (live) cb.batch->ended.fetch_add(1, std::memory_order_acq_rel);
		cb.batch   = nullptr;
		cb.cb_pair = UINT32_MAX;
	}
	return s.real.end(command);
}

inline VKAPI_ATTR VkResult VKAPI_CALL HookCreatePool(VkDevice device, const VkCommandPoolCreateInfo* info,
                                                     const VkAllocationCallbacks* allocator, VkCommandPool* pool) {
	State&         s      = *g_state.load(std::memory_order_acquire);
	const VkResult result = s.real.create_pool(device, info, allocator, pool);
	if (result == VK_SUCCESS && info != nullptr && pool != nullptr) {
		std::lock_guard lock(s.cb_mutex);
		s.pool_family[(uint64_t)(*pool)] = info->queueFamilyIndex; // NOLINT
	}
	return result;
}
inline VKAPI_ATTR VkResult VKAPI_CALL HookAllocate(VkDevice device, const VkCommandBufferAllocateInfo* info,
                                                   VkCommandBuffer* commands) {
	State&         s      = *g_state.load(std::memory_order_acquire);
	const VkResult result = s.real.allocate(device, info, commands);
	if (result != VK_SUCCESS || info == nullptr || commands == nullptr) return result;
	std::lock_guard lock(s.cb_mutex);
	const auto      found  = s.pool_family.find((uint64_t)(info->commandPool)); // NOLINT
	const uint32_t  family = found != s.pool_family.end() ? found->second : s.graphics_family;
	const bool      ok     = family < s.family_bits.size() && s.family_bits[family] != 0;
	for (uint32_t i = 0; i < info->commandBufferCount; ++i) {
		auto& cb      = s.cbs[commands[i]];
		cb.timestamps = ok;
		cb.batch      = nullptr;
		cb.cb_pair    = UINT32_MAX;
	}
	return result;
}

// The reader's side.
struct Window {
	std::unordered_map<Key, Agg, KeyHash> aggs;
	Totals                                totals;
};
struct Taking {
	uint32_t                             slot      = 0;
	uint64_t                             closed_ms = 0;
	bool                                 taken     = false;
	std::vector<Rec>                     recs;
	std::vector<std::array<uint64_t, 2>> times;
	std::vector<uint8_t>                 done;
	size_t                               first = 0; // the first not done
};

inline void Release(State& s, uint32_t pair) {
	s.real.reset(s.device, s.pool, pair * 2u, 2u);
	s.pair_busy[pair].store(0, std::memory_order_release);
}

// The window's records taken and read: aggregated into `w`.
inline void Aggregate(State& s, Taking& t, Window& w) {
	const auto ns = [&](uint64_t value) { return static_cast<uint64_t>(static_cast<double>(value & s.mask) * s.period); };
	std::vector<std::pair<uint64_t, uint64_t>> intervals;
	for (size_t i = 0; i < t.recs.size(); ++i) {
		const Rec& rec = t.recs[i];
		if (t.done[i] == 0) {
			++w.totals.lost;
			Orphan(s, rec.pair);
			continue;
		}
		Release(s, rec.pair);
		const uint64_t begin = ns(t.times[i][0]), end = ns(t.times[i][1]);
		if (end < begin) continue; // (a wrapped counter)
		const uint64_t duration = end - begin;
		if (rec.op == OpCommandBuffer) {
			++w.totals.cbs;
			w.totals.cb_sum_ns += duration;
			intervals.emplace_back(begin, end);
			continue;
		}
		Key key;
		key.op       = rec.op;
		key.pipe     = rec.pipe;
		key.wave     = rec.wave;
		key.emulated = rec.emulated;
		key.vs       = rec.vs;
		key.main     = rec.main;
		key.a        = rec.a;
		key.b        = rec.b;
		key.c        = rec.c;
		auto& agg    = w.aggs[key];
		++agg.calls;
		agg.ns += duration;
		agg.max_ns = (std::max)(agg.max_ns, duration);
		agg.work += rec.work;
		++w.totals.calls;
		w.totals.attributed_ns += duration;
		if (rec.op == OpDispatch || rec.op == OpDispatchIndirect) w.totals.dispatch_ns += duration;
		else w.totals.draw_ns += duration;
	}
	std::sort(intervals.begin(), intervals.end());
	uint64_t covered = 0, reach = 0;
	for (const auto& [begin, end]: intervals) {
		const uint64_t from = (std::max)(begin, reach);
		if (end > from) covered += end - from;
		reach = (std::max)(reach, end);
	}
	w.totals.busy_ns += covered;
	if (!intervals.empty()) w.totals.span_ns += reach - intervals.front().first;
	++w.totals.frames;
}

// True once the window is read (or given up on) and aggregated.
inline bool TryFinish(State& s, Taking& t, Window& w, uint64_t now) {
	Batch& batch = s.batches[t.slot];
	if (!t.taken) {
		const bool ended = batch.ended.load(std::memory_order_acquire) >= batch.begun.load(std::memory_order_acquire);
		if (!ended && now - t.closed_ms < 2000) return false;
		{
			std::lock_guard lock(batch.mutex);
			batch.gen.fetch_add(1, std::memory_order_acq_rel); // (later appends are orphans)
			t.recs.swap(batch.recs);
			batch.recs.clear();
		}
		t.taken = true;
		t.times.assign(t.recs.size(), {});
		t.done.assign(t.recs.size(), 0);
		t.first = 0;
	}
	bool all = true;
	for (size_t i = t.first; i < t.recs.size(); ++i) {
		if (t.done[i] != 0) continue;
		uint64_t       data[4] {};
		const VkResult result =
		    s.real.results(s.device, s.pool, t.recs[i].pair * 2u, 2u, sizeof(data), data, 2u * sizeof(uint64_t),
		                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
		if ((result != VK_SUCCESS && result != VK_NOT_READY) || data[1] == 0 || data[3] == 0) {
			if (all) t.first = i;
			all = false;
			continue;
		}
		t.times[i] = {data[0], data[2]};
		t.done[i]  = 1;
	}
	if (!all && now - t.closed_ms < 3000) return false;
	Aggregate(s, t, w);
	t.recs.clear();
	t.taken = false;
	return true;
}

inline void Report(State& s, Window& w, uint64_t now, uint64_t window_ms, uint64_t dropped, uint64_t orphans) {
	const auto&  t      = w.totals;
	const double frames = t.frames != 0 ? static_cast<double>(t.frames) : 1.0;
	const double busy   = t.busy_ns != 0 ? static_cast<double>(t.busy_ns) : 1.0;
	Desport::EmitLine(fmt::format(
	    "DES_GPU_PROF total t={} window_ms={} frames={} every={} cbs={} calls={} keys={} attributed_ms={:.3f} draw_ms={:.3f} "
	    "dispatch_ms={:.3f} busy_ms={:.3f} cb_sum_ms={:.3f} span_ms={:.3f} frame_attributed_ms={:.3f} frame_busy_ms={:.3f} "
	    "frame_span_ms={:.3f} coverage_pct={:.1f} dropped={} lost={} orphans={}\n",
	    now, window_ms, t.frames, s.every, t.cbs, t.calls, w.aggs.size(), static_cast<double>(t.attributed_ns) / 1e6,
	    static_cast<double>(t.draw_ns) / 1e6, static_cast<double>(t.dispatch_ns) / 1e6, static_cast<double>(t.busy_ns) / 1e6,
	    static_cast<double>(t.cb_sum_ns) / 1e6, static_cast<double>(t.span_ns) / 1e6,
	    static_cast<double>(t.attributed_ns) / 1e6 / frames, static_cast<double>(t.busy_ns) / 1e6 / frames,
	    static_cast<double>(t.span_ns) / 1e6 / frames,
	    t.busy_ns != 0 ? 100.0 * static_cast<double>(t.attributed_ns) / busy : 0.0, dropped, t.lost, orphans));
	std::vector<std::pair<const Key*, const Agg*>> order;
	order.reserve(w.aggs.size());
	for (const auto& [key, agg]: w.aggs) order.emplace_back(&key, &agg);
	const size_t count = (std::min)(order.size(), static_cast<size_t>(s.top));
	std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), order.end(),
	                  [](const auto& x, const auto& y) { return x.second->ns > y.second->ns; });
	for (size_t i = 0; i < count; ++i) {
		const Key& key     = *order[i].first;
		const Agg& agg     = *order[i].second;
		const bool compute = key.op == OpDispatch || key.op == OpDispatchIndirect;
		std::string shape;
		if (key.op == OpDispatchIndirect) shape = "groups=indirect";
		else if (compute) shape = fmt::format("groups={}x{}x{}", key.a, key.b, key.c);
		else shape = fmt::format("target={}x{}", key.a, key.b);
		std::string extra;
		if (compute) extra = key.wave == 0 ? " wave=-" : fmt::format(" wave={}{}", key.wave, key.emulated != 0 ? "e" : "");
		else extra = fmt::format(" vs={:016x} verts={:.0f}", key.vs,
		                         static_cast<double>(agg.work) / static_cast<double>(agg.calls));
		Desport::EmitLine(fmt::format(
		    "DES_GPU_PROF rank={} t={} kind={} pipe={} hash={:016x} {} calls={} calls_per_frame={:.1f} gpu_ms={:.3f} "
		    "ms_per_frame={:.3f} pct={:.2f} avg_us={:.2f} max_us={:.1f}{}\n",
		    i + 1, now, OpName(key.op), PipeName(key.pipe), key.main, shape, agg.calls,
		    static_cast<double>(agg.calls) / frames, static_cast<double>(agg.ns) / 1e6,
		    static_cast<double>(agg.ns) / 1e6 / frames, 100.0 * static_cast<double>(agg.ns) / busy,
		    static_cast<double>(agg.ns) / 1e3 / static_cast<double>(agg.calls), static_cast<double>(agg.max_ns) / 1e3,
		    extra));
	}
}

inline void ReaderLoop(const std::stop_token& stop, State& s) {
	LocalPlatform::SetThreadName("Kyty.GpuProf");
	auto           window       = std::make_unique<Window>();
	uint64_t       window_start = Desport::ElapsedMilliseconds();
	uint64_t       last_flips   = s.flips->load(std::memory_order_relaxed);
	uint64_t       open_flip    = 0;
	int            open_slot    = -1;
	uint32_t       next_slot    = 0;
	uint64_t       dropped_seen = 0, orphans_seen = 0;
	std::array<bool, BatchSlots> in_use {};
	std::deque<Taking>           closed;
	while (!stop.stop_requested()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		const uint64_t now   = Desport::ElapsedMilliseconds();
		const uint64_t flips = s.flips->load(std::memory_order_acquire);
		if (open_slot >= 0) {
			if (flips != open_flip) { // the sampled frame ends at the next flip
				s.open.store(nullptr, std::memory_order_release);
				Taking taking;
				taking.slot      = static_cast<uint32_t>(open_slot);
				taking.closed_ms = now;
				closed.push_back(std::move(taking));
				open_slot = -1;
			}
		} else if (flips != last_flips && flips % s.every == 0 && closed.size() < BatchSlots - 1) {
			for (uint32_t i = 0; i < BatchSlots; ++i) {
				const uint32_t slot = (next_slot + i) % BatchSlots;
				if (in_use[slot]) continue;
				Batch& batch = s.batches[slot];
				{
					std::lock_guard lock(batch.mutex);
					for (const auto& rec: batch.recs) Orphan(s, rec.pair); // (late appends of an earlier use)
					batch.recs.clear();
					batch.begun.store(0, std::memory_order_relaxed);
					batch.ended.store(0, std::memory_order_relaxed);
					batch.gen.fetch_add(1, std::memory_order_acq_rel);
				}
				in_use[slot] = true;
				open_slot    = static_cast<int>(slot);
				open_flip    = flips;
				next_slot    = slot + 1;
				s.open.store(&batch, std::memory_order_release);
				break;
			}
		}
		last_flips = flips;
		while (!closed.empty() && TryFinish(s, closed.front(), *window, now)) {
			in_use[closed.front().slot] = false;
			closed.pop_front();
		}
		{ // Pairs whose window was given up on: surely done (or never run) 5 s later.
			std::lock_guard lock(s.orphan_mutex);
			auto            keep = s.orphans.begin();
			for (auto it = s.orphans.begin(); it != s.orphans.end(); ++it) {
				if (now - it->second >= 5000) Release(s, it->first);
				else *keep++ = *it;
			}
			s.orphans.erase(keep, s.orphans.end());
		}
		if (now - window_start < uint64_t {s.seconds} * 1000u) continue;
		const uint64_t dropped = s.dropped.load(std::memory_order_relaxed);
		const uint64_t orphans = s.orphaned.load(std::memory_order_relaxed);
		if (window->totals.frames != 0) Report(s, *window, now, now - window_start, dropped - dropped_seen, orphans - orphans_seen);
		window       = std::make_unique<Window>();
		window_start = now;
		dropped_seen = dropped;
		orphans_seen = orphans;
	}
	s.open.store(nullptr, std::memory_order_release);
}

inline uint32_t Number(const char* name, uint32_t fallback, uint32_t low, uint32_t high) {
	const char* text = std::getenv(name);
	if (text == nullptr || *text == '\0') return fallback;
	char*      end   = nullptr;
	const auto value = std::strtoul(text, &end, 10);
	if (*end != '\0' || value < low || value > high) {
		std::printf("DES_GPU_PROF %s=%s ignored (%u to %u), using %u\n", name, text, low, high, fallback);
		return fallback;
	}
	return static_cast<uint32_t>(value);
}

// After LocalVulkanRecording::Install and the device-fault bind hook, before any command pool or command: `dispatch` is
// the table every command ends in (LocalVulkanRecording::HookDispatch()). Logs "DES_GPU_PROF on=<0|1> ..." either way.
inline void Install(vk::detail::DispatchLoaderDynamic& dispatch, vk::PhysicalDevice physical, VkDevice device,
                    uint32_t graphics_family, const std::atomic<uint64_t>* flips) {
	if (!Requested()) {
		std::printf("DES_GPU_PROF on=0\n");
		std::fflush(stdout);
		return;
	}
	if (g_state.load(std::memory_order_acquire) != nullptr) return;
	auto* s = new State();
	for (const auto& family: physical.getQueueFamilyProperties()) s->family_bits.push_back(family.timestampValidBits);
	const uint32_t bits = graphics_family < s->family_bits.size() ? s->family_bits[graphics_family] : 0u;
	auto&          r    = s->real;
	r.bind                  = dispatch.vkCmdBindPipeline;
	r.draw                  = dispatch.vkCmdDraw;
	r.draw_indexed          = dispatch.vkCmdDrawIndexed;
	r.draw_indirect         = dispatch.vkCmdDrawIndirect;
	r.draw_indexed_indirect = dispatch.vkCmdDrawIndexedIndirect;
	r.mesh                  = dispatch.vkCmdDrawMeshTasksEXT;
	r.dispatch              = dispatch.vkCmdDispatch;
	r.dispatch_indirect     = dispatch.vkCmdDispatchIndirect;
	r.begin                 = dispatch.vkBeginCommandBuffer;
	r.end                   = dispatch.vkEndCommandBuffer;
	r.begin_rendering       = dispatch.vkCmdBeginRendering;
	r.end_rendering         = dispatch.vkCmdEndRendering;
	r.create_pool           = dispatch.vkCreateCommandPool;
	r.allocate              = dispatch.vkAllocateCommandBuffers;
	r.timestamp             = dispatch.vkCmdWriteTimestamp;
	r.results               = dispatch.vkGetQueryPoolResults;
	r.reset                 = dispatch.vkResetQueryPool;
	const char* missing     = nullptr;
	if (bits == 0) missing = "no_timestamps_on_graphics_family";
	else if (r.bind == nullptr || r.draw == nullptr || r.draw_indexed == nullptr || r.draw_indirect == nullptr ||
	         r.draw_indexed_indirect == nullptr || r.dispatch == nullptr || r.dispatch_indirect == nullptr ||
	         r.begin == nullptr || r.end == nullptr || r.begin_rendering == nullptr || r.end_rendering == nullptr ||
	         r.create_pool == nullptr || r.allocate == nullptr || r.timestamp == nullptr || r.results == nullptr ||
	         r.reset == nullptr)
		missing = "missing_entry_point";
	VkQueryPoolCreateInfo query {};
	query.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	query.queryCount = Pairs * 2u;
	if (missing == nullptr && dispatch.vkCreateQueryPool(device, &query, nullptr, &s->pool) != VK_SUCCESS)
		missing = "create_failed";
	if (missing != nullptr) {
		std::printf("DES_GPU_PROF on=0 reason=%s\n", missing);
		std::fflush(stdout);
		delete s;
		return;
	}
	r.reset(device, s->pool, 0, Pairs * 2u); // (hostQueryReset: vulkanWindow.cpp)
	s->device          = device;
	s->graphics_family = graphics_family;
	s->period          = static_cast<double>(physical.getProperties().limits.timestampPeriod);
	s->mask            = bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << bits) - 1u;
	s->every           = Number("KYTY_GPU_PROFILE_EVERY", 30, 1, 100000);
	s->seconds         = Number("KYTY_GPU_PROFILE_SECONDS", 10, 1, 3600);
	s->top             = Number("KYTY_GPU_PROFILE_TOP", 25, 1, 1000);
	s->flips           = flips;
	s->pair_busy       = std::make_unique<std::atomic<uint8_t>[]>(Pairs);
	for (uint32_t i = 0; i < Pairs; ++i) s->pair_busy[i].store(0, std::memory_order_relaxed);
	g_state.store(s, std::memory_order_release);
	dispatch.vkCmdBindPipeline        = HookBind;
	dispatch.vkCmdDraw                = HookDraw;
	dispatch.vkCmdDrawIndexed         = HookDrawIndexed;
	dispatch.vkCmdDrawIndirect        = HookDrawIndirect;
	dispatch.vkCmdDrawIndexedIndirect = HookDrawIndexedIndirect;
	if (r.mesh != nullptr) dispatch.vkCmdDrawMeshTasksEXT = HookMesh;
	dispatch.vkCmdDispatch            = HookDispatch;
	dispatch.vkCmdDispatchIndirect    = HookDispatchIndirect;
	dispatch.vkBeginCommandBuffer     = HookBegin;
	dispatch.vkEndCommandBuffer       = HookEnd;
	dispatch.vkCmdBeginRendering      = HookBeginRendering;
	dispatch.vkCmdEndRendering        = HookEndRendering;
	dispatch.vkCreateCommandPool      = HookCreatePool;
	dispatch.vkAllocateCommandBuffers = HookAllocate;
	s->reader = std::jthread([s](const std::stop_token& stop) { ReaderLoop(stop, *s); });
	std::printf("DES_GPU_PROF on=1 every=%u seconds=%u top=%u pairs=%u period_ns=%.3f valid_bits=%u mesh=%u\n", s->every,
	            s->seconds, s->top, Pairs, s->period, bits, r.mesh != nullptr ? 1u : 0u);
	std::fflush(stdout);
}

// A scheduler shuts down: the reader stops before the device goes away; no window opens any more.
inline void Stop() {
	auto* s = g_state.load(std::memory_order_acquire);
	if (s == nullptr || s->stopped.exchange(true, std::memory_order_acq_rel)) return;
	s->reader.request_stop();
	if (s->reader.joinable()) s->reader.join();
}

} // namespace GpuProfile
