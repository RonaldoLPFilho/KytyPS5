#pragma once
// desport: KYTY_ASYNC_COMPUTE. The guest's compute-ring dispatches on a Vulkan queue of their own, so that a guest read of
// their results does not wait behind the graphics work submitted before them (DES_READBACK_LAT, perf-ab LAT1: a read
// waited ~2.7 ms on the GPU with ~23 submissions ahead of it; every guest queue went through one VkQueue).
//   0 (default)  nothing changes; no queue is created.
//   1            queue 0 of a compute-only family (RTX 5080: family 2), created with the device (vulkanWindow.cpp).
//   2            one more queue of the graphics family (for comparison: every command is legal there).
// KYTY_ASYNC_COMPUTE_UNSAFE=1 (diagnostic only): drops the label causality bound below. An upper bound of the gain;
// it may race (a guest that synchronized through a label the CPU saw early).
//
// Design (masterSemaphore.h, commandScheduler.h, renderCompute.cpp, bufferCache.cpp):
// - Epochs. The master timeline's tick stays the one fence of every cache and deferred operation: tick X is the epoch of
//   everything recorded while CurrentTick() == X. The graphics submission of X signals the graphics timeline with X, as
//   before; the async submission recorded during X (at most one) signals the async timeline with X. X is complete when
//   both are (MasterSemaphore::Refresh: min(graphics, oldest unfinished async epoch - 1)); Wait(X) waits for both.
// - Graphics after async: every graphics submission waits for the async timeline at the last async epoch (a total
//   order: whatever an async dispatch wrote, any later graphics command or cache operation sees).
// - Async after graphics: an async submission waits for the graphics timeline at its dependency epoch D < X only, the
//   largest of: the in-flight graphics GPU writes to the buffer ranges its dispatches bind (BufferCache write log:
//   shader writes, uploads, joins, image copies); the guest's sync: the last graphics label (CP memory write) written
//   before the guest enqueued the compute submission (causality: the guest can only have acted on labels written
//   before; KYTY_ASYNC_COMPUTE_UNSAFE drops this one) and the labels its WAIT_REG_MEMs passed on (their epochs:
//   g_labels); the same guest queue's earlier work that went to the graphics queue.
// - Eligibility: a dispatch goes to the async buffer only on the normal path (table dispatches are skipped on compute
//   rings), with no images, no GDS, no device-address reads, nothing recorded and no engine write noted while its
//   resources were prepared (uploads, joins, layout transitions: those went to the graphics buffer), the write log on
//   (KYTY_READBACK_QUEUE), no pending sparse-binding wait, speculation off, D below the open epoch. Else it is recorded
//   into the graphics buffer as before (a fallback, counted by reason in the DES_ASYNC_COMPUTE line).
// - Reads: a guest readback whose bytes' in-flight writers are all async copies on the transfer queue after the async
//   timeline (BufferCache::ReadbackQueueReady), even with KYTY_READBACK_QUEUE=1.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
}

namespace AsyncCompute {

// The environment, read once (device creation reads it before the GPU thread's switches are applied).
inline uint32_t RequestedMode() {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_ASYNC_COMPUTE");
		if (value == nullptr || value[0] == '\0') return 0u;
		const auto parsed = std::strtoul(value, nullptr, 10);
		return parsed == 1 || parsed == 2 ? static_cast<uint32_t>(parsed) : 0u;
	}();
	return mode;
#else
	return 0; // (needs the recording layer's work counters)
#endif
}
inline bool Unsafe() {
	static const bool unsafe = [] {
		const char* value = std::getenv("KYTY_ASYNC_COMPUTE_UNSAFE");
		return value != nullptr && value[0] == '1';
	}();
	return unsafe;
}

// Set once the command scheduler has its async queue (CommandScheduler::CommandScheduler): the master timeline's next
// tick, i.e. the open epoch (MasterSemaphore::m_current_tick).
inline std::atomic<bool>                          g_on {false};
inline std::atomic<const std::atomic<uint64_t>*> g_epoch {nullptr};
inline bool                                       On() {
    return g_on.load(std::memory_order_relaxed);
}
inline uint64_t CurrentEpoch() {
	const auto* epoch = g_epoch.load(std::memory_order_acquire);
	return epoch != nullptr ? epoch->load(std::memory_order_acquire) : 0;
}

// Counters (the DES_ASYNC_COMPUTE line every 5 s: Report, commandScheduler.cpp).
enum Id : uint32_t {
	Candidates,   // compute-ring dispatches that reached the normal path's resource preparation
	Async,        // ... recorded into the async buffer
	FbNoLog,      // fallbacks: the write log is off or not dated yet (KYTY_READBACK_QUEUE)
	FbImages,     // ... the dispatch binds images
	FbDma,        // ... device-address reads (uses_dma)
	FbGds,        // ... GDS
	FbWork,       // ... its preparation recorded commands (uploads, copies, barriers, transitions)
	FbNotes,      // ... its preparation noted engine writes (an upload prologue's copies, joins)
	FbSparse,     // ... a sparse binding the next submission must wait for
	FbTaa,        // ... the DLSS TAA dispatch
	FbDepOpen,    // ... its dependency is in the open graphics epoch (not submitted yet)
	FbEpoch,      // ... the epoch already had its async submission, or changed during the preparation
	TableSkipped, // compute-ring dispatches kept off the table path
	Submits,      // async submissions
	SubmitsWait,  // ... that wait for the graphics timeline
	DepBacklog,   // sum over those of: dependency epoch - completed graphics epoch at the submission
	DepLabel,     // the dependency was set by: the guest's label before the enqueue,
	DepWrites,    // in-flight graphics writes to the bound ranges,
	DepWait,      // a WAIT_REG_MEM's label,
	DepOwn,       // the same guest queue's graphics-queue work
	GfxWaits,     // graphics submissions that wait for the async timeline
	EpochFlushes, // graphics flushes made to give an async buffer a fresh epoch
	RbAsync,      // guest readbacks copied on the transfer queue after the async timeline
	RbAsyncNs,    // the guest thread's wait for those copies
	RbMixed,      // readbacks refused the transfer queue: in-flight writes of both queues
	Slices,       // compute slices translated while it is on
	Count
};
inline std::atomic<uint64_t> g_values[Count] {};
inline void                  Add(Id id, uint64_t n = 1) {
    g_values[id].fetch_add(n, std::memory_order_relaxed);
}
void Report(); // any thread; prints at most every 5 s

// Labels: every CP memory write (WRITE_DATA, end-of-pipe, ...: GuestGpu's WriteGuestMemory) with the graphics epoch that
// makes it true - the open epoch for the graphics processor's, the writer queue's last graphics-queue epoch for a
// compute processor's (0: all its work went to the async queue). GPU thread only, but g_gfx_label_epoch.
struct LabelEntry {
	uint64_t address = 0;
	uint64_t epoch   = 0;
};
constexpr uint32_t                      LabelBits = 12;
inline std::array<LabelEntry, 1u << LabelBits> g_labels {};
// The largest epoch an overwritten entry held: what a label the table no longer knows may need.
inline uint64_t              g_label_evicted = 0;
// The largest graphics epoch of a label written so far; read by guest threads at their enqueue.
inline std::atomic<uint64_t> g_gfx_label_epoch {0};
inline LabelEntry&           LabelSlot(uint64_t address) {
    return g_labels[static_cast<uint32_t>(((address >> 2u) * 0x9e3779b97f4a7c15ull) >> (64u - LabelBits))];
}
inline void NoteLabel(uint64_t address, uint64_t size, uint64_t epoch) {
	// (Published before the memory write: a guest that sees the label sees the epoch - stores stay in order on x86.)
	if (epoch != 0) {
		auto known = g_gfx_label_epoch.load(std::memory_order_relaxed);
		while (known < epoch && !g_gfx_label_epoch.compare_exchange_weak(known, epoch, std::memory_order_seq_cst)) {
		}
	}
	const uint64_t first = address & ~uint64_t {3};
	const uint64_t words = (address + size - first + 3) / 4;
	for (uint64_t i = 0; i < words; ++i) {
		if (i == 64) { // (a long write: its other words as unknown labels)
			g_label_evicted = std::max(g_label_evicted, epoch);
			break;
		}
		auto& entry = LabelSlot(first + i * 4);
		if (entry.address != first + i * 4) g_label_evicted = std::max(g_label_evicted, entry.epoch);
		entry = {first + i * 4, epoch};
	}
}
inline uint64_t LabelEpoch(uint64_t address, uint64_t size) {
	uint64_t       epoch = 0;
	const uint64_t first = address & ~uint64_t {3};
	for (uint64_t at = first; at < address + size; at += 4) {
		const auto& entry = LabelSlot(at);
		epoch             = std::max(epoch, entry.address == at ? entry.epoch : g_label_evicted);
	}
	return epoch;
}

// The compute slice the GPU thread translates (GuestGpu::Process): its guest queue, its dependency bounds, the work
// counter at its start and the work recorded into the async buffer since (the rest went to the graphics queue).
struct Slice {
	bool     active     = false;
	uint32_t queue      = 0;
	uint64_t label_dep  = 0;
	uint64_t wait_dep   = 0;
	uint64_t work_start = 0;
	uint64_t async_work = 0;
};
inline thread_local Slice t_slice;
// GPU thread: per guest queue, the last epoch its work went to the graphics queue.
inline std::array<uint64_t, 64> g_queue_gfx_epoch {};

inline uint64_t RecordedWork() {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	return LocalVulkanRecording::RecordedWork();
#else
	return 0;
#endif
}
// A compute-ring dispatch can go to the async buffer: on, in a compute slice, speculation off.
inline bool SliceActive() {
	return On() && t_slice.active && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0;
}
// GPU thread, compute slice: the epoch of the latest work this queue recorded on the graphics queue.
inline uint64_t QueueGfxEpoch() {
	auto&          slice = t_slice;
	const uint64_t work  = RecordedWork();
	if (work - slice.work_start > slice.async_work) {
		g_queue_gfx_epoch[slice.queue & 63u] = CurrentEpoch();
		slice.work_start                     = work;
		slice.async_work                     = 0;
	}
	return g_queue_gfx_epoch[slice.queue & 63u];
}
// GPU thread, a CP memory write (not a speculation's).
inline void NoteCpWrite(uint64_t address, uint64_t size) {
	if (!On()) return;
	NoteLabel(address, size, t_slice.active ? QueueGfxEpoch() : CurrentEpoch());
}
// GPU thread, a compute processor's WAIT_REG_MEM that passed.
inline void NoteWaitPassed(uint64_t address, uint64_t size) {
	if (!On() || !t_slice.active) return;
	t_slice.wait_dep = std::max(t_slice.wait_dep, LabelEpoch(address, size));
}
inline void BeginSlice(uint32_t queue, uint64_t label_dep) {
	t_slice = {.active     = true,
	           .queue      = queue,
	           .label_dep  = Unsafe() ? 0 : label_dep,
	           .wait_dep   = 0,
	           .work_start = RecordedWork(),
	           .async_work = 0};
	Add(Slices);
}
inline void EndSlice() {
	if (!t_slice.active) return;
	(void)QueueGfxEpoch();
	t_slice.active = false;
}

} // namespace AsyncCompute
