// SPDX-License-Identifier: GPL-2.0-only
// desport: Demon's Souls port layer (milestones, call tracing, game-specific HLE overrides).
#include "loader/desport.h"

#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Desport {

namespace {

const auto g_process_start = std::chrono::steady_clock::now();

uint64_t ElapsedMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now() - g_process_start)
	                                 .count());
}

std::mutex            g_milestone_mutex;
std::set<std::string> g_milestones;

} // namespace

void Milestone(std::string_view name, std::string_view detail) {
	{
		std::scoped_lock lock(g_milestone_mutex);
		if (!g_milestones.emplace(name).second) {
			return;
		}
	}
	Log::WriteToConsoleAndLog(
	    detail.empty() ? fmt::format("DES_MILESTONE {} t={}\n", name, ElapsedMs())
	                   : fmt::format("DES_MILESTONE {} t={} {}\n", name, ElapsedMs(), detail));
}

void FlipMilestone(bool from_gpu) {
	static std::atomic_uint64_t flips {0};
	const auto                  n = flips.fetch_add(1) + 1;
	if (n == 1 || n == 60 || n == 600 || n == 3600) {
		Milestone(fmt::format("flip_{}", n), from_gpu ? "source=gpu" : "source=cpu");
	}
	// Frame rate every 5 s (DES_FPS lines, read by tools/log_triage.py); the thread that wins the
	// exchange of the window start logs it.
	static std::atomic_uint64_t window_start_ms {0};
	static std::atomic_uint64_t window_start_flips {0};
	const auto                  now   = ElapsedMs();
	auto                        start = window_start_ms.load(std::memory_order_relaxed);
	if (start == 0) {
		window_start_ms.compare_exchange_strong(start, now);
		window_start_flips.store(n);
	} else if (now - start >= 5000 && window_start_ms.compare_exchange_strong(start, now)) {
		const auto frames = n - window_start_flips.exchange(n);
		Log::WriteToConsoleAndLog(
		    fmt::format("DES_FPS t={} fps={:.1f}\n", now, 1000.0 * static_cast<double>(frames) /
		                                                     static_cast<double>(now - start)));
		Stats::LogEngineCounters(now, frames);
	}
}

namespace Stats {

void LogEngineCounters(uint64_t now_ms, uint64_t frames) {
	using Common::DebugCounters::Counter;
	if (!Enabled()) {
		return;
	}
	if (!Common::DebugCounters::Enabled()) {
		// The upstream counters feed --perf-overlay; DES_STATS turns them on for the log too.
		Common::DebugCounters::SetEnabled(true);
		return;
	}
	static constexpr std::pair<Counter, const char*> Logged[] = {
	    {Counter::Draws, "draws"},
	    {Counter::Dispatches, "dispatches"},
	    {Counter::QueueSubmits, "queue_submits"},
	    {Counter::GpuWaits, "gpu_waits"},
	    {Counter::GpuWaitNs, "gpu_wait_ns"},
	    {Counter::GpuFaults, "gpu_faults"},
	    {Counter::BufferDownloadBytes, "buffer_download_bytes"},
	    {Counter::BufferUploadBytes, "buffer_upload_bytes"},
	    {Counter::TextureDownloadBytes, "texture_download_bytes"},
	    {Counter::TextureUploadBytes, "texture_upload_bytes"},
	    {Counter::ColorMetadataReadbacks, "color_metadata_readbacks"},
	    {Counter::ColorMetadataReadbackNs, "color_metadata_readback_ns"},
	    {Counter::ShadersCompiled, "shaders_compiled"},
	    {Counter::PipelinesCreated, "pipelines_created"},
	};
	static uint64_t previous[std::size(Logged)] {};
	std::string     line = fmt::format("DES_COUNTERS t={} frames={}", now_ms, frames);
	for (size_t i = 0; i < std::size(Logged); i++) {
		const auto value = Common::DebugCounters::Get(Logged[i].first);
		line += fmt::format(" {}={}", Logged[i].second, value - previous[i]);
		previous[i] = value;
	}
	Log::WriteToConsoleAndLog(line + "\n");
}

} // namespace Stats

namespace Stats {

namespace {

struct Totals {
	uint64_t calls = 0;
	uint64_t ns    = 0;
	uint64_t bytes = 0;
};

struct State {
	std::mutex                                       mutex;
	std::unordered_map<std::string, Totals>          totals;
	std::unordered_map<std::string, std::unordered_map<uint64_t, Totals>> regions;
	uint64_t                                         window_start_ms = 0;
};

State& GetState() {
	static State state;
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("DES_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void Add(const char* name, uint64_t ns, uint64_t bytes, uint64_t vaddr) {
	if (!Enabled()) {
		return;
	}
	auto&            state = GetState();
	std::string      out;
	{
		std::scoped_lock lock(state.mutex);
		auto&            t = state.totals[name];
		t.calls++;
		t.ns += ns;
		t.bytes += bytes;
		if (vaddr != 0) {
			auto& r = state.regions[name][vaddr >> 16u];
			r.calls++;
			r.ns += ns;
		}
		const auto now = ElapsedMs();
		if (state.window_start_ms == 0) {
			state.window_start_ms = now;
		}
		if (now - state.window_start_ms < 5000) {
			return;
		}
		state.window_start_ms = now;
		for (const auto& [n, v]: state.totals) {
			out += fmt::format("DES_STAT t={} name={} calls={} ms={:.1f} bytes={}\n", now, n, v.calls,
			                   static_cast<double>(v.ns) / 1e6, v.bytes);
		}
		for (auto& [n, regions]: state.regions) {
			std::vector<std::pair<uint64_t, Totals>> top(regions.begin(), regions.end());
			std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
			top.resize(std::min<size_t>(top.size(), 8));
			for (const auto& [region, v]: top) {
				out += fmt::format("DES_STAT_ADDR t={} name={} region=0x{:x} calls={} ms={:.1f}\n", now, n,
				                   region << 16u, v.calls, static_cast<double>(v.ns) / 1e6);
			}
		}
		state.totals.clear();
		state.regions.clear();
	}
	Log::WriteToConsoleAndLog(out);
}

Scope::Scope(const char* name, uint64_t bytes, uint64_t vaddr)
    : m_name(name), m_bytes(bytes), m_vaddr(vaddr), m_start_ns(Enabled() ? NowNs() : 0) {}

Scope::~Scope() {
	if (m_start_ns != 0) {
		Add(m_name, static_cast<uint64_t>(NowNs() - m_start_ns), m_bytes, m_vaddr);
	}
}

} // namespace Stats

void FileOpenMilestone(const char* guest_path) {
	static std::atomic_bool done {false};
	if (done.load(std::memory_order_relaxed) || guest_path == nullptr) {
		return;
	}
	const std::string_view path(guest_path);
	if (!path.starts_with("/app0/") || path.starts_with("/app0/sce_module/") ||
	    path.starts_with("/app0/sce_sys/") || path == "/app0/eboot.bin") {
		return;
	}
	if (!done.exchange(true)) {
		Milestone("first_data_file", fmt::format("path={}", path));
	}
}

namespace Trace {

namespace {

// One block: 2048 16-byte thunks (32 KiB), then one 64-byte {counter, target} line per thunk
// on separate pages: counting never writes to a cache line that holds code or another counter.
// The increment is not locked: under contention a count is a lower bound, while presence and
// first use (what Phase 6 needs) stay exact.
constexpr uint64_t ThunkSize      = 16;
constexpr uint64_t DataStride     = 64;
constexpr uint64_t EntriesInBlock = 2048;
constexpr uint64_t BlockCodeSize  = EntriesInBlock * ThunkSize;
constexpr uint64_t BlockSize      = BlockCodeSize + EntriesInBlock * DataStride;
constexpr uint64_t SampleMs       = 50;
constexpr uint64_t WriteEveryMs   = 1000;

struct Entry {
	uint64_t    slot     = 0;
	uint64_t    thunk    = 0;
	uint64_t*   counter  = nullptr;
	uint64_t*   target   = nullptr;
	int64_t     first_ms = -1;
	std::string name;
	std::string dbg_name;
	std::string program;
};

std::mutex                           g_mutex;
std::vector<uint64_t>                g_blocks;
std::vector<std::unique_ptr<Entry>>  g_entries;
std::unordered_map<uint64_t, Entry*> g_by_slot;
std::unordered_map<uint64_t, Entry*> g_by_thunk;
std::filesystem::path                g_output;
std::atomic_bool                     g_sampler_started {false};

std::string JsonEscape(std::string_view text) {
	std::string out;
	for (const char c: text) {
		switch (c) {
			case '"': out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n"; break;
			default:
				if (static_cast<unsigned char>(c) < 0x20) {
					out += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
				} else {
					out += c;
				}
		}
	}
	return out;
}

void Sample() {
	const auto now = static_cast<int64_t>(ElapsedMs());
	for (auto& entry: g_entries) {
		if (entry->first_ms < 0 && __atomic_load_n(entry->counter, __ATOMIC_RELAXED) != 0) {
			entry->first_ms = now;
		}
	}
}

void WriteFile() {
	std::string text =
	    fmt::format("{{\n \"tool\": \"desport_trace\", \"format_version\": 1, \"elapsed_ms\": {}, "
	                "\"sample_interval_ms\": {}, \"counts\": \"lower_bound\",\n \"entries\": [",
	                ElapsedMs(), SampleMs);
	bool first = true;
	for (const auto& entry: g_entries) {
		const auto name = std::string_view(entry->name);
		const auto nid  = name.substr(0, name.find_first_of("[#"));
		text += fmt::format("{}\n  {{\"nid\": \"{}\", \"symbol\": \"{}\", \"function\": \"{}\", "
		                    "\"program\": \"{}\", \"slot\": {}, \"count\": {}, \"first_ms\": {}}}",
		                    first ? "" : ",", JsonEscape(nid), JsonEscape(entry->name),
		                    JsonEscape(entry->dbg_name), JsonEscape(entry->program), entry->slot,
		                    __atomic_load_n(entry->counter, __ATOMIC_RELAXED),
		                    entry->first_ms < 0 ? std::string("null")
		                                        : std::to_string(entry->first_ms));
		first = false;
	}
	text += "\n ]\n}\n";
	auto temporary = g_output;
	temporary += ".tmp";
	{
		std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
		out << text;
	}
	std::error_code error;
	std::filesystem::rename(temporary, g_output, error);
}

void Flush() {
	std::scoped_lock lock(g_mutex);
	if (g_entries.empty()) {
		return;
	}
	Sample();
	WriteFile();
}

void StartSampler() {
	if (g_sampler_started.exchange(true)) {
		return;
	}
	const char* file = std::getenv("DES_TRACE_FILE");
	g_output         = (file != nullptr && *file != '\0') ? std::filesystem::path(file)
	                                                      : std::filesystem::path("calls.json");
	std::atexit(Flush);
	std::at_quick_exit(Flush);
	// The engine leaves with std::_Exit on fatal errors, so the file is rewritten every second.
	std::thread([] {
		uint64_t ticks = 0;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::milliseconds(SampleMs));
			std::scoped_lock lock(g_mutex);
			Sample();
			if (++ticks % (WriteEveryMs / SampleMs) == 0) {
				WriteFile();
			}
		}
	}).detach();
	Log::WriteToConsoleAndLog(
	    fmt::format("DES_TRACE: counting PLT calls -> {}\n", g_output.string()));
}

Entry* NewEntry() {
	const auto index = g_entries.size();
	if (index % EntriesInBlock == 0) {
		const auto block = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, BlockSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "desport_trace");
		if (block == 0) {
			return nullptr;
		}
		g_blocks.push_back(block);
	}
	const auto block = g_blocks.back();
	const auto slot  = index % EntriesInBlock;
	auto       entry = std::make_unique<Entry>();
	entry->thunk     = block + slot * ThunkSize;
	entry->counter   = reinterpret_cast<uint64_t*>(block + BlockCodeSize + slot * DataStride);
	entry->target    = entry->counter + 1;
	*entry->counter  = 0;

	// inc qword [rip+counter]; jmp qword [rip+target]; int3 x3
	uint8_t    code[ThunkSize] = {0x48, 0xff, 0x05, 0, 0, 0,    0,    0xff,
	                              0x25, 0,    0,    0, 0, 0xcc, 0xcc, 0xcc};
	const auto counter_disp    = static_cast<int32_t>(reinterpret_cast<int64_t>(entry->counter) -
	                                                  static_cast<int64_t>(entry->thunk + 7));
	const auto target_disp     = static_cast<int32_t>(reinterpret_cast<int64_t>(entry->target) -
	                                                  static_cast<int64_t>(entry->thunk + 13));
	std::memcpy(code + 3, &counter_disp, sizeof(counter_disp));
	std::memcpy(code + 9, &target_disp, sizeof(target_disp));
	std::memcpy(reinterpret_cast<void*>(entry->thunk), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(entry->thunk, ThunkSize);

	auto* raw = entry.get();
	g_entries.push_back(std::move(entry));
	g_by_thunk[raw->thunk] = raw;
	return raw;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("DES_TRACE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

uint64_t Wrap(uint64_t slot_vaddr, uint64_t target, const std::string& name,
              const std::string& dbg_name, const std::filesystem::path& program) {
	if (!Enabled() || target == 0) {
		return target;
	}
	StartSampler();
	std::scoped_lock lock(g_mutex);
	target       = g_by_thunk.contains(target) ? *g_by_thunk[target]->target : target;
	Entry* entry = nullptr;
	if (auto it = g_by_slot.find(slot_vaddr); it != g_by_slot.end()) {
		entry = it->second;
	} else {
		entry = NewEntry();
		if (entry == nullptr) {
			return target;
		}
		entry->slot           = slot_vaddr;
		entry->program        = program.filename().string();
		g_by_slot[slot_vaddr] = entry;
	}
	entry->name     = name;
	entry->dbg_name = dbg_name;
	__atomic_store_n(entry->target, target, __ATOMIC_RELEASE);
	return entry->thunk;
}

uint64_t Unwrap(uint64_t value) {
	if (!Enabled()) {
		return value;
	}
	std::scoped_lock lock(g_mutex);
	const auto       it = g_by_thunk.find(value);
	return it == g_by_thunk.end() ? value : *it->second->target;
}

} // namespace Trace

bool FindGameHook(const std::string& nid, Loader::SymbolType type, Loader::SymbolRecord* out) {
	static const auto hooks = [] {
		auto        symbols = std::make_unique<Loader::SymbolDatabase>();
		std::string title_id;
		Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
		RegisterGameHooks(symbols.get(), title_id);
		return symbols;
	}();
	if (const auto* record = hooks->FindByNid(nid, type); record != nullptr) {
		*out = *record;
		return true;
	}
	return false;
}

} // namespace Desport
