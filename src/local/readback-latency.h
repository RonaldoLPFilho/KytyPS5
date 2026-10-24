#pragma once
// desport: KYTY_READBACK_LATENCY. Where a guest readback's wait goes, and a fix for the part spent behind the graphics
// command buffers.
//   0 (default)  nothing (no counters, no change).
//   1            counters: a "DES_READBACK_LAT ..." line every 5 s (BufferCache::ReportReadbackLatency).
//   2            1, plus the yield: the graphics queue's translation stops at a draw/dispatch boundary (after at least
//                KYTY_READBACK_YIELD_WORK draws or dispatches, default 16) while an async-compute submission waits for
//                the render thread, flushes what it translated and lets that submission be translated first
//                (GuestGpu::ComputeReady, CommandProcessor::ProcessPm4). The guest's compute rings run concurrently
//                with graphics on the real GPU: any interleaving at a packet boundary is one the hardware may run, and
//                the processor already switches queues mid command buffer at its waits.
// Per guest read that waited for a GPU tick (`lat_*`): the tick it waited for is followed from the render thread's
// submission (SubmitBuffers: enqueue), the recording worker's vkQueueSubmit (ReplaySubmit: driver) to the end of the
// wait (done): submit_us = enqueue - start, worker_us = driver - max(start, enqueue), gpu_us = done - max(start,
// driver); gpu_backlog = the submissions the GPU still had before that tick when it was handed to the driver;
// drv_backlog = the submissions the worker still had to hand over when the read began.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_readback_latency_mode;
extern volatile std::atomic<uint32_t> kyty_local_readback_yield_work;
}

namespace ReadbackLatency {

inline bool Stats() {
	return kyty_local_readback_latency_mode.load(std::memory_order_relaxed) != 0;
}
inline bool YieldOn() { // (not Yield: winbase.h defines a Yield() macro)
	return kyty_local_readback_latency_mode.load(std::memory_order_relaxed) >= 2;
}
inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

enum Id : uint32_t {
	Reads,         // guest reads that waited for a GPU tick
	ReadsAsync,    // ... whose last sync point (label write) came from an async-compute queue
	ReadsFlushed,  // ... whose tick was still being recorded (the read submitted it)
	ReadsLost,     // ... whose tick left the time ring (no breakdown)
	SubmitNs,      // sums over the reads with a breakdown
	WorkerNs,
	GpuNs,
	TotalNs,
	GpuBacklog,    // sum of submissions ahead on the GPU at the driver hand-over
	DrvBacklog,    // sum of submissions the worker still had at the read's start
	TickWork,      // sum of the draws and dispatches recorded into the awaited tick's command buffer
	WaitLt500,     // the read's wait for the tick: < 0.5 ms, < 1, < 2, < 4, >= 4 ms
	WaitLt1000,
	WaitLt2000,
	WaitLt4000,
	WaitGe4000,
	Submits,       // every submission handed to the driver
	SubmitLagNs,   // sum of enqueue -> driver over them (the worker's lag)
	SubmitLagMaxNs,
	WorkerBusyNs,  // the recording worker replaying chunks
	CqSubs,        // async-compute submissions (first slice)
	CqWaitNs,      // sum of enqueue -> first slice over them (waiting for the render thread)
	CqWaitMaxNs,
	GfxSlices,     // graphics submission slices (GuestGpu::Process calls)
	GfxSliceNs,
	GfxSliceMaxNs,
	Yields,        // KYTY_READBACK_LATENCY=2: graphics translations stopped for an async-compute submission
	Count
};
inline std::atomic<uint64_t> g_values[Count] {};
inline void Add(Id id, uint64_t n = 1) {
	g_values[id].fetch_add(n, std::memory_order_relaxed);
}
inline void Max(Id id, uint64_t n) {
	auto known = g_values[id].load(std::memory_order_relaxed);
	while (known < n && !g_values[id].compare_exchange_weak(known, n, std::memory_order_relaxed)) {
	}
}

// The command processor's last sync point (EOP / WRITE_DATA label): from an async-compute queue?
inline std::atomic<bool> g_last_sync_async {false};
inline void NoteSyncPoint(bool async_queue) {
	if (Stats()) g_last_sync_async.store(async_queue, std::memory_order_relaxed);
}

// Times of the recent ticks (master timeline values), by tick modulo the ring.
struct TickTimes {
	std::atomic<uint64_t> tick {0};
	std::atomic<uint64_t> enqueue_ns {0};
	std::atomic<uint64_t> driver_ns {0};
	std::atomic<uint64_t> gpu_tick {0}; // the GPU's timeline value right after the driver hand-over
	std::atomic<uint64_t> work {0};
};
constexpr size_t           TickRing = 8192;
inline TickTimes           g_ticks[TickRing];
// Render thread: the draws and dispatches of the command buffer being submitted (CommandScheduler::Submit).
inline thread_local uint64_t t_submit_work = 0;

// Render thread (SubmitBuffers): the tick is handed to the recording stream (or submitted, without it).
inline void NoteEnqueue(uint64_t tick) {
	if (!Stats()) return;
	auto& entry = g_ticks[tick % TickRing];
	entry.tick.store(0, std::memory_order_relaxed);
	entry.driver_ns.store(0, std::memory_order_relaxed);
	entry.gpu_tick.store(0, std::memory_order_relaxed);
	entry.work.store(t_submit_work, std::memory_order_relaxed);
	entry.enqueue_ns.store(NowNs(), std::memory_order_relaxed);
	entry.tick.store(tick, std::memory_order_release);
	t_submit_work = 0;
}
// Recording worker (ReplaySubmit) or render thread: vkQueueSubmit of the tick returned; `gpu_tick`: the timeline then.
inline void NoteDriver(uint64_t tick, uint64_t gpu_tick) {
	if (!Stats()) return;
	const uint64_t now   = NowNs();
	auto&          entry = g_ticks[tick % TickRing];
	if (entry.tick.load(std::memory_order_acquire) != tick) return;
	entry.gpu_tick.store(gpu_tick, std::memory_order_relaxed);
	entry.driver_ns.store(now, std::memory_order_release);
	const uint64_t enqueue = entry.enqueue_ns.load(std::memory_order_relaxed);
	const uint64_t lag     = now > enqueue ? now - enqueue : 0;
	Add(Submits);
	Add(SubmitLagNs, lag);
	Max(SubmitLagMaxNs, lag);
}

// A guest read that waited for `tick`: `start` (the read's start), `done` (the wait's end); `flushed`: the read
// submitted the tick; `driver_tick`: the last tick handed to the driver when the read began.
inline void NoteRead(uint64_t tick, uint64_t start, uint64_t wait_start, uint64_t done, bool flushed,
                     uint64_t driver_tick) {
	Add(Reads);
	if (g_last_sync_async.load(std::memory_order_relaxed)) Add(ReadsAsync);
	if (flushed) Add(ReadsFlushed);
	const uint64_t wait = done > wait_start ? done - wait_start : 0;
	Add(wait < 500000 ? WaitLt500 : wait < 1000000 ? WaitLt1000 : wait < 2000000 ? WaitLt2000 : wait < 4000000 ? WaitLt4000 : WaitGe4000);
	if (tick > driver_tick) Add(DrvBacklog, tick - driver_tick);
	auto&          entry   = g_ticks[tick % TickRing];
	const uint64_t before  = entry.tick.load(std::memory_order_acquire);
	const uint64_t enqueue = entry.enqueue_ns.load(std::memory_order_relaxed);
	const uint64_t driver  = entry.driver_ns.load(std::memory_order_acquire);
	const uint64_t gpu     = entry.gpu_tick.load(std::memory_order_relaxed);
	const uint64_t work    = entry.work.load(std::memory_order_relaxed);
	if (before != tick || entry.tick.load(std::memory_order_acquire) != tick || enqueue == 0 || driver == 0 || done < start) {
		Add(ReadsLost);
		return;
	}
	const uint64_t from_enqueue = enqueue > start ? enqueue : start;
	const uint64_t from_driver  = driver > from_enqueue ? driver : from_enqueue;
	Add(SubmitNs, enqueue > start ? enqueue - start : 0);
	Add(WorkerNs, driver > from_enqueue ? driver - from_enqueue : 0);
	Add(GpuNs, done > from_driver ? done - from_driver : 0);
	Add(TotalNs, done - start);
	Add(GpuBacklog, tick > gpu + 1 ? tick - gpu - 1 : 0);
	Add(TickWork, work);
}

} // namespace ReadbackLatency
