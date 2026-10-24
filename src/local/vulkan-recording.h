#pragma once

#include <atomic> // desport: KYTY_SUBMIT_COALESCE
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace vk::detail { class DispatchLoaderDynamic; }

// Vulkan command recording on a worker thread (KYTY_VULKAN_RECORDING). The
// renderer remains the sole resource/cache owner; a producer scope transfers
// only fully copied Vulkan call arguments.
namespace LocalVulkanRecording {
void Install();
// The Vulkan entry points without recording: for a queue the recording worker does not own.
const vk::detail::DispatchLoaderDynamic& DirectDispatch();
// Replaces the vkCmdBindPipeline every bind ends in (recorded, packet or direct: the device-loss breadcrumbs of
// device-fault.h) and returns the one replaced. After Install, before any command is recorded.
void* ReplaceBindPipeline(void* entry);
// Complete this producer's CPU recording before handing its Vulkan objects to
// another thread. A receiver's drain cannot flush the sender's thread-local queue.
// This does not wait for GPU execution.
void Drain();
struct Segment {
    const void* data = nullptr;
    size_t size = 0;
};
using ReplayPacket = void (*)(std::span<const Segment>, const vk::detail::DispatchLoaderDynamic&);
// Draws and descriptor encoding are recorded as whole packets.
bool PacketsEnabled();
// Segments are copied transactionally. An optional immutable owner is retained
// until replay finishes; callers never lend stack or mutable-cache pointers.
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner = {});
// Preserve order when a packet cannot fit. Does not retain any argument.
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments);
// Command-buffer begin/end and queue submission, queued in order with the recorded
// commands instead of draining the recording queue first (KYTY_DEFERRED_SUBMIT).
// A submit packet is published at once: other threads may wait on its tick.
bool DeferredSubmitEnabled();
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish);
// Deferred queue submissions: counted when queued and when the worker has handed them to
// the driver. WaitDeferredSubmits blocks until every submission queued so far reached the
// driver (not the GPU). A present must not depend on a signal still in the worker's queue:
// the Windows driver then blocks inside vkQueuePresentKHR, holding the queue lock the
// worker needs for that very submission.
void NoteDeferredSubmitQueued();
void NoteDeferredSubmitDone();
void WaitDeferredSubmits();
// Deferred submissions queued and handed to the driver so far (diagnostics).
uint64_t DeferredSubmitsQueued();
uint64_t DeferredSubmitsDone();
// desport: begin KYTY_SUBMIT_COALESCE (commandScheduler.cpp). A recording worker may hold graphics submissions to hand
// several to the driver as one. While it holds some, its wait for work calls `poll(false)` instead of blocking, which
// returns true while submissions are still held; `poll(true)` hands them all over at once.
using IdlePoll = bool (*)(bool force);
void SetIdlePoll(IdlePoll poll); // the calling worker's (thread-local)
bool OnWorker();                 // the calling thread is a recording worker
// This thread is about to make a queue submission that is not a held graphics one (the readback queue's copy, an
// async-compute submission): on a worker, the held submissions go first (one of them may be what it waits for).
void BeforeOtherSubmit();
// Producer side: the main scheduler's deferred submissions queued so far and handed to the driver so far (the worker's
// q numbers), and the q every held submission up to must be handed over (raised by waiters).
inline std::atomic<bool>     g_coalesce_on {false};
inline std::atomic<uint64_t> g_main_queued {0};
inline std::atomic<uint64_t> g_main_handed {0};
inline std::atomic<uint64_t> g_want_q {0};
inline void RaiseWantQ(uint64_t q) {
    auto known = g_want_q.load(std::memory_order_relaxed);
    while (known < q && !g_want_q.compare_exchange_weak(known, q, std::memory_order_seq_cst)) {}
}
// desport: end KYTY_SUBMIT_COALESCE
uint64_t StateEpoch();
// Commands the calling thread recorded that do GPU work or synchronize (draws, dispatches,
// copies, clears, rendering scopes, barriers): unchanged between two barriers = no work in between.
uint64_t WorkCalls();
// Work calls and packets (draws, descriptors) the calling thread recorded: unchanged = nothing recorded.
uint64_t RecordedWork();
// Commands that change memory or images (draws, dispatches, copies, clears) and packets the calling thread
// recorded: unchanged = nothing but barriers and rendering scopes recorded.
uint64_t RecordedWrites();
class ProducerScope {
public:
    ProducerScope();
    ~ProducerScope();
    ProducerScope(const ProducerScope&) = delete;
    ProducerScope& operator=(const ProducerScope&) = delete;
};
}
