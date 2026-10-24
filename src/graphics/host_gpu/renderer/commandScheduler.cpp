#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/timer.h" // desport: KYTY_GPU_TIMESTAMPS=2 (host counter)
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionManager.h"
#include "async-upload.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#include "device-fault.h"
#endif
#include "live-trace-gpu.h"
#include "readback-latency.h" // desport: KYTY_READBACK_LATENCY
#include "async-compute.h"    // desport: KYTY_ASYNC_COMPUTE
#include "gpu-profile.h"      // desport: KYTY_GPU_PROFILE
#include "loader/desport.h"   // desport: KYTY_ASYNC_COMPUTE (DES_ASYNC_COMPUTE line)
#include "local-platform.h"
#include "live-control.h" // desport: KYTY_GPU_TIMESTAMPS (flips)
#include "graphics/host_gpu/renderer/renderContext.h" // desport: KYTY_GPU_TIMESTAMPS (the renderer's scheduler)

#include <algorithm>
#include <array> // desport
#include <atomic>
#include <chrono> // desport
#include <memory> // desport
#include <cstdlib> // desport
#include <cstring> // desport
#include <cstdio>
#include <optional>
#include <fmt/format.h> // desport: KYTY_ASYNC_COMPUTE
#include <mutex>
#include <string>
#include <thread> // desport
#include <utility> // desport
#include <vector> // desport

extern "C" {
// Dispatches recorded per submission (KYTY_DISPATCH_BATCH).
volatile std::atomic_uint32_t kyty_local_dispatch_batch {32};
// Draws recorded per submission (KYTY_DRAW_BATCH, 0: no limit): about 0.5 ms of translated draws.
volatile std::atomic_uint32_t kyty_local_draw_batch {256};
// Buffer uploads of pages dirty since before the open command buffer began go into its upload prologue
// (CommandScheduler::UploadPrologue).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_upload_prologue {1};
// 0: refresh the master timeline before every drain attempt.
// 1: prove the pending-operation queue empty before the timeline query and the
//    operation lock. Draws and dispatches call this with nothing to retire.
volatile std::atomic<uint32_t> kyty_local_pending_drain_mode {0};
}

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;
static thread_local CommandScheduler::Recorder* t_recorder           = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
	if (result == vk::Result::eErrorDeviceLost) DeviceFault::Report();
}

// desport: KYTY_GPU_TIMESTAMPS / KYTY_SUBMIT_COALESCE (defined after the deferred submission).
void DesportSubmitInit(GraphicContext& graphics);
bool DesportCoalesceQueued();
namespace GpuTs {
void Stop();
} // namespace GpuTs

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master, uint32_t family)
    : m_graphics(graphics), m_master(master) {
	if (family == UINT32_MAX) family = graphics.queue_family; // desport: KYTY_ASYNC_COMPUTE (the async queue's family)
	EXIT_IF(family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this), m_prologue(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_tick_monitor([this](std::stop_token stop) {
	      LocalPlatform::SetThreadName("Kyty.TickMon");
	      {
		      vk::QueryPoolCreateInfo info {};
		      info.queryType  = vk::QueryType::eTimestamp;
		      info.queryCount = LiveTrace::TimestampSlots * 2u;
		      vk::QueryPool pool {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &pool) == vk::Result::eSuccess) {
			      LiveTrace::g_timestamp_period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
			      LiveTrace::g_timestamp_pool   = static_cast<VkQueryPool>(pool);
		      }
		      // Per-draw marks (`tracem`): reset on the host, read after tracing stops.
		      info.queryCount = LiveTrace::MarkSlots;
		      vk::QueryPool marks {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &marks) == vk::Result::eSuccess) {
			      m_graphics.device.resetQueryPool(marks, 0, LiveTrace::MarkSlots);
			      LiveTrace::g_mark_pool = static_cast<VkQueryPool>(marks);
			      static vk::Device device = m_graphics.device;
			      LiveTrace::g_read_marks = [](uint32_t count) {
				      const auto            pool = static_cast<VkQueryPool>(LiveTrace::g_mark_pool);
				      std::vector<uint64_t> values(count);
				      if (count != 0 &&
				          device.getQueryPoolResults(pool, 0, count, count * sizeof(uint64_t), values.data(),
				                                     sizeof(uint64_t),
				                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
				              vk::Result::eSuccess) {
					      for (uint32_t i = 0; i < count; ++i)
						      LiveTrace::Append(LiveTrace::GpuMarkValue, i,
						                        static_cast<uint64_t>(static_cast<double>(values[i]) *
						                                              LiveTrace::g_timestamp_period));
				      }
				      device.resetQueryPool(pool, 0, LiveTrace::MarkSlots);
			      };
		      }
	      }
	      uint64_t next = 0;
	      while (!stop.stop_requested()) {
		      if (!LiveTrace::g_on.load(std::memory_order_relaxed)) {
			      next = 0;
			      std::this_thread::sleep_for(std::chrono::milliseconds(5));
			      continue;
		      }
		      if (next == 0) next = m_master.KnownGpuTick() + 1;
		      vk::SemaphoreWaitInfo wait {};
		      const auto            semaphore = m_master.Handle();
		      wait.semaphoreCount           = 1;
		      wait.pSemaphores              = &semaphore;
		      wait.pValues                  = &next;
		      if (m_graphics.device.waitSemaphores(&wait, 2000000) == vk::Result::eSuccess) {
			      LiveTrace::Event(LiveTrace::TickDone, next);
			      auto& mapped = LiveTrace::g_tick_slot[next % LiveTrace::TimestampSlots];
			      if (const auto slot = mapped.exchange(0); slot != 0 && LiveTrace::g_timestamp_pool != nullptr) {
				      uint64_t times[2] {};
				      if (m_graphics.device.getQueryPoolResults(
				              static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool), (slot - 1u) * 2u, 2u,
				              sizeof(times), times, sizeof(uint64_t), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
					      LiveTrace::Event(LiveTrace::GpuSpan, next);
					      LiveTrace::Event(LiveTrace::GpuSpanNs,
					                       static_cast<uint64_t>(static_cast<double>(times[0]) * LiveTrace::g_timestamp_period),
					                       static_cast<uint64_t>(static_cast<double>(times[1]) * LiveTrace::g_timestamp_period));
				      }
			      }
			      ++next;
		      }
	      }
      }),
      m_async_command(*this) {
	// desport: KYTY_GPU_TIMESTAMPS / KYTY_SUBMIT_COALESCE: the renderer's scheduler only (not the presenter's).
	m_main = &context.GetCommandScheduler() == this;
	if (m_main) DesportSubmitInit(graphics);
	// desport: KYTY_ASYNC_COMPUTE: the async queue's timeline, pool and epoch source (async-compute.h).
	if (graphics.async_queue != nullptr && AsyncCompute::RequestedMode() != 0) {
		m_master.EnableAsync();
		m_async_pool = std::make_unique<CommandPool>(graphics, m_master, graphics.async_family);
		AsyncCompute::g_epoch.store(&m_master.CurrentTickValue(), std::memory_order_release);
		AsyncCompute::g_on.store(true, std::memory_order_release);
	}
}

void CommandScheduler::CompleteDispatch() {
	// Expose command-processor/GPU overlap without a host fence. Bound the
	// recording batch; resource retirement still uses the submission timeline.
	// The boundary costs one all-commands dependency, which lets no part of the
	// next batch begin before this one drains, so its size trades that drain
	// against how early the queue receives work. Dispatches inside a batch keep
	// the same chain barriers at any size. (A recorder's buffers are submitted later, whole.)
	if (t_recorder != nullptr) return;
	const auto batch = std::clamp(kyty_local_dispatch_batch.load(std::memory_order_relaxed), 1u, 65536u);
	if (++m_recorded_dispatches < batch) {
		return;
	}
	// While the GPU still runs work handed to it the batch goes on (GpuBusy, asked at the batch's end and every 4
	// dispatches after it), up to 8 batches.
	if (m_recorded_dispatches < 8 * batch &&
	    ((m_recorded_dispatches != batch && (m_recorded_dispatches & 3u) != 0) || GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// As CompleteDispatch for draws: a draw-heavy guest command buffer reaches the GPU in parts while it is still being
// translated. Whole, the GPU idled through its translation (3-4.5 ms for the big passes) and ran it after, late into the
// next frame, where the next frame's culling readbacks waited for it.
void CommandScheduler::CompleteDraws(uint32_t draws) {
	const auto batch = kyty_local_draw_batch.load(std::memory_order_relaxed);
	if (t_recorder != nullptr || batch == 0 || (m_recorded_draws += draws) < batch) {
		return;
	}
	// As CompleteDispatch's: on while the GPU is busy (asked at the batch's end and about every 16 draws after it), up
	// to 8 batches.
	const bool end = m_recorded_draws - draws < batch;
	if (m_recorded_draws < 8 * batch && ((!end && (m_recorded_draws & 15u) >= draws && draws < 16) || GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// A batch boundary gives the GPU work early, where it would idle through the batch's translation; while it still runs
// work it was handed, the boundary gives it nothing it lacks and costs a drain on the GPU (~8 us between back-to-back
// command buffers) and a submission on the recording worker (~13 us, half of it in the kernel; at 1-1 standing 141
// submissions a frame were ~1 ms of the GPU's and ~1.8 ms of the worker's, whose lag left the GPU idle ~1.35 ms a frame
// with work already submitted). The batch then goes on until the GPU has run what it was handed. Same-process A/B, 3-4
// rounds each: 1-1 standing 60.2 -> 62.4 fps, low1 52.8 -> 56.0 (a boot paced at 60: low1 54.2 -> 55.6); Latria spin
// 74.2 -> 75.9, low1 53.9 -> 55.4; Shrine standing unchanged (75.3 / 75.3).
bool CommandScheduler::GpuBusy() {
	if (m_main && DesportCoalesceQueued()) return true; // desport: KYTY_SUBMIT_COALESCE
	const auto handed = m_driver_tick.load(std::memory_order_acquire);
	// desport: KYTY_ASYNC_COMPUTE: the graphics queue's own work (the known tick also waits for async epochs).
	const auto done = [&] { return m_master.AsyncEnabled() ? m_master.IsGfxFree(handed) : m_master.IsFree(handed); };
	if (done()) return false;
	m_master.Refresh();
	return !done();
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	GpuTs::Stop(); // desport: KYTY_GPU_TIMESTAMPS (the first scheduler to shut down stops the reader)
	GpuProfile::Stop(); // desport: KYTY_GPU_PROFILE (likewise)
	SubmitAsync(); // desport: KYTY_ASYNC_COMPUTE
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	if (t_recorder != nullptr) {
		BeginRecording(*t_recorder, registers, user_config, shaders);
		return;
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	auto& command = t_recorder != nullptr ? *t_recorder->command : m_command;
	if (Active() && !command.IsInvalid()) {
		command.EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	if (t_recorder != nullptr) throw RecorderRefusal {"FlushAndWait"};
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	if (t_recorder != nullptr) throw RecorderRefusal {"Finish"};
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	if (t_recorder != nullptr) {
		// (A recorder's own work is not submitted: waiting for it would never end.)
		if (tick >= PendingTick) throw RecorderRefusal {"Wait"};
		m_master.Wait(tick);
		return;
	}
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	if (t_recorder != nullptr) return; // (the owning thread retires them)
	// The pending queue is empty for the large majority of draws and dispatches.
	// Refresh() issues a Vulkan timeline query and the drain takes the operation
	// mutex, so let an atomic count prove there is nothing to retire first. The
	// count is only a hint: the queue is still re-checked under the lock.
	if (kyty_local_pending_drain_mode.load(std::memory_order_relaxed) != 0 &&
	    m_pending_operation_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_operation_count.fetch_sub(1, std::memory_order_relaxed);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	if (t_recorder != nullptr) throw RecorderRefusal {"DeferOperation"};
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		m_pending_operation_count.fetch_add(1, std::memory_order_release);
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	if (t_recorder != nullptr) throw RecorderRefusal {"DeferPriorityOperation"};
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		LiveTrace::Event(LiveTrace::GpuDone, operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

bool CommandScheduler::Active() const noexcept {
	return (t_recorder != nullptr ? *t_recorder->command : m_command).m_registers != nullptr;
}

uint64_t CommandScheduler::CurrentTick() const noexcept {
	return t_recorder != nullptr ? t_recorder->pending_tick : m_master.CurrentTick();
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return t_recorder != nullptr ? *t_recorder->command : m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid() || !m_prologue.IsInvalid());
	++m_command_serial;
	m_command_clock    = g_cpu_dirty_clock.load(std::memory_order_acquire);
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	LiveTrace::g_mark_command = static_cast<VkCommandBuffer>(m_command.m_buffer);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	m_desport_writes_begin = LocalVulkanRecording::RecordedWrites(); // desport: KYTY_GPU_TIMESTAMPS
#endif
	return m_command;
}

// desport: begin KYTY_GPU_TIMESTAMPS. KYTY_GPU_TIMESTAMPS=1: every submission of a scheduler on the graphics queue is
// wrapped in two pre-recorded command buffers of its own, a GPU timestamp at its start (top of pipe, after a reset of
// its two queries) and one at its end (all commands), in a ring of 4096 query pairs. A reader thread takes them once the
// submission's timeline tick is reached (never a wait on the queries) and every 5 s writes "DES_GPU_TS t=<ms>
// window_ms= subs= ticks= frames= subs_per_frame= ticks_per_frame= busy_us= busy_p50_us= busy_p95_us= busy_max_us=
// small_subs= small_pct= nowrite= nowrite_busy_us= gaps= gap_us= gap_p50_us= gap_p95_us= gap_max_us= overlaps= busy_ms=
// span_ms= util_pct= dropped=": per vkQueueSubmit of ours (subs; ticks = the timeline ticks they signalled, more than
// subs when KYTY_SUBMIT_COALESCE merges), its GPU time from start to end (busy), the gap from the end of one to the start
// of the next one of ours (gaps; a negative one is an overlap: the next started before the previous ended), the union
// of the busy intervals against the GPU clock between the first start and the last end (util_pct), the submissions
// under 50 us (small_subs) and those whose command buffers recorded no draw/dispatch/copy/clear (nowrite). Submissions of
// others on the queue (Streamline's, frame generation's) are not wrapped: they fall into the gaps. Never touches the
// async-compute or the readback queue.
//
// desport: KYTY_GPU_TIMESTAMPS=2 adds, per wrapped submission: a "pre" timestamp (all commands, in a batch of its own
// without waits, put ahead of a submission that waits on semaphores: when the work before it on the queue was done),
// so head - pre is the time the queue sat in the submission's semaphore waits (the head is a top-of-pipe timestamp in
// the waiting batch itself: after the waits), a "tail-top" one (top of pipe at the start of the tail buffer: the front
// end issued all of the submission's commands; busy = issue + drain), the host time of the vkQueueSubmit, and, with
// VK_KHR/EXT_calibrated_timestamps (enabled at device creation for mode 2), the host<->GPU clock offset (recalibrated
// every second). Every 5 s "DES_GPU_WAIT t=<ms> window_ms= subs= cbs= cbs_per_frame= present_subs= waited=
// waited_pct= wait_us= wait_p95_us= wait_max_us= wait_ms= w_async= w_async_us= w_gfx= w_gfx_us= w_present=
// w_present_us= w_other= w_other_us= issue_us= drain_us= calibrated= cal_dev_us= qlat_n= qlat_us= qlat_p50_us=
// qlat_p95_us= qlat_max_us= idle_submits= idle_pct= launch_us= launch_p50_us= launch_p95_us= launch_max_us=":
// waited = submissions with semaphore waits (w_*: by the semaphore waited for: the async-compute timeline, a graphics
// timeline, a presenter submission's swapchain acquire, other = a sparse binding's timeline or unknown; a submission
// waiting on several counts in each, its wait once in each); qlat = vkQueueSubmit on the host to the submission
// reaching the GPU (pre, else head), queue backlog included; launch = the same for submissions handed over while the
// GPU had finished the previous wrapped one (idle_submits): the driver's and the OS's latency alone.
namespace {
namespace GpuTs {
constexpr uint32_t Slots   = 4096;
constexpr uint32_t Buckets = 2048; // 2 us each (the last: 4.094 ms and up)
constexpr uint32_t Queries = 4;    // desport: per slot: head, tail, pre, tail-top
enum Kind : uint8_t { KindAsync = 1, KindGfx = 2, KindPresent = 4, KindOther = 8 }; // desport: KYTY_GPU_TIMESTAMPS=2
// desport: KYTY_GPU_TIMESTAMPS=2: the host counter at the hand-over, semaphore waits (count, Kind bits), whether a pre
// buffer was put ahead, a renderer's submission (not the presenter's), its command buffers.
struct Extra {
	uint64_t host  = 0;
	uint32_t waits = 0;
	uint8_t  kinds = 0;
	bool     pre   = false;
	bool     main  = true;
	uint32_t cbs   = 0;
};
struct Entry {
	std::atomic<uint64_t> published {0}; // seq + 1 once filled (under the queue lock)
	MasterSemaphore*      master = nullptr;
	uint64_t              tick   = 0;
	uint32_t              ticks  = 1;
	uint32_t              writes = UINT32_MAX;
	Extra                 extra; // desport: KYTY_GPU_TIMESTAMPS=2
};
struct Window {
	uint64_t n = 0, busy_sum = 0, busy_max = 0, small_subs = 0, nowrite = 0, nowrite_busy = 0, ticks = 0;
	uint64_t gaps = 0, gap_sum = 0, gap_max = 0, overlaps = 0;
	uint64_t union_ns = 0, first = 0, last_end = 0, prev_end = 0, prev_seq = UINT64_MAX;
	std::array<uint32_t, Buckets> busy_hist {}, gap_hist {};
	static uint32_t Bucket(uint64_t ns) { return static_cast<uint32_t>(std::min<uint64_t>(ns / 2000u, Buckets - 1)); }
	static double   Percentile(const std::array<uint32_t, Buckets>& hist, uint64_t count, double p) {
		if (count == 0) return 0.0;
		const auto target = static_cast<uint64_t>(static_cast<double>(count) * p);
		uint64_t   seen   = 0;
		for (uint32_t i = 0; i < Buckets; ++i) {
			seen += hist[i];
			if (seen > target) return static_cast<double>(i) * 2.0 + 1.0;
		}
		return static_cast<double>(Buckets) * 2.0;
	}
	void Add(uint64_t seq, uint64_t begin, uint64_t end, uint32_t entry_ticks, uint32_t writes) {
		if (end < begin) { // (a wrapped counter: not counted, and no gap across it)
			prev_seq = UINT64_MAX;
			return;
		}
		const uint64_t busy = end - begin;
		++n;
		ticks += entry_ticks;
		busy_sum += busy;
		busy_max = std::max(busy_max, busy);
		++busy_hist[Bucket(busy)];
		if (busy < 50000) ++small_subs;
		if (writes == 0) {
			++nowrite;
			nowrite_busy += busy;
		}
		if (prev_seq != UINT64_MAX && prev_seq + 1 == seq) {
			if (begin < prev_end) {
				++overlaps;
			} else {
				const uint64_t gap = begin - prev_end;
				++gaps;
				gap_sum += gap;
				gap_max = std::max(gap_max, gap);
				++gap_hist[Bucket(gap)];
			}
		}
		if (first == 0) first = begin;
		const uint64_t from = std::max(begin, last_end);
		if (end > from) union_ns += end - from;
		last_end = std::max(last_end, end);
		prev_end = end;
		prev_seq = seq;
	}
	// desport: begin KYTY_GPU_TIMESTAMPS=2 (DES_GPU_WAIT)
	struct KindStats {
		uint64_t subs = 0, ns = 0;
	};
	std::array<KindStats, 4>      by_kind {}; // async, gfx, present, other
	uint64_t                      x_subs = 0, cbs = 0, present_subs = 0, waited = 0, wait_ns = 0, wait_max = 0;
	uint64_t                      split_n = 0, issue_ns = 0, drain_ns = 0;
	uint64_t                      qlat_n = 0, qlat_ns = 0, qlat_max = 0, idle_submits = 0, launch_ns = 0, launch_max = 0;
	std::array<uint32_t, Buckets> wait_hist {}, qlat_hist {}, launch_hist {};
	// Before Add (prev_end/prev_seq are the previous submission's). pre/top: 0 when not read. host_gpu: the hand-over
	// on the GPU clock (0: no calibration).
	void AddExtra(uint64_t seq, uint64_t head, uint64_t end, uint64_t pre, uint64_t top, uint64_t host_gpu,
	              const Extra& e) {
		if (end < head) return;
		++x_subs;
		cbs += e.cbs;
		if (!e.main) ++present_subs;
		if (e.waits != 0 && pre != 0) {
			const uint64_t wait = head > pre ? head - pre : 0;
			++waited;
			wait_ns += wait;
			wait_max = std::max(wait_max, wait);
			++wait_hist[Bucket(wait)];
			for (uint32_t k = 0; k < 4; ++k) {
				if ((e.kinds & (1u << k)) == 0) continue;
				++by_kind[k].subs;
				by_kind[k].ns += wait;
			}
		}
		if (top != 0 && top >= head && end >= top) {
			++split_n;
			issue_ns += top - head;
			drain_ns += end - top;
		}
		if (host_gpu != 0) {
			const uint64_t start = pre != 0 ? std::min(pre, head) : head;
			const uint64_t qlat  = start > host_gpu ? start - host_gpu : 0;
			++qlat_n;
			qlat_ns += qlat;
			qlat_max = std::max(qlat_max, qlat);
			++qlat_hist[Bucket(qlat)];
			if (prev_seq != UINT64_MAX && prev_seq + 1 == seq && prev_end < host_gpu) {
				++idle_submits;
				launch_ns += qlat;
				launch_max = std::max(launch_max, qlat);
				++launch_hist[Bucket(qlat)];
			}
		}
	}
	// desport: end KYTY_GPU_TIMESTAMPS=2
};
struct State {
	VkDevice                                 device   = VK_NULL_HANDLE;
	const vk::detail::DispatchLoaderDynamic* d        = nullptr;
	VkQueryPool                              pool     = VK_NULL_HANDLE;
	VkCommandPool                            commands = VK_NULL_HANDLE;
	std::array<VkCommandBuffer, Slots>       heads {}, tails {};
	double                                   period = 1.0;
	uint64_t                                 mask   = ~uint64_t {0};
	uint64_t                                 next   = 0; // under the graphics queue lock
	std::atomic<uint64_t>                    read {0};
	std::array<Entry, Slots>                 entries;
	std::atomic<uint64_t>                    dropped {0};
	std::atomic<bool>                        stopped {false};
	std::jthread                             reader;
	// desport: KYTY_GPU_TIMESTAMPS=2: mode (1, 2), the pre buffers, the calibration call (null: none) and the host
	// counter's frequency.
	uint32_t                                 mode = 1;
	std::array<VkCommandBuffer, Slots>       pres {};
	PFN_vkGetCalibratedTimestampsKHR         calibrate = nullptr;
	uint64_t                                 host_frequency = 1;
};
std::atomic<State*> g_state {nullptr};

// desport: KYTY_GPU_TIMESTAMPS: 0 off, 1 on, 2 on with the wait/launch split (DES_GPU_WAIT).
uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_GPU_TIMESTAMPS");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return 0u;
		return std::strcmp(value, "2") == 0 ? 2u : 1u;
	}();
	return mode;
}
bool Requested() {
	return Mode() != 0;
}

// desport: KYTY_GPU_TIMESTAMPS=2: a host counter value in ns.
uint64_t HostNs(const State& s, uint64_t counter) {
	const uint64_t f = s.host_frequency;
	return counter / f * 1000000000ull + counter % f * 1000000000ull / f;
}

// desport: KYTY_GPU_TIMESTAMPS=2: the GPU clock (ns) minus the host counter (ns), from the best of 3 calibrated reads;
// false when the call failed. `deviation`: its maximum deviation (ns).
bool Calibrate(const State& s, int64_t& offset, uint64_t& deviation) {
	if (s.calibrate == nullptr) return false;
	VkCalibratedTimestampInfoKHR infos[2] {};
	for (auto& info: infos) info.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
	infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
#if defined(_WIN32)
	infos[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
	infos[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
	bool ok = false;
	for (int attempt = 0; attempt < 3; ++attempt) {
		uint64_t values[2] {};
		uint64_t dev = 0;
		if (s.calibrate(s.device, 2, infos, values, &dev) != VK_SUCCESS) continue;
		if (ok && dev >= deviation) continue;
		const auto gpu = static_cast<int64_t>(static_cast<double>(values[0] & s.mask) * s.period);
		offset         = gpu - static_cast<int64_t>(HostNs(s, values[1]));
		deviation      = dev;
		ok             = true;
	}
	return ok;
}

void ReaderLoop(const std::stop_token& stop, State& s) {
	LocalPlatform::SetThreadName("Kyty.GpuTs");
	auto     window       = std::make_unique<Window>();
	uint64_t window_start = Desport::ElapsedMilliseconds();
	uint64_t flips_start  = LiveControl::g_flips.load(std::memory_order_relaxed);
	uint64_t dropped_seen = 0;
	// desport: KYTY_GPU_TIMESTAMPS=2: the clock offset (recalibrated every second).
	int64_t  cal_offset = 0;
	uint64_t cal_dev = 0, cal_ms = 0, cal_dev_max = 0;
	bool     calibrated = false;
	while (!stop.stop_requested()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (s.mode == 2 && s.calibrate != nullptr && (cal_ms == 0 || Desport::ElapsedMilliseconds() - cal_ms >= 1000)) {
			cal_ms = Desport::ElapsedMilliseconds();
			if (Calibrate(s, cal_offset, cal_dev)) {
				calibrated  = true;
				cal_dev_max = std::max(cal_dev_max, cal_dev);
			}
		}
		for (;;) {
			const uint64_t seq   = s.read.load(std::memory_order_relaxed);
			auto&          entry = s.entries[seq % Slots];
			if (entry.published.load(std::memory_order_acquire) != seq + 1) break;
			if (!entry.master->IsGfxFree(entry.tick)) {
				entry.master->Refresh();
				if (!entry.master->IsGfxFree(entry.tick)) break;
			}
			uint64_t       times[2] {};
			const uint32_t slot   = static_cast<uint32_t>(seq % Slots);
			const VkResult result = s.d->vkGetQueryPoolResults(s.device, s.pool, slot * Queries, 2u, sizeof(times), times,
			                                                   sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
			if (result == VK_NOT_READY) break;
			// desport: KYTY_GPU_TIMESTAMPS=2: the pre (when put ahead) and tail-top timestamps.
			uint64_t pre_raw = 0, top_raw = 0;
			bool     have_pre = false, have_top = false;
			if (s.mode == 2 && result == VK_SUCCESS) {
				if (entry.extra.pre) {
					const VkResult r = s.d->vkGetQueryPoolResults(s.device, s.pool, slot * Queries + 2u, 1u, sizeof(pre_raw),
					                                              &pre_raw, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
					if (r == VK_NOT_READY) break;
					have_pre = r == VK_SUCCESS;
				}
				const VkResult r = s.d->vkGetQueryPoolResults(s.device, s.pool, slot * Queries + 3u, 1u, sizeof(top_raw),
				                                              &top_raw, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
				if (r == VK_NOT_READY) break;
				have_top = r == VK_SUCCESS;
			}
			const uint32_t entry_ticks = entry.ticks, writes = entry.writes;
			const Extra    extra       = entry.extra; // desport: KYTY_GPU_TIMESTAMPS=2 (before the slot may be reused)
			s.read.store(seq + 1, std::memory_order_release); // (the slot may be reused from here on)
			if (result != VK_SUCCESS) {
				window->prev_seq = UINT64_MAX;
				continue;
			}
			const auto ns = [&](uint64_t value) {
				return static_cast<uint64_t>(static_cast<double>(value & s.mask) * s.period);
			};
			if (s.mode == 2) {
				uint64_t host_gpu = 0;
				if (calibrated) {
					const int64_t value = static_cast<int64_t>(HostNs(s, extra.host)) + cal_offset;
					host_gpu            = value > 0 ? static_cast<uint64_t>(value) : 0;
				}
				window->AddExtra(seq, ns(times[0]), ns(times[1]), have_pre ? ns(pre_raw) : 0, have_top ? ns(top_raw) : 0,
				                 host_gpu, extra);
			}
			window->Add(seq, ns(times[0]), ns(times[1]), entry_ticks, writes);
		}
		const uint64_t now = Desport::ElapsedMilliseconds();
		if (now - window_start < 5000) continue;
		const auto&    w       = *window;
		const uint64_t flips   = LiveControl::g_flips.load(std::memory_order_relaxed);
		const uint64_t frames  = flips - flips_start;
		const uint64_t dropped = s.dropped.load(std::memory_order_relaxed);
		const uint64_t span    = w.last_end > w.first ? w.last_end - w.first : 0;
		const auto     per     = [](uint64_t sum, uint64_t count, double scale) {
            return count != 0 ? static_cast<double>(sum) / scale / static_cast<double>(count) : 0.0;
		};
		Desport::EmitLine(fmt::format(
		    "DES_GPU_TS t={} window_ms={} subs={} ticks={} frames={} subs_per_frame={:.1f} ticks_per_frame={:.1f} "
		    "busy_us={:.1f} busy_p50_us={:.0f} busy_p95_us={:.0f} busy_max_us={:.1f} small_subs={} small_pct={:.1f} nowrite={} "
		    "nowrite_busy_us={:.1f} gaps={} gap_us={:.1f} gap_p50_us={:.0f} gap_p95_us={:.0f} gap_max_us={:.1f} overlaps={} "
		    "busy_ms={:.1f} span_ms={:.1f} util_pct={:.1f} dropped={}\n",
		    now, now - window_start, w.n, w.ticks, frames, per(w.n, frames, 1.0), per(w.ticks, frames, 1.0),
		    per(w.busy_sum, w.n, 1e3), Window::Percentile(w.busy_hist, w.n, 0.5), Window::Percentile(w.busy_hist, w.n, 0.95),
		    static_cast<double>(w.busy_max) / 1e3, w.small_subs, w.n != 0 ? 100.0 * static_cast<double>(w.small_subs) / static_cast<double>(w.n) : 0.0,
		    w.nowrite, per(w.nowrite_busy, w.nowrite, 1e3), w.gaps, per(w.gap_sum, w.gaps, 1e3),
		    Window::Percentile(w.gap_hist, w.gaps, 0.5), Window::Percentile(w.gap_hist, w.gaps, 0.95),
		    static_cast<double>(w.gap_max) / 1e3, w.overlaps, static_cast<double>(w.union_ns) / 1e6,
		    static_cast<double>(span) / 1e6, span != 0 ? 100.0 * static_cast<double>(w.union_ns) / static_cast<double>(span) : 0.0,
		    dropped - dropped_seen));
		if (s.mode == 2) { // desport: KYTY_GPU_TIMESTAMPS=2
			const auto pct = [](uint64_t part, uint64_t whole) {
				return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
			};
			const auto& k = w.by_kind;
			Desport::EmitLine(fmt::format(
			    "DES_GPU_WAIT t={} window_ms={} subs={} cbs={} cbs_per_frame={:.1f} present_subs={} waited={} waited_pct={:.2f} "
			    "wait_us={:.1f} wait_p95_us={:.0f} wait_max_us={:.1f} wait_ms={:.2f} w_async={} w_async_us={:.1f} w_gfx={} "
			    "w_gfx_us={:.1f} w_present={} w_present_us={:.1f} w_other={} w_other_us={:.1f} issue_us={:.1f} drain_us={:.1f} "
			    "calibrated={} cal_dev_us={:.1f} qlat_n={} qlat_us={:.1f} qlat_p50_us={:.0f} qlat_p95_us={:.0f} qlat_max_us={:.1f} "
			    "idle_submits={} idle_pct={:.1f} launch_us={:.1f} launch_p50_us={:.0f} launch_p95_us={:.0f} launch_max_us={:.1f}\n",
			    now, now - window_start, w.x_subs, w.cbs, per(w.cbs, frames, 1.0), w.present_subs, w.waited,
			    pct(w.waited, w.x_subs), per(w.wait_ns, w.waited, 1e3), Window::Percentile(w.wait_hist, w.waited, 0.95),
			    static_cast<double>(w.wait_max) / 1e3, static_cast<double>(w.wait_ns) / 1e6, k[0].subs,
			    per(k[0].ns, k[0].subs, 1e3), k[1].subs, per(k[1].ns, k[1].subs, 1e3), k[2].subs, per(k[2].ns, k[2].subs, 1e3),
			    k[3].subs, per(k[3].ns, k[3].subs, 1e3), per(w.issue_ns, w.split_n, 1e3), per(w.drain_ns, w.split_n, 1e3),
			    calibrated ? 1 : 0, static_cast<double>(cal_dev_max) / 1e3, w.qlat_n, per(w.qlat_ns, w.qlat_n, 1e3),
			    Window::Percentile(w.qlat_hist, w.qlat_n, 0.5), Window::Percentile(w.qlat_hist, w.qlat_n, 0.95),
			    static_cast<double>(w.qlat_max) / 1e3, w.idle_submits, pct(w.idle_submits, w.qlat_n),
			    per(w.launch_ns, w.idle_submits, 1e3), Window::Percentile(w.launch_hist, w.idle_submits, 0.5),
			    Window::Percentile(w.launch_hist, w.idle_submits, 0.95), static_cast<double>(w.launch_max) / 1e3));
			cal_dev_max = 0;
		}
		window       = std::make_unique<Window>();
		window_start = now;
		flips_start  = flips;
		dropped_seen = dropped;
	}
}

// The renderer's scheduler, once (KYTY_GPU_TIMESTAMPS set). Logs "DES_GPU_TS on=<0|1> ..." either way.
void Init(GraphicContext& graphics) {
	if (g_state.load(std::memory_order_acquire) != nullptr) return;
	if (!Requested()) {
		std::puts("DES_GPU_TS on=0");
		std::fflush(stdout);
		return;
	}
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits = graphics.queue_family < families.size() ? families[graphics.queue_family].timestampValidBits : 0u;
	if (bits == 0) {
		std::printf("DES_GPU_TS on=0 reason=no_timestamps_on_queue_family\n");
		std::fflush(stdout);
		return;
	}
	auto* s = new State();
	s->mode = Mode(); // desport: KYTY_GPU_TIMESTAMPS=2
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	s->d = &LocalVulkanRecording::DirectDispatch(); // (never recorded: these buffers are filled once, here)
	if (s->d->vkCmdWriteTimestamp == nullptr) s->d = &VULKAN_HPP_DEFAULT_DISPATCHER;
#else
	s->d = &VULKAN_HPP_DEFAULT_DISPATCHER;
#endif
	const auto& d = *s->d;
	s->device     = static_cast<VkDevice>(graphics.device);
	s->period     = static_cast<double>(graphics.GetPhysicalDeviceProperties().limits.timestampPeriod);
	s->mask       = bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << bits) - 1u;
	VkQueryPoolCreateInfo query {};
	query.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	query.queryCount = Slots * Queries; // desport: KYTY_GPU_TIMESTAMPS=2 (head, tail, pre, tail-top per slot)
	VkCommandPoolCreateInfo pool {};
	pool.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool.queueFamilyIndex = graphics.queue_family;
	if (d.vkCreateQueryPool(s->device, &query, nullptr, &s->pool) != VK_SUCCESS ||
	    d.vkCreateCommandPool(s->device, &pool, nullptr, &s->commands) != VK_SUCCESS) {
		std::printf("DES_GPU_TS on=0 reason=create_failed\n");
		std::fflush(stdout);
		return; // (leaked: a diagnostic that failed to start)
	}
	if (d.vkResetQueryPool != nullptr) d.vkResetQueryPool(s->device, s->pool, 0, Slots * Queries);
	VkCommandBufferAllocateInfo allocate {};
	allocate.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate.commandPool        = s->commands;
	allocate.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = Slots;
	EXIT_NOT_IMPLEMENTED(d.vkAllocateCommandBuffers(s->device, &allocate, s->heads.data()) != VK_SUCCESS ||
	                     d.vkAllocateCommandBuffers(s->device, &allocate, s->tails.data()) != VK_SUCCESS);
	if (s->mode == 2) EXIT_NOT_IMPLEMENTED(d.vkAllocateCommandBuffers(s->device, &allocate, s->pres.data()) != VK_SUCCESS);
	const bool ts2 = d.vkCmdWriteTimestamp2 != nullptr;
	const auto stamp = [&](VkCommandBuffer command, bool top, uint32_t query_index) {
		if (ts2)
			d.vkCmdWriteTimestamp2(command, top ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			                       s->pool, query_index);
		else
			d.vkCmdWriteTimestamp(command, top ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			                      s->pool, query_index);
	};
	// Resubmitted once the reader took the slot's last use (its tick was reached): no simultaneous use, no reset.
	VkCommandBufferBeginInfo begin {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	for (uint32_t slot = 0; slot < Slots; ++slot) {
		const uint32_t q = slot * Queries; // desport: KYTY_GPU_TIMESTAMPS=2
		EXIT_NOT_IMPLEMENTED(d.vkBeginCommandBuffer(s->heads[slot], &begin) != VK_SUCCESS);
		d.vkCmdResetQueryPool(s->heads[slot], s->pool, q, 2u);
		stamp(s->heads[slot], true, q);
		EXIT_NOT_IMPLEMENTED(d.vkEndCommandBuffer(s->heads[slot]) != VK_SUCCESS);
		EXIT_NOT_IMPLEMENTED(d.vkBeginCommandBuffer(s->tails[slot], &begin) != VK_SUCCESS);
		if (s->mode == 2) { // desport: the front end reached the tail (issued the submission's commands)
			d.vkCmdResetQueryPool(s->tails[slot], s->pool, q + 3u, 1u);
			stamp(s->tails[slot], true, q + 3u);
		}
		stamp(s->tails[slot], false, q + 1u);
		EXIT_NOT_IMPLEMENTED(d.vkEndCommandBuffer(s->tails[slot]) != VK_SUCCESS);
		if (s->mode == 2) { // desport: the work before it on the queue done (in a batch of its own, ahead of the waits)
			EXIT_NOT_IMPLEMENTED(d.vkBeginCommandBuffer(s->pres[slot], &begin) != VK_SUCCESS);
			d.vkCmdResetQueryPool(s->pres[slot], s->pool, q + 2u, 1u);
			stamp(s->pres[slot], false, q + 2u);
			EXIT_NOT_IMPLEMENTED(d.vkEndCommandBuffer(s->pres[slot]) != VK_SUCCESS);
		}
	}
	// desport: begin KYTY_GPU_TIMESTAMPS=2. The calibration (VK_KHR/EXT_calibrated_timestamps, enabled at device creation
	// for mode 2 when the device has it) and the host counter's frequency (Common::Timer: QPC on Windows).
	const char* calibration = "off";
	if (s->mode == 2) {
		s->host_frequency = std::max<uint64_t>(Common::Timer::QueryPerformanceFrequency(), 1);
		calibration       = "no_extension";
		if (graphics.desport_calibrated_timestamps != 0 && d.vkGetDeviceProcAddr != nullptr) {
			const bool khr = graphics.desport_calibrated_timestamps == 1;
			s->calibrate   = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                d.vkGetDeviceProcAddr(s->device, khr ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT"));
			calibration    = s->calibrate == nullptr ? "no_entry_point" : (khr ? "khr" : "ext");
			// (The host domain must be one the device calibrates against.)
			const auto domains_call = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR != nullptr
			                              ? VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
			                              : VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
			if (s->calibrate != nullptr && domains_call != nullptr) {
				const auto physical = static_cast<VkPhysicalDevice>(graphics.physical_device);
				uint32_t   count    = 0;
				std::vector<VkTimeDomainKHR> domains;
				if (domains_call(physical, &count, nullptr) == VK_SUCCESS && count != 0) {
					domains.resize(count);
					if (domains_call(physical, &count, domains.data()) != VK_SUCCESS) count = 0;
					domains.resize(count);
				}
#if defined(_WIN32)
				const VkTimeDomainKHR host_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
				const VkTimeDomainKHR host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
				const bool device_ok = std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_DEVICE_KHR) != domains.end();
				const bool host_ok   = std::find(domains.begin(), domains.end(), host_domain) != domains.end();
				if (!device_ok || !host_ok) {
					s->calibrate = nullptr;
					calibration  = "no_host_domain";
				}
			}
		}
	}
	// desport: end KYTY_GPU_TIMESTAMPS=2
	s->reader = std::jthread([s](const std::stop_token& stop) { ReaderLoop(stop, *s); });
	g_state.store(s, std::memory_order_release);
	std::printf("DES_GPU_TS on=1 mode=%u slots=%u period_ns=%.3f valid_bits=%u ts2=%u calibration=%s\n", s->mode, Slots,
	            s->period, bits, ts2 ? 1u : 0u, calibration);
	std::fflush(stdout);
}

// Under the graphics queue's lock, right before its vkQueueSubmit: the ring slot whose head/tail buffers wrap the
// submission (signalling `tick` of `master`, `ticks` timeline ticks merged, `writes` writing commands), or UINT32_MAX.
// desport: KYTY_GPU_TIMESTAMPS=2: `extra` (waits, their kinds, main, cbs; the host time and `pre` are set here). With
// extra.pre set on return the caller puts Pre(slot) in a batch of its own ahead of the waiting one.
uint32_t Begin(MasterSemaphore* master, uint64_t tick, uint32_t ticks, uint32_t writes, Extra extra = {}) {
	auto* s = g_state.load(std::memory_order_acquire);
	if (s == nullptr || s->stopped.load(std::memory_order_acquire)) return UINT32_MAX;
	const uint64_t seq = s->next;
	if (seq - s->read.load(std::memory_order_acquire) >= Slots) {
		s->dropped.fetch_add(1, std::memory_order_relaxed);
		return UINT32_MAX;
	}
	auto& entry  = s->entries[seq % Slots];
	entry.master = master;
	entry.tick   = tick;
	entry.ticks  = ticks;
	entry.writes = writes;
	// desport: KYTY_GPU_TIMESTAMPS=2
	extra.pre   = s->mode == 2 && extra.waits != 0;
	extra.host  = s->mode == 2 ? Common::Timer::QueryPerformanceCounter() : 0;
	entry.extra = extra;
	entry.published.store(seq + 1, std::memory_order_release);
	s->next = seq + 1;
	return static_cast<uint32_t>(seq % Slots);
}
VkCommandBuffer Head(uint32_t slot) { return g_state.load(std::memory_order_relaxed)->heads[slot]; }
VkCommandBuffer Tail(uint32_t slot) { return g_state.load(std::memory_order_relaxed)->tails[slot]; }
// desport: KYTY_GPU_TIMESTAMPS=2: the slot's pre buffer (put ahead, in a batch of its own, when Begin set extra.pre).
VkCommandBuffer Pre(uint32_t slot) { return g_state.load(std::memory_order_relaxed)->pres[slot]; }
bool            PreWanted(uint32_t slot) {
    return slot != UINT32_MAX && g_state.load(std::memory_order_relaxed)->entries[slot].extra.pre;
}
// desport: KYTY_GPU_TIMESTAMPS=2: the Kind bits of `count` semaphores a submission of `master` waits for (`main`: the
// renderer's scheduler; the presenter's waits for the swapchain image).
uint8_t WaitKinds(const MasterSemaphore* master, bool main, const VkSemaphore* semaphores, uint32_t count) {
	uint8_t kinds = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const auto semaphore = vk::Semaphore(semaphores[i]);
		if (master != nullptr && master->AsyncHandle() && semaphore == master->AsyncHandle()) kinds |= KindAsync;
		else if (master != nullptr && semaphore == master->Handle()) kinds |= KindGfx;
		else if (!main) kinds |= KindPresent;
		else kinds |= KindOther;
	}
	return kinds;
}
// A scheduler shuts down: the reader stops before the timelines it reads go away (no more slots are given out).
void Stop() {
	auto* s = g_state.load(std::memory_order_acquire);
	if (s == nullptr || s->stopped.exchange(true, std::memory_order_acq_rel)) return;
	s->reader.request_stop();
	if (s->reader.joinable()) s->reader.join();
}
} // namespace GpuTs
} // namespace
// desport: end KYTY_GPU_TIMESTAMPS

#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace {
// vkEndCommandBuffer + vkQueueSubmit executed by the recording worker, in order with
// the commands it has replayed. The tick is assigned on the producer as before.
struct DeferredSubmit {
	VkCommandBuffer      commands[CommandScheduler::MaxSubmitEntries];
	bool                 ended[CommandScheduler::MaxSubmitEntries]; // recorded and ended on another thread
	uint32_t             count;
	VkQueue              queue;
	Common::Mutex*       queue_mutex;
	uint32_t             waits, signals;
	VkSemaphore          wait_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             wait_ticks[SubmitInfo::MaxSemaphores];
	VkPipelineStageFlags wait_stages[SubmitInfo::MaxSemaphores];
	VkSemaphore          signal_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             signal_ticks[SubmitInfo::MaxSemaphores];
	uint64_t             tick;
	uint64_t             upload_sequence; // KYTY_ASYNC_UPLOAD copies these buffers read
	uint32_t             timestamp_slot;  // live trace (the last buffer the stream ends)
	std::atomic<uint64_t>* driver_tick;   // CommandScheduler::m_driver_tick (GpuBusy)
	MasterSemaphore*       master;        // desport: KYTY_READBACK_LATENCY (the timeline at the hand-over)
	bool                   async;         // desport: KYTY_ASYNC_COMPUTE (an async-queue submission: tick = its epoch)
	uint64_t               q;             // desport: KYTY_SUBMIT_COALESCE (the main scheduler's number; 0: other)
	uint32_t               writes;        // desport: KYTY_GPU_TIMESTAMPS (writing commands; UINT32_MAX: unknown)
	bool                   main;          // desport: KYTY_GPU_TIMESTAMPS=2 (the renderer's scheduler, not the presenter's)
};

// desport: begin KYTY_SUBMIT_COALESCE. KYTY_SUBMIT_COALESCE=<n> (0/absent: off; 1: 16; 2-64: n): the recording worker
// holds the renderer's plain graphics submissions (no semaphore wait, only the timeline signal) and hands up to n of them
// to the driver as one vkQueueSubmit - one batch, its command buffers in order, one timeline signal with the last tick
// (KYTY_SUBMIT_COALESCE_SIGNALS=each: one batch per tick, each with its own signal, in the same call). Waiting for an
// earlier tick is waiting for a later value of the same timeline: every waiter is still satisfied, at most when the
// whole batch completes. The worker holds while the GPU still has KYTY_SUBMIT_COALESCE_DEPTH (default 2) of its batches
// it has not finished, and hands the held ones over at once when: n are held, a thread waits for one of their ticks
// (MasterSemaphore::Wait), a wait for everything queued is asked (WaitDeferredSubmits: presents; another thread's direct
// submission), the oldest is KYTY_SUBMIT_COALESCE_US old (default 500), or any other submission is about to be made on
// the worker (an async-compute one, a readback copy: they may wait on the graphics timeline, and a wait-before-signal
// stalls the Windows driver). "DES_SUBMIT_COALESCE ..." every 5 s.
namespace Coalesce {
struct Config {
	uint32_t max     = 0;
	uint32_t depth   = 2;
	uint64_t hold_ns = 500000;
	bool     each    = false;
};
const Config& Get() {
	static const Config config = [] {
		const auto number = [](const char* name, uint64_t fallback, uint64_t low, uint64_t high) {
			const char* text = std::getenv(name);
			if (text == nullptr || *text == '\0') return fallback;
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 10);
			if (*end != '\0' || value < low || value > high) {
				std::fprintf(stderr, "%s must be a value from %llu to %llu\n", name, static_cast<unsigned long long>(low),
				             static_cast<unsigned long long>(high));
				std::abort();
			}
			return static_cast<uint64_t>(value);
		};
		Config c;
		const uint64_t max = number("KYTY_SUBMIT_COALESCE", 0, 0, 64);
		c.max              = static_cast<uint32_t>(max == 1 ? 16 : max);
		c.depth            = static_cast<uint32_t>(number("KYTY_SUBMIT_COALESCE_DEPTH", 2, 1, 8));
		c.hold_ns          = number("KYTY_SUBMIT_COALESCE_US", 500, 20, 20000) * 1000u;
		const char* signals = std::getenv("KYTY_SUBMIT_COALESCE_SIGNALS");
		c.each              = signals != nullptr && std::strcmp(signals, "each") == 0;
		return c;
	}();
	return config;
}
bool On() { return Get().max != 0; }

enum Id : uint32_t {
	Calls,       // vkQueueSubmit calls of the worker's graphics hand-overs (held batches and single ones)
	Ticks,       // timeline ticks they carried
	Batches,     // held batches handed over
	Merged,      // ticks in them
	MaxBatch,    // the largest (max)
	HoldNs,      // sum of the oldest's age at the hand-over
	ReasonCap,   // why a held batch went: n held
	ReasonTick,  // a thread waits for one of its ticks
	ReasonAll,   // a wait for everything queued
	ReasonAge,   // the oldest reached KYTY_SUBMIT_COALESCE_US
	ReasonGpu,   // the GPU had fewer than KYTY_SUBMIT_COALESCE_DEPTH batches left
	ReasonForce, // another submission on the worker (readback copy, async compute), or its end
	ReasonOther, // a graphics submission that cannot be held (it waits, or signals more than the timeline)
	GpuBusyHeld, // CommandScheduler::GpuBusy: busy because submissions were queued and not handed over
	Count
};
std::atomic<uint64_t> g_values[Count] {};
void Add(Id id, uint64_t n = 1) { g_values[id].fetch_add(n, std::memory_order_relaxed); }

void Report() {
	static std::atomic<uint64_t> last_ms {0};
	static uint64_t              last[Count] {};
	static std::mutex            mutex;
	const uint64_t               now      = Desport::ElapsedMilliseconds();
	uint64_t                     previous = last_ms.load(std::memory_order_relaxed);
	if (previous != 0 && now - previous < 5000) return;
	std::unique_lock lock(mutex, std::try_to_lock);
	if (!lock.owns_lock()) return;
	previous = last_ms.load(std::memory_order_relaxed);
	if (previous != 0 && now - previous < 5000) return;
	uint64_t d[Count];
	for (size_t i = 0; i < Count; ++i) {
		const uint64_t value = g_values[i].load(std::memory_order_relaxed);
		d[i]                 = i == MaxBatch ? value : value - last[i];
		last[i]              = value;
	}
	g_values[MaxBatch].store(0, std::memory_order_relaxed);
	last_ms.store(now, std::memory_order_relaxed);
	if (previous == 0) return;
	const auto per = [](uint64_t sum, uint64_t n, double scale) {
		return n != 0 ? static_cast<double>(sum) / scale / static_cast<double>(n) : 0.0;
	};
	Desport::EmitLine(fmt::format(
	    "DES_SUBMIT_COALESCE t={} window_ms={} calls={} ticks={} ticks_per_call={:.2f} batches={} merged={} max_batch={} "
	    "hold_us={:.1f} r_cap={} r_tick={} r_all={} r_age={} r_gpu={} r_force={} r_other={} gpubusy_held={}\n",
	    now, now - previous, d[Calls], d[Ticks], per(d[Ticks], d[Calls], 1.0), d[Batches], d[Merged], d[MaxBatch],
	    per(d[HoldNs], d[Batches], 1e3), d[ReasonCap], d[ReasonTick], d[ReasonAll], d[ReasonAge], d[ReasonGpu],
	    d[ReasonForce], d[ReasonOther], d[GpuBusyHeld]));
}

constexpr uint32_t MaxHeldCommands = 256;
// The worker's held submissions and the ticks of its last hand-overs (the GPU's backlog in batches).
struct Held {
	std::vector<DeferredSubmit> subs;
	uint32_t                    commands     = 0;
	uint64_t                    first_ns     = 0;
	uint64_t                    refreshed_ns = 0;
	std::array<uint64_t, 8>     handed {};
	uint32_t                    handed_next = 0;
};
thread_local Held t_held;
// Scratch of HandOver (worker thread or a producer's fallback).
thread_local std::vector<VkCommandBuffer>               t_commands;
thread_local std::vector<VkSubmitInfo>                  t_infos;
thread_local std::vector<VkTimelineSemaphoreSubmitInfo> t_timelines;
} // namespace Coalesce
// desport: end KYTY_SUBMIT_COALESCE

// desport: KYTY_SUBMIT_COALESCE / KYTY_GPU_TIMESTAMPS. Hands `n` deferred submissions (their buffers ended) to the driver
// in one vkQueueSubmit: n == 1 exactly as given; n > 1 (held: no waits, only the timeline signal) as one batch signalling
// the last tick, or one batch each (KYTY_SUBMIT_COALESCE_SIGNALS=each). Graphics-queue ones are wrapped in the
// timestamp buffers (KYTY_GPU_TIMESTAMPS).
void HandOver(const DeferredSubmit* subs, uint32_t n, const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& first = subs[0];
	const auto& last  = subs[n - 1];
	const bool  each  = n > 1 && Coalesce::Get().each;
	uint32_t    writes = 0;
	for (uint32_t i = 0; i < n; ++i) {
		writes = subs[i].writes == UINT32_MAX || writes == UINT32_MAX ? UINT32_MAX : writes + subs[i].writes;
		for (uint32_t j = 0; j < subs[i].signals; ++j) {
			if (subs[i].signal_ticks[j] != subs[i].tick) continue;
			DeviceFault::NoteSubmit((uint64_t)(subs[i].signal_semaphores[j]), subs[i].tick, subs[i].commands, // NOLINT
			                        subs[i].count, subs[i].wait_semaphores, subs[i].wait_ticks, subs[i].waits);
			break;
		}
	}
	auto& commands  = Coalesce::t_commands;
	auto& infos     = Coalesce::t_infos;
	auto& timelines = Coalesce::t_timelines;
	VkResult result;
	{
		Common::LockGuard lock(*first.queue_mutex);
		// (The timestamp ring's order is the queue's: its slot is taken under the queue's lock.)
		// desport: KYTY_GPU_TIMESTAMPS=2: the waits (the first's: held ones have none), their kinds, the buffers.
		GpuTs::Extra extra;
		extra.waits = first.waits;
		extra.kinds = GpuTs::WaitKinds(first.master, first.main, first.wait_semaphores, first.waits);
		extra.main  = first.main;
		for (uint32_t i = 0; i < n; ++i) extra.cbs += subs[i].count;
		const uint32_t slot = first.async ? UINT32_MAX : GpuTs::Begin(last.master, last.tick, n, writes, extra);
		commands.clear();
		infos.assign(each ? n : 1u, VkSubmitInfo {});
		timelines.assign(each ? n : 1u, VkTimelineSemaphoreSubmitInfo {});
		std::array<uint32_t, 64 + 1> starts {}; // (each: where batch i's buffers begin in `commands`)
		if (slot != UINT32_MAX) commands.push_back(GpuTs::Head(slot));
		for (uint32_t i = 0; i < n; ++i) {
			if (i != 0) starts[i] = static_cast<uint32_t>(commands.size());
			commands.insert(commands.end(), subs[i].commands, subs[i].commands + subs[i].count);
		}
		if (slot != UINT32_MAX) commands.push_back(GpuTs::Tail(slot));
		starts[each ? n : 1u] = static_cast<uint32_t>(commands.size());
		for (uint32_t b = 0; b < infos.size(); ++b) {
			// One batch: the first's waits (none when held) and the last's signals. Each: batch b, its own.
			const auto& waiter = each ? subs[b] : first;
			const auto& signal = each ? subs[b] : last;
			auto&       timeline = timelines[b];
			timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
			timeline.waitSemaphoreValueCount   = waiter.waits;
			timeline.pWaitSemaphoreValues      = waiter.wait_ticks;
			timeline.signalSemaphoreValueCount = signal.signals;
			timeline.pSignalSemaphoreValues    = signal.signal_ticks;
			auto& info                = infos[b];
			info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.pNext                = &timeline;
			info.waitSemaphoreCount   = waiter.waits;
			info.pWaitSemaphores      = waiter.wait_semaphores;
			info.pWaitDstStageMask    = waiter.wait_stages;
			info.commandBufferCount   = starts[b + 1] - starts[b];
			info.pCommandBuffers      = commands.data() + starts[b];
			info.signalSemaphoreCount = signal.signals;
			info.pSignalSemaphores    = signal.signal_semaphores;
		}
		// desport: KYTY_GPU_TIMESTAMPS=2: the pre timestamp in a batch of its own (no waits, no signals) ahead of the
		// waiting one: head - pre is the time the queue sat in the waits.
		VkCommandBuffer pre = VK_NULL_HANDLE;
		if (GpuTs::PreWanted(slot)) {
			pre = GpuTs::Pre(slot);
			VkSubmitInfo info {};
			info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.commandBufferCount = 1;
			info.pCommandBuffers    = &pre;
			infos.insert(infos.begin(), info);
		}
		result = dispatch.vkQueueSubmit(first.queue, static_cast<uint32_t>(infos.size()), infos.data(), VK_NULL_HANDLE);
	}
	if (result != VK_SUCCESS) {
		if (result == VK_ERROR_DEVICE_LOST) DeviceFault::Report();
		EXIT("deferred vkQueueSubmit failed: %d, tick=%" PRIu64 "\n", static_cast<int>(result), last.tick);
	}
	last.driver_tick->store(last.tick, std::memory_order_release);
	if (ReadbackLatency::Stats() && !last.async) { // desport: KYTY_READBACK_LATENCY (graphics ticks only)
		last.master->Refresh();
		const uint64_t known = last.master->KnownGpuTick();
		for (uint32_t i = 0; i < n; ++i) ReadbackLatency::NoteDriver(subs[i].tick, known);
	}
	for (uint32_t i = 0; i < n; ++i) LocalVulkanRecording::NoteDeferredSubmitDone();
	if (last.q != 0) {
		auto& handed = LocalVulkanRecording::g_main_handed;
		for (auto known = handed.load(std::memory_order_relaxed);
		     known < last.q && !handed.compare_exchange_weak(known, last.q, std::memory_order_seq_cst);) {
		}
		if (LocalVulkanRecording::OnWorker()) {
			auto& held                                                       = Coalesce::t_held;
			held.handed[held.handed_next++ % static_cast<uint32_t>(held.handed.size())] = last.tick;
			Coalesce::Add(Coalesce::Calls);
			Coalesce::Add(Coalesce::Ticks, n);
			Coalesce::Report();
		}
	}
}

// desport: begin KYTY_SUBMIT_COALESCE (worker thread)
namespace Coalesce {
bool Holdable(const DeferredSubmit& submit) {
	return Get().max != 0 && submit.q != 0 && !submit.async && submit.waits == 0 && submit.signals == 1 &&
	       submit.signal_ticks[0] == submit.tick;
}
void Flush(Id reason) {
	auto& held = t_held;
	if (held.subs.empty()) return;
	const auto n   = static_cast<uint32_t>(held.subs.size());
	const auto age = ReadbackLatency::NowNs() - held.first_ns;
	HandOver(held.subs.data(), n, LocalVulkanRecording::DirectDispatch());
	Add(reason);
	Add(Batches);
	Add(Merged, n);
	Add(HoldNs, age);
	for (auto known = g_values[MaxBatch].load(std::memory_order_relaxed);
	     known < n && !g_values[MaxBatch].compare_exchange_weak(known, n, std::memory_order_relaxed);) {
	}
	held.subs.clear();
	held.commands = 0;
	Report();
}
// Hands the held ones over when one of the reasons holds; true while some are still held.
bool Evaluate(bool refresh) {
	auto& held = t_held;
	if (held.subs.empty()) return false;
	const auto&    first  = held.subs.front();
	const auto&    config = Get();
	const uint64_t now    = ReadbackLatency::NowNs();
	Id             reason = Count;
	if (held.subs.size() >= config.max) {
		reason = ReasonCap;
	} else if (first.master->WantTick() >= first.tick) {
		reason = ReasonTick;
	} else if (LocalVulkanRecording::g_want_q.load(std::memory_order_seq_cst) >= first.q) {
		reason = ReasonAll;
	} else if (now - held.first_ns >= config.hold_ns) {
		reason = ReasonAge;
	} else {
		if (refresh || now - held.refreshed_ns >= 10000) {
			first.master->Refresh();
			held.refreshed_ns = now;
		}
		const uint64_t gpu         = first.master->KnownGfxTick();
		uint32_t       outstanding = 0;
		for (const auto tick: held.handed) outstanding += tick > gpu ? 1u : 0u;
		if (outstanding < config.depth) reason = ReasonGpu;
	}
	if (reason == Count) return true;
	Flush(reason);
	return false;
}
bool Poll(bool force) {
	if (force) {
		Flush(ReasonForce);
		return false;
	}
	return Evaluate(false);
}
void Hold(const DeferredSubmit& submit) {
	auto& held = t_held;
	if (!held.subs.empty() && held.commands + submit.count > MaxHeldCommands) Flush(ReasonCap);
	if (held.subs.empty()) {
		held.first_ns = ReadbackLatency::NowNs();
		LocalVulkanRecording::SetIdlePoll(&Poll);
	}
	held.subs.push_back(submit);
	held.commands += submit.count;
	(void)Evaluate(true);
}
} // namespace Coalesce
// desport: end KYTY_SUBMIT_COALESCE

void ReplaySubmit(std::span<const LocalVulkanRecording::Segment> segments,
                  const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& submit = *static_cast<const DeferredSubmit*>(segments[0].data);
	AsyncUpload::Wait(submit.upload_sequence);
	LiveTrace::Event(LiveTrace::GpuSubmit, submit.tick, 1);
	for (uint32_t i = 0; i < submit.count; ++i) {
		if (submit.ended[i]) continue;
		if (submit.timestamp_slot != UINT32_MAX && i + 1 == submit.count && LiveTrace::g_timestamp_pool != nullptr) {
			dispatch.vkCmdWriteTimestamp(submit.commands[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			                             static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool),
			                             submit.timestamp_slot * 2u + 1u);
			LiveTrace::g_tick_slot[submit.tick % LiveTrace::TimestampSlots].store(submit.timestamp_slot + 1u,
			                                                                        std::memory_order_release);
		}
		if (dispatch.vkEndCommandBuffer(submit.commands[i]) != VK_SUCCESS) {
			EXIT("deferred vkEndCommandBuffer failed, tick=%" PRIu64 "\n", submit.tick);
		}
	}
	// desport: begin KYTY_SUBMIT_COALESCE. On the worker: held if it may be, else after what is held (the queue's and
	// the timeline's order). A producer's own submission (its stream was full) after everything the worker holds.
	if (LocalVulkanRecording::OnWorker()) {
		if (Coalesce::Holdable(submit)) {
			Coalesce::Hold(submit);
			return;
		}
		Coalesce::Flush(submit.async ? Coalesce::ReasonForce : Coalesce::ReasonOther);
	} else if (submit.q > 1 && Coalesce::On()) {
		LocalVulkanRecording::RaiseWantQ(submit.q - 1);
		while (LocalVulkanRecording::g_main_handed.load(std::memory_order_acquire) < submit.q - 1) std::this_thread::yield();
	}
	// desport: end KYTY_SUBMIT_COALESCE
	HandOver(&submit, 1, dispatch);
}
// A recorded buffer dropped before submission, in order with the commands replayed into it.
void ReplayDiscard(std::span<const LocalVulkanRecording::Segment> segments,
                   const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto command = *static_cast<const VkCommandBuffer*>(segments[0].data);
	DeviceFault::NoteDiscard(command);
	if (dispatch.vkResetCommandBuffer(command, 0) != VK_SUCCESS) {
		EXIT("deferred vkResetCommandBuffer failed\n");
	}
}
} // namespace
#endif

// desport: begin KYTY_GPU_TIMESTAMPS / KYTY_SUBMIT_COALESCE
namespace {
// The renderer's scheduler starts both (their boot lines: "DES_GPU_TS on=...", "DES_SUBMIT_COALESCE on=...").
void DesportSubmitInit(GraphicContext& graphics) {
	GpuTs::Init(graphics);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	const auto& config = Coalesce::Get();
	LocalVulkanRecording::g_coalesce_on.store(config.max != 0, std::memory_order_seq_cst);
	std::printf("DES_SUBMIT_COALESCE on=%u max=%u depth=%u hold_us=%llu signals=%s (needs KYTY_DEFERRED_SUBMIT=1)\n",
	            config.max != 0 ? 1u : 0u, config.max, config.depth, static_cast<unsigned long long>(config.hold_ns / 1000u),
	            config.each ? "each" : "last");
#else
	std::printf("DES_SUBMIT_COALESCE on=0 reason=no_recording_worker\n");
#endif
	std::fflush(stdout);
}
// GpuBusy: submissions of the renderer queued and not handed to the driver yet (held, or the worker behind) are work the
// GPU gets anyway: a batch boundary gives it nothing.
bool DesportCoalesceQueued() {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (!Coalesce::On()) return false;
	if (LocalVulkanRecording::g_main_queued.load(std::memory_order_acquire) >
	    LocalVulkanRecording::g_main_handed.load(std::memory_order_acquire)) {
		Coalesce::Add(Coalesce::GpuBusyHeld);
		return true;
	}
#endif
	return false;
}
} // namespace
// desport: end KYTY_GPU_TIMESTAMPS / KYTY_SUBMIT_COALESCE

vk::CommandBuffer CommandScheduler::UploadPrologue(uint64_t last_dirty, uint64_t written_serial) {
	if (t_recorder != nullptr || kyty_local_upload_prologue.load(std::memory_order_relaxed) == 0 ||
	    m_command.IsInvalid() || last_dirty > m_command_clock || written_serial == m_command_serial)
		return nullptr;
	if (m_prologue.IsInvalid()) {
		// (Committed for the open buffer's submission: the pool reuses it once that completes.)
		m_prologue.m_buffer = m_command_pool.Commit();
		m_prologue.Begin();
		// The copies after everything submitted before (which may read or write their destinations).
		VulkanMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		                                    {}, 1, &before, 0, nullptr, 0, nullptr);
	}
	return m_prologue.Handle();
}

CommandScheduler::SubmitEntry CommandScheduler::ClosePrologue() {
	if (m_prologue_hook) m_prologue_hook(m_prologue.Handle());
	// Everything after the copies (the open buffer's commands) sees them.
	VulkanMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {},
	                                    1, &after, 0, nullptr, 0, nullptr);
	const SubmitEntry entry {m_prologue.m_buffer, false, UINT32_MAX};
	m_prologue.m_buffer = nullptr;
	return entry;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	if (t_recorder != nullptr) throw RecorderRefusal {"Submit"};
	EXIT_IF(m_command.IsInvalid());
	m_command.EndRendering();
	// A compute chain's pending barrier closes this buffer (left pending, the next Begin recorded it into a buffer
	// that had not begun).
	(void)m_command.Handle();
	std::array<SubmitEntry, 2> entries {};
	size_t                     n = 0;
	if (!m_prologue.IsInvalid()) entries[n++] = ClosePrologue();
	entries[n++]          = {m_command.m_buffer, false, m_command.m_timestamp_slot};
	ReadbackLatency::t_submit_work = m_recorded_draws + m_recorded_dispatches; // desport: KYTY_READBACK_LATENCY
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// desport: KYTY_GPU_TIMESTAMPS: draws, dispatches, copies and clears recorded into it (and its prologue).
	m_desport_submit_writes = static_cast<uint32_t>(
	    std::min<uint64_t>(LocalVulkanRecording::RecordedWrites() - m_desport_writes_begin, UINT32_MAX - 1u));
#endif
	const auto tick       = SubmitBuffers({entries.data(), n}, submit);
	m_command.m_buffer    = nullptr;
	m_recorded_dispatches = 0;
	m_recorded_draws      = 0;
	return tick;
}

void CommandScheduler::WaitBeforeNextSubmit(vk::Semaphore semaphore, uint64_t value) {
	// (One semaphore at a time: a later value of the same one replaces the earlier; another one is waited for now.)
	if (m_next_wait_semaphore == semaphore) {
		m_next_wait_value = std::max(m_next_wait_value, value);
		return;
	}
	if (m_next_wait_semaphore != nullptr) {
		vk::SemaphoreWaitInfo wait {};
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &m_next_wait_semaphore;
		wait.pValues        = &m_next_wait_value;
		EXIT_NOT_IMPLEMENTED(m_graphics.device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess);
	}
	m_next_wait_semaphore = semaphore;
	m_next_wait_value     = value;
}

uint64_t CommandScheduler::SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit) {
	const uint32_t desport_writes = std::exchange(m_desport_submit_writes, UINT32_MAX); // desport: KYTY_GPU_TIMESTAMPS
	if (m_next_wait_semaphore != nullptr) {
		if (submit.num_wait_semaphores < SubmitInfo::MaxSemaphores) {
			submit.AddWait(m_next_wait_semaphore, m_next_wait_value);
		} else {
			vk::SemaphoreWaitInfo wait {};
			wait.semaphoreCount = 1;
			wait.pSemaphores    = &m_next_wait_semaphore;
			wait.pValues        = &m_next_wait_value;
			EXIT_NOT_IMPLEMENTED(m_graphics.device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess);
		}
		m_next_wait_semaphore = nullptr;
	}
	// desport: begin KYTY_ASYNC_COMPUTE. The epoch's async buffer goes first; this graphics submission then waits for the
	// async timeline at the last async epoch (graphics after async in total order; once per async epoch: the graphics
	// queue keeps its order).
	if (m_async_pool != nullptr) {
		SubmitAsync();
		const uint64_t last = m_master.LastAsyncEpoch();
		if (last > m_gfx_async_waited) {
			if (!m_master.IsAsyncFree(last)) {
				if (submit.num_wait_semaphores < SubmitInfo::MaxSemaphores) {
					submit.AddWait(m_master.AsyncHandle(), last);
				} else {
					vk::SemaphoreWaitInfo wait {};
					const auto            semaphore = m_master.AsyncHandle();
					wait.semaphoreCount             = 1;
					wait.pSemaphores                = &semaphore;
					wait.pValues                    = &last;
					EXIT_NOT_IMPLEMENTED(m_graphics.device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess);
				}
				AsyncCompute::Add(AsyncCompute::GfxWaits);
			}
			m_gfx_async_waited = last;
		}
	}
	// desport: end KYTY_ASYNC_COMPUTE
	// The work may run once submitted: the pages it writes are read-protected first.
	PageManager::FlushDeferredProtection();
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	EXIT_IF(m_graphics.queue == nullptr || entries.empty() || entries.size() > MaxSubmitEntries);

#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (LocalVulkanRecording::DeferredSubmitEnabled()) {
		DeferredSubmit deferred {};
		// This thread alone allocates ticks and the worker submits them in stream order: no queue
		// lock here (it only waited for the worker's vkQueueSubmit, 1.2% of the render thread).
		deferred.tick = m_master.NextTick();
		ReadbackLatency::NoteEnqueue(deferred.tick); // desport: KYTY_READBACK_LATENCY
		deferred.master = &m_master;                 // desport
		deferred.async  = false;                     // desport: KYTY_ASYNC_COMPUTE
		deferred.writes = desport_writes;            // desport: KYTY_GPU_TIMESTAMPS
		deferred.main   = m_main;                    // desport: KYTY_GPU_TIMESTAMPS=2
		// desport: KYTY_SUBMIT_COALESCE: the renderer's submissions are numbered in queue order (the worker may hold them).
		deferred.q = m_main && Coalesce::On()
		                 ? LocalVulkanRecording::g_main_queued.fetch_add(1, std::memory_order_seq_cst) + 1
		                 : 0;
		submit.AddSignal(m_master.Handle(), deferred.tick);
		deferred.count          = static_cast<uint32_t>(entries.size());
		deferred.timestamp_slot = UINT32_MAX;
		deferred.driver_tick    = &m_driver_tick;
		for (size_t i = 0; i < entries.size(); ++i) {
			deferred.commands[i] = entries[i].buffer;
			deferred.ended[i]    = entries[i].ended;
			if (!entries[i].ended) deferred.timestamp_slot = entries[i].timestamp_slot;
		}
		deferred.upload_sequence = AsyncUpload::SubmitSequence();
		deferred.queue           = m_graphics.queue;
		deferred.queue_mutex     = &m_graphics.queue_mutex;
		deferred.waits           = submit.num_wait_semaphores;
		deferred.signals         = submit.num_signal_semaphores;
		for (uint32_t i = 0; i < deferred.waits; ++i) {
			deferred.wait_semaphores[i] = submit.wait_semaphores[i];
			deferred.wait_ticks[i]      = submit.wait_ticks[i];
			deferred.wait_stages[i]     = static_cast<VkPipelineStageFlags>(submit.wait_stages[i]);
		}
		for (uint32_t i = 0; i < deferred.signals; ++i) {
			deferred.signal_semaphores[i] = submit.signal_semaphores[i];
			deferred.signal_ticks[i]      = submit.signal_ticks[i];
		}
		const LocalVulkanRecording::Segment segments[] {{&deferred, sizeof(deferred)}};
		LocalVulkanRecording::NoteDeferredSubmitQueued();
		if (!LocalVulkanRecording::EnqueueDeferred(ReplaySubmit, segments, true)) {
			// Keep order: everything recorded so far, then this exact submit.
			LocalVulkanRecording::Drain();
			ReplaySubmit(segments, VULKAN_HPP_DEFAULT_DISPATCHER);
		}
		LiveTrace::Event(LiveTrace::GpuSubmit, deferred.tick);
		return deferred.tick;
	}
#endif
	// desport: KYTY_GPU_TIMESTAMPS: room for the timestamp buffers around them (buffers[0] and the one after the last).
	std::array<vk::CommandBuffer, MaxSubmitEntries + 2> buffers {};
	for (size_t i = 0; i < entries.size(); ++i) {
		buffers[i + 1] = entries[i].buffer;
		if (!entries[i].ended) EXIT_NOT_IMPLEMENTED(buffers[i + 1].end() != vk::Result::eSuccess);
	}
	auto& graphics = m_graphics;
	AsyncUpload::Drain();

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		ReadbackLatency::NoteEnqueue(tick); // desport: KYTY_READBACK_LATENCY
		submit.AddSignal(m_master.Handle(), tick);
		// desport: KYTY_GPU_TIMESTAMPS (the slot is taken under the queue's lock: the ring's order is the queue's).
		GpuTs::Extra ts_extra; // desport: KYTY_GPU_TIMESTAMPS=2
		ts_extra.waits = submit.num_wait_semaphores;
		ts_extra.kinds = GpuTs::WaitKinds(&m_master, m_main, reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()),
		                                  submit.num_wait_semaphores);
		ts_extra.main  = m_main;
		ts_extra.cbs   = static_cast<uint32_t>(entries.size());
		const uint32_t ts_slot  = GpuTs::Begin(&m_master, tick, 1, desport_writes, ts_extra);
		uint32_t       ts_first = 1, ts_count = static_cast<uint32_t>(entries.size());
		if (ts_slot != UINT32_MAX) {
			buffers[0]            = vk::CommandBuffer(GpuTs::Head(ts_slot));
			buffers[ts_count + 1] = vk::CommandBuffer(GpuTs::Tail(ts_slot));
			ts_first              = 0;
			ts_count += 2;
		}

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = ts_count;                  // desport: KYTY_GPU_TIMESTAMPS
		submit_info.pCommandBuffers      = buffers.data() + ts_first; // desport: KYTY_GPU_TIMESTAMPS
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
		DeviceFault::NoteSubmit((uint64_t)(static_cast<VkSemaphore>(m_master.Handle())), tick, // NOLINT
		                        reinterpret_cast<const VkCommandBuffer*>(buffers.data() + 1), // desport
		                        static_cast<uint32_t>(entries.size()),
		                        reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()), submit.wait_ticks.data(),
		                        submit.num_wait_semaphores);
#endif

		// desport: KYTY_GPU_TIMESTAMPS=2: the pre timestamp in a batch of its own ahead of the waiting one.
		std::array<vk::SubmitInfo, 2> ts_infos {vk::SubmitInfo {}, submit_info};
		vk::CommandBuffer             ts_pre {};
		uint32_t                      ts_first_info = 1;
		if (GpuTs::PreWanted(ts_slot)) {
			ts_pre                         = vk::CommandBuffer(GpuTs::Pre(ts_slot));
			ts_infos[0].commandBufferCount = 1;
			ts_infos[0].pCommandBuffers    = &ts_pre;
			ts_first_info                  = 0;
		}
		result = graphics.queue.submit(2u - ts_first_info, ts_infos.data() + ts_first_info, nullptr);
		m_driver_tick.store(tick, std::memory_order_release);
	}
	if (ReadbackLatency::Stats() && result == vk::Result::eSuccess) { // desport: KYTY_READBACK_LATENCY
		m_master.Refresh();
		ReadbackLatency::NoteDriver(tick, m_master.KnownGpuTick());
	}

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	return tick;
}

std::unique_ptr<CommandScheduler::Recorder> CommandScheduler::CreateRecorder() {
	auto                      recorder = std::make_unique<Recorder>();
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = m_graphics.queue_family;
	create.flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	EXIT_NOT_IMPLEMENTED(m_graphics.device.createCommandPool(&create, nullptr, &recorder->pool) != vk::Result::eSuccess);
	recorder->command = std::unique_ptr<CommandBuffer>(new CommandBuffer(*this));
	return recorder;
}

void CommandScheduler::DestroyRecorder(std::unique_ptr<Recorder> recorder) {
	if (!recorder) return;
	for (const auto& [buffer, tick]: recorder->buffers)
		if (tick != UINT64_MAX && tick != 0) m_master.Wait(tick);
	m_graphics.device.destroyCommandPool(recorder->pool, nullptr);
}

void CommandScheduler::SetThreadRecorder(Recorder* recorder) noexcept {
	t_recorder = recorder;
}

CommandScheduler::Recorder* CommandScheduler::ThreadRecorder() noexcept {
	return t_recorder;
}

void CommandScheduler::BeginRecording(Recorder& recorder, HW::Context& registers, HW::UserConfig& user_config,
                                      HW::Shader& shaders) {
	auto& command = *recorder.command;
	command.Bind(registers, user_config, shaders);
	if (!command.IsInvalid()) return;
	// A buffer whose last submission completed, else a new one.
	size_t index = recorder.buffers.size();
	for (size_t i = 0; i < recorder.buffers.size(); ++i) {
		const auto tick = recorder.buffers[i].second;
		if (tick != UINT64_MAX && (tick == 0 || m_master.IsFree(tick))) {
			index = i;
			break;
		}
	}
	if (index == recorder.buffers.size()) {
		m_master.Refresh();
		for (size_t i = 0; i < recorder.buffers.size() && index == recorder.buffers.size(); ++i)
			if (recorder.buffers[i].second != UINT64_MAX && m_master.IsFree(recorder.buffers[i].second)) index = i;
	}
	if (index == recorder.buffers.size()) {
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = recorder.pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		vk::CommandBuffer buffer;
		EXIT_NOT_IMPLEMENTED(m_graphics.device.allocateCommandBuffers(&allocate, &buffer) != vk::Result::eSuccess);
		recorder.buffers.emplace_back(buffer, 0);
	}
	recorder.buffers[index].second = UINT64_MAX;
	recorder.open                  = index;
	command.m_buffer               = recorder.buffers[index].first;
	command.Begin();
}

void CommandScheduler::EndRecording(Recorder& recorder) {
	auto& command = *recorder.command;
	if (command.IsInvalid()) return;
	command.EndRendering();
	(void)command.Handle(); // (a pending compute chain barrier)
	Recorder::Recorded recorded {.index = recorder.open, .timestamp_slot = command.m_timestamp_slot};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// Through a recording stream the commands are replayed later: the submission ends the buffer, in order.
	const bool streamed = LocalVulkanRecording::DeferredSubmitEnabled() && !recorder.own_thread;
#else
	const bool streamed = false;
#endif
	if (!streamed) {
		EXIT_NOT_IMPLEMENTED(command.m_buffer.end() != vk::Result::eSuccess);
		recorded.ended = true;
	}
	recorder.recorded.push_back(recorded);
	recorder.open    = SIZE_MAX;
	command.m_buffer = nullptr;
}

uint64_t CommandScheduler::SubmitRecorded(Recorder& recorder, size_t count) {
	EXIT_IF(t_recorder != nullptr || !recorder.command->IsInvalid() || count > recorder.recorded.size());
	// This scheduler's open buffer first (its commands precede the recorded ones), its next one begun after them:
	// what is recorded into a buffer is fenced with the tick the buffer is submitted with (CurrentTick).
	CheckActive();
	const bool open = !m_command.IsInvalid();
	// (One submission for the open buffer and the recorded ones after it.)
	std::array<SubmitEntry, MaxSubmitEntries> entries {};
	size_t                                    n = 0;
	if (open) {
		m_command.EndRendering();
		(void)m_command.Handle(); // (a pending compute chain barrier closes it, as Submit's)
		if (!m_prologue.IsInvalid()) entries[n++] = ClosePrologue();
		entries[n++]          = {m_command.m_buffer, false, m_command.m_timestamp_slot};
		m_command.m_buffer    = nullptr;
		m_recorded_dispatches = 0;
		m_recorded_draws      = 0;
	}
	uint64_t tick = 0;
	for (size_t i = 0, first = 0; i < count; ++i) {
		const auto& recorded = recorder.recorded[i];
		entries[n++]         = {recorder.buffers[recorded.index].first, recorded.ended, recorded.timestamp_slot};
		if (n == MaxSubmitEntries || i + 1 == count) {
			tick = SubmitBuffers({entries.data(), n}, {});
			for (size_t j = first; j <= i; ++j) recorder.buffers[recorder.recorded[j].index].second = tick;
			first = i + 1;
			n     = 0;
		}
	}
	if (n != 0) (void)SubmitBuffers({entries.data(), n}, {}); // (the open buffer alone: nothing recorded)
	recorder.recorded.erase(recorder.recorded.begin(), recorder.recorded.begin() + static_cast<ptrdiff_t>(count));
	if (open) BeginNext();
	return tick;
}

void CommandScheduler::DiscardRecorded(Recorder& recorder) {
	EndRecording(recorder);
	for (const auto& recorded: recorder.recorded) {
		const VkCommandBuffer command = recorder.buffers[recorded.index].first;
		if (recorder.own_thread) { // (ended: begun again, the pool resets it)
			recorder.buffers[recorded.index].second = 0;
			continue;
		}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		const LocalVulkanRecording::Segment segments[] {{&command, sizeof(command)}};
		if (!recorded.ended && LocalVulkanRecording::EnqueueDeferred(ReplayDiscard, segments, false)) {
			recorder.buffers[recorded.index].second = 0; // (reset by the stream, before any later use of it there)
			continue;
		}
		if (!recorded.ended) LocalVulkanRecording::Drain();
#endif
		EXIT_NOT_IMPLEMENTED(vk::CommandBuffer {command}.reset() != vk::Result::eSuccess);
		recorder.buffers[recorded.index].second = 0;
	}
	recorder.recorded.clear();
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

// desport: begin KYTY_ASYNC_COMPUTE (async-compute.h). GPU thread.
CommandBuffer& CommandScheduler::AsyncCommand(uint64_t dependency) {
	EXIT_IF(m_async_pool == nullptr || t_recorder != nullptr);
	EXIT_IF(dependency >= m_master.CurrentTick());
	if (m_async_command.IsInvalid()) {
		// (One async submission an epoch: the async timeline's values only grow. RenderExecutor::DispatchDirect checks.)
		EXIT_IF(AsyncEpochUsed());
		m_async_command.m_buffer      = m_async_pool->Commit();
		m_async_command.m_registers   = m_command.m_registers;
		m_async_command.m_user_config = m_command.m_user_config;
		m_async_command.m_shaders     = m_command.m_shaders;
		m_async_command.Begin();
		m_async_dispatches = 0;
	}
	m_async_dependency = std::max(m_async_dependency, dependency);
	return m_async_command;
}

void CommandScheduler::CompleteAsyncDispatch() {
	const auto batch = std::clamp(kyty_local_dispatch_batch.load(std::memory_order_relaxed), 1u, 65536u);
	// (A graphics flush submits the async buffer first, and opens the next epoch for the next one.)
	if (++m_async_dispatches >= batch) Flush();
}

void CommandScheduler::SubmitAsync() {
	if (m_async_command.IsInvalid()) return;
	EXIT_IF(t_recorder != nullptr || m_async_pool == nullptr);
	m_async_command.EndRendering();
	const uint64_t work_before = AsyncCompute::RecordedWork();
	(void)m_async_command.Handle(); // (a pending compute chain barrier closes it)
	AsyncCompute::t_slice.async_work += AsyncCompute::RecordedWork() - work_before; // (the async buffer's work)
	const uint64_t epoch = m_master.CurrentTick();
	SubmitInfo     submit;
	if (m_async_dependency != 0 && !m_master.IsGfxFree(m_async_dependency)) m_master.Refresh();
	if (m_async_dependency != 0 && !m_master.IsGfxFree(m_async_dependency)) {
		submit.AddWait(m_master.Handle(), m_async_dependency);
		AsyncCompute::Add(AsyncCompute::SubmitsWait);
		AsyncCompute::Add(AsyncCompute::DepBacklog, m_async_dependency - m_master.KnownGfxTick());
	}
	submit.AddSignal(m_master.AsyncHandle(), epoch);
	// (Before the hand-over: no Refresh may take the epoch for complete with the graphics timeline alone.)
	m_master.NoteAsyncEpoch(epoch);
	PageManager::FlushDeferredProtection();
	const vk::CommandBuffer buffer = m_async_command.m_buffer;
	auto&                   graphics = m_graphics;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (LocalVulkanRecording::DeferredSubmitEnabled()) {
		DeferredSubmit deferred {};
		deferred.tick            = epoch;
		deferred.master          = &m_master;
		deferred.async           = true;
		deferred.q               = 0;          // desport: KYTY_SUBMIT_COALESCE
		deferred.writes          = UINT32_MAX; // desport: KYTY_GPU_TIMESTAMPS
		deferred.main            = m_main;     // desport: KYTY_GPU_TIMESTAMPS=2
		deferred.count           = 1;
		deferred.commands[0]     = buffer;
		deferred.ended[0]        = false;
		deferred.timestamp_slot  = UINT32_MAX; // (the live trace maps graphics ticks to slots)
		deferred.driver_tick     = &m_async_driver_tick;
		deferred.upload_sequence = AsyncUpload::SubmitSequence();
		deferred.queue           = graphics.async_queue;
		deferred.queue_mutex     = &graphics.async_queue_mutex;
		deferred.waits           = submit.num_wait_semaphores;
		deferred.signals         = submit.num_signal_semaphores;
		for (uint32_t i = 0; i < deferred.waits; ++i) {
			deferred.wait_semaphores[i] = submit.wait_semaphores[i];
			deferred.wait_ticks[i]      = submit.wait_ticks[i];
			deferred.wait_stages[i]     = static_cast<VkPipelineStageFlags>(submit.wait_stages[i]);
		}
		for (uint32_t i = 0; i < deferred.signals; ++i) {
			deferred.signal_semaphores[i] = submit.signal_semaphores[i];
			deferred.signal_ticks[i]      = submit.signal_ticks[i];
		}
		const LocalVulkanRecording::Segment segments[] {{&deferred, sizeof(deferred)}};
		LocalVulkanRecording::NoteDeferredSubmitQueued();
		if (!LocalVulkanRecording::EnqueueDeferred(ReplaySubmit, segments, true)) {
			LocalVulkanRecording::Drain();
			ReplaySubmit(segments, VULKAN_HPP_DEFAULT_DISPATCHER);
		}
	} else
#endif
	{
		EXIT_NOT_IMPLEMENTED(buffer.end() != vk::Result::eSuccess);
		AsyncUpload::Drain();
		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();
		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();
		vk::Result result;
		{
			Common::LockGuard lock(graphics.async_queue_mutex);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			DeviceFault::NoteSubmit((uint64_t)(static_cast<VkSemaphore>(m_master.AsyncHandle())), epoch, // NOLINT
			                        reinterpret_cast<const VkCommandBuffer*>(&buffer), 1,
			                        reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()),
			                        submit.wait_ticks.data(), submit.num_wait_semaphores);
#endif
			result = graphics.async_queue.submit(1, &submit_info, nullptr);
			m_async_driver_tick.store(epoch, std::memory_order_release);
		}
		if (result != vk::Result::eSuccess) ReportVulkanFatal("vkQueueSubmit (async compute)", result, epoch, 0, 0, 0, 0, 0, 0, 0);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	}
	m_async_command.m_buffer = nullptr;
	m_async_epoch_submitted  = epoch;
	m_async_dependency       = 0;
	m_async_dispatches       = 0;
	AsyncCompute::Add(AsyncCompute::Submits);
	AsyncCompute::Report();
}
// desport: end KYTY_ASYNC_COMPUTE

} // namespace Libs::Graphics

// desport: KYTY_ASYNC_COMPUTE (async-compute.h). Any thread: every 5 s, "DES_ASYNC_COMPUTE t=<ms> window_ms=<ms>
// candidates= async= fb_nolog= fb_images= fb_dma= fb_gds= fb_work= fb_notes= fb_sparse= fb_taa= fb_dep_open= fb_epoch=
// table_skipped= submits= submits_wait= dep_backlog= dep_label= dep_writes= dep_wait= dep_own= gfx_waits= epoch_flushes=
// rb_async= rb_async_wait_us= rb_mixed= slices=" (counts of the window; dep_backlog: the mean, over the submissions that
// wait, of the graphics epochs still ahead of their dependency at the submission - compare lat_gpu_backlog;
// rb_async_wait_us: the mean guest wait of those readbacks).
void AsyncCompute::Report() {
	namespace A = AsyncCompute;
	static std::atomic<uint64_t> last_ms {0};
	static uint64_t              last[A::Count] {};
	static std::mutex            mutex;
	const uint64_t               now      = Desport::ElapsedMilliseconds();
	uint64_t                     previous = last_ms.load(std::memory_order_relaxed);
	if (previous != 0 && now - previous < 5000) return;
	std::unique_lock lock(mutex, std::try_to_lock);
	if (!lock.owns_lock()) return;
	previous = last_ms.load(std::memory_order_relaxed);
	if (previous != 0 && now - previous < 5000) return;
	uint64_t d[A::Count];
	for (size_t i = 0; i < A::Count; ++i) {
		const uint64_t value = A::g_values[i].load(std::memory_order_relaxed);
		d[i]                 = value - last[i];
		last[i]              = value;
	}
	last_ms.store(now, std::memory_order_relaxed);
	if (previous == 0) return;
	const auto mean = [](uint64_t sum, uint64_t n, double scale) {
		return n != 0 ? static_cast<double>(sum) / scale / static_cast<double>(n) : 0.0;
	};
	std::string out = fmt::format(
	    "DES_ASYNC_COMPUTE t={} window_ms={} candidates={} async={} fb_nolog={} fb_images={} fb_dma={} fb_gds={} fb_work={} "
	    "fb_notes={} fb_sparse={} fb_taa={} fb_dep_open={} fb_epoch={} table_skipped={}",
	    now, now - previous, d[A::Candidates], d[A::Async], d[A::FbNoLog], d[A::FbImages], d[A::FbDma], d[A::FbGds],
	    d[A::FbWork], d[A::FbNotes], d[A::FbSparse], d[A::FbTaa], d[A::FbDepOpen], d[A::FbEpoch], d[A::TableSkipped]);
	out += fmt::format(" submits={} submits_wait={} dep_backlog={:.2f} dep_label={} dep_writes={} dep_wait={} dep_own={} "
	                   "gfx_waits={} epoch_flushes={} rb_async={} rb_async_wait_us={:.1f} rb_mixed={} slices={}\n",
	                   d[A::Submits], d[A::SubmitsWait], mean(d[A::DepBacklog], d[A::SubmitsWait], 1.0), d[A::DepLabel],
	                   d[A::DepWrites], d[A::DepWait], d[A::DepOwn], d[A::GfxWaits], d[A::EpochFlushes], d[A::RbAsync],
	                   mean(d[A::RbAsyncNs], d[A::RbAsync], 1e3), d[A::RbMixed], d[A::Slices]);
	Desport::EmitLine(out);
}
