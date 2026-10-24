#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Finish a complete dispatch; may submit, but never waits for the GPU.
	void           CompleteDispatch();
	// Finish `draws` complete draws; may submit (KYTY_DRAW_BATCH), but never waits for the GPU.
	void           CompleteDraws(uint32_t draws);
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// The next submission (of the open command buffer or any later one) waits on the GPU for `semaphore` (a timeline) to
	// reach `value`: a queue operation the commands being recorded depend on (a sparse binding, Buffer::MakeResident).
	void WaitBeforeNextSubmit(vk::Semaphore semaphore, uint64_t value);
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept;
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	// A buffer upload (BufferCache::UploadDirtyRanges) of pages whose regions were last made CPU-dirty at `last_dirty`
	// (g_cpu_dirty_clock), no later than the open buffer began: those pages were dirty all the time since, so no
	// command recorded into the open buffer read or wrote them (every use of a CPU-dirty range synchronizes it first),
	// and their copies may run before all of its commands. They go into this buffer, submitted ahead of the open one:
	// one dependency each way per submission, where an upload in order ended the render pass and drained the queue on
	// both sides. Not into a buffer the open buffer recorded a copy into (`written_serial`, Buffer::written_serial:
	// a buffer created now gets its old buffers' bytes by a copy, an image download writes more than the range synced).
	// A shader's write synchronized its pages first: those are not among the pages still CPU-dirty.
	// Null: the upload is recorded in order (pages dirtied since, a speculation's recorder, switch off).
	[[nodiscard]] vk::CommandBuffer UploadPrologue(uint64_t last_dirty, uint64_t written_serial);
	// Called with the upload prologue's buffer as it closes, before its closing barrier (BufferCache: the prologue's
	// copies gathered as one indirect copy).
	void SetPrologueHook(std::function<void(vk::CommandBuffer)> hook) { m_prologue_hook = std::move(hook); }
	// Advances with every command buffer this scheduler begins.
	[[nodiscard]] uint64_t CommandSerial() const noexcept { return m_command_serial; }
	// A speculative translation's (src/graphics/guest_gpu/speculation.cpp): its recorder's pending tick (from
	// PendingTick on), which no GPU completion reaches; what it fences gets the real tick when it is committed.
	[[nodiscard]] uint64_t         CurrentTick() const noexcept;
	[[nodiscard]] bool             IsFree(uint64_t tick);
	// desport: KYTY_READBACK_LATENCY: the last tick handed to the driver (any thread).
	[[nodiscard]] uint64_t DriverTick() const noexcept { return m_driver_tick.load(std::memory_order_acquire); }
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

	static constexpr uint64_t PendingTick = uint64_t {1} << 62u;
	// Thrown on a thread with a recorder for a call that would submit or wait (the translation is dropped).
	struct RecorderRefusal {
		const char* what = nullptr;
	};

	// The command buffers of a speculative translation, recorded on the thread that made it current
	// (SetThreadRecorder) and submitted later, in guest order, by the thread that owns the scheduler
	// (SubmitRecorded). While it is current, that thread's recording calls (Current, BeginRendering, ...)
	// use it; calls that submit or wait for the GPU must not be made.
	struct Recorder {
		struct Recorded {
			size_t   index          = 0;     // in buffers
			bool     ended          = false; // else the submission ends it (recorded through a recording stream)
			uint32_t timestamp_slot = UINT32_MAX;
		};
		vk::CommandPool                pool = nullptr;
		std::unique_ptr<CommandBuffer> command;
		size_t                         open = SIZE_MAX; // the buffer `command` records into
		// Each buffer and the tick of its last submission (UINT64_MAX: being recorded or recorded, not submitted).
		std::vector<std::pair<vk::CommandBuffer, uint64_t>> buffers;
		std::vector<Recorded>                               recorded; // in order
		// Its thread ends its buffers (that thread, not the submitting one, owns the pool: a discarded buffer is
		// begun again as it is, the pool resets it).
		bool own_thread = false;
		// What its translation fences (CurrentTick): PendingTick and up, one value per translation in flight.
		uint64_t pending_tick = PendingTick;
	};
	[[nodiscard]] std::unique_ptr<Recorder> CreateRecorder();
	void                                    DestroyRecorder(std::unique_ptr<Recorder> recorder);
	static void                             SetThreadRecorder(Recorder* recorder) noexcept;
	[[nodiscard]] static Recorder*          ThreadRecorder() noexcept;
	// Begins a command buffer bound to these registers / ends the open one (kept for SubmitRecorded).
	void BeginRecording(Recorder& recorder, HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void EndRecording(Recorder& recorder);
	// The first `count` recorded buffers, in order, after this scheduler's open one (submitted first): the tick of
	// the last.
	uint64_t SubmitRecorded(Recorder& recorder, size_t count);
	static constexpr size_t MaxSubmitEntries = 8; // command buffers in one submission
	// Drops what was recorded (never submitted): the buffers are free again.
	void DiscardRecorded(Recorder& recorder);

	// desport: begin KYTY_ASYNC_COMPUTE (src/local/async-compute.h). The async queue is up.
	[[nodiscard]] bool AsyncReady() const noexcept { return m_async_pool != nullptr; }
	// The open epoch already had its async submission (another needs a graphics flush first: at most one an epoch).
	[[nodiscard]] bool AsyncEpochUsed() const noexcept { return m_async_epoch_submitted == m_master.CurrentTick(); }
	[[nodiscard]] bool AsyncOpen() const noexcept { return !m_async_command.IsInvalid(); }
	// The async buffer of the open epoch (begun if needed); its submission waits for the graphics timeline at `dependency`
	// (and at the dependencies given before it), which must be below the open epoch.
	[[nodiscard]] CommandBuffer& AsyncCommand(uint64_t dependency);
	// A dispatch was recorded into the async buffer: after KYTY_DISPATCH_BATCH of them it is submitted (with a graphics
	// flush: the next one gets a fresh epoch).
	void CompleteAsyncDispatch();
	// Submits the async buffer, if open, signalling the async timeline with the open epoch. Any graphics submission does
	// this first (SubmitBuffers).
	void SubmitAsync();
	// A sparse binding the next submission must wait for (WaitBeforeNextSubmit) is pending.
	[[nodiscard]] bool HasPendingSubmitWait() const noexcept { return m_next_wait_semaphore != nullptr; }
	// desport: end KYTY_ASYNC_COMPUTE

private:
	class CommandPool {
	public:
		// desport: KYTY_ASYNC_COMPUTE: `family` (default: the graphics queue's).
		CommandPool(GraphicContext& graphics, MasterSemaphore& master, uint32_t family = UINT32_MAX);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	// Submits the buffers, in order, in one submission (each ended already when `ended`), the next tick signalled.
	struct SubmitEntry {
		vk::CommandBuffer buffer;
		bool              ended          = false;
		uint32_t          timestamp_slot = UINT32_MAX;
	};
	uint64_t SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit);
	uint64_t SubmitBuffer(vk::CommandBuffer buffer, bool ended, uint32_t timestamp_slot, SubmitInfo submit) {
		const SubmitEntry entry {buffer, ended, timestamp_slot};
		return SubmitBuffers({&entry, 1}, submit);
	}

	// The open buffer's upload prologue (UploadPrologue), ended and put ahead of it in its submission.
	[[nodiscard]] SubmitEntry ClosePrologue();
	// The GPU still runs work of this scheduler's the driver has (CompleteDispatch, CompleteDraws).
	[[nodiscard]] bool GpuBusy();
	std::function<void(vk::CommandBuffer)> m_prologue_hook;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	CommandBuffer                m_prologue;
	uint64_t                     m_command_clock  = 0; // g_cpu_dirty_clock when m_command began
	uint64_t                     m_command_serial = 1;
	// WaitBeforeNextSubmit: the semaphore and the value the next submission waits for (null: none).
	vk::Semaphore                m_next_wait_semaphore = nullptr;
	uint64_t                     m_next_wait_value     = 0;
	uint32_t                     m_recorded_dispatches = 0;
	uint32_t                     m_recorded_draws      = 0;
	// The last tick handed to the driver (its vkQueueSubmit returned: on the recording worker for a deferred one).
	std::atomic<uint64_t>        m_driver_tick {0};
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	// Count of live entries in m_pending_operations, updated under
	// m_operation_mutex and read lock-free as a "nothing to retire" hint.
	std::atomic<uint32_t> m_pending_operation_count {0};
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	// Local diagnostic: records every tick's completion while the live trace runs.
	std::jthread                 m_tick_monitor;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	// desport: KYTY_ASYNC_COMPUTE: the async queue family's pool (null: off), the async buffer of the open epoch, the
	// graphics epoch its submission waits for, the epoch of the last async submission, the dispatches in the open buffer,
	// the last async epoch a graphics submission waited for, the last async epoch handed to the driver.
	std::unique_ptr<CommandPool> m_async_pool;
	CommandBuffer                m_async_command;
	uint64_t                     m_async_dependency      = 0;
	uint64_t                     m_async_epoch_submitted = 0;
	uint32_t                     m_async_dispatches      = 0;
	uint64_t                     m_gfx_async_waited      = 0;
	std::atomic<uint64_t>        m_async_driver_tick {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
