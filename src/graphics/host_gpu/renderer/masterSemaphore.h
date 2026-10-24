#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <deque>
#include <mutex>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	void Wait(uint64_t tick);
	// desport: KYTY_SUBMIT_COALESCE: the highest tick a thread waits for (Wait), which a recording worker holding
	// submissions hands over at once (commandScheduler.cpp).
	void NoteWant(uint64_t tick) noexcept {
		auto known = m_want_tick.load(std::memory_order_relaxed);
		while (known < tick && !m_want_tick.compare_exchange_weak(known, tick, std::memory_order_seq_cst)) {
		}
	}
	[[nodiscard]] uint64_t WantTick() const noexcept { return m_want_tick.load(std::memory_order_seq_cst); }

	// desport: begin KYTY_ASYNC_COMPUTE (src/local/async-compute.h). A tick is an epoch: the graphics submission of X
	// signals Handle() with X, the async submission recorded during X (at most one) signals AsyncHandle() with X.
	// KnownGpuTick / IsFree / Wait then mean both: the graphics timeline and every async epoch up to the tick.
	void EnableAsync();
	// The next tick (CurrentTick) as the atomic other threads may read (AsyncCompute::g_epoch).
	[[nodiscard]] const std::atomic<uint64_t>& CurrentTickValue() const noexcept { return m_current_tick; }
	[[nodiscard]] bool          AsyncEnabled() const noexcept { return m_async_semaphore != nullptr; }
	[[nodiscard]] vk::Semaphore AsyncHandle() const noexcept { return m_async_semaphore; }
	// Render thread, before the async submission of `epoch` is handed over (epochs only grow).
	void                        NoteAsyncEpoch(uint64_t epoch);
	[[nodiscard]] uint64_t      LastAsyncEpoch() const noexcept { return m_async_last.load(std::memory_order_acquire); }
	// The graphics timeline alone, and the async one, as last read (Refresh).
	[[nodiscard]] uint64_t      KnownGfxTick() const noexcept { return m_gfx_tick.load(std::memory_order_acquire); }
	[[nodiscard]] uint64_t      KnownAsyncTick() const noexcept { return m_async_tick.load(std::memory_order_acquire); }
	[[nodiscard]] bool          IsGfxFree(uint64_t tick) const noexcept { return KnownGfxTick() >= tick; }
	[[nodiscard]] bool          IsAsyncFree(uint64_t epoch) const noexcept { return KnownAsyncTick() >= epoch; }
	// desport: end KYTY_ASYNC_COMPUTE

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::atomic<uint64_t> m_want_tick {0}; // desport: KYTY_SUBMIT_COALESCE
	// desport: KYTY_ASYNC_COMPUTE: the async timeline, the async epochs issued and not known complete (oldest first),
	// the last one issued, the graphics and async timelines as last read.
	vk::Semaphore         m_async_semaphore = nullptr;
	std::mutex            m_async_mutex;
	std::deque<uint64_t>  m_async_pending;
	std::atomic<uint64_t> m_async_last {0};
	std::atomic<uint64_t> m_gfx_tick {0};
	std::atomic<uint64_t> m_async_tick {0};
	void                  WaitBoth(uint64_t tick, uint64_t async_epoch);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
