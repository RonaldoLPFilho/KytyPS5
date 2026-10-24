#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "live-census.h"
#include "device-fault.h"

#include <algorithm>
#include <cstdlib>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
	if (m_async_semaphore != nullptr) { // desport: KYTY_ASYNC_COMPUTE
		m_graphics.device.destroySemaphore(m_async_semaphore, nullptr);
	}
}

// desport: KYTY_ASYNC_COMPUTE. The async queue's timeline (masterSemaphore.h).
void MasterSemaphore::EnableAsync() {
	if (m_async_semaphore != nullptr) return;
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;
	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;
	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_async_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_async_semaphore == nullptr);
}

// desport: KYTY_ASYNC_COMPUTE. Render thread: the async submission of `epoch` is about to be handed over (before it, so
// that no Refresh takes the epoch for complete with the graphics timeline alone).
void MasterSemaphore::NoteAsyncEpoch(uint64_t epoch) {
	std::lock_guard lock(m_async_mutex);
	EXIT_IF(epoch <= m_async_last.load(std::memory_order_relaxed));
	m_async_pending.push_back(epoch);
	m_async_last.store(epoch, std::memory_order_release);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	// (A lost device's timeline reads UINT64_MAX: the first sign of the loss, often.)
	if (result == vk::Result::eErrorDeviceLost || counter == UINT64_MAX) DeviceFault::Report();
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	// desport: begin KYTY_ASYNC_COMPUTE. The graphics timeline alone; with the async queue the known tick is also below
	// the oldest async epoch not complete (an epoch's async part signals the async timeline). The graphics timeline is
	// read first: an async epoch issued after that is above it (its graphics submission comes after it).
	const auto raise = [](std::atomic<uint64_t>& value, uint64_t to) {
		auto known_value = value.load(std::memory_order_acquire);
		while (known_value < to &&
		       !value.compare_exchange_weak(known_value, to, std::memory_order_release, std::memory_order_relaxed)) {
		}
	};
	raise(m_gfx_tick, counter);
	if (m_async_semaphore != nullptr) {
		uint64_t   async_counter = 0;
		const auto async_result  = m_graphics.device.getSemaphoreCounterValue(m_async_semaphore, &async_counter);
		if (async_result == vk::Result::eErrorDeviceLost || async_counter == UINT64_MAX) DeviceFault::Report();
		EXIT_NOT_IMPLEMENTED(async_result != vk::Result::eSuccess);
		raise(m_async_tick, async_counter);
		std::lock_guard lock(m_async_mutex);
		while (!m_async_pending.empty() && m_async_pending.front() <= async_counter) m_async_pending.pop_front();
		if (!m_async_pending.empty()) counter = std::min(counter, m_async_pending.front() - 1);
	}
	// desport: end KYTY_ASYNC_COMPUTE
	raise(m_gpu_tick, counter);
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}
	NoteWant(tick); // desport: KYTY_SUBMIT_COALESCE (a held submission of the tick is handed over now)
	// desport: KYTY_ASYNC_COMPUTE: and the last async epoch up to the tick still pending.
	uint64_t async_epoch = 0;
	if (m_async_semaphore != nullptr) {
		std::lock_guard lock(m_async_mutex);
		for (const auto epoch: m_async_pending) {
			if (epoch > tick) break;
			async_epoch = epoch;
		}
	}
	WaitBoth(tick, async_epoch);
}

// desport: KYTY_ASYNC_COMPUTE: Wait's wait, on the graphics timeline at the tick and the async one at its epoch (0: none).
void MasterSemaphore::WaitBoth(uint64_t tick, uint64_t async_epoch) {
	LiveCensus::Scope census(LiveCensus::GpuWait, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
	                         reinterpret_cast<uint64_t>(__builtin_return_address(1)));
	LiveCensus::WaitScope waiting(LiveCensus::WaitGpu);
	const vk::Semaphore   semaphores[2] {m_semaphore, m_async_semaphore};
	const uint64_t        values[2] {tick, async_epoch};
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = async_epoch != 0 ? 2u : 1u;
	wait_info.pSemaphores    = semaphores;
	wait_info.pValues        = values;

	// (A tick that does not come says every 5 s where the timeline is: a stalled GPU, 8 GB sim 10-12. KYTY_STALL_SECONDS:
	// another period - under Windows' 2 s GPU timeout, a hung GPU is reported before the reset.)
	static const uint64_t StallSeconds = [] {
		const char* value = std::getenv("KYTY_STALL_SECONDS");
		const auto  seconds = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
		return seconds != 0 ? seconds : uint64_t {5};
	}();
	for (uint32_t round = 1;; ++round) {
		const auto result = m_graphics.device.waitSemaphores(&wait_info, StallSeconds * 1000000000ull);
		if (result == vk::Result::eSuccess) break;
		if (result == vk::Result::eErrorDeviceLost) {
			// (A diagnostic's state too: what the GPU was in when it was reset, if it still reads.)
			if (const auto dump = DeviceFault::g_stall_dump.load(std::memory_order_acquire); dump != nullptr) {
				uint64_t now = 0;
				(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &now);
				dump(now);
			}
			DeviceFault::Report();
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eTimeout);
		uint64_t now = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &now);
		std::printf("STALL graphics timeline: tick %llu not reached after %llu s, at %llu\n",
		            static_cast<unsigned long long>(tick), static_cast<unsigned long long>(round * StallSeconds),
		            static_cast<unsigned long long>(now));
		if (async_epoch != 0) { // desport: KYTY_ASYNC_COMPUTE
			uint64_t async_now = 0;
			(void)m_graphics.device.getSemaphoreCounterValue(m_async_semaphore, &async_now);
			std::printf("STALL async-compute timeline: epoch %llu not reached, at %llu\n",
			            static_cast<unsigned long long>(async_epoch), static_cast<unsigned long long>(async_now));
		}
		std::fflush(stdout);
		if (round == 1) {
			DeviceFault::ReportStall((uint64_t)(static_cast<VkSemaphore>(m_semaphore)), now, [&](uint64_t semaphore) { // NOLINT
				uint64_t value = 0;
				(void)m_graphics.device.getSemaphoreCounterValue(vk::Semaphore(reinterpret_cast<VkSemaphore>(semaphore)), &value);
				return value;
			});
		}
	}
	Refresh();
}

} // namespace Libs::Graphics
