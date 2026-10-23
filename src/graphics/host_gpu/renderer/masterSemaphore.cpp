#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "live-census.h"
#include "device-fault.h"

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
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	// (A lost device's timeline reads UINT64_MAX: the first sign of the loss, often.)
	if (result == vk::Result::eErrorDeviceLost || counter == UINT64_MAX) DeviceFault::Report();
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	LiveCensus::Scope census(LiveCensus::GpuWait, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
	                         reinterpret_cast<uint64_t>(__builtin_return_address(1)));
	LiveCensus::WaitScope waiting(LiveCensus::WaitGpu);
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

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
