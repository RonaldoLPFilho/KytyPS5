#pragma once

// DLSS frame generation (KYTY_FRAMEGEN) on Windows: guest address-space reservation stops at a
// ceiling to leave room below 1 TiB for NVIDIA's optical-flow and NGX allocations. Hold the free
// parts of that range while guest memory is being reserved, then release them immediately before
// vkCreateDevice so the driver can claim it.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

namespace LocalDriverHole {

inline std::vector<std::pair<uint64_t, uint64_t>>& Regions() {
	static std::vector<std::pair<uint64_t, uint64_t>> regions;
	return regions;
}

// Reserves the free parts of [begin, end) (64 KiB granularity).
inline void Hold(uint64_t begin, uint64_t end) {
#if defined(_WIN32)
	constexpr uint64_t Granularity = 0x10000;
	uint64_t           held        = 0;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0) break;
		const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto region_end  = std::min<uint64_t>(end, region_base + static_cast<uint64_t>(info.RegionSize));
		const auto start       = (std::max<uint64_t>(region_base, at) + Granularity - 1) & ~(Granularity - 1);
		const auto stop        = region_end & ~(Granularity - 1);
		if (info.State == MEM_FREE && stop > start &&
		    VirtualAlloc(reinterpret_cast<void*>(start), stop - start, MEM_RESERVE, PAGE_NOACCESS) != nullptr) {
			Regions().emplace_back(start, stop - start);
			held += stop - start;
		}
		if (region_end <= at) break;
		at = region_end;
	}
	std::printf("Frame generation: holding %llu MiB of address space for the driver\n",
	            static_cast<unsigned long long>(held >> 20u));
	std::fflush(stdout);
#else
	(void)begin;
	(void)end;
#endif
}

// Gives the held room back once, just before Vulkan creates the device.
inline void Release() {
#if defined(_WIN32)
	for (const auto& [base, size]: Regions()) {
		VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE);
	}
#endif
	Regions().clear();
}

} // namespace LocalDriverHole
