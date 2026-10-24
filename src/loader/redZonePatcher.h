// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_RED_ZONE_PATCHER_H_
#define KYTY_LOADER_RED_ZONE_PATCHER_H_

#include "common/common.h"

#include <span>
#include <vector>

namespace Loader {

struct RedZonePatchResult {
	uint64_t function_count                           = 0;
	uint64_t instruction_count                        = 0;
	uint64_t red_zone_function_count                  = 0;
	uint64_t memory_instruction_count                 = 0;
	uint64_t short_memory_instruction_count           = 0;
	uint64_t patched_memory_instruction_count         = 0;
	uint64_t stack_dependent_memory_instruction_count = 0;
	uint64_t control_flow_memory_instruction_count    = 0;
	uint64_t unrelocatable_memory_instruction_count   = 0;
	uint64_t indirect_red_zone_function_count         = 0;
	uint64_t swept_instruction_count                  = 0;
	uint64_t live_call_count                          = 0;
	uint64_t gap_swept_function_count                 = 0; // desport: functions whose unreached code was swept
};

void RegisterRedZonePatchModule(void* module_ptr, uint64_t module_size, void* trampoline_area_ptr,
                                uint64_t trampoline_area_size);
void UnregisterRedZonePatchModule(void* module_ptr);

RedZonePatchResult PatchRedZoneMemoryInstructions(uint64_t segment_addr, uint64_t segment_size,
                                                  std::span<const uintptr_t> function_starts);

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

// desport: KYTY_REDZONE_FAULT_AUDIT=1 - which guest faults could have overwritten live red-zone data: a fault
// at a memory instruction the patcher left unprotected while the red zone is live, or in code it never
// analyzed (not reached from an .eh_frame function start). Built at patch time, read by the exception handler.
[[nodiscard]] bool RedZoneFaultAuditEnabled(); // desport
// desport: the exception handler, for a guest access fault it is about to handle at `rip`: prints a
// RED_ZONE_AUDIT line once per address (up to 256) when the fault is one of those.
void RedZoneFaultAuditReport(uint64_t rip); // desport

} // namespace Loader

#endif /* KYTY_LOADER_RED_ZONE_PATCHER_H_ */
