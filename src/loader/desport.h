// SPDX-License-Identifier: GPL-2.0-only
// desport: Demon's Souls port layer (milestones, call tracing, game-specific HLE overrides).
#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_DESPORT_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_DESPORT_H_

#include "loader/symbolDatabase.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace Desport {

// Writes "DES_MILESTONE <name> t=<ms> [detail]" to the console and the log once per name.
void Milestone(std::string_view name, std::string_view detail = {});
// Counts accepted flip requests and emits flip_1, flip_60, flip_600 and flip_3600.
void FlipMilestone(bool from_gpu);
// The first guest file opened under /app0 that is not part of the executable or sce_sys.
void FileOpenMilestone(const char* guest_path);

namespace Trace {

// DES_TRACE=1 counts every PLT call through a per-slot thunk and writes DES_TRACE_FILE
// (default calls.json in the working directory) every second and at exit.
bool Enabled();
// Returns the value to store in the PLT slot: a counting thunk that jumps to `target`.
// Wrapping the same slot again retargets its existing thunk.
uint64_t Wrap(uint64_t slot_vaddr, uint64_t target, const std::string& name,
              const std::string& dbg_name, const std::filesystem::path& program);
// The real target when `value` is a tracing thunk; `value` otherwise.
uint64_t Unwrap(uint64_t value);

} // namespace Trace

// Game-specific HLE overrides from game/hooks, consulted before the engine's own HLE.
bool FindGameHook(const std::string& nid, Loader::SymbolType type, Loader::SymbolRecord* out);
// Implemented in game/hooks (or desport/gameHooksDefault.cpp when there is no game layer).
void RegisterGameHooks(Loader::SymbolDatabase* symbols, const std::string& title_id);

} // namespace Desport

#endif /* EMULATOR_INCLUDE_EMULATOR_LOADER_DESPORT_H_ */
