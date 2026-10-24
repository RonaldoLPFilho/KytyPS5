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

// desport: the same line the milestones go out with (log and stdout, flushed) and the process's age in ms.
void     EmitLine(const std::string& text);
uint64_t ElapsedMilliseconds();

// desport: crash record (desportCrash.cpp): a minidump in DES_CRASH_DIR (default: the working directory) and a
// "DES_CRASH t=<ms> kind=<exception|exit|device_lost> code=0x.. pc=0x.. module=.. thread=<tid> name=.. dump=.. message=.."
// line, once per process.
namespace Crash {

// Hooks Common::ReportCrash (EXIT/EXIT_IF/NOT_IMPLEMENTED), the unhandled-exception filter, abort and std::terminate.
void Install();
// The faulting context of a host exception the engine is about to end with EXIT (kept per thread for the dump).
void NoteException(const void* native_context, uint32_t code, uint64_t pc);

} // namespace Crash

// desport: periodic memory report (DES_MEM=1 or DES_STATS=1): "DES_MEM t=<ms> name=<category> bytes=<n> count=<n>" lines
// every DES_MEM_INTERVAL_S seconds (default 30), read by tools/log_triage.py. Owners of memory register a provider that
// calls the sink once per category; providers run on the reporter thread, so they must read thread-safe figures only.
namespace Mem {

using Sink     = void (*)(const char* name, uint64_t bytes, uint64_t count);
using Provider = void (*)(Sink sink);

bool Enabled();
void Register(Provider provider);   // no-op when disabled; starts the reporter on the first call
void Unregister(Provider provider); // waits for a report in progress

} // namespace Mem

// desport: movie pacing. Bink movies advance one frame per flip, so with a 120 Hz virtual vblank they play twice as
// fast. While a .bk2 file is open (the game keeps its movie open for the whole playback), the flip queue flips at most
// every MovieFlipFloor()+1 vblanks (~60 flips a second). KYTY_MOVIE_PACING=0 turns it off.
void MovieFileOpened(const char* guest_path);
void MovieFileClosed(const char* guest_path);
// Vblanks to leave between two flips while a movie plays (0: none, not playing, 60 Hz or less, or turned off).
int MovieFlipFloor();

namespace Stats {

// DES_STATS=1: per-name call count, total time and bytes, logged every 5 s as
// "DES_STAT t=<ms> name=<n> calls=<c> ms=<total> bytes=<b>" plus, for names recorded with an
// address, "DES_STAT_ADDR t=<ms> name=<n> region=0x<64 KiB region> calls=<c> ms=<total>" (top 8).
bool Enabled();
void Add(const char* name, uint64_t ns, uint64_t bytes = 0, uint64_t vaddr = 0);

// Times a scope when stats are enabled.
class Scope {
public:
	explicit Scope(const char* name, uint64_t bytes = 0, uint64_t vaddr = 0);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	const char* m_name;
	uint64_t    m_bytes;
	uint64_t    m_vaddr;
	int64_t     m_start_ns;
};

} // namespace Stats

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
