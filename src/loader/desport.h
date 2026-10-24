// SPDX-License-Identifier: GPL-2.0-only
// desport: Demon's Souls port layer (milestones, call tracing, game-specific HLE overrides).
#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_DESPORT_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_DESPORT_H_

#include "loader/symbolDatabase.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace Loader {
struct Program;
} // namespace Loader // desport: ultrawide

namespace Desport {

// desport: KYTY_ULTRAWIDE=<width>x<height>|<aspect> (desportUltrawide.cpp): the render setup's 16:9 constant of the main
// program becomes the given aspect ratio (found by its bytes; logs "DES_ULTRAWIDE aspect=<x> applied|skipped <reason>").
// With KYTY_ULTRAWIDE=<w>x<h>, KYTY_ULTRAWIDE_RENDER=1 makes the 4K modes output and render that shape and
// KYTY_ULTRAWIDE_HUD=native|16x9|<w>x<h> sets the UI system's screen size (desport: native ultrawide phases).
void InstallUltrawide(Loader::Program* program);

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
void MovieFileRead(const char* guest_path); // desport: a .bk2 no longer read for 3 s no longer counts as playing
// Vblanks to leave between two flips while a movie plays (0: none, not playing, 60 Hz or less, or turned off).
int MovieFlipFloor();
// desport: a .bk2 file is open (a Bink movie plays), whatever KYTY_MOVIE_PACING says (ultrawide movie pillarbox).
bool MoviePlaying();
// desport: KYTY_ULTRAWIDE_MOVIES (desportUltrawide.cpp): the horizontal scale (< 1) that keeps a 16:9 movie at 16:9 in a
// wider target (`target_width` x `target_height`), or 0 when movies are left to stretch (or not ultrawide).
float UltrawideMovieScale(uint32_t target_width, uint32_t target_height);
// desport: unified MoviePlaying() (declared above): KYTY_FPS_LIMIT also keeps flips on vblank ticks while it is true.

// desport: KYTY_FPS_LIMIT (desportFps.cpp): frame rate without the vblank as a ceiling.
//   unset    the engine as before: a flip only on a vblank tick (the ceiling is --vblank-frequency).
//   0        no limit: a finished frame is flipped at once, between vblank ticks.
//   <n>      at most n flips a second (20..1000), between vblank ticks.
//   display  at most the display's refresh rate (queried from the window's display); with frame generation, the
//            display's rate / the frames DLSS-G shows per flip (the presented rate is then the display's).
// desport: with frame generation every limit counts flips (rendered frames); <n> is not divided (SetFrameGeneration).
// The vblank keeps its rate (--vblank-frequency); movies (.bk2 open), the game's own sceVideoOutSetFlipRate(1|2) and
// KYTY_FLIP_RATE keep flipping on vblank ticks. With a limit, the game's time step is freed from its 1/59.94 s floor
// through console variables (KYTY_FPS_GAME_CONVARS, see GameConvarLines). Logs "DES_FPS_LIMIT ..." at boot.
namespace Fps {

// A limit is set (any value of KYTY_FPS_LIMIT that parsed).
bool Requested();
// Flips follow the limit now (requested and, for "display", the display's rate known).
bool Pacing();
// The flip allowed by the limit is due (always true without a period).
bool Due();
// Steady-clock ns of the next allowed flip (0: now / no period).
uint64_t NextDueNs();
uint64_t NowNs();
// A flip was presented (`between_vblanks`: outside a vblank tick).
void OnFlip(bool between_vblanks);
// "display": the window's display refresh rate (Hz), 0 while unknown. Pacing starts once a rate is set.
bool NeedsDisplayRate();
void SetDisplayRate(uint32_t hz);
// desport: vblank thread, every tick: the frames DLSS-G shows per flip (FrameGen::PresentedFrames() while frame
// generation is enabled, else 1). "display" re-derives its flip limit from it; each change logs
// "DES_FPS_LIMIT fg t=<ms> mode=<display|fixed|unlimited> ... fg=<n> effective=<fps> ...". No-op without a limit.
void SetFrameGeneration(uint32_t presented_frames);
// The boot line: "DES_FPS_LIMIT mode=<off|unlimited|fixed|display> ...".
void LogBoot(uint32_t vblank_hz);
// Every 30 s while pacing: "DES_FPS_LIMIT stats t=<ms> between_vblanks=<n> on_vblank=<n> movie_hold=<0|1>".
void MaybeLogStats();
// "+name=value\n" lines added to the game's command-line file (loader/gameArgs.cpp) while a limit is requested:
// KYTY_FPS_GAME_CONVARS (default "gameMinDeltaTime=0.001;gameDeltaTimeCorrectionTolorance=0.1"; "0" = none), minus the
// names KYTY_GAME_CONVARS already sets.
std::string GameConvarLines();

} // namespace Fps

// desport: KYTY_FAULT_CLASSIFY=1 (PA: where do the guest faults come from?): atomic counters in GpuResourceManager::HandleFault, by
// access, mapping kind, outcome and thread, reported every 5 s as one line:
// "DES_FAULTS t=<ms> write_direct=<n> write_flexible=<n> write_pooled=<n> write_other=<n> read=<n> other=<n> window=<n>
// restore=<n> invalidate=<n> readback=<n> guest=<n> gpu_thread=<n>" (the counts of the interval, not totals).
namespace Faults {

enum Kind : uint32_t {
	WriteDirect,   // write fault, page in direct memory (a view of the backing section)
	WriteFlexible, // ... in flexible memory (private commit)
	WritePooled,   // ... in pooled memory
	WriteOther,    // ... in none of those (unmapped by now)
	Read,          // read fault (PAGE_NOACCESS page the GPU wrote)
	Other,         // execute or unknown access
	Window,        // outcome: served by the CPU write window
	Restore,       // outcome: protection nothing watched, given back (RestoreIfUnwatched)
	Invalidate,    // outcome: buffer and texture caches invalidated
	Readback,      // outcome: GPU-written bytes read back
	Guest,         // the faulting thread is not the emulator's GPU thread
	GpuThread,     // the faulting thread is the emulator's GPU thread
	KindCount
};

bool Enabled();
void Add(Kind kind) noexcept; // starts the reporter on the first call; no-op when disabled

} // namespace Faults

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
