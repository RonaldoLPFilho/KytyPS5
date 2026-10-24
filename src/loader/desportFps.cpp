// SPDX-License-Identifier: GPL-2.0-only
// desport: KYTY_FPS_LIMIT=0|<n>|display - the frame rate without the vblank as its ceiling (PLAN.md
// Fase 7 item 4).
//
// Where the ceiling came from (static reading of the engine and of eboot.bin 01.007.000):
//   - The engine flips at most once per virtual vblank tick (VideoOutDriver::PresentThread calls
//   FlipQueue::Flip once
//     per tick), and the game waits for its flips (GPU flips +
//     sceVideoOutIsFlipPending/WaitVblank), so the game's frame rate is capped at
//     --vblank-frequency (60 by default, 120 in the launcher's "up to 120 fps" mode).
//   - The game itself does not cap in play: its TargetFramesPerSecond console variable defaults to
//   0 (FPS_Unlocked;
//     1/2/3 = 60/30/20), which the video-out pacing function (0x90b3a0 in 01.007.000) turns into
//     sceVideoOutSetFlipRate(0) and an expected refresh of 59.94 Hz.
//   - But its time step is floored there: the time manager (0xb280e0) computes expectedDeltaTime =
//   1/59.94 s and clamps
//     the frame's delta time to at least gameMinDeltaTime, whose default (-1) means
//     "expectedDeltaTime". Above ~60 fps every frame still advances the game by 16.7 ms: the game
//     runs fast (the ceremony88 fork noted that a 240 Hz vblank changes the game's speed). With the
//     optional delta-time correction (gameDeltaTimeCorrectionTolorance, 0.5) the step is quantized
//     to 0.5 x expectedDeltaTime, a floor of 8.3 ms (120 fps).
//   Fix without patching code: the game reads its command-line file into its console at boot
//   (loader/gameArgs.cpp, KYTY_GAME_CONVARS), so a limit also sets gameMinDeltaTime=0.001 (floor 1
//   ms) and gameDeltaTimeCorrectionTolorance=0.1 (quantum 1.7 ms); KYTY_FPS_GAME_CONVARS replaces
//   or ("0") removes them. Physics keeps its fixed 60 Hz step (physicsUseFixedTimestep=1,
//   physicsBaseFixedTimestepOnRefreshRate=1, physicsMaxUpdateStepsPerFrame=1).
//
// desport: with frame generation (DLSS-G, FrameGen::Enabled()) the limit counts flips, i.e. rendered frames; DLSS-G adds
// its generated frames at the present. "display" then limits flips to display rate / FrameGen::PresentedFrames() (the
// frames DLSS-G shows per flip, as it last reported them, about once a second; it follows F2 and a minimized window),
// so the presented rate is the display's. <n> stays a limit on flips: the presented rate is about n x the frames shown
// per flip (a launcher that wants "n presented" passes n / (1 + generated)). Each change logs
// "DES_FPS_LIMIT fg t=<ms> mode=<..> fg=<frames per flip> effective=<flip limit> presented=<..>".
// The Reflex frame limiter (slReflexSetOptions frameLimitUs) is not used: these flips are paced here already.
#include "loader/desport.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <string>

namespace Desport::Fps {

namespace {

enum class Mode { Off, Unlimited, Fixed, Display, Invalid };

struct Settings {
	Mode        mode = Mode::Off;
	uint32_t    fps  = 0; // Fixed
	std::string raw;
};

constexpr uint32_t kMinFps = 20;
constexpr uint32_t kMaxFps = 1000;
// A flip may come this much before its due time (the vblank thread polls; a tick can land just
// before it).
constexpr uint64_t kSlackNs = 250'000;

const Settings& Get() {
	static const Settings settings = [] {
		Settings    s;
		const char* value = std::getenv("KYTY_FPS_LIMIT");
		if (value == nullptr || *value == '\0') {
			return s;
		}
		s.raw = value;
		std::string lower;
		for (const char c: s.raw)
			lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		if (lower == "display") {
			s.mode = Mode::Display;
		} else if (lower == "0" || lower == "unlimited" || lower == "uncapped") {
			s.mode = Mode::Unlimited;
		} else {
			char*      end = nullptr;
			const long n   = std::strtol(lower.c_str(), &end, 10);
			if (end != nullptr && *end == '\0' && n > 0) {
				s.mode = Mode::Fixed;
				s.fps  = static_cast<uint32_t>(std::clamp<long>(n, kMinFps, kMaxFps));
			} else {
				s.mode = Mode::Invalid;
			}
		}
		return s;
	}();
	return settings;
}

std::atomic<uint64_t> g_period_ns {0};   // 0: no limit
std::atomic_bool      g_pacing {false};  // Pacing()
std::atomic<uint64_t> g_next_due_ns {0}; // the vblank thread writes it (OnFlip)
std::atomic<uint64_t> g_between_vblanks {0};
std::atomic<uint64_t> g_on_vblank {0};
std::atomic<uint64_t> g_last_stats_ns {0};
// desport: frame generation (SetFrameGeneration): frames DLSS-G shows per flip (1: none) and, for "display", the
// display's rate once known. Written by the vblank thread only.
std::atomic<uint32_t> g_fg_frames {1};
std::atomic<uint32_t> g_display_hz {0};

void StartPacing(uint64_t period_ns) {
	g_period_ns.store(period_ns, std::memory_order_relaxed);
	g_pacing.store(true, std::memory_order_release);
}

std::string ToLower(std::string text) {
	for (auto& c: text)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return text;
}

// desport: "display" with frame generation: flips (rendered frames) at display rate / frames shown per flip, so the
// presented rate is the display's. The period of that flip rate, kept within kMinFps..kMaxFps.
uint64_t DisplayPeriodNs(uint32_t hz, uint32_t fg_frames) {
	const uint64_t period  = 1'000'000'000ull * std::max<uint32_t>(fg_frames, 1) / std::max<uint32_t>(hz, 1);
	return std::clamp<uint64_t>(period, 1'000'000'000ull / kMaxFps, 1'000'000'000ull / kMinFps);
}

double PeriodFps(uint64_t period_ns) {
	return period_ns == 0 ? 0.0 : 1e9 / static_cast<double>(period_ns);
}

} // namespace

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool Requested() {
	const auto mode = Get().mode;
	return mode == Mode::Unlimited || mode == Mode::Fixed || mode == Mode::Display;
}

bool Pacing() {
	return g_pacing.load(std::memory_order_acquire);
}

uint64_t NextDueNs() {
	return g_period_ns.load(std::memory_order_relaxed) == 0
	           ? 0
	           : g_next_due_ns.load(std::memory_order_relaxed);
}

bool Due() {
	const auto due = NextDueNs();
	return due == 0 || NowNs() + kSlackNs >= due;
}

void OnFlip(bool between_vblanks) {
	if (!Pacing()) {
		return;
	}
	(between_vblanks ? g_between_vblanks : g_on_vblank).fetch_add(1, std::memory_order_relaxed);
	const auto period = g_period_ns.load(std::memory_order_relaxed);
	if (period == 0) {
		return;
	}
	// Keep the cadence (n flips a second on average); a flip later than a whole period starts it
	// again from now.
	const auto now  = NowNs();
	const auto next = g_next_due_ns.load(std::memory_order_relaxed);
	g_next_due_ns.store(next == 0 || now > next + period ? now + period : next + period,
	                    std::memory_order_relaxed);
}

bool NeedsDisplayRate() {
	return Get().mode == Mode::Display && !Pacing();
}

void SetDisplayRate(uint32_t hz) {
	if (!NeedsDisplayRate() || hz == 0) {
		return;
	}
	// desport: with frame generation the flip limit is the display's rate / frames shown per flip (SetFrameGeneration).
	const auto fg     = g_fg_frames.load(std::memory_order_relaxed);
	const auto period = DisplayPeriodNs(hz, fg);
	g_display_hz.store(hz, std::memory_order_relaxed);
	StartPacing(period);
	EmitLine(fmt::format("DES_FPS_LIMIT display t={} display={}Hz limit={:.1f}fps fg={} effective={:.1f}fps\n",
	                     ElapsedMilliseconds(), hz, PeriodFps(period), fg, PeriodFps(period)));
}

void SetFrameGeneration(uint32_t presented_frames) {
	const auto& s  = Get();
	const auto  fg = std::clamp<uint32_t>(presented_frames, 1, 8);
	if (!Requested() || g_fg_frames.exchange(fg, std::memory_order_relaxed) == fg) {
		return;
	}
	// desport: effective = the flip (rendered frame) limit now; presented = what the display gets with DLSS-G.
	switch (s.mode) {
		case Mode::Display: {
			const auto hz = g_display_hz.load(std::memory_order_relaxed);
			if (hz == 0) {
				return; // (SetDisplayRate applies it once the display's rate is known)
			}
			const auto period = DisplayPeriodNs(hz, fg);
			g_period_ns.store(period, std::memory_order_relaxed);
			EmitLine(fmt::format("DES_FPS_LIMIT fg t={} mode=display display={}Hz fg={} effective={:.1f}fps "
			                     "presented={:.1f}fps\n",
			                     ElapsedMilliseconds(), hz, fg, PeriodFps(period), PeriodFps(period) * fg));
			return;
		}
		case Mode::Fixed:
			EmitLine(fmt::format("DES_FPS_LIMIT fg t={} mode=fixed limit={}fps fg={} effective={}fps presented={}fps\n",
			                     ElapsedMilliseconds(), s.fps, fg, s.fps, s.fps * fg));
			return;
		case Mode::Unlimited:
			EmitLine(fmt::format("DES_FPS_LIMIT fg t={} mode=unlimited fg={} effective=unlimited\n", ElapsedMilliseconds(),
			                     fg));
			return;
		default: return;
	}
}

void LogBoot(uint32_t vblank_hz) {
	const auto& s = Get();
	switch (s.mode) {
		case Mode::Off:
			EmitLine(fmt::format("DES_FPS_LIMIT mode=off vblank={}Hz (KYTY_FPS_LIMIT unset: one "
			                     "flip per vblank tick)\n",
			                     vblank_hz));
			return;
		case Mode::Invalid:
			EmitLine(fmt::format("DES_FPS_LIMIT mode=off vblank={}Hz (KYTY_FPS_LIMIT=\"{}\" not "
			                     "understood: 0|<n>|display)\n",
			                     vblank_hz, s.raw));
			return;
		case Mode::Unlimited: StartPacing(0); break;
		case Mode::Fixed: StartPacing(1'000'000'000ull / s.fps); break;
		case Mode::Display: break; // pacing starts with SetDisplayRate
	}
	std::string convars = GameConvarLines();
	std::replace(convars.begin(), convars.end(), '\n', ' ');
	if (!convars.empty() && convars.back() == ' ') convars.pop_back();
	EmitLine(fmt::format(
	    // desport: fg=display_divided: with frame generation "display" limits flips to display rate / frames shown per
	    // flip; <n> always limits flips (rendered frames)
	    "DES_FPS_LIMIT mode={} limit={} vblank={}Hz movies=vblank fg=display_divided game_convars=\"{}\"\n",
	    s.mode == Mode::Unlimited ? "unlimited"
	    : s.mode == Mode::Fixed   ? "fixed"
	                              : "display",
	    s.mode == Mode::Unlimited ? std::string("none")
	    : s.mode == Mode::Fixed   ? fmt::format("{}fps", s.fps)
	                              : std::string("display-refresh"),
	    vblank_hz, convars));
}

void MaybeLogStats() {
	if (!Pacing()) {
		return;
	}
	const auto now  = NowNs();
	const auto last = g_last_stats_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_last_stats_ns.store(now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 30'000'000'000ull) {
		return;
	}
	g_last_stats_ns.store(now, std::memory_order_relaxed);
	EmitLine(
	    fmt::format("DES_FPS_LIMIT stats t={} between_vblanks={} on_vblank={} movie_hold={}\n",
	                ElapsedMilliseconds(), g_between_vblanks.exchange(0, std::memory_order_relaxed),
	                g_on_vblank.exchange(0, std::memory_order_relaxed), MoviePlaying() ? 1 : 0));
}

std::string GameConvarLines() {
	static const std::string lines = [] {
		if (!Requested()) {
			return std::string();
		}
		const char*       custom = std::getenv("KYTY_FPS_GAME_CONVARS");
		const std::string list =
		    custom != nullptr ? custom
		                      : "gameMinDeltaTime=0.001;gameDeltaTimeCorrectionTolorance=0.1";
		if (list.empty() || list == "0") {
			return std::string();
		}
		const char*       user_value = std::getenv("KYTY_GAME_CONVARS");
		const std::string user       = ToLower(user_value != nullptr ? user_value : "");
		std::string       out;
		for (size_t at = 0; at < list.size();) {
			auto end = list.find_first_of(";,", at);
			if (end == std::string::npos) end = list.size();
			const auto first = list.find_first_not_of(' ', at);
			const auto last  = list.find_last_not_of(' ', end - 1);
			if (first != std::string::npos && first < end && last >= first) {
				const auto entry = list.substr(first, last - first + 1);
				const auto name  = ToLower(entry.substr(0, entry.find('=')));
				// A name KYTY_GAME_CONVARS sets itself wins (it is added by gameArgs.cpp already).
				bool taken = false;
				for (size_t pos = user.find(name); pos != std::string::npos;
				     pos        = user.find(name, pos + 1)) {
					const bool starts = pos == 0 || user[pos - 1] == ';' || user[pos - 1] == ',' ||
					                    user[pos - 1] == ' ';
					const auto after  = pos + name.size();
					if (starts && after < user.size() &&
					    (user[after] == '=' || user[after] == ' ')) {
						taken = true;
						break;
					}
				}
				if (!taken) out += "+" + entry + "\n";
			}
			at = end + 1;
		}
		return out;
	}();
	return lines;
}

} // namespace Desport::Fps
