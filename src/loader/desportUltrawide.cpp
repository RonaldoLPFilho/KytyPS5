// SPDX-License-Identifier: GPL-2.0-only
// desport: KYTY_ULTRAWIDE=<width>x<height> (or a decimal aspect ratio, e.g. 2.3889): Demon's Souls' render setup stores
// the 3D view's aspect ratio with "mov dword [rbx+4], 16/9" (c7 43 04 39 8e e3 3f, eboot.bin 01.007.000: the one such
// instruction in the file). The constant becomes width / height of the chosen shape: a wider field of view drawn
// anamorphically into the game's 16:9 buffers, which the window stretches back (docs/CURRENT-PLAYABLE.md "21:9 / 5K").
// The HUD, the menus and the movies stretch with it. It is what the ceremony88 fork's launcher did with a game patch
// file (--game-patch, offset 0x90ae7f), found by its bytes instead of an offset: no patch file, any aspect ratio.
//
// desport: native ultrawide phases, found statically in eboot.bin
// 01.007.000 and patched by signature, never by offset:
//   KYTY_ULTRAWIDE_RENDER=1 - the game's display configuration (the video-settings query and the two mode switches that
//     fill it: render size, output size, maximum size) produces an output of <width>x<height> instead of 3840x2160 for
//     the 4K modes, and a render size of the same shape: the scene, the post-processing, the TAA/DLSS output, the UI
//     target (kRTT_UI_Onscreen is made at the output size) and the presented image are all <width>x<height>.
//     The render height is KYTY_RENDER_HEIGHT, else the KYTY_DLSS mode's share of <height>, else <height>.
//   KYTY_ULTRAWIDE_HUD=native|16x9|<w>x<h> - the UI system's screen size (a float pair it picks, 3840x2160 for the 4K
//     modes, before laying out its scenes; it has an override of its own the game never sets): native = the output
//     size (the UI target's size with KYTY_ULTRAWIDE_RENDER), 16x9 = <height>*16/9 x <height>. Read from a copy of the
//     constant in memory near the program, the one load redirected to it (the original constant has other readers).
// Every patch logs "DES_ULTRAWIDE <what> applied|skipped <reason>".
#include "loader/desport.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsRenderSize.h"
#include "loader/elf.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace Desport {

namespace {

void Report(const std::string& line) {
	LOGF("%s\n", line.c_str());
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

std::string AspectText(double aspect) {
	char text[48];
	std::snprintf(text, sizeof(text), "DES_ULTRAWIDE aspect=%.4f", aspect);
	return text;
}

// "3440x1440", "21:9" or "2.3889" -> the aspect ratio; 0 when the text is none of those.
double ParseAspect(const char* text) {
	char*  end   = nullptr;
	double first = std::strtod(text, &end);
	if (end == text || first <= 0.0) return 0.0;
	if (*end == 'x' || *end == 'X' || *end == ':') {
		const char* rest   = end + 1;
		char*       end2   = nullptr;
		const double second = std::strtod(rest, &end2);
		return end2 != rest && second > 0.0 ? first / second : 0.0;
	}
	return *end == '\0' ? first : 0.0;
}

// desport: "<width>x<height>" in pixels (both 64..8192); false for anything else (an aspect ratio, "21:9").
bool ParsePixels(const char* text, uint32_t* width, uint32_t* height) {
	char*               end = nullptr;
	const unsigned long w   = std::strtoul(text, &end, 10);                     // desport
	if (end == text || (*end != 'x' && *end != 'X')) return false;              // desport
	const char*         rest = end + 1;                                         // desport
	const unsigned long h    = std::strtoul(rest, &end, 10);                    // desport
	if (end == rest || *end != '\0' || w < 64 || h < 64 || w > 8192 || h > 8192) return false; // desport
	*width  = static_cast<uint32_t>(w);                                         // desport
	*height = static_cast<uint32_t>(h);                                         // desport
	return true;                                                                // desport
}

// desport: the target of a RIP-relative operand: its 32-bit displacement at `disp`, the next instruction at `next`.
uint64_t RipTarget(uint64_t disp, uint64_t next) {
	int32_t value = 0;                                                          // desport
	std::memcpy(&value, reinterpret_cast<const void*>(disp), sizeof(value));    // desport
	return next + static_cast<int64_t>(value);                                  // desport
}

bool Readable(uint64_t address, uint64_t size) {
	return Libs::Graphics::HostMemoryRangeIsReadable(address, size); // desport
}

bool HoldsPair(uint64_t address, uint32_t first, uint32_t second) {
	uint32_t pair[2] {};                                                        // desport
	if (!Readable(address, sizeof(pair))) return false;                         // desport
	std::memcpy(pair, reinterpret_cast<const void*>(address), sizeof(pair));    // desport
	return pair[0] == first && pair[1] == second;                               // desport
}

bool HoldsWords(uint64_t address, const std::array<uint32_t, 5>& words) {
	std::array<uint32_t, 5> held {};                                             // desport
	if (!Readable(address, sizeof(held))) return false;                         // desport
	std::memcpy(held.data(), reinterpret_cast<const void*>(address), sizeof(held)); // desport
	return held == words;                                                       // desport
}

void WritePair(uint64_t address, uint32_t first, uint32_t second) {
	const uint32_t pair[2] {first, second};                                     // desport
	std::memcpy(reinterpret_cast<void*>(address), pair, sizeof(pair));          // desport
}

void WriteWord(uint64_t address, uint32_t value) {
	std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value)); // desport
}

// desport: width for `height` at `aspect`, a multiple of 8, at most `limit`.
uint32_t WidthFor(uint32_t height, double aspect, uint32_t limit) {
	const auto width = static_cast<uint32_t>(std::lround(height * aspect / 8.0)) * 8u; // desport
	return std::min(width, limit);                                                       // desport
}

// desport: the render height for an output <height> high: KYTY_RENDER_HEIGHT (at most <height>), else the KYTY_DLSS
// mode's share of it (as DemonsSoulsRenderSize does for 2160: quality 2/3, balanced 0.58, performance 1/2, ultra 1/3),
// else <height> itself.
uint32_t RenderHeightFor(uint32_t height) {
	if (const char* text = std::getenv("KYTY_RENDER_HEIGHT"); text != nullptr && *text != '\0') {
		const auto value = static_cast<uint32_t>(std::strtoul(text, nullptr, 10)); // desport
		if (value >= 360 && value <= height) return value;                         // desport
	}
	if (const char* mode = std::getenv("KYTY_DLSS"); mode != nullptr && *mode != '\0') {
		const auto at_2160 = Loader::DemonsSoulsRenderSize::DlssRenderHeight(mode); // desport (0: off, dlaa, unknown)
		if (at_2160 != 0) return std::max<uint32_t>(360, (height * at_2160 / 2160u) & ~1u); // desport
	}
	return height; // desport
}

// desport: the display configuration's places, located by signature (eboot.bin 01.007.000):
//   * the mode switches' 4K case (two copies: the setter and the constructor): "vmovsd xmm1, [output pair]; vmovsd
//     xmm0, [render pair]; jmp" (the pattern DemonsSoulsRenderSize uses), output 3840x2160 - the output constant is
//     shared by every 4K case, the render constant is the performance mode's (2560x1440, or DemonsSoulsRenderSize's);
//   * the dynamic-resolution case: the same loads followed by "mov byte [rbp+x], 1" (render 2688x1512);
//   * the video-settings query's four per-mode tables (render width/height, output width/height; modes 1-5);
//   * the maximum size immediates (movabs 3840 | 2160 << 32) next to the switches.
struct DisplaySites {
	uint64_t              output_constant = 0, render_constant = 0, dynamic_constant = 0; // desport
	uint64_t              render_widths = 0, render_heights = 0, output_widths = 0, output_heights = 0; // desport
	std::vector<uint64_t> maximum_immediates;                                  // desport: imm64 addresses
};

std::optional<DisplaySites> FindDisplaySites(const Loader::Program& program, std::string* why) {
	DisplaySites sites;
	std::vector<uint64_t> switch_sites; // desport
	for (const auto site: Loader::GuestCode::Find(program, Loader::GuestCode::Pattern(Loader::DemonsSoulsRenderSize::ModeCase), 64)) {
		const auto output = RipTarget(site + 4, site + 8);   // desport
		const auto render = RipTarget(site + 12, site + 16); // desport
		if (!HoldsPair(output, 3840, 2160) || HoldsPair(render, 3840, 2160)) continue; // desport: the 4K-output, upscaled case
		if ((sites.output_constant != 0 && sites.output_constant != output) ||
		    (sites.render_constant != 0 && sites.render_constant != render)) {
			*why = "mode switches disagree on their constants"; // desport
			return std::nullopt;                                 // desport
		}
		sites.output_constant = output; // desport
		sites.render_constant = render; // desport
		switch_sites.push_back(site);   // desport
	}
	if (switch_sites.empty()) {
		*why = "the 4K mode case of the display configuration not found (another game version, or already patched)"; // desport
		return std::nullopt;                                                                                           // desport
	}
	for (const auto site: Loader::GuestCode::Find(program, Loader::GuestCode::Pattern(
	                                                           "c5 fb 10 0d ?? ?? ?? ?? c5 fb 10 05 ?? ?? ?? ?? c6 45 ?? 01 eb"),
	                                              8)) {
		const auto output = RipTarget(site + 4, site + 8);   // desport
		const auto render = RipTarget(site + 12, site + 16); // desport
		if (output == sites.output_constant && HoldsPair(render, 2688, 1512)) sites.dynamic_constant = render; // desport
	}
	// desport: the query: lea r8,[render heights]; lea r9,[output widths]; lea r11,[output heights]; mov r8d,[r8+4*r10];
	// mov r9d,[r9+4*r10] ... lea rdi,[render widths]; mov edi,[rdi+4*r10].
	const auto query = Loader::GuestCode::FindUnique(program, Loader::GuestCode::Pattern(
	    "4c 8d 05 ?? ?? ?? ?? 4c 8d 0d ?? ?? ?? ?? 4c 8d 1d ?? ?? ?? ?? 47 8b 04 90 47 8b 0c 91"));
	const auto widths = Loader::GuestCode::FindUnique(program, Loader::GuestCode::Pattern("48 8d 3d ?? ?? ?? ?? 42 8b 3c 97"));
	if (!query || !widths || *widths < *query || *widths - *query > 0x80) {
		*why = "the video-settings query not found"; // desport
		return std::nullopt;                         // desport
	}
	sites.render_heights = RipTarget(*query + 3, *query + 7);    // desport
	sites.output_widths  = RipTarget(*query + 10, *query + 14);  // desport
	sites.output_heights = RipTarget(*query + 17, *query + 21);  // desport
	sites.render_widths  = RipTarget(*widths + 3, *widths + 7);  // desport
	if (!HoldsWords(sites.output_widths, {1920, 1920, 3840, 3840, 3840}) ||
	    !HoldsWords(sites.output_heights, {1080, 1080, 2160, 2160, 2160}) || !Readable(sites.render_widths, 20) ||
	    !Readable(sites.render_heights, 20)) {
		*why = "the video-settings tables hold unexpected sizes"; // desport
		return std::nullopt;                                      // desport
	}
	// desport: the maximum size, near the switches (the setter, the query and the constructor have one each).
	const auto low  = *std::min_element(switch_sites.begin(), switch_sites.end()) - 0x2000; // desport
	const auto high = *std::max_element(switch_sites.begin(), switch_sites.end()) + 0x2000; // desport
	for (const auto site: Loader::GuestCode::Find(program, Loader::GuestCode::Pattern("48 ?? 00 0f 00 00 70 08 00 00"), 16)) {
		if (site >= low && site <= high) sites.maximum_immediates.push_back(site + 2); // desport
	}
	return sites; // desport
}

std::string SizeText(uint32_t width, uint32_t height) {
	return std::to_string(width) + "x" + std::to_string(height); // desport
}

// desport: KYTY_ULTRAWIDE_RENDER=1; true when applied.
bool InstallRender(const Loader::Program& program, uint32_t width, uint32_t height, double aspect) {
	std::string why;                                                        // desport
	const auto  sites = FindDisplaySites(program, &why);                    // desport
	if (!sites) {
		Report("DES_ULTRAWIDE render skipped: " + why); // desport
		return false;                                   // desport
	}
	const uint32_t render_height = RenderHeightFor(height);                      // desport
	const uint32_t render_width  = WidthFor(render_height, aspect, width);       // desport
	const uint32_t dynamic_height = (height * 7u / 10u) & ~1u;                  // desport: 1512 of 2160
	const uint32_t dynamic_width  = WidthFor(dynamic_height, aspect, width);     // desport
	// Output of every 4K mode, render of the performance mode (the game's default), of the native 4K mode and of the
	// dynamic mode, in both the switches and the query's tables (index = mode - 1).
	WritePair(sites->output_constant, width, height);                           // desport
	WritePair(sites->render_constant, render_width, render_height);             // desport
	if (sites->dynamic_constant != 0) WritePair(sites->dynamic_constant, dynamic_width, dynamic_height); // desport
	for (uint32_t index = 2; index < 5; ++index) {
		WriteWord(sites->output_widths + 4 * index, width);   // desport
		WriteWord(sites->output_heights + 4 * index, height); // desport
	}
	WriteWord(sites->render_widths + 4 * 2, render_width);    // desport
	WriteWord(sites->render_heights + 4 * 2, render_height);  // desport
	WriteWord(sites->render_widths + 4 * 3, width);           // desport
	WriteWord(sites->render_heights + 4 * 3, height);         // desport
	WriteWord(sites->render_widths + 4 * 4, dynamic_width);   // desport
	WriteWord(sites->render_heights + 4 * 4, dynamic_height); // desport
	// The maximum size (what the dynamic mode allocates its targets at) only grows.
	std::string maximum = "kept 3840x2160";                    // desport
	if (width > 3840 || height > 2160) {
		const uint64_t value = std::max<uint64_t>(width, 3840) | (std::max<uint64_t>(height, 2160) << 32u); // desport
		for (const auto immediate: sites->maximum_immediates) {
			std::memcpy(reinterpret_cast<void*>(immediate), &value, sizeof(value));                        // desport
			(void)Common::VirtualMemory::FlushInstructionCache(immediate, sizeof(value));                  // desport
		}
		maximum = SizeText(std::max<uint32_t>(width, 3840), std::max<uint32_t>(height, 2160)) + " at " +
		          std::to_string(sites->maximum_immediates.size()) + " sites"; // desport
	}
	Report("DES_ULTRAWIDE render applied: output " + SizeText(width, height) + ", render " +
	       SizeText(render_width, render_height) + ", dynamic " + SizeText(dynamic_width, dynamic_height) +
	       (sites->dynamic_constant != 0 ? "" : " (its switch case not found)") + ", maximum " + maximum); // desport
	return true; // desport
}

// desport: KYTY_ULTRAWIDE_HUD; `render` = KYTY_ULTRAWIDE_RENDER applied (the UI target is then <width>x<height>).
void InstallHud(const Loader::Program& program, const char* setting, uint32_t width, uint32_t height, bool render) {
	uint32_t ui_width = 0, ui_height = 0; // desport
	const std::string value = setting;    // desport
	if (value == "native" || value == "16x9") {
		if (!render || width == 0) {
			Report("DES_ULTRAWIDE hud skipped: " + value + " needs KYTY_ULTRAWIDE=<width>x<height> and KYTY_ULTRAWIDE_RENDER=1"); // desport
			return;                                                                                                          // desport
		}
		ui_height = height;                                                               // desport
		ui_width  = value == "native" ? width : (height * 16u / 9u + 1u) & ~1u;           // desport
	} else if (!ParsePixels(setting, &ui_width, &ui_height)) {
		Report("DES_ULTRAWIDE hud skipped: KYTY_ULTRAWIDE_HUD=" + value + " is not native, 16x9 or <width>x<height>"); // desport
		return;                                                                                                       // desport
	}
	// desport: the UI's screen size: cmp eax,3; setb al; vmovd; vpbroadcastb; vpslld 31; vblendvps xmm0, xmm1 (1920x1080),
	// [rip+3840x2160 vector], xmm0 - the 4K modes (3-5) take the memory operand. The vector has other readers, so the one
	// load is pointed at a copy (16 bytes: the size, then the original's other two floats) in memory near the program.
	const auto site = Loader::GuestCode::FindUnique(program, Loader::GuestCode::Pattern(
	    "83 f8 03 0f 92 c0 c5 f9 6e c0 c4 e2 79 78 c0 c5 f9 72 f0 1f c4 e3 71 4a 05 ?? ?? ?? ?? 00"));
	if (!site) {
		Report("DES_ULTRAWIDE hud skipped: the UI screen-size selection not found"); // desport
		return;                                                                      // desport
	}
	const uint64_t disp     = *site + 25;                  // desport
	const uint64_t next     = *site + 30;                  // desport: after the is4 byte
	const uint64_t original = RipTarget(disp, next);       // desport
	float          vector[4] {};                           // desport
	if (!Readable(original, sizeof(vector))) {
		Report("DES_ULTRAWIDE hud skipped: the UI size constant is not readable"); // desport
		return;                                                                    // desport
	}
	std::memcpy(vector, reinterpret_cast<const void*>(original), sizeof(vector)); // desport
	if (vector[0] != 3840.0f || vector[1] != 2160.0f) {
		Report("DES_ULTRAWIDE hud skipped: the UI size constant is not 3840x2160 (patched already?)"); // desport
		return;                                                                                       // desport
	}
	constexpr uint64_t CopyOffset = 0x8100000; // desport: past the program and the GPU-pages cave (0x8080000 + 0x8000)
	constexpr uint64_t CopySize   = 0x1000;    // desport
	if (program.mapped_size >= CopyOffset || program.base_vaddr > UINT64_MAX - CopyOffset - CopySize) {
		Report("DES_ULTRAWIDE hud skipped: no room for the UI size copy"); // desport
		return;                                                            // desport
	}
	const auto requested = program.base_vaddr + CopyOffset; // desport
	const auto copy      = Libs::LibKernel::Memory::AllocateRuntimeMemory(requested, CopySize, Common::VirtualMemory::Mode::ReadWrite,
	                                                                      "desport_ultrawide_hud", true); // desport
	if (copy != requested) {
		if (copy != 0) Libs::LibKernel::Memory::FreeGuestMemory(copy, CopySize);      // desport
		Report("DES_ULTRAWIDE hud skipped: could not allocate the UI size copy near the program"); // desport
		return;                                                                       // desport
	}
	const int64_t displacement = static_cast<int64_t>(copy) - static_cast<int64_t>(next); // desport
	if (displacement < INT32_MIN || displacement > INT32_MAX) {
		Libs::LibKernel::Memory::FreeGuestMemory(copy, CopySize);                   // desport
		Report("DES_ULTRAWIDE hud skipped: the UI size copy is out of reach");      // desport
		return;                                                                     // desport
	}
	vector[0] = static_cast<float>(ui_width);                                        // desport
	vector[1] = static_cast<float>(ui_height);                                       // desport
	std::memcpy(reinterpret_cast<void*>(copy), vector, sizeof(vector));               // desport
	const auto disp32 = static_cast<int32_t>(displacement);                          // desport
	std::memcpy(reinterpret_cast<void*>(disp), &disp32, sizeof(disp32));              // desport
	if (!Common::VirtualMemory::FlushInstructionCache(disp, sizeof(disp32))) {
		Report("DES_ULTRAWIDE hud skipped: could not finalize the write"); // desport
		return;                                                            // desport
	}
	Report("DES_ULTRAWIDE hud applied: UI screen size " + SizeText(ui_width, ui_height) + " (4K modes; was 3840x2160)" +
	       (render ? "" : " - without KYTY_ULTRAWIDE_RENDER the UI target stays 3840x2160")); // desport
}

} // namespace

void InstallUltrawide(Loader::Program* program) {
	const char* setting = std::getenv("KYTY_ULTRAWIDE");
	if (setting == nullptr || *setting == '\0') return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;

	const double aspect = ParseAspect(setting);
	if (aspect < 1.0 || aspect > 8.0) {
		Report(std::string("DES_ULTRAWIDE aspect=") + setting + " skipped: expected <width>x<height> or an aspect ratio from 1 to 8");
		return;
	}
	const auto sites = Loader::GuestCode::Find(*program, Loader::GuestCode::Pattern("c7 43 04 39 8e e3 3f"), 3);
	if (sites.size() != 1) {
		Report(AspectText(aspect) + (sites.empty() ? " skipped: render setup constant not found (another game version, or patched)"
		                                           : " skipped: render setup constant not unique (" + std::to_string(sites.size()) + " sites)"));
		return;
	}
	const float value = static_cast<float>(aspect);
	std::memcpy(reinterpret_cast<void*>(sites[0] + 3), &value, sizeof(value));
	if (!Common::VirtualMemory::FlushInstructionCache(sites[0] + 3, sizeof(value))) {
		Report(AspectText(aspect) + " skipped: could not finalize the write");
		return;
	}
	Report(AspectText(aspect) + " applied");

	// desport: native phases (see the top of the file).
	uint32_t width = 0, height = 0;                                       // desport
	const bool pixels = ParsePixels(setting, &width, &height);            // desport
	bool       render = false;                                            // desport
	if (const char* phase = std::getenv("KYTY_ULTRAWIDE_RENDER"); phase != nullptr && std::strcmp(phase, "1") == 0) {
		if (!pixels) {
			Report("DES_ULTRAWIDE render skipped: KYTY_ULTRAWIDE must be <width>x<height> (pixels), not an aspect ratio"); // desport
		} else {
			render = InstallRender(*program, width, height, aspect); // desport
		}
	}
	if (const char* hud = std::getenv("KYTY_ULTRAWIDE_HUD"); hud != nullptr && *hud != '\0' && std::strcmp(hud, "0") != 0) {
		InstallHud(*program, hud, pixels ? width : 0, pixels ? height : 0, render); // desport
	} else if (render) {
		Report("DES_ULTRAWIDE hud skipped: KYTY_ULTRAWIDE_HUD not set (the UI lays out for 3840x2160 into the " +
		       SizeText(width, height) + " UI target; try KYTY_ULTRAWIDE_HUD=native)"); // desport
	}
}

} // namespace Desport
