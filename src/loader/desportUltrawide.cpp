// SPDX-License-Identifier: GPL-2.0-only
// desport: KYTY_ULTRAWIDE=<width>x<height> (or a decimal aspect ratio, e.g. 2.3889): Demon's Souls' render setup stores
// the 3D view's aspect ratio with "mov dword [rbx+4], 16/9" (c7 43 04 39 8e e3 3f, eboot.bin 01.007.000: the one such
// instruction in the file). The constant becomes width / height of the chosen shape: a wider field of view drawn
// anamorphically into the game's 16:9 buffers, which the window stretches back (docs/CURRENT-PLAYABLE.md "21:9 / 5K").
// The HUD, the menus and the movies stretch with it. It is what the ceremony88 fork's launcher did with a game patch
// file (--game-patch, offset 0x90ae7f), found by its bytes instead of an offset: no patch file, any aspect ratio.
#include "loader/desport.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/elf.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "common/logging/log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
}

} // namespace Desport
