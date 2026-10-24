#include "frame-gen.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg> // desport: DLSS SR log helper
#include <memory>  // desport: KYTY_DLSS_UPSCALE images
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

#ifdef KYTY_FRAMEGEN_STREAMLINE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h> // desport: DLSS Super Resolution / DLAA
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <filesystem>
#include <string>
#include <vector>
#endif

namespace Libs::Graphics::FrameGen {

namespace {

struct Sources {
	ImageId  depth, motion;
	uint64_t depth_address = 0, motion_address = 0;
	bool     valid = false;
};

int      g_generate    = 0;     // frames to generate per rendered frame (0 = off)
bool     g_ready       = false; // Streamline loaded and the feature available
bool     g_fg_runtime_toggle = false;
bool     g_fps_hud_runtime_toggle = false;
std::atomic_bool g_fg_active {false};
std::atomic_bool g_fg_options_dirty {false};
std::atomic_bool g_fps_hud_active {false};
std::atomic<uint64_t> g_last_present_guest_frame {0};
std::atomic<uint64_t> g_runtime_toggle_count {0};
const auto g_process_start_time = std::chrono::steady_clock::now();
Camera   g_camera;              // Kyty.Gpu: the current guest frame
Sources  g_sources;             // Kyty.Gpu: the current guest frame
uint64_t g_guest_frame = 0;     // Kyty.Gpu

// desport: DLSS SR (KYTY_DLSS). g_sr_mode is a sl::DLSSMode value (0 = off), set at Initialize.
uint32_t         g_sr_mode     = 0;        // desport
const char*      g_sr_mode_name = "off";   // desport
bool             g_sr_ready    = false;    // desport: the feature is loaded and supported (OnDevice)
std::atomic_bool g_sr_failed {false};      // desport: given up (the game's TAA runs from then on)
uint32_t         g_sr_upscale  = 0;        // desport: KYTY_DLSS_UPSCALE=1|2 (SrUpscaleMode), 0: off
uint64_t         g_sr_frame    = 0;        // desport: Kyty.Gpu, guest frames (EndSrFrame / PrepareInputs)
// desport: phase B (KYTY_DLSS with DLSS-G): Streamline frame indices come from one counter for every thread (the SR on
// Kyty.Gpu, the present thread), so a guest frame's SR evaluate, constants, tags, markers and present share one index.
std::atomic<uint32_t> g_next_frame_index {0}; // desport
uint32_t         g_frame_token_index   = 0;     // desport: Kyty.Gpu, the current guest frame's index (0: none yet)
bool             g_frame_constants_set = false; // desport: Kyty.Gpu, the SR set the current guest frame's constants
const char*      g_reflex_name = "off";         // desport: the applied Reflex mode (DES_DLSS)

uint32_t NextFrameIndex() { // desport
	uint32_t index = g_next_frame_index.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (index == 0) index = g_next_frame_index.fetch_add(1, std::memory_order_relaxed) + 1u; // (0 means none)
	return index;
}

// desport: the boot line for triage (tools/log_triage.py reads DES_* lines): what DLSS SR, DLSS-G and Reflex do.
void LogDesDlss(const char* streamline) {
	const bool combined = g_ready && g_sr_ready && g_sr_mode != 0;
	// desport: upscale=<0|1> (KYTY_DLSS_UPSCALE); the sizes follow in "DES_DLSS upscale=1 render=WxH output=WxH" when
	// the composite first samples the DLSS output.
	const int upscale = g_sr_mode != 0 ? static_cast<int>(g_sr_upscale) : 0; // desport
	std::printf("DES_DLSS sr=%s fg=%d reflex=%s sr_ready=%d fg_ready=%d combined=%d sr_viewport=%d f2=%d upscale=%d "
	            "streamline=%s\n",
	            g_sr_mode != 0 ? g_sr_mode_name : "off", g_generate, g_ready ? g_reflex_name : "off",
	            g_sr_ready ? 1 : 0, g_ready ? 1 : 0, combined ? 1 : 0, combined ? 0 : 1, g_fg_runtime_toggle ? 1 : 0,
	            upscale, streamline);
	std::fflush(stdout);
	LOGF("DES_DLSS sr=%s fg=%d reflex=%s sr_ready=%d fg_ready=%d combined=%d upscale=%d streamline=%s\n",
	     g_sr_mode != 0 ? g_sr_mode_name : "off", g_generate, g_ready ? g_reflex_name : "off", g_sr_ready ? 1 : 0,
	     g_ready ? 1 : 0, combined ? 1 : 0, upscale, streamline);
}

struct CaptureStats {
	uint32_t frames = 0;
	uint32_t camera_captures = 0;
	uint32_t taa_dispatches = 0;
	uint32_t no_camera = 0;
	uint32_t no_taa = 0;
	uint32_t replaced_images = 0;
	uint32_t valid_inputs = 0;
	uint32_t learned_cameras = 0;  // frames whose camera came from a learned layout
	uint32_t held_cameras = 0;     // frames that held the last camera
	uint32_t rejected_cameras = 0; // listed-shader captures that were not a camera
};
CaptureStats g_capture_stats;

// A world->clip matrix of the game's form (Decompose): finite, a w row (the view direction), a z row of about
// (0, 0, 0, near) (reversed Z), x and y rows of the projection's scale; the previous matrix alike and of
// about the same scale; a jitter below a pixel or so. Rejects constants a listed layout misreads.
bool PlausibleCamera(const Camera& camera) {
	const auto finite = [](const std::array<float, 16>& m) {
		return std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); });
	};
	if (!finite(camera.view_proj) || !finite(camera.prev_view_proj) || !std::isfinite(camera.jitter_ndc[0]) ||
	    !std::isfinite(camera.jitter_ndc[1])) {
		return false;
	}
	const auto length = [](const std::array<float, 16>& m, uint32_t row) {
		return std::sqrt(m[row * 4] * m[row * 4] + m[row * 4 + 1] * m[row * 4 + 1] + m[row * 4 + 2] * m[row * 4 + 2]);
	};
	const auto& m = camera.view_proj;
	const float w = length(m, 3);
	if (w < 1e-6f || length(m, 2) > 1e-2f * w || !(m[11] > 0.0f) || length(m, 0) < 1e-3f * w ||
	    length(m, 1) < 1e-3f * w) {
		return false;
	}
	const auto& p = camera.prev_view_proj;
	const auto near_ratio = [](float a, float b, float tolerance) {
		return a > 0.0f && b > 0.0f && std::fabs(a / b - 1.0f) <= tolerance;
	};
	if (!near_ratio(length(p, 3), w, 0.1f) || !near_ratio(length(p, 0), length(m, 0), 0.5f) ||
	    !near_ratio(length(p, 1), length(m, 1), 0.5f)) {
		return false;
	}
	return std::fabs(camera.jitter_ndc[0]) < 0.01f && std::fabs(camera.jitter_ndc[1]) < 0.01f;
}

// Camera layouts of vertex shaders not in CameraShaders, learned while playing: in a frame where a listed shader
// gave the camera, another shader whose constants hold that same matrix, bit for bit (each row found on its own:
// some shaders order the rows differently), three frames in a row with the same offsets, is a camera shader too.
// A learned shader is checked again in every frame that also has a listed camera, and whenever it draws twice in
// a frame (an object's world->clip matrix differs between objects): one difference and it is dropped for good.
// The Nexus, for one, draws its G-buffer with shaders the list lacks (camera found in 8 of 300 frames there).
// KYTY_FRAMEGEN_LEARN=0 turns it off.
struct LearnedLayout {
	enum class State : uint8_t { Empty, Learning, Camera, Rejected };
	uint64_t                hash  = 0;
	State                   state = State::Empty;
	std::array<uint16_t, 4> rows {};      // word offsets of the current matrix's rows x, y, z, w
	std::array<uint16_t, 3> prev_rows {}; // rows x, y and w of the previous matrix (has_prev)
	uint16_t                jitter = 0;   // (has_jitter)
	bool                    has_prev = false, has_jitter = false;
	uint8_t                 confirmations = 0;
	uint8_t                 misses        = 0;
	uint64_t                snapshot_frame = ~0ull; // the capture frame of its last snapshot (one a frame)
};
// Every vertex shader not in CameraShaders takes an entry (camera, rejected or still learning): the game has about
// 1400 of them in a session, and with 512 entries new areas' camera shaders found no free slot (40 bytes an entry).
constexpr size_t                            LearnedTableSize = 4096;
std::array<LearnedLayout, LearnedTableSize> g_learned;              // Kyty.Gpu
bool                                        g_learning_enabled = true;
uint32_t                                    g_learned_count    = 0; // layouts promoted to cameras
// SRTs of undecided shaders drawn before the frame's listed camera, compared at the frame's end.
struct LearnSnapshot {
	static constexpr uint32_t MaxWords = 128;
	LearnedLayout*                  layout = nullptr;
	uint32_t                        words  = 0;
	std::array<uint32_t, MaxWords> srt {};
};
std::array<LearnSnapshot, 16> g_learn_snapshots; // Kyty.Gpu
uint32_t                      g_learn_snapshot_count = 0;
uint64_t                      g_capture_frame        = 0; // Kyty.Gpu: frames gathered (PrepareInputs)
// The camera a learned layout read this frame, the shader it came from.
Camera   g_learned_camera;
uint64_t g_learned_camera_hash = 0;
// The last frame's camera from a shader (listed or learned): the previous matrix of a learned layout without one,
// and what a frame without any camera holds.
Camera   g_last_camera;
uint64_t g_last_camera_frame = 0;
bool     g_hold_enabled      = true; // KYTY_FRAMEGEN_CAMERA_HOLD=0 turns the hold off

LearnedLayout* FindLearned(uint64_t hash) {
	constexpr size_t mask = LearnedTableSize - 1;
	const auto       slot = static_cast<size_t>((hash ^ (hash >> 29u)) & mask);
	for (size_t probe = 0; probe < 64; ++probe) {
		auto& entry = g_learned[(slot + probe) & mask];
		if (entry.hash == hash) return &entry;
		if (entry.state == LearnedLayout::State::Empty) {
			entry.hash  = hash;
			entry.state = LearnedLayout::State::Learning;
			return &entry;
		}
	}
	return nullptr; // (a crowded neighborhood: not learned)
}

// The first offset of `count` words equal to `values` (bit for bit), or -1.
int32_t FindWords(std::span<const uint32_t> srt, const float* values, uint32_t count) {
	if (srt.size() < count) return -1;
	for (size_t at = 0; at + count <= srt.size(); ++at) {
		if (std::memcmp(&srt[at], values, count * sizeof(float)) == 0) return static_cast<int32_t>(at);
	}
	return -1;
}

void ReadMatrixRows(std::span<const uint32_t> srt, const std::array<uint16_t, 4>& rows, std::array<float, 16>& out) {
	for (uint32_t row = 0; row < 4; ++row) std::memcpy(&out[row * 4], &srt[rows[row]], 4 * sizeof(float));
}

// A learned layout's reading of `srt`: the current matrix, the previous one (when the layout has none: the last
// frame's camera, else no motion) and the jitter (else none).
bool ReadLearnedCamera(const LearnedLayout& layout, std::span<const uint32_t> srt, Camera& camera) {
	uint32_t needed = 0;
	for (const auto row: layout.rows) needed = std::max<uint32_t>(needed, row + 4u);
	if (layout.has_prev) {
		for (const auto row: layout.prev_rows) needed = std::max<uint32_t>(needed, row + 4u);
	}
	if (layout.has_jitter) needed = std::max<uint32_t>(needed, layout.jitter + 2u);
	if (srt.size() < needed) return false;
	ReadMatrixRows(srt, layout.rows, camera.view_proj);
	if (layout.has_prev) {
		std::memcpy(&camera.prev_view_proj[0], &srt[layout.prev_rows[0]], 4 * sizeof(float));
		std::memcpy(&camera.prev_view_proj[4], &srt[layout.prev_rows[1]], 4 * sizeof(float));
		std::memcpy(&camera.prev_view_proj[12], &srt[layout.prev_rows[2]], 4 * sizeof(float));
		for (uint32_t i = 0; i < 4; ++i) camera.prev_view_proj[8 + i] = camera.view_proj[8 + i];
	} else if (g_last_camera.valid && g_last_camera_frame + 1 == g_capture_frame) {
		camera.prev_view_proj = g_last_camera.view_proj;
		for (uint32_t i = 0; i < 4; ++i) camera.prev_view_proj[8 + i] = camera.view_proj[8 + i];
	} else {
		camera.prev_view_proj = camera.view_proj;
	}
	camera.jitter_ndc = {0.0f, 0.0f};
	if (layout.has_jitter) std::memcpy(camera.jitter_ndc.data(), &srt[layout.jitter], 2 * sizeof(float));
	camera.valid = true;
	return true;
}

// Learned layouts last across launches: each one is appended to this file when learned (or dropped), and the
// file is read at start, so a shader is learned once ever. A loaded layout is checked like a learned one and
// dropped (and the drop recorded) on a difference. Lines: "camera <hash> x y z w <prev x y w | - - -> <jitter | ->",
// "reject <hash>"; a later line about a shader replaces an earlier one. KYTY_FRAMEGEN_LAYOUT_FILE=<path> elsewhere.
const char* LayoutFilePath() {
	static const std::string path = [] {
		const char* env = std::getenv("KYTY_FRAMEGEN_LAYOUT_FILE");
		return std::string(env != nullptr && *env != '\0' ? env : "_PipelineCache/framegen-camera-layouts.txt");
	}();
	return path.c_str();
}

void WriteLayout(FILE* file, const LearnedLayout& layout) {
	if (layout.state == LearnedLayout::State::Camera) {
		std::fprintf(file, "camera %016" PRIx64 " %u %u %u %u", layout.hash, layout.rows[0], layout.rows[1],
		             layout.rows[2], layout.rows[3]);
		if (layout.has_prev) {
			std::fprintf(file, " %u %u %u", layout.prev_rows[0], layout.prev_rows[1], layout.prev_rows[2]);
		} else {
			std::fprintf(file, " - - -");
		}
		if (layout.has_jitter) {
			std::fprintf(file, " %u\n", layout.jitter);
		} else {
			std::fprintf(file, " -\n");
		}
	} else {
		std::fprintf(file, "reject %016" PRIx64 "\n", layout.hash);
	}
}

void SaveLayout(const LearnedLayout& layout) {
	FILE* file = std::fopen(LayoutFilePath(), "a");
	if (file == nullptr) return;
	WriteLayout(file, layout);
	std::fclose(file);
}

// The file rewritten with a line a shader (it only grows while playing: a learned shader, later dropped, has both).
void CompactLayoutFile() {
	const std::string path      = LayoutFilePath();
	const std::string temporary = path + ".tmp";
	FILE*             file      = std::fopen(temporary.c_str(), "w");
	if (file == nullptr) return;
	std::fprintf(file, "# Frame generation camera layouts (frame-gen.cpp LoadLayouts); the emulator appends what it learns.\n");
	for (const auto& layout: g_learned) {
		if (layout.state == LearnedLayout::State::Camera || layout.state == LearnedLayout::State::Rejected) {
			WriteLayout(file, layout);
		}
	}
	const bool written = std::fflush(file) == 0;
	std::fclose(file);
	std::error_code error;
	if (written) std::filesystem::rename(temporary, path, error);
	if (!written || error) std::filesystem::remove(temporary, error);
}

void LoadLayouts() {
	FILE* file = std::fopen(LayoutFilePath(), "r");
	if (file == nullptr) return;
	uint32_t cameras = 0, rejects = 0;
	char     line[256];
	while (std::fgets(line, sizeof(line), file) != nullptr) {
		char     kind[16] {};
		uint64_t hash = 0;
		int      used = 0;
		if (std::sscanf(line, "%15s %" SCNx64 "%n", kind, &hash, &used) != 2) continue;
		auto* layout = FindLearned(hash);
		if (layout == nullptr) continue;
		if (std::strcmp(kind, "reject") == 0) {
			if (layout->state == LearnedLayout::State::Camera && g_learned_count > 0) --g_learned_count;
			layout->state = LearnedLayout::State::Rejected;
			continue;
		}
		if (std::strcmp(kind, "camera") != 0) continue;
		// The remaining fields: 4 row offsets, 3 previous-row offsets or "-", a jitter offset or "-".
		std::array<int, 8> values {};
		std::array<bool, 8> present {};
		const char*        at = line + used;
		bool               ok = true;
		for (size_t i = 0; i < values.size() && ok; ++i) {
			while (*at == ' ' || *at == '\t') ++at;
			if (*at == '-') {
				present[i] = false;
				++at;
			} else {
				char* end = nullptr;
				const long value = std::strtol(at, &end, 10);
				ok         = end != at && value >= 0 && value < 0xffff;
				values[i]  = static_cast<int>(value);
				present[i] = true;
				at         = end;
			}
		}
		if (!ok || !present[0] || !present[1] || !present[2] || !present[3]) continue;
		for (uint32_t row = 0; row < 4; ++row) layout->rows[row] = static_cast<uint16_t>(values[row]);
		layout->has_prev = present[4] && present[5] && present[6];
		// (Previous rows read inside the current ones: learned before that was excluded; the last frame's camera
		// stands in for them.)
		for (uint32_t prev = 4; prev < 7 && layout->has_prev; ++prev) {
			for (uint32_t row = 0; row < 4; ++row) {
				if (values[prev] < values[row] + 4 && values[row] < values[prev] + 4) layout->has_prev = false;
			}
		}
		for (uint32_t row = 0; row < 3; ++row) {
			layout->prev_rows[row] = layout->has_prev ? static_cast<uint16_t>(values[4 + row]) : 0;
		}
		layout->has_jitter = present[7];
		for (uint32_t row = 0; row < 4 && layout->has_jitter; ++row) {
			if (values[7] < values[row] + 4 && values[row] < values[7] + 2) layout->has_jitter = false;
		}
		layout->jitter = layout->has_jitter ? static_cast<uint16_t>(values[7]) : 0;
		if (layout->state != LearnedLayout::State::Camera) ++g_learned_count;
		layout->state = LearnedLayout::State::Camera;
	}
	std::fclose(file);
	for (const auto& layout: g_learned) {
		cameras += layout.state == LearnedLayout::State::Camera ? 1u : 0u;
		rejects += layout.state == LearnedLayout::State::Rejected ? 1u : 0u;
	}
	std::printf("Frame generation: loaded %u camera layouts and %u rejected shaders from %s\n", cameras, rejects,
	            LayoutFilePath());
	std::fflush(stdout);
	CompactLayoutFile();
}

// Compares an undecided shader's constants with the frame's listed camera.
void LearnFrom(LearnedLayout& layout, std::span<const uint32_t> srt, const Camera& camera) {
	if (layout.state != LearnedLayout::State::Learning) return;
	std::array<int32_t, 4> rows {};
	for (uint32_t row = 0; row < 4; ++row) rows[row] = FindWords(srt, &camera.view_proj[row * 4], 4);
	if (std::any_of(rows.begin(), rows.end(), [](int32_t at) { return at < 0; })) {
		layout.confirmations = 0;
		if (++layout.misses >= 8) layout.state = LearnedLayout::State::Rejected;
		return;
	}
	// The previous matrix and the jitter are looked for outside the current matrix's rows: with the camera still,
	// the previous rows equal the current ones bit for bit, and the first match was the current row (a layout
	// whose "previous" matrix never moved).
	const auto outside = [&](const float* values, uint32_t count) {
		for (size_t at = 0; at + count <= srt.size(); ++at) {
			const bool overlaps = std::any_of(rows.begin(), rows.end(), [&](int32_t row) {
				return static_cast<int32_t>(at) < row + 4 && row < static_cast<int32_t>(at + count);
			});
			if (!overlaps && std::memcmp(&srt[at], values, count * sizeof(float)) == 0) return static_cast<int32_t>(at);
		}
		return -1;
	};
	const std::array<int32_t, 3> prev {outside(&camera.prev_view_proj[0], 4), outside(&camera.prev_view_proj[4], 4),
	                                   outside(&camera.prev_view_proj[12], 4)};
	const bool    has_prev = std::all_of(prev.begin(), prev.end(), [](int32_t at) { return at >= 0; });
	const bool    jittered = camera.jitter_ndc[0] != 0.0f || camera.jitter_ndc[1] != 0.0f;
	const int32_t jitter   = jittered ? outside(camera.jitter_ndc.data(), 2) : -1;
	const auto    same = [&] {
		for (uint32_t row = 0; row < 4; ++row) {
			if (layout.rows[row] != rows[row]) return false;
		}
		if (has_prev && layout.has_prev) {
			for (uint32_t row = 0; row < 3; ++row) {
				if (layout.prev_rows[row] != prev[row]) return false;
			}
		}
		return !(jitter >= 0 && layout.has_jitter && layout.jitter != jitter);
	};
	if (layout.confirmations == 0 || !same()) {
		layout.confirmations = 0;
		layout.has_prev      = false;
		layout.has_jitter    = false;
	}
	for (uint32_t row = 0; row < 4; ++row) layout.rows[row] = static_cast<uint16_t>(rows[row]);
	if (has_prev) {
		for (uint32_t row = 0; row < 3; ++row) layout.prev_rows[row] = static_cast<uint16_t>(prev[row]);
		layout.has_prev = true;
	}
	if (jitter >= 0) {
		layout.jitter     = static_cast<uint16_t>(jitter);
		layout.has_jitter = true;
	}
	if (++layout.confirmations < 3) return;
	layout.state = LearnedLayout::State::Camera;
	++g_learned_count;
	std::printf("Frame generation: learned camera layout VS=%016" PRIx64 " rows=%u,%u,%u,%u prev=%s%d,%d,%d jitter=%d\n",
	            layout.hash, layout.rows[0], layout.rows[1], layout.rows[2], layout.rows[3],
	            layout.has_prev ? "" : "none:", layout.has_prev ? layout.prev_rows[0] : -1,
	            layout.has_prev ? layout.prev_rows[1] : -1, layout.has_prev ? layout.prev_rows[2] : -1,
	            layout.has_jitter ? static_cast<int>(layout.jitter) : -1);
	std::fflush(stdout);
	SaveLayout(layout);
}

void RejectLearned(LearnedLayout& layout, const char* why) {
	if (layout.state == LearnedLayout::State::Camera && g_learned_count > 0) --g_learned_count;
	layout.state = LearnedLayout::State::Rejected;
	std::printf("Frame generation: dropped learned camera layout VS=%016" PRIx64 " (%s)\n", layout.hash, why);
	std::fflush(stdout);
	SaveLayout(layout);
}

struct PresentTagStats {
	uint32_t presents = 0;
	uint32_t tagged_inputs = 0;
	uint32_t no_inputs = 0;
	uint32_t reused_frames = 0;
	uint32_t tag_failures = 0;
	uint32_t sr_constants = 0; // desport: presents whose constants the DLSS SR set (phase B)
	int32_t  last_tag_result = 0;
	uint64_t first_guest_frame = 0;
	uint64_t last_guest_frame = 0;
	uint32_t color_input_extent_matches = 0;
	uint32_t color_input_extent_mismatches = 0;
	vk::Extent2D last_color_extent {};
	vk::Extent2D last_input_extent {};
	vk::Extent2D last_output_extent {};
	vk::Extent2D last_region_extent {};
};
PresentTagStats g_present_tag_stats;

void ReportCaptureStats() {
	if (g_capture_stats.frames < 300) {
		return;
	}
	std::printf("Frame generation: input capture last %u guest frames: camera=%u learned=%u held=%u TAA_dispatches=%u "
	            "valid=%u missing_camera=%u missing_TAA=%u replaced_images=%u rejected_cameras=%u learned_layouts=%u\n",
	            g_capture_stats.frames, g_capture_stats.camera_captures, g_capture_stats.learned_cameras,
	            g_capture_stats.held_cameras, g_capture_stats.taa_dispatches, g_capture_stats.valid_inputs,
	            g_capture_stats.no_camera, g_capture_stats.no_taa, g_capture_stats.replaced_images,
	            g_capture_stats.rejected_cameras, g_learned_count);
	std::fflush(stdout);
	g_capture_stats = {};
}

float AsFloat(uint32_t word) {
	float value = 0.0f;
	std::memcpy(&value, &word, sizeof(value));
	return value;
}

bool IsDepthFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32: return true;
		default: return false;
	}
}

void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect,
                vk::ImageLayout layout, vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = image.state.pl_stage;
	barrier.srcAccessMask       = image.state.access_mask;
	barrier.dstStageMask        = stage;
	barrier.dstAccessMask       = access;
	barrier.oldLayout           = image.state.layout;
	barrier.newLayout           = layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// (Re)create `image` as a sampled copy target like `source`; with `attachment`, a color
// attachment too where the format allows it (*attachment tells).
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view,
               const VulkanImage& source, bool* attachment = nullptr) {
	if (image.image != nullptr && image.format == source.format &&
	    image.extent.width == source.extent.width && image.extent.height == source.extent.height) {
		return;
	}
	if (view != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
	}
	if (image.image != nullptr) {
		graphics.DeleteImage(image);
	}
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {source.extent.width, source.extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = source.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	if (attachment != nullptr) {
		*attachment = !IsDepthFormat(source.format) &&
		              static_cast<bool>(graphics.physical_device.getFormatProperties(source.format).optimalTilingFeatures &
		                                vk::FormatFeatureFlagBits::eColorAttachment);
		if (*attachment) {
			create.usage |= vk::ImageUsageFlagBits::eColorAttachment;
		}
	}
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image)) {
		EXIT("frame generation: cannot allocate a %ux%u input image\n", source.extent.width,
		     source.extent.height);
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = source.format;
	view_info.subresourceRange = {IsDepthFormat(source.format) ? vk::ImageAspectFlagBits::eDepth
	                                                           : vk::ImageAspectFlagBits::eColor,
	                              0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr, &view),
	                     "frame generation input view");
}

void CopyInto(vk::CommandBuffer command, Image& source, VulkanImage& target) {
	const auto aspect = IsDepthFormat(source.backing.format) ? vk::ImageAspectFlagBits::eDepth
	                                                         : vk::ImageAspectFlagBits::eColor;
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transition(command, target, aspect, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {aspect, 0, 0, 1};
	copy.dstSubresource = {aspect, 0, 0, 1};
	copy.extent         = target.extent;
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transition(command, target, aspect, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

// KYTY_FPS_HUD: frames shown per presented frame, and the panel's normalized rectangle (16 bits
// per coordinate: x0, y0, x1, y1; 0 = none).
std::atomic<uint32_t> g_presented_frames {1};
std::atomic<uint64_t> g_static_rect {0};

// No motion under the KYTY_FPS_HUD panel: DLSS-G would move it with the scene behind it.
void ZeroStaticRect(vk::CommandBuffer command, VulkanImage& motion, vk::ImageView view) {
	const auto packed = g_static_rect.load(std::memory_order_relaxed);
	if (packed == 0) {
		return;
	}
	const auto coordinate = [&](uint32_t index, uint32_t size) {
		return static_cast<int32_t>(((packed >> (16u * index)) & 0xffffu) * size / 0xffffu);
	};
	const int32_t x0 = coordinate(0, motion.extent.width), y0 = coordinate(1, motion.extent.height);
	const int32_t x1 = coordinate(2, motion.extent.width), y1 = coordinate(3, motion.extent.height);
	if (x1 <= x0 || y1 <= y0) {
		return;
	}
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
	           vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo color {};
	color.imageView   = view;
	color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color.loadOp      = vk::AttachmentLoadOp::eLoad;
	color.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = vk::Extent2D {motion.extent.width, motion.extent.height};
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &color;
	command.beginRendering(rendering);
	vk::ClearAttachment clear {};
	clear.aspectMask       = vk::ImageAspectFlagBits::eColor;
	clear.colorAttachment  = 0;
	clear.clearValue.color = vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
	vk::ClearRect rect {};
	rect.rect       = vk::Rect2D {{x0, y0}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
	rect.layerCount = 1;
	command.clearAttachments(1, &clear, 1, &rect);
	command.endRendering();
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// Streamline entry points (sl.interposer.dll, loaded at run time) and feature functions.
struct Api {
	PFun_slInit*                 init                = nullptr;
	PFun_slIsFeatureSupported*   is_feature_supported = nullptr;
	PFun_slIsFeatureLoaded*      is_feature_loaded   = nullptr;
	PFun_slGetFeatureFunction*   get_feature_function = nullptr;
	PFun_slGetNewFrameToken*     get_new_frame_token = nullptr;
	PFun_slSetConstants*         set_constants       = nullptr;
	PFun_slSetTagForFrame*       set_tag_for_frame   = nullptr;
	PFun_slDLSSGSetOptions*      dlssg_set_options   = nullptr;
	PFun_slDLSSGGetState*        dlssg_get_state     = nullptr;
	PFun_slReflexSetOptions*     reflex_set_options  = nullptr;
	PFun_slReflexSleep*          reflex_sleep        = nullptr;
	PFun_slPCLSetMarker*         pcl_set_marker      = nullptr;
	PFun_slEvaluateFeature*      evaluate_feature    = nullptr; // desport: DLSS SR
	PFun_slAllocateResources*    allocate_resources  = nullptr; // desport: DLSS SR
	PFun_slDLSSGetOptimalSettings* dlss_get_optimal_settings = nullptr; // desport: DLSS SR
	PFun_slDLSSSetOptions*       dlss_set_options    = nullptr; // desport: DLSS SR
};
// slReflexSleep once a presented frame, before its markers: NVIDIA's Reflex guide requires it every frame (whatever
// the Reflex mode), and DLSS-G's pacer on Vulkan relies on Reflex. Without it every DLSS-G present blocked ~34 ms
// (instead of the ~0.2 ms the DLSS-G guide gives), so the game made 30 frames a second with frame generation on.
// KYTY_FRAMEGEN_REFLEX_SLEEP=0 leaves it out; KYTY_FRAMEGEN_REFLEX=off|on|boost the Reflex mode (default on).
bool   g_reflex_sleep = true;
double g_sleep_ms_sum = 0.0, g_sleep_ms_max = 0.0; // present thread, the 300-present window
Api              g_api;
sl::FrameToken*  g_token = nullptr; // present thread: the frame being presented
uint32_t         g_frame_index = 0;
bool             g_options_set = false;
bool             g_options_fg_enabled = false;
uint32_t         g_presents_without_inputs = 0;
uint32_t         g_presents_with_inputs    = 0;
vk::Extent2D     g_options_backbuffer {};
vk::Extent2D     g_options_render {};
vk::Format       g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t         g_presents = 0;
std::mutex       g_token_mutex;
std::mutex       g_streamline_mutex;
// Matrices in Streamline's convention: row-major, row vectors (p' = p * M).
using Mat4 = sl::float4x4;

Mat4 Transposed(const std::array<float, 16>& m) { // our column-vector matrix -> SL
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		r[i] = sl::float4(m[i], m[4 + i], m[8 + i], m[12 + i]);
	}
	return r;
}

float At(const sl::float4& v, uint32_t i) {
	return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

Mat4 Multiply(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		float row[4] {};
		for (uint32_t j = 0; j < 4; ++j) {
			for (uint32_t k = 0; k < 4; ++k) row[j] += At(a[i], k) * At(b[k], j);
		}
		r[i] = sl::float4(row[0], row[1], row[2], row[3]);
	}
	return r;
}

bool Inverted(const Mat4& m, Mat4& out) {
	double a[4][8] {};
	for (uint32_t i = 0; i < 4; ++i) {
		for (uint32_t j = 0; j < 4; ++j) a[i][j] = At(m[i], j);
		a[i][4 + i] = 1.0;
	}
	for (uint32_t c = 0; c < 4; ++c) {
		uint32_t pivot = c;
		for (uint32_t r = c + 1; r < 4; ++r) {
			if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
		}
		if (std::fabs(a[pivot][c]) < 1e-12) return false;
		for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
		const double scale = 1.0 / a[c][c];
		for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
		for (uint32_t r = 0; r < 4; ++r) {
			if (r == c) continue;
			const double f = a[r][c];
			for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
		}
	}
	for (uint32_t i = 0; i < 4; ++i) {
		out[i] = sl::float4(static_cast<float>(a[i][4]), static_cast<float>(a[i][5]),
		                    static_cast<float>(a[i][6]), static_cast<float>(a[i][7]));
	}
	return true;
}

// The game's world->clip matrix is P * V with a perspective P whose w row is (0, 0, 1, 0) (w =
// view depth) and whose z row is constant (reversed-Z, infinite far). Split it into camera
// axes, position and projection terms.
struct CameraFrame {
	sl::float3 right, up, forward, position;
	float      sx = 1.0f, sy = 1.0f, cx = 0.0f, cy = 0.0f, z_near = 0.1f;
	bool       valid = false;
};

CameraFrame Decompose(const std::array<float, 16>& m) {
	CameraFrame c;
	const auto row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return c;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf = r3[3] / flen;
	const auto split = [&](const std::array<double, 4>& r, double& s, double& off, double axis[3], double& t) {
		off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
		double v[3] {r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
		s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (s < 1e-9) return false;
		for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
		t = (r[3] / flen - off * tf) / s;
		return true;
	};
	double sx = 0, cx = 0, sy = 0, cy = 0, x[3] {}, y[3] {}, tx = 0, ty = 0;
	if (!split(r0, sx, cx, x, tx) || !split(r1, sy, cy, y, ty)) return c;
	// Up = forward x right (Streamline's convention); a flipped clip y shows up as a negative sy.
	const double up[3] {f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2], f[0] * x[1] - f[1] * x[0]};
	if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) {
		sy = -sy;
		ty = -ty;
	}
	c.right    = sl::float3(float(x[0]), float(x[1]), float(x[2]));
	c.up       = sl::float3(float(up[0]), float(up[1]), float(up[2]));
	c.forward  = sl::float3(float(f[0]), float(f[1]), float(f[2]));
	// View translation (tx, ty, tf) = -axes . position.
	const double pos[3] {-(tx * x[0] + ty * up[0] + tf * f[0]), -(tx * x[1] + ty * up[1] + tf * f[1]),
	                     -(tx * x[2] + ty * up[2] + tf * f[2])};
	c.position = sl::float3(float(pos[0]), float(pos[1]), float(pos[2]));
	c.sx       = float(sx);
	c.sy       = float(sy);
	c.cx       = float(cx);
	c.cy       = float(cy);
	c.z_near   = float(r2[3] / flen);
	c.valid    = true;
	return c;
}

// Streamline's "Failed to find global tag" errors: a present in DLSS-G mode without its inputs (a short gap
// BeforePresent keeps the mode through). Counted, the first few shown.
std::atomic<uint32_t> g_missing_tag_errors {0};

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	// KYTY_FRAMEGEN_VERBOSE: every message on stdout (the run log), not only errors.
	static const bool verbose = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr;
	if (type == sl::LogType::eError && !verbose && message != nullptr &&
	    std::strstr(message, "Failed to find global tag") != nullptr &&
	    g_missing_tag_errors.fetch_add(1, std::memory_order_relaxed) >= 4) {
		return;
	}
	if (type == sl::LogType::eError || verbose) {
		std::printf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
		std::fflush(stdout);
	}
}

template <typename F>
bool FeatureFunction(sl::Feature feature, const char* name, F*& function) {
	void* pointer = nullptr;
	if (g_api.get_feature_function(feature, name, pointer) != sl::Result::eOk || pointer == nullptr) {
		LOGF("frame generation: %s is unavailable\n", name);
		return false;
	}
	function = reinterpret_cast<F*>(pointer);
	return true;
}
#endif

} // namespace

bool Enabled() {
	return g_ready && g_fg_active.load(std::memory_order_acquire);
}

bool PresentIntegrationReady() {
	return g_ready && (g_fg_runtime_toggle || g_fg_active.load(std::memory_order_acquire));
}

bool ToggleFrameGeneration() {
	if (!g_ready || !g_fg_runtime_toggle) {
		return false;
	}
	const bool was_enabled = g_fg_active.load(std::memory_order_relaxed);
	g_fg_active.store(!was_enabled, std::memory_order_release);
	g_fg_options_dirty.store(true, std::memory_order_release);
	const auto toggle = g_runtime_toggle_count.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                           std::chrono::steady_clock::now() - g_process_start_time)
	                           .count();
	const auto last_guest_frame = g_last_present_guest_frame.load(std::memory_order_acquire);
	LOGF("Frame generation toggle: %s -> %s toggle=%" PRIu64 " uptime_ms=%" PRId64
	     " last_guest_frame=%" PRIu64 "\n",
	     was_enabled ? "on" : "off", was_enabled ? "off" : "on", toggle, uptime_ms,
	     last_guest_frame);
	std::printf("Frame generation toggle: %s -> %s toggle=%" PRIu64 " uptime_ms=%" PRId64
	            " last_guest_frame=%" PRIu64 "\n",
	            was_enabled ? "on" : "off", was_enabled ? "off" : "on", toggle, uptime_ms,
	            last_guest_frame);
	std::fflush(stdout);
	return true;
}

bool FpsHudEnabled() {
	return g_fps_hud_active.load(std::memory_order_acquire);
}

bool ToggleFpsHud() {
	if (!g_fps_hud_runtime_toggle) {
		return false;
	}
	const bool was_enabled = g_fps_hud_active.load(std::memory_order_relaxed);
	g_fps_hud_active.store(!was_enabled, std::memory_order_release);
	LOGF("FPS counter toggle: %s -> %s\n", was_enabled ? "on" : "off", was_enabled ? "off" : "on");
	std::printf("FPS counter toggle: %s -> %s\n", was_enabled ? "on" : "off", was_enabled ? "off" : "on");
	std::fflush(stdout);
	return true;
}

const char* RuntimeStatus() {
	static char status[80];
	const bool fg_active = g_fg_active.load(std::memory_order_acquire);
	const bool hud_active = g_fps_hud_active.load(std::memory_order_acquire);
	if (g_fg_runtime_toggle || fg_active) {
		std::snprintf(status, sizeof(status), " | FG: %s%s", fg_active ? "on" : "off",
		              g_fg_runtime_toggle ? " (F2)" : "");
	} else {
		status[0] = '\0';
	}
	if (g_fps_hud_runtime_toggle) {
		const size_t used = std::strlen(status);
		std::snprintf(status + used, sizeof(status) - used, " | FPS: %s (F1)", hud_active ? "on" : "off");
	}
	return status;
}

bool CaptureRequested() {
	return g_ready;
}

bool Requested() {
	// Present-id is enabled for the Reflex/frame-generation path.
	// desport: DLSS SR loads Streamline too, whose common plugins enable VK_NV_low_latency2 (needs present-id).
	return g_generate > 0 || g_fg_runtime_toggle || g_sr_mode != 0;
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	// desport: the DLSS SR path also runs with Streamline's command-list state tracking off.
	if ((g_generate == 0 && !g_fg_runtime_toggle && g_sr_mode == 0) || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
		return;
	}
	// The real loader: Streamline itself forwards to it.
	HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
	auto*   get_device_proc = loader != nullptr ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	                                                reinterpret_cast<void*>(GetProcAddress(loader, "vkGetDeviceProcAddr")))
	                                          : nullptr;
	if (get_device_proc == nullptr) {
		return;
	}
	auto&      dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto native     = static_cast<VkDevice>(device);
	uint32_t   replaced   = 0;
	const auto direct     = [&](auto& function, const char* name) {
        if (auto* pointer = get_device_proc(native, name); pointer != nullptr) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(pointer);
            ++replaced;
        }
	};
	direct(dispatcher.vkBeginCommandBuffer, "vkBeginCommandBuffer");
	direct(dispatcher.vkCmdBindPipeline, "vkCmdBindPipeline");
	direct(dispatcher.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
	direct(dispatcher.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
	std::printf("Frame generation: %u command-buffer calls bypass Streamline\n", replaced);
#else
	(void)device;
#endif
}

void Initialize() {
	const char* value = std::getenv("KYTY_FRAMEGEN");
	const int requested_frames = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
	g_fg_runtime_toggle = std::getenv("KYTY_FRAMEGEN_RUNTIME_TOGGLE") != nullptr;
	const auto switched_off = [](const char* name) {
		const char* text = std::getenv(name);
		return text != nullptr && std::strcmp(text, "0") == 0;
	};
	g_learning_enabled = !switched_off("KYTY_FRAMEGEN_LEARN");
	g_hold_enabled     = !switched_off("KYTY_FRAMEGEN_CAMERA_HOLD");
	// desport: begin KYTY_DLSS (values of sl::DLSSMode: 1 performance, 2 balanced, 3 quality, 4 ultra performance,
	// 6 DLAA); every line to 'desport: end' is new.
	if (const char* dlss = std::getenv("KYTY_DLSS"); dlss != nullptr && *dlss != '\0') {
		const std::string text = dlss;
		struct Mode { const char* name; uint32_t value; };
		constexpr Mode modes[] {{"dlaa", 6}, {"quality", 3}, {"balanced", 2}, {"performance", 1},
		                        {"ultra_performance", 4}};
		for (const auto& mode: modes) {
			if (text == mode.name) {
				g_sr_mode      = mode.value;
				g_sr_mode_name = mode.name;
			}
		}
		if (g_sr_mode == 0 && text != "off" && text != "0") {
			std::printf("DLSS: KYTY_DLSS=%s is not off|dlaa|quality|balanced|performance|ultra_performance; off\n", dlss);
		}
		// desport: phase B: with frame generation too, both share one frame index per guest frame (SrWithFrameGeneration).
	}
	// desport: KYTY_DLSS_UPSCALE=1|2: real upscaling (SrUpscaleMode; 2: the scene composite at the output size).
	if (const char* upscale = std::getenv("KYTY_DLSS_UPSCALE");
	    upscale != nullptr && (std::strcmp(upscale, "1") == 0 || std::strcmp(upscale, "2") == 0)) {
		g_sr_upscale = g_sr_mode != 0 ? static_cast<uint32_t>(upscale[0] - '0') : 0u;
		if (g_sr_upscale == 0) std::printf("DLSS: KYTY_DLSS_UPSCALE=%s needs KYTY_DLSS=<mode>; off\n", upscale);
	}
	// desport: end KYTY_DLSS
	if (g_learning_enabled && (requested_frames > 0 || g_fg_runtime_toggle || g_sr_mode != 0)) { // desport: SR too
		LoadLayouts();
	}
	g_fps_hud_runtime_toggle = std::getenv("KYTY_FPS_HUD_RUNTIME_TOGGLE") != nullptr;
	const char* fps_hud = std::getenv("KYTY_FPS_HUD");
	g_fps_hud_active.store(fps_hud != nullptr && *fps_hud != '\0' && std::strcmp(fps_hud, "0") != 0,
	                       std::memory_order_relaxed);
	g_generate        = requested_frames > 0 ? requested_frames : (g_fg_runtime_toggle ? 1 : 0);
	g_fg_active.store(requested_frames > 0, std::memory_order_relaxed);
	if (g_fg_runtime_toggle) {
		LOGF("Frame generation: runtime toggle enabled; F2 starts %s\n",
		     g_fg_active.load(std::memory_order_relaxed) ? "on" : "off");
		std::printf("Frame generation: runtime toggle enabled; F2 starts %s\n",
		            g_fg_active.load(std::memory_order_relaxed) ? "on" : "off");
	}
	if (g_generate == 0 && !g_fg_runtime_toggle && g_sr_mode == 0) { // desport: Streamline for DLSS SR too
		LogDesDlss("unused"); // desport
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// Streamline SDK binaries: explicit override, portable package, or an in-tree build.
	std::filesystem::path dir;
	if (const char* env = std::getenv("KYTY_STREAMLINE_DIR"); env != nullptr && *env != '\0') {
		dir = env;
	} else {
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		const auto exe_dir = std::filesystem::path(exe).parent_path();
		const std::array candidates {
		    exe_dir / "streamline" / "bin" / "x64",
		    exe_dir / "_Build" / "deps" / "streamline" / "sdk" / "bin" / "x64",
		    exe_dir.parent_path() / "deps" / "streamline" / "sdk" / "bin" / "x64",
		};
		const auto found = std::find_if(candidates.begin(), candidates.end(), [](const auto& candidate) {
			return std::filesystem::exists(candidate / "sl.interposer.dll");
		});
		dir = found != candidates.end() ? *found : candidates[1];
	}
	const auto interposer = dir / "sl.interposer.dll";
	if (std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr) {
		// Graphics modules already loaded before Streamline (it must come first).
		for (const char* name: {"vulkan-1.dll", "nvoglv64.dll", "opengl32.dll", "dxgi.dll", "d3d11.dll", "d3d12.dll",
		                        "d3d9.dll", "nvapi64.dll", "nvcuda.dll", "nvofapi64.dll", "sl.interposer.dll"}) {
			if (GetModuleHandleA(name) != nullptr) std::printf("Frame generation: %s already loaded\n", name);
		}
	}
	HMODULE    module     = LoadLibraryW(interposer.c_str());
	if (module == nullptr) {
		LOGF("frame generation: cannot load %s\n", interposer.string().c_str());
		std::printf("Frame generation: cannot load %s\n", interposer.string().c_str());
		g_generate = 0;
		g_sr_mode  = 0; // desport: no Streamline, no DLSS SR
		LogDesDlss("missing"); // desport
		return;
	}
	const auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    reinterpret_cast<void*>(GetProcAddress(module, name)));
		return function != nullptr;
	};
	if (!load(g_api.init, "slInit") || !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
	    !load(g_api.is_feature_loaded, "slIsFeatureLoaded") ||
	    !load(g_api.get_feature_function, "slGetFeatureFunction") ||
	    !load(g_api.get_new_frame_token, "slGetNewFrameToken") ||
	    !load(g_api.set_constants, "slSetConstants") ||
	    !load(g_api.set_tag_for_frame, "slSetTagForFrame")) {
		LOGF("frame generation: sl.interposer.dll lacks an entry point\n");
		std::printf("Frame generation: sl.interposer.dll lacks an entry point\n"); // desport
		g_generate = 0;
		g_sr_mode  = 0; // desport: no Streamline, no DLSS SR
		LogDesDlss("bad_interposer"); // desport
		return;
	}
	static const std::wstring plugin_dir = dir.wstring();
	static const std::wstring log_dir    = (dir.parent_path() / "logs").wstring();
	static const wchar_t*     plugin_paths[] {plugin_dir.c_str()};
	// KYTY_FRAMEGEN_FEATURES=dlssg,reflex,pcl: optional diagnostic feature subset.
	static std::vector<sl::Feature> features;
	if (g_generate != 0 || g_fg_runtime_toggle) {
		features.insert(features.end(), {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL});
	}
	if (const char* list = std::getenv("KYTY_FRAMEGEN_FEATURES"); list != nullptr) {
		const std::string text = list;
		features.clear();
		if (text.find("dlssg") != std::string::npos) features.push_back(sl::kFeatureDLSS_G);
		if (text.find("reflex") != std::string::npos) features.push_back(sl::kFeatureReflex);
		if (text.find("pcl") != std::string::npos) features.push_back(sl::kFeaturePCL);
	}
	// desport: begin DLSS Super Resolution (KYTY_DLSS) feature and its evaluate entry points (new lines to 'end').
	if (g_sr_mode != 0) {
		if (std::find(features.begin(), features.end(), sl::kFeatureDLSS) == features.end()) {
			features.push_back(sl::kFeatureDLSS);
		}
		if (!load(g_api.evaluate_feature, "slEvaluateFeature") ||
		    !load(g_api.allocate_resources, "slAllocateResources")) {
			LOGF("DLSS: sl.interposer.dll lacks slEvaluateFeature/slAllocateResources; DLSS SR off\n");
			std::printf("DLSS: sl.interposer.dll lacks slEvaluateFeature/slAllocateResources; DLSS SR off\n");
			g_sr_mode = 0;
			std::erase(features, sl::kFeatureDLSS);
		}
		if (features.empty()) {
			g_generate = 0;
			LogDesDlss("no_features"); // desport
			return;
		}
	}
	// desport: end DLSS Super Resolution feature
	sl::Preferences           preferences {};
	preferences.showConsole        = false;
	preferences.logLevel           = std::getenv("KYTY_FRAMEGEN_VERBOSE") ? sl::LogLevel::eVerbose
	                                                                      : sl::LogLevel::eDefault;
	preferences.pathsToPlugins     = plugin_paths;
	preferences.numPathsToPlugins  = 1;
	preferences.pathToLogsAndData  = nullptr;
	preferences.logMessageCallback = LogMessage;
	preferences.flags              = sl::PreferenceFlags::eDisableCLStateTracking |
	                    sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	preferences.featuresToLoad    = features.data();
	preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
	preferences.engine            = sl::EngineType::eCustom;
	preferences.engineVersion     = "1.0";
	preferences.projectId         = "6a3f1d5e-4b7c-4f2a-9d8e-1c2b3a4d5e6f";
	preferences.renderAPI         = sl::RenderAPI::eVulkan;
	if (const auto result = g_api.init(preferences, sl::kSDKVersion); result != sl::Result::eOk) {
		LOGF("frame generation: slInit failed (%d)\n", static_cast<int>(result));
		std::printf("Frame generation: slInit failed (%d)\n", static_cast<int>(result));
		g_generate = 0;
		g_sr_mode  = 0; // desport: no Streamline, no DLSS SR
		LogDesDlss("init_failed"); // desport
		return;
	}
	// SDL loads the interposer as the Vulkan loader: every instance/device/swapchain call then
	// goes through Streamline, which adds what DLSS-G needs.
	SetEnvironmentVariableW(L"SDL_VULKAN_LIBRARY", interposer.c_str());
	_putenv_s("SDL_VULKAN_LIBRARY", interposer.string().c_str());
	LOGF("Streamline loaded from %s, frame generation=%d, DLSS SR=%s\n", // desport: DLSS SR mode
	     dir.string().c_str(), g_generate, g_sr_mode != 0 ? g_sr_mode_name : "off");
#else
	LOGF("frame generation: this build has no Streamline support\n");
	g_generate = 0;
	g_sr_mode  = 0; // desport: no Streamline, no DLSS SR
	LogDesDlss("not_built"); // desport
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
	if (g_generate == 0 && !g_fg_runtime_toggle && g_sr_mode == 0) { // desport: DLSS SR too
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// desport: the DES_DLSS line once the device's features are known, whichever way OnDevice ends.
	struct DesDlssAtExit {
		~DesDlssAtExit() { LogDesDlss("loaded"); }
	} des_dlss_at_exit;
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
	// desport: begin DLSS SR, independent of DLSS-G (phase A runs it without frame generation); new lines to 'end'.
	if (g_sr_mode != 0) {
		bool sr_loaded = false;
		if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) != sl::Result::eOk ||
		    g_api.is_feature_loaded(sl::kFeatureDLSS, sr_loaded) != sl::Result::eOk || !sr_loaded) {
			LOGF("DLSS: DLSS Super Resolution is not supported on this device; the game's TAA stays\n");
			std::printf("DLSS: DLSS Super Resolution is not supported on this device; the game's TAA stays\n");
		} else if (FeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", g_api.dlss_get_optimal_settings) &&
		           FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options)) {
			g_sr_ready = true;
			LOGF("DLSS: Super Resolution ready, mode %s\n", g_sr_mode_name);
			std::printf("DLSS: Super Resolution ready, mode %s\n", g_sr_mode_name);
			std::fflush(stdout);
		}
	}
	if (g_generate == 0 && !g_fg_runtime_toggle) { // desport: DLSS SR only, no DLSS-G setup
		return;
	}
	// desport: end DLSS SR
	if (g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter) != sl::Result::eOk ||
	    g_api.is_feature_loaded(sl::kFeatureDLSS_G, loaded) != sl::Result::eOk || !loaded) {
		LOGF("frame generation: DLSS-G is not supported on this device\n");
		std::printf("Frame generation: DLSS-G is not supported on this device\n");
		return;
	}
	if (!FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
	    !FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
	    !FeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
	    !FeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
		return;
	}
	if (!FeatureFunction(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep)) {
		g_api.reflex_sleep = nullptr;
	}
	if (const char* sleep = std::getenv("KYTY_FRAMEGEN_REFLEX_SLEEP"); sleep != nullptr && std::strcmp(sleep, "0") == 0) {
		g_reflex_sleep = false;
	}
	sl::ReflexOptions reflex {};
	reflex.mode            = sl::ReflexMode::eLowLatency;
	const char* reflex_env = std::getenv("KYTY_FRAMEGEN_REFLEX");
	if (reflex_env != nullptr && std::strcmp(reflex_env, "off") == 0) {
		if (g_sr_ready) {
			// desport: phase B: Reflex stays on with DLSS SR and frame generation together (DLSS-G paces with it).
			LOGF("DLSS: KYTY_FRAMEGEN_REFLEX=off ignored with DLSS SR and frame generation; Reflex on\n");
			std::printf("DLSS: KYTY_FRAMEGEN_REFLEX=off ignored with DLSS SR and frame generation; Reflex on\n");
		} else {
			reflex.mode = sl::ReflexMode::eOff;
		}
	} else if (reflex_env != nullptr && std::strcmp(reflex_env, "boost") == 0) {
		reflex.mode = sl::ReflexMode::eLowLatencyWithBoost;
	}
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
		std::printf("Frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result)); // desport
		g_reflex_name = "failed"; // desport
	} else { // desport: DES_DLSS
		g_reflex_name = reflex.mode == sl::ReflexMode::eOff ? "off"
		                : reflex.mode == sl::ReflexMode::eLowLatency ? "on" : "boost";
	}
	std::printf("Frame generation: Reflex %s, slReflexSleep %s\n",
	            reflex.mode == sl::ReflexMode::eOff ? "off" : reflex.mode == sl::ReflexMode::eLowLatency ? "on" : "on+boost",
	            g_api.reflex_sleep == nullptr ? "unavailable" : g_reflex_sleep ? "called every frame" : "not called");
	g_ready = true;
	LOGF("frame generation: DLSS-G ready\n");
	std::printf("Frame generation: DLSS-G ready (%d generated frame(s) per frame)\n", g_generate);
	std::fflush(stdout);
#endif
}

void BeforePresent(const FrameInputs* inputs, vk::Extent2D color_extent, vk::Extent2D backbuffer,
                   vk::Rect2D region, vk::Format format, uint32_t image_count, bool reusing_last,
                   uint32_t token_index, bool constants_set) { // desport: phase B
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	std::unique_lock api_lock(g_streamline_mutex); // desport: released around slReflexSleep (the SR evaluates meanwhile)
	const sl::ViewportHandle viewport {0};
	// A minimized window presents a 0x0 back buffer: DLSS-G cannot be created for it (NGX 0xbad0000a every
	// present until the window is back), so it is off while minimized and on again with the window.
	const bool               minimized = backbuffer.width == 0 || backbuffer.height == 0;
	const bool               usable    = inputs != nullptr && inputs->valid && !minimized;
	// DLSS-G's mode follows F2. Switching it off and on again flushes its history and costs a
	// hitch, so a short gap in the inputs (a few frames) keeps it on: those presents carry no
	// tags (Streamline shows them as they are). Only a longer one (menus, loading, movies) turns
	// it off; 10 presents in a row with inputs turn it on again (fades into loading and movies run
	// the TAA pass on some frames only: switching on at the first one flapped the mode ~30 times in
	// a few seconds). With the camera held between captures (PrepareInputs) gaps come only from
	// frames without the game's TAA pass.
	constexpr uint32_t ModeOffAfterPresents = 30;
	constexpr uint32_t ModeOnAfterPresents  = 10;
	if (usable) {
		g_presents_without_inputs = 0;
		if (g_presents_with_inputs < std::numeric_limits<uint32_t>::max()) ++g_presents_with_inputs;
	} else {
		g_presents_with_inputs = 0;
		if (g_presents_without_inputs < std::numeric_limits<uint32_t>::max()) ++g_presents_without_inputs;
	}
	const bool fg_active = g_fg_active.load(std::memory_order_acquire);
	const bool fg_on     = g_options_set && g_options_fg_enabled;
	const bool enable_fg = fg_active && !minimized &&
	                       (fg_on ? (usable || g_presents_without_inputs <= ModeOffAfterPresents)
	                              : (usable && g_presents_with_inputs >= ModeOnAfterPresents));
	const bool mode_changed = !g_options_set || g_options_fg_enabled != enable_fg;
	// Minimized, DLSS-G only hears the switch off (once, with the last real size). A new back buffer format (HDR on
	// or off) sets the options again too.
	if (minimized ? (g_options_set && g_options_fg_enabled)
	              : (!g_options_set || g_fg_options_dirty.exchange(false, std::memory_order_acq_rel) ||
	                 g_options_fg_enabled != enable_fg || g_options_format != format ||
	                 (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render_extent)))) {
		sl::DLSSGOptions options {};
		options.mode                = enable_fg ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
		options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
		options.numBackBuffers      = image_count;
		// The off switch for a minimized window keeps the last real size (0x0 is not a valid color buffer).
		options.colorWidth          = minimized ? g_options_backbuffer.width : backbuffer.width;
		options.colorHeight         = minimized ? g_options_backbuffer.height : backbuffer.height;
		options.colorBufferFormat   = static_cast<uint32_t>(format);
		if (usable) {
			options.mvecDepthWidth    = inputs->render_extent.width;
			options.mvecDepthHeight   = inputs->render_extent.height;
			options.mvecBufferFormat  = static_cast<uint32_t>(inputs->motion.format);
			options.depthBufferFormat = static_cast<uint32_t>(inputs->depth.format);
			g_options_render          = inputs->render_extent;
		}
		if (const auto result = g_api.dlssg_set_options(viewport, options); result != sl::Result::eOk) {
			LOGF("frame generation: slDLSSGSetOptions failed (%d)\n", static_cast<int>(result));
		} else {
			if (mode_changed) {
				LOGF("frame generation: DLSS-G mode %s (F2 state and inputs)\n", enable_fg ? "on" : "off");
				std::printf("Frame generation: DLSS-G mode %s (F2 state and inputs)\n",
				            enable_fg ? "on" : "off");
				std::fflush(stdout);
			}
			g_options_set        = true;
			g_options_fg_enabled = enable_fg;
			if (!minimized) g_options_backbuffer = backbuffer;
			g_options_format = format;
		}
	}

	std::lock_guard lock(g_token_mutex);
	if (minimized) {
		g_token = nullptr; // no frame for Streamline: the present goes out untagged
		return;
	}
	// desport: phase B: the frame index the DLSS SR evaluated this guest frame with (carried in FrameInputs), so its
	// constants serve DLSS-G too; a frame without one (blank, reused, before the SR ran) or one older than the last
	// present (indices must grow at the present) takes a new index and its own constants.
	bool constants_ready = false;
	if (SrWithFrameGeneration() && token_index != 0 && token_index > g_frame_index) {
		g_frame_index   = token_index;
		constants_ready = constants_set;
	} else {
		g_frame_index = NextFrameIndex();
	}
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		static std::atomic_bool logged {false}; // desport: say it once
		if (!logged.exchange(true)) {
			LOGF("frame generation: slGetNewFrameToken failed on the present thread (frame %u); presents go untagged\n",
			     g_frame_index);
			std::printf("Frame generation: slGetNewFrameToken failed on the present thread (frame %u); presents go "
			            "untagged\n",
			            g_frame_index);
		}
		return;
	}
	// desport: (g_token stays this index's slot through the sleep below: the renderer thread takes at most one new
	// index a guest frame, far fewer than Streamline's ring of MAX_FRAMES_IN_FLIGHT = 6 tokens.)
	auto& token = *g_token;
	if (g_reflex_sleep && g_api.reflex_sleep != nullptr) {
		api_lock.unlock(); // desport: the renderer thread's DLSS SR evaluate does not wait for the Reflex sleep
		const auto sleep_begin = std::chrono::steady_clock::now();
		g_api.reflex_sleep(token);
		const double sleep_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sleep_begin).count();
		g_sleep_ms_sum += sleep_ms;
		g_sleep_ms_max = std::max(g_sleep_ms_max, sleep_ms);
		api_lock.lock(); // desport
	}
	++g_present_tag_stats.presents;
	if (usable) {
		g_last_present_guest_frame.store(inputs->guest_frame, std::memory_order_release);
		++g_present_tag_stats.tagged_inputs;
		g_present_tag_stats.last_color_extent = color_extent;
		g_present_tag_stats.last_input_extent = inputs->render_extent;
		g_present_tag_stats.last_output_extent = backbuffer;
		g_present_tag_stats.last_region_extent = region.extent;
		if (color_extent.width == inputs->render_extent.width &&
		    color_extent.height == inputs->render_extent.height) {
			++g_present_tag_stats.color_input_extent_matches;
		} else {
			++g_present_tag_stats.color_input_extent_mismatches;
		}
		if (g_present_tag_stats.first_guest_frame == 0) {
			g_present_tag_stats.first_guest_frame = inputs->guest_frame;
		}
		g_present_tag_stats.last_guest_frame = inputs->guest_frame;
	} else if (reusing_last) {
		++g_present_tag_stats.reused_frames;
	} else {
		++g_present_tag_stats.no_inputs;
	}
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);

	sl::Constants constants {};
	const auto    camera = usable ? Decompose(inputs->camera.view_proj) : CameraFrame {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	constants.cameraViewToClip = identity;
	constants.clipToCameraView = identity;
	constants.clipToLensClip   = identity;
	constants.clipToPrevClip   = identity;
	constants.prevClipToClip   = identity;
	constants.jitterOffset     = sl::float2(0.0f, 0.0f);
	constants.mvecScale        = sl::float2(1.0f, 1.0f);
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos        = sl::float3(0, 0, 0);
	constants.cameraUp         = sl::float3(0, 1, 0);
	constants.cameraRight      = sl::float3(1, 0, 0);
	constants.cameraFwd        = sl::float3(0, 0, 1);
	constants.cameraNear       = 0.1f;
	constants.cameraFar        = 10000.0f;
	constants.cameraFOV        = 1.0f;
	constants.cameraAspectRatio = 16.0f / 9.0f;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted       = sl::Boolean::eTrue;
	constants.cameraMotionIncluded = sl::Boolean::eTrue;
	constants.motionVectors3D     = sl::Boolean::eFalse;
	constants.reset               = usable ? sl::Boolean::eFalse : sl::Boolean::eTrue;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated   = sl::Boolean::eFalse;
	constants.motionVectorsJittered  = sl::Boolean::eFalse;
	if (usable && camera.valid) {
		// Projection in view space: x' = sx*x + cx*z, y' = sy*y + cy*z, z' = near, w' = z.
		Mat4 projection;
		projection[0] = sl::float4(camera.sx, 0, 0, 0);
		projection[1] = sl::float4(0, camera.sy, 0, 0);
		projection[2] = sl::float4(camera.cx, camera.cy, 0, 1);
		projection[3] = sl::float4(0, 0, camera.z_near, 0);
		constants.cameraViewToClip = projection;
		Inverted(projection, constants.clipToCameraView);
		const auto current  = Transposed(inputs->camera.view_proj);
		const auto previous = Transposed(inputs->camera.prev_view_proj);
		Mat4       clip_to_world;
		if (Inverted(current, clip_to_world)) {
			constants.clipToPrevClip = Multiply(clip_to_world, previous);
			Inverted(constants.clipToPrevClip, constants.prevClipToClip);
		}
		const float width  = static_cast<float>(inputs->render_extent.width);
		const float height = static_cast<float>(inputs->render_extent.height);
		constants.jitterOffset = sl::float2(inputs->camera.jitter_ndc[0] * width * 0.5f,
		                                    -inputs->camera.jitter_ndc[1] * height * 0.5f);
		static const bool pixels_mv = [] {
			const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
			return mv != nullptr && std::string(mv) == "px";
		}();
		// The game's vectors point along the motion, in UV units: its TAA reads the history at
		// uv - mv. DLSS wants them towards the previous position (uv + mv = previous uv), so they
		// are negated; with them as they were, generated frames moved the scene the wrong way
		// when the camera turned. KYTY_FRAMEGEN_MV_SIGN=1 keeps the game's sign (a comparison).
		static const float mv_sign = [] {
			const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
		}();
		constants.mvecScale = pixels_mv ? sl::float2(mv_sign / width, mv_sign / height) : sl::float2(mv_sign, mv_sign);
		constants.cameraPos   = camera.position;
		constants.cameraUp    = camera.up;
		constants.cameraRight = camera.right;
		constants.cameraFwd   = camera.forward;
		constants.cameraNear  = camera.z_near;
		constants.cameraFar   = 100000.0f;
		constants.cameraFOV   = 2.0f * std::atan(1.0f / std::fabs(camera.sy));
		constants.cameraAspectRatio = std::fabs(camera.sy / camera.sx);
	}
	if (constants_ready) {
		++g_present_tag_stats.sr_constants; // desport: phase B, set once by the DLSS SR for this frame index
	} else if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk &&
	           g_presents < 8) {
		LOGF("frame generation: slSetConstants failed (%d)\n", static_cast<int>(result));
	}

	if (usable) {
		sl::Resource depth(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->depth.image), nullptr,
		                   static_cast<VkImageView>(inputs->depth_view),
		                   static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		depth.width        = inputs->depth.extent.width;
		depth.height       = inputs->depth.extent.height;
		depth.nativeFormat = static_cast<uint32_t>(inputs->depth.format);
		depth.mipLevels    = 1;
		depth.arrayLayers  = 1;
		depth.flags        = 0;
		depth.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->depth.usage));
		sl::Resource motion(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->motion.image), nullptr,
		                    static_cast<VkImageView>(inputs->motion_view),
		                    static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		motion.width        = inputs->motion.extent.width;
		motion.height       = inputs->motion.extent.height;
		motion.nativeFormat = static_cast<uint32_t>(inputs->motion.format);
		motion.mipLevels    = 1;
		motion.arrayLayers  = 1;
		motion.flags        = 0;
		motion.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->motion.usage));
		const sl::Extent full {0, 0, inputs->render_extent.width, inputs->render_extent.height};
		const sl::Extent game {static_cast<uint32_t>(region.offset.y), static_cast<uint32_t>(region.offset.x),
		                       region.extent.width, region.extent.height};
		// The backbuffer is tagged only when the game image covers part of it (KYTY_PRESENT_ASPECT).
		const bool      partial = region.offset.x != 0 || region.offset.y != 0 ||
		                     region.extent.width != backbuffer.width || region.extent.height != backbuffer.height;
		sl::ResourceTag tags[] {
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &full),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
		                    &full),
		    sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent,
		                    &game),
		};
		const auto result = g_api.set_tag_for_frame(token, viewport, tags, partial ? 3u : 2u, nullptr);
		if (result != sl::Result::eOk) {
			++g_present_tag_stats.tag_failures;
			g_present_tag_stats.last_tag_result = static_cast<int32_t>(result);
		}
	}
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
#else
	(void)inputs;
	(void)backbuffer;
	(void)region;
	(void)format;
	(void)image_count;
	(void)token_index;   // desport
	(void)constants_set; // desport
#endif
}

void AfterPresent() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || g_token == nullptr) {
		return;
	}
	std::lock_guard api_lock(g_streamline_mutex);
	// desport: the token for the present's index again (the renderer thread's DLSS SR may have taken new tokens from
	// Streamline's small ring since BeforePresent; the same index as the current one returns the same token).
	sl::FrameToken* present_token = nullptr;
	if (g_api.get_new_frame_token(present_token, &g_frame_index) != sl::Result::eOk || present_token == nullptr) {
		present_token = g_token;
	}
	g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *present_token);
	const auto presents = ++g_presents;
	if (presents % 300 == 0) {
		const auto& stats = g_present_tag_stats;
		LOGF("frame generation: present tags last %u presents: tagged=%u no_inputs=%u reused=%u "
		     "tag_failures=%u last_result=%d guest_frame=%" PRIu64 "..%" PRIu64 "\n",
		     stats.presents, stats.tagged_inputs, stats.no_inputs, stats.reused_frames,
		     stats.tag_failures, stats.last_tag_result, stats.first_guest_frame, stats.last_guest_frame);
		std::printf("Frame generation: present tags last %u presents: tagged=%u no_inputs=%u reused=%u "
		            "tag_failures=%u sr_constants=%u last_result=%d missing_tag_errors=%u mode=%s guest_frame=%" PRIu64
		            "..%" PRIu64 "\n", // desport: sr_constants (phase B)
		            stats.presents, stats.tagged_inputs, stats.no_inputs, stats.reused_frames,
		            stats.tag_failures, stats.sr_constants, stats.last_tag_result,
		            g_missing_tag_errors.exchange(0, std::memory_order_relaxed), g_options_fg_enabled ? "on" : "off",
		            stats.first_guest_frame, stats.last_guest_frame);
		if (stats.tagged_inputs != 0) {
			LOGF("Frame generation input alignment: color=%ux%u depth_motion=%ux%u output=%ux%u region=%ux%u "
			     "color_input_matches=%u mismatches=%u across %u tagged presents\n",
			     stats.last_color_extent.width, stats.last_color_extent.height,
			     stats.last_input_extent.width, stats.last_input_extent.height,
			     stats.last_output_extent.width, stats.last_output_extent.height,
			     stats.last_region_extent.width, stats.last_region_extent.height,
			     stats.color_input_extent_matches, stats.color_input_extent_mismatches,
			     stats.tagged_inputs);
			std::printf("Frame generation input alignment: color=%ux%u depth_motion=%ux%u output=%ux%u region=%ux%u "
			            "color_input_matches=%u mismatches=%u across %u tagged presents\n",
			            stats.last_color_extent.width, stats.last_color_extent.height,
			            stats.last_input_extent.width, stats.last_input_extent.height,
			            stats.last_output_extent.width, stats.last_output_extent.height,
			            stats.last_region_extent.width, stats.last_region_extent.height,
			            stats.color_input_extent_matches, stats.color_input_extent_mismatches,
			            stats.tagged_inputs);
		}
		std::printf("Frame generation: reflex_sleep avg/max=%.2f/%.2f ms\n",
		            stats.presents != 0 ? g_sleep_ms_sum / stats.presents : 0.0, g_sleep_ms_max);
		g_sleep_ms_sum = g_sleep_ms_max = 0.0;
		std::fflush(stdout);
		g_present_tag_stats = {};
	}
	if (presents % 30 != 1) {
		return;
	}
	sl::DLSSGState state {};
	if (g_api.dlssg_get_state(sl::ViewportHandle {0}, state, nullptr) != sl::Result::eOk) {
		return;
	}
	g_presented_frames.store(std::max(state.numFramesActuallyPresented, 1u), std::memory_order_relaxed);
	if (presents % 600 == 1) {
		LOGF("frame generation: status=0x%x presented=%u max_generated=%u vram=%" PRIu64 " MiB\n",
		     static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		     state.numFramesToGenerateMax, state.estimatedVRAMUsageInBytes >> 20u);
		std::printf("Frame generation: status=0x%x presented=%u max_generated=%u\n",
		            static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		            state.numFramesToGenerateMax);
		std::fflush(stdout);
	}
#endif
}

uint32_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

void SetStaticRect(float x0, float y0, float x1, float y1) {
	const auto pack = [](float value, uint32_t index) {
		return static_cast<uint64_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 65535.0f)) << (16u * index);
	};
	g_static_rect.store(pack(x0, 0) | pack(y0, 1) | pack(x1, 2) | pack(y1, 3), std::memory_order_relaxed);
}

void OnDraw(uint64_t vs_hash, std::span<const uint32_t> srt) {
	if (!Enabled() && !SrActive()) { // desport: the DLSS SR needs the camera too
		return;
	}
	const auto layout = std::find_if(CameraShaders.begin(), CameraShaders.end(),
	                                 [vs_hash](const CameraShaderLayout& candidate) {
	                                  return candidate.hash == vs_hash;
	                                 });
	if (layout == CameraShaders.end()) {
		if (!g_learning_enabled || srt.size() < 16) {
			return;
		}
		auto* learned = FindLearned(vs_hash);
		if (learned == nullptr) {
			return;
		}
		switch (learned->state) {
			case LearnedLayout::State::Camera: {
				Camera camera;
				if (!ReadLearnedCamera(*learned, srt, camera)) return;
				if (!g_learned_camera.valid) {
					if (PlausibleCamera(camera)) {
						g_learned_camera      = camera;
						g_learned_camera_hash = vs_hash;
					}
				} else if (g_learned_camera_hash == vs_hash &&
				           std::memcmp(camera.view_proj.data(), g_learned_camera.view_proj.data(),
				                       sizeof(camera.view_proj)) != 0) {
					// Two draws, two matrices: an object's world->clip matrix, not the camera.
					RejectLearned(*learned, "differs between draws");
					g_learned_camera.valid = false;
				}
				return;
			}
			case LearnedLayout::State::Learning:
				if (learned->snapshot_frame == g_capture_frame) return; // one look a frame
				learned->snapshot_frame = g_capture_frame;
				if (g_camera.valid) {
					LearnFrom(*learned, srt, g_camera);
				} else if (g_learn_snapshot_count < g_learn_snapshots.size()) {
					// The frame's camera comes later: compared at the frame's end.
					auto& snapshot  = g_learn_snapshots[g_learn_snapshot_count++];
					snapshot.layout = learned;
					snapshot.words  = static_cast<uint32_t>(std::min<size_t>(srt.size(), LearnSnapshot::MaxWords));
					std::memcpy(snapshot.srt.data(), srt.data(), snapshot.words * sizeof(uint32_t));
				}
				return;
			default: return;
		}
	}
	if (g_camera.valid) {
		return;
	}
	const uint32_t view_proj_offset = layout->view_proj_offset;
	const uint32_t prev_view_proj_offset = layout->prev_view_proj_offset;
	const uint32_t jitter_offset = layout->jitter_offset;
	if (srt.size() < view_proj_offset + 16 || srt.size() < prev_view_proj_offset + 12 ||
	    srt.size() < jitter_offset + 2) {
		return;
	}
	Camera camera;
	for (uint32_t i = 0; i < 16; ++i) {
		camera.view_proj[i] = AsFloat(srt[view_proj_offset + i]);
	}
	// The previous matrix has rows x, y, w; its z row is taken from the current one.
	for (uint32_t i = 0; i < 4; ++i) {
		camera.prev_view_proj[i]      = AsFloat(srt[prev_view_proj_offset + i]);
		camera.prev_view_proj[4 + i]  = AsFloat(srt[prev_view_proj_offset + 4 + i]);
		camera.prev_view_proj[8 + i]  = camera.view_proj[8 + i];
		camera.prev_view_proj[12 + i] = AsFloat(srt[prev_view_proj_offset + 8 + i]);
	}
	camera.jitter_ndc = {AsFloat(srt[jitter_offset]), AsFloat(srt[jitter_offset + 1])};
	if (!PlausibleCamera(camera)) {
		++g_capture_stats.rejected_cameras; // (the next listed draw of the frame may still give it)
		return;
	}
	++g_capture_stats.camera_captures;
	camera.valid = true;
	g_camera     = camera;
}

void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address) {
	if ((!Enabled() && !SrActive()) || !IsTaaShader(cs_hash)) { // desport: DLSS SR too
		return;
	}
	++g_capture_stats.taa_dispatches;
	g_sources = {depth, motion, depth_address, motion_address, true};
}

void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs) {
	inputs.valid = false;
	// desport: phase B: every guest frame carries its Streamline frame index to the present (the one the DLSS SR
	// evaluated with, else a new one taken here, in guest-frame order), F2 on or off.
	inputs.token_index   = 0;
	inputs.constants_set = false;
	if (SrWithFrameGeneration()) {
		if (g_frame_token_index == 0) g_frame_token_index = NextFrameIndex();
		inputs.token_index    = g_frame_token_index;
		inputs.constants_set  = g_frame_constants_set;
		g_frame_token_index   = 0;
		g_frame_constants_set = false;
	}
	if (!Enabled()) {
		if (SrActive()) { // desport: F2 has DLSS-G off, the DLSS SR still runs: its camera ends at the flip
			g_sources.valid = false;
			EndSrFrame();
			return;
		}
		// Keep the present-thread Streamline lifecycle alive for F2, but avoid copying the
		// game's depth and motion images while frame generation is switched off.
		g_sources.valid        = false;
		g_camera.valid         = false;
		g_learned_camera.valid = false;
		g_learn_snapshot_count = 0;
		++g_capture_frame;
		return;
	}
	const auto sources = g_sources;
	auto       camera  = g_camera;
	auto       source  = CameraSource::Table;
	if (camera.valid) {
		// Shaders that drew before the listed camera, and a learned layout drawn with it, against it.
		for (uint32_t i = 0; i < g_learn_snapshot_count; ++i) {
			auto& snapshot = g_learn_snapshots[i];
			LearnFrom(*snapshot.layout, std::span<const uint32_t>(snapshot.srt.data(), snapshot.words), camera);
		}
		if (g_learned_camera.valid && std::memcmp(g_learned_camera.view_proj.data(), camera.view_proj.data(),
		                                          sizeof(camera.view_proj)) != 0) {
			if (auto* learned = FindLearned(g_learned_camera_hash); learned != nullptr) {
				RejectLearned(*learned, "differs from the listed camera");
			}
		}
	} else if (g_learned_camera.valid) {
		camera = g_learned_camera;
		source = CameraSource::Learned;
	}
	if (camera.valid) {
		g_last_camera       = camera;
		g_last_camera_frame = g_capture_frame;
	} else if (g_hold_enabled && g_last_camera.valid && sources.valid) {
		// No camera shader drew: the last camera, still, and no jitter. The motion vectors carry the
		// camera's motion as well (cameraMotionIncluded); DLSS-G keeps generating instead of its
		// mode going off and on with every view the camera shaders miss.
		camera                = g_last_camera;
		camera.prev_view_proj = camera.view_proj;
		camera.jitter_ndc     = {0.0f, 0.0f};
		source                = CameraSource::Held;
	}
	const bool ready = sources.valid && camera.valid;
	g_sources.valid        = false;
	g_camera.valid         = false;
	g_learned_camera.valid = false;
	g_learn_snapshot_count = 0;
	++g_capture_frame;
	++g_sr_frame; // desport: phase B, the DLSS SR's frame count (EndSrFrame's, which does not run with DLSS-G)
	++g_guest_frame;
	++g_capture_stats.frames;
	if (!ready) {
		g_capture_stats.no_taa += sources.valid ? 0u : 1u;
		g_capture_stats.no_camera += camera.valid ? 0u : 1u;
		ReportCaptureStats();
		return;
	}
	auto& depth  = cache.GetImage(sources.depth);
	auto& motion = cache.GetImage(sources.motion);
	if (depth.info.data.address != sources.depth_address ||
	    motion.info.data.address != sources.motion_address ||
	    depth.backing.extent.width != motion.backing.extent.width ||
	    depth.backing.extent.height != motion.backing.extent.height) {
		++g_capture_stats.replaced_images;
		ReportCaptureStats();
		return; // the images were replaced after the TAA pass
	}
	Configure(graphics, inputs.depth, inputs.depth_view, depth.backing);
	Configure(graphics, inputs.motion, inputs.motion_view, motion.backing, &inputs.motion_attachment);
	command.EndRendering();
	auto vk_command = command.Handle();
	CopyInto(vk_command, depth, inputs.depth);
	CopyInto(vk_command, motion, inputs.motion);
	if (inputs.motion_attachment) {
		ZeroStaticRect(vk_command, inputs.motion, inputs.motion_view);
	}
	inputs.render_extent = {motion.backing.extent.width, motion.backing.extent.height};
	inputs.camera        = camera;
	inputs.camera_source = source;
	inputs.guest_frame   = g_guest_frame;
	inputs.valid         = true;
	++g_capture_stats.valid_inputs;
	g_capture_stats.learned_cameras += source == CameraSource::Learned ? 1u : 0u;
	g_capture_stats.held_cameras += source == CameraSource::Held ? 1u : 0u;
	ReportCaptureStats();
}

void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs) {
	for (auto* view: {&inputs.depth_view, &inputs.motion_view}) {
		if (*view != nullptr) {
			graphics.device.destroyImageView(*view, nullptr);
			*view = nullptr;
		}
	}
	for (auto* image: {&inputs.depth, &inputs.motion}) {
		if (image->image != nullptr) {
			graphics.DeleteImage(*image);
		}
	}
	inputs.valid = false;
}

// desport: begin DLSS SR/DLAA (KYTY_DLSS), every line to 'desport: end' is new
// desport: DLSS Super Resolution / DLAA (KYTY_DLSS) in place of the game's TAA dispatch (docs/DLSS_PLAN.md, phase
// A: without DLSS-G). Renderer thread only. The TAA dispatch's images: depth (slot 1) and motion vectors (slot 3) as
// DLSS-G takes them; the color input and the output were not confirmed by a frame capture yet, so they are chosen
// from the shader's metadata (output: the written storage image; color: a sampled color image of the motion
// vectors' extent), KYTY_DLSS_*_SLOT overriding, and logged for checking against a capture.
bool SrRequested() {
	return g_sr_mode != 0;
}

bool SrActive() {
	return g_sr_ready && g_sr_mode != 0 && !g_sr_failed.load(std::memory_order_relaxed);
}

bool SrWithFrameGeneration() { // desport: phase B
	return g_ready && SrActive();
}

bool SrUpscale() { // desport: KYTY_DLSS_UPSCALE
	return g_sr_upscale != 0 && SrActive();
}

void EndSrFrame() {
	if (!SrActive()) {
		return;
	}
	// desport: as PrepareInputs does for DLSS-G: OnDraw keeps the frame's first camera, so it ends at the flip.
	if (g_camera.valid) {
		for (uint32_t i = 0; i < g_learn_snapshot_count; ++i) {
			auto& snapshot = g_learn_snapshots[i];
			LearnFrom(*snapshot.layout, std::span<const uint32_t>(snapshot.srt.data(), snapshot.words), g_camera);
		}
		g_last_camera       = g_camera;
		g_last_camera_frame = g_capture_frame;
	} else if (g_learned_camera.valid) {
		g_last_camera       = g_learned_camera;
		g_last_camera_frame = g_capture_frame;
	}
	g_camera.valid         = false;
	g_learned_camera.valid = false;
	g_learn_snapshot_count = 0;
	++g_capture_frame;
	++g_sr_frame;
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
namespace {

// desport: best guess for the TAA dispatch's color slot, used only when the metadata does not single one out.
constexpr int      SrGuessColorSlot = 0;
constexpr int      SrDepthSlot      = 1; // (renderCompute.cpp's FrameGen::OnDispatch)
constexpr int      SrMotionSlot     = 3;
constexpr uint32_t SrViewport       = 1; // (DLSS-G uses viewport 0)

// desport: one log line per kind of problem (each falls back to the game's TAA for that dispatch).
enum SrIssue : uint32_t {
	SrIssueSlots    = 1u << 0,
	SrIssueOutput   = 1u << 1,
	SrIssueColor    = 1u << 2,
	SrIssueRange    = 1u << 3,
	SrIssueOptions  = 1u << 4,
	SrIssueToken    = 1u << 5,
	SrIssueEvaluate = 1u << 6,
	SrIssueTwice    = 1u << 7,
};
std::atomic<uint32_t> g_sr_issues {0};

void SrIssueOnce(uint32_t issue, const char* format, ...) {
	if ((g_sr_issues.fetch_or(issue, std::memory_order_relaxed) & issue) != 0) {
		return;
	}
	char    text[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	LOGF("DLSS: %s; the game's TAA runs instead\n", text);
	std::printf("DLSS: %s; the game's TAA runs instead\n", text);
	std::fflush(stdout);
}

int SlotSetting(const char* name, int fallback) {
	const char* text = std::getenv(name);
	if (text == nullptr || *text == '\0') return fallback;
	char*      end   = nullptr;
	const long value = std::strtol(text, &end, 10);
	return end != text && *end == '\0' && value >= 0 && value < 64 ? static_cast<int>(value) : fallback;
}

bool Setting(const char* name, bool fallback) {
	const char* text = std::getenv(name);
	if (text == nullptr || *text == '\0') return fallback;
	return std::strcmp(text, "0") != 0;
}

bool HdrFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR16G16B16Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eR32G32B32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eE5B9G9R9UfloatPack32: return true;
		default: return false;
	}
}

// desport: KYTY_DLSS_PRESET. Default: DLSS 4.5 (nvngx_dlss 310.x), the second-generation transformer: preset M,
// and L for ultra performance (sl_dlss.h names them the defaults of performance and ultra performance). "sdk": the
// SDK/driver's own default per mode (K for DLAA/quality/balanced). J, K, L, M: that preset in every mode.
struct SrPresets {
	sl::DLSSPreset dlaa, quality, balanced, performance, ultra_performance;
	const char*    name;
};
SrPresets ReadPresets() {
	const char*       text  = std::getenv("KYTY_DLSS_PRESET");
	const std::string value = text != nullptr ? text : "";
	const auto        all   = [](sl::DLSSPreset preset, const char* name) {
        return SrPresets {preset, preset, preset, preset, preset, name};
	};
	if (value == "sdk") return all(sl::DLSSPreset::eDefault, "sdk default");
	if (value == "J" || value == "j") return all(sl::DLSSPreset::ePresetJ, "J");
	if (value == "K" || value == "k") return all(sl::DLSSPreset::ePresetK, "K");
	if (value == "L" || value == "l") return all(sl::DLSSPreset::ePresetL, "L");
	if (value == "M" || value == "m") return all(sl::DLSSPreset::ePresetM, "M");
	if (!value.empty() && value != "default") {
		std::printf("DLSS: KYTY_DLSS_PRESET=%s is not default|sdk|J|K|L|M; default\n", text);
	}
	return {sl::DLSSPreset::ePresetM, sl::DLSSPreset::ePresetM, sl::DLSSPreset::ePresetM, sl::DLSSPreset::ePresetM,
	        sl::DLSSPreset::ePresetL, "DLSS 4.5 (M; L for ultra performance)"};
}

// desport: KYTY_DLSS_JITTER_SIGN=<x><y>, '+' or '-': the jitter in render pixels is (x * jx * w / 2, y * jy * h / 2)
// of the game's NDC jitter (default "+-", DLSS-G's conversion in BeforePresent; not validated for SR).
std::array<float, 2> JitterSign() {
	const char*          text = std::getenv("KYTY_DLSS_JITTER_SIGN");
	std::array<float, 2> sign {1.0f, -1.0f};
	if (text != nullptr && std::strlen(text) >= 2) {
		for (int i = 0; i < 2; ++i) sign[i] = text[i] == '-' ? -1.0f : 1.0f;
	}
	return sign;
}

sl::Resource SrResource(const SrSlot& slot) {
	sl::Resource resource(sl::ResourceType::eTex2d, static_cast<VkImage>(slot.image), nullptr,
	                      static_cast<VkImageView>(slot.view),
	                      static_cast<uint32_t>(static_cast<VkImageLayout>(slot.layout)));
	resource.width        = slot.extent.width;
	resource.height       = slot.extent.height;
	resource.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(slot.view_format));
	resource.mipLevels    = 1;
	resource.arrayLayers  = 1;
	resource.flags        = 0;
	resource.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(slot.usage));
	return resource;
}

// desport: the last evaluated frame and the applied options (Kyty.Gpu).
struct SrLast {
	bool         valid = false;
	uint64_t     frame = 0;
	vk::Extent2D output {}, render {};
	int          color = -1, output_slot = -1;
	CameraFrame  camera;
	uint32_t     logged_configs = 0;
	uint64_t     signature      = 0;
};
struct SrOptions {
	bool         set = false, allocated = false, hdr = true;
	vk::Extent2D output {};
	uint32_t     min_w = 0, min_h = 0, max_w = 0, max_h = 0; // the render size range (0: unknown)
};
struct SrStats {
	uint32_t evaluated = 0, resets = 0, held = 0, learned = 0;
};
SrLast    g_sr_last;
SrOptions g_sr_options;
SrStats   g_sr_stats;

// desport: KYTY_DLSS_UPSCALE (Kyty.Gpu). An image the engine owns, replaced when its size or format changes; a replaced
// image is left allocated (it may still be in flight): at most UpscaleMaxImages per kind and run.
constexpr uint32_t UpscaleMaxImages = 4;
struct OwnedImage {
	std::vector<std::unique_ptr<VulkanImage>> images;
	VulkanImage*  current = nullptr;
	vk::ImageView view    = nullptr;
};
// desport: the frame's TAA output (and the DLSS image of the output size written from its inputs); with
// KYTY_DLSS_UPSCALE=2 also the scene composite's render-size target (the draw that samples the TAA output into a
// target of the TAA output's size, 01.007.000 ps 0x95318d27a44b0dd5), its copy at the output size (the same draw
// recorded again, reading the DLSS image), and the final pass that scales the composite to the output (#257, ps
// 0x37db6bd367380e9c: scene and UI into the output buffer), whose largest texture (the UI layer) is the output size.
struct SrUpscaleState {
	uint64_t     taa_address = 0;
	vk::Extent2D taa_extent {}, render {}, output {};
	uint64_t     frame = UINT64_MAX; // g_sr_frame of the last TAA dispatch
	bool         ready = false;      // DLSS wrote `dlss` from that dispatch's inputs
	OwnedImage   dlss, composite;
	uint64_t     composite_address = 0, composite_ps = 0, final_address = 0, final_ps = 0;
	vk::Extent2D composite_extent {};
	uint64_t     composite_frame = UINT64_MAX;       // the frame the composite target was seen in
	uint64_t     composite_ready_frame = UINT64_MAX; // the frame the output-size composite was recorded in
	uint32_t     substituted = 0, replays = 0, logged_readers = 0, logged_writers = 0;
	uint64_t     last_reader = 0, last_writer = 0;
	bool         announced = false;
};
SrUpscaleState g_up;

// desport: one line per kind of upscale problem (the game's own upscale stays for that frame).
void UpscaleNoteOnce(uint32_t bit, const char* format, ...) {
	static std::atomic<uint32_t> noted {0};
	if ((noted.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
	char    text[384];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	LOGF("DLSS upscale: %s\n", text);
	std::printf("DLSS upscale: %s\n", text);
	std::fflush(stdout);
}

// desport: `owned` holds an image of `format` and `extent` with `usage` (and a view of it); false when it cannot.
bool EnsureOwned(GraphicContext& graphics, OwnedImage& owned, vk::Format format, vk::Extent2D extent,
                 vk::ImageUsageFlags usage, vk::FormatFeatureFlags needed, const char* what, uint32_t note) {
	auto* image = owned.current;
	if (image != nullptr && image->format == format && image->extent.width == extent.width &&
	    image->extent.height == extent.height) {
		return true;
	}
	if (owned.images.size() >= UpscaleMaxImages) {
		UpscaleNoteOnce(note, "the %s image changed size more than %u times; no more images", what, UpscaleMaxImages);
		return false;
	}
	if ((graphics.physical_device.getFormatProperties(format).optimalTilingFeatures & needed) != needed) {
		UpscaleNoteOnce(note << 1u, "format %u cannot be the %s image", static_cast<uint32_t>(format), what);
		return false;
	}
	auto                fresh = std::make_unique<VulkanImage>();
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, *fresh)) {
		UpscaleNoteOnce(note << 2u, "cannot allocate the %ux%u %s image", extent.width, extent.height, what);
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = fresh->image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageView view         = nullptr;
	if (graphics.device.createImageView(&view_info, nullptr, &view) != vk::Result::eSuccess || view == nullptr) {
		graphics.DeleteImage(*fresh);
		UpscaleNoteOnce(note << 2u, "cannot create the %ux%u %s image's view", extent.width, extent.height, what);
		return false;
	}
	owned.current = fresh.get();
	owned.view    = view;
	owned.images.push_back(std::move(fresh));
	return true;
}

// desport: the output-size image DLSS writes, of `format` (the TAA output's) and `extent`, in General.
bool PrepareUpscaled(GraphicContext& graphics, vk::CommandBuffer command, vk::Format format, vk::Extent2D extent) {
	if (!EnsureOwned(graphics, g_up.dlss, format, extent,
	                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
	                     vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
	                 vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eSampledImage, "DLSS output",
	                 1u << 0)) {
		return false;
	}
	auto& image = *g_up.dlss.current;
	if (image.state.layout != vk::ImageLayout::eGeneral) {
		Transition(command, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		           vk::PipelineStageFlagBits2::eComputeShader);
	}
	return true;
}

sl::Resource UpscaledResource() {
	const auto&  image = *g_up.dlss.current;
	sl::Resource resource(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
	                      static_cast<VkImageView>(g_up.dlss.view), static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL));
	resource.width        = image.extent.width;
	resource.height       = image.extent.height;
	resource.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
	resource.mipLevels    = 1;
	resource.arrayLayers  = 1;
	resource.flags        = 0;
	resource.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
	return resource;
}

// desport: `a` at least `b` in both directions and larger in one.
bool LargerExtent(vk::Extent2D a, vk::Extent2D b) {
	return a.width >= b.width && a.height >= b.height && (a.width > b.width || a.height > b.height);
}

uint64_t UpscalePsFilter() {
	static const uint64_t filter = [] {
		const char* text = std::getenv("KYTY_DLSS_UPSCALE_PS");
		return text != nullptr && *text != '\0' ? std::strtoull(text, nullptr, 16) : 0ull;
	}();
	return filter;
}

// desport: up to 24 different lines about the draws that read the TAA output or the composite (and write the final
// target), to check the chain against a capture.
void UpscaleReader(const char* what, uint64_t address, vk::Extent2D extent, uint64_t ps_hash, vk::Extent2D target,
                   const char* verdict) {
	const uint64_t reader = ps_hash ^ (uint64_t {target.width} << 32u) ^ (uint64_t {target.height} << 48u) ^
	                        reinterpret_cast<uintptr_t>(verdict) ^ reinterpret_cast<uintptr_t>(what);
	if (reader == g_up.last_reader || g_up.logged_readers >= 24) return;
	g_up.last_reader = reader;
	++g_up.logged_readers;
	LOGF("DLSS upscale: %s 0x%" PRIx64 " %ux%u read by ps 0x%016" PRIx64 " into %ux%u: %s\n", what, address,
	     extent.width, extent.height, ps_hash, target.width, target.height, verdict);
	std::printf("DLSS upscale: %s 0x%" PRIx64 " %ux%u read by ps 0x%016" PRIx64 " into %ux%u: %s\n", what, address,
	            extent.width, extent.height, ps_hash, target.width, target.height, verdict);
	std::fflush(stdout);
}

void UpscaleAnnounce(const char* how, uint64_t ps_hash) {
	if (g_up.announced) return;
	g_up.announced = true;
	LOGF("DES_DLSS upscale=%u render=%ux%u output=%ux%u %s=0x%016" PRIx64 " mode=%s\n", g_sr_upscale,
	     g_up.render.width, g_up.render.height, g_up.output.width, g_up.output.height, how, ps_hash, g_sr_mode_name);
	std::printf("DES_DLSS upscale=%u render=%ux%u output=%ux%u %s=0x%016" PRIx64 " mode=%s\n", g_sr_upscale,
	            g_up.render.width, g_up.render.height, g_up.output.width, g_up.output.height, how, ps_hash,
	            g_sr_mode_name);
	std::fflush(stdout);
}

} // namespace
#endif

bool EvaluateSr(GraphicContext& graphics, vk::CommandBuffer command, uint64_t cs_hash, std::span<const SrSlot> slots,
                bool upscale) { // desport: upscale (KYTY_DLSS_UPSCALE)
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!SrActive() || !IsTaaShader(cs_hash)) {
		return false;
	}
	if (upscale) { // desport: KYTY_DLSS_UPSCALE: nothing from an earlier dispatch is sampled from here on
		g_up.ready = false;
		g_up.frame = g_sr_frame;
	}
	static const int  color_setting  = SlotSetting("KYTY_DLSS_COLOR_SLOT", -1);
	static const int  output_setting = SlotSetting("KYTY_DLSS_OUTPUT_SLOT", -1);
	static const int  depth_slot     = SlotSetting("KYTY_DLSS_DEPTH_SLOT", SrDepthSlot);
	static const int  motion_slot    = SlotSetting("KYTY_DLSS_MOTION_SLOT", SrMotionSlot);
	static const auto presets        = ReadPresets();
	static const auto jitter_sign    = JitterSign();
	static const bool auto_exposure  = Setting("KYTY_DLSS_AUTOEXPOSURE", true);
	static const bool hdr_forced     = std::getenv("KYTY_DLSS_HDR") != nullptr && *std::getenv("KYTY_DLSS_HDR") != '\0';
	static const bool hdr_setting    = Setting("KYTY_DLSS_HDR", true);
	static const float cut_distance  = [] {
        const char* text = std::getenv("KYTY_DLSS_CUT_DISTANCE");
        return text != nullptr && std::atof(text) > 0 ? static_cast<float>(std::atof(text)) : 8.0f;
	}();
	static const bool pixels_mv = [] {
		const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
		return mv != nullptr && std::string(mv) == "px";
	}();
	static const float mv_sign = [] {
		const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
		return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
	}();

	const auto count  = static_cast<int>(slots.size());
	const auto usable = [&](int i) {
		return i >= 0 && i < count && slots[i].image != nullptr && slots[i].view != nullptr &&
		       slots[i].layout != vk::ImageLayout::eUndefined;
	};
	if (!usable(depth_slot) || !usable(motion_slot)) {
		SrIssueOnce(SrIssueSlots,
		            "TAA dispatch 0x%016" PRIx64 " has no depth (slot %d) or motion vectors (slot %d) among its %d images",
		            cs_hash, depth_slot, motion_slot, count);
		return false;
	}
	const vk::Extent2D render = slots[motion_slot].extent;
	// Output: the written storage image (the largest, if several), unless KYTY_DLSS_OUTPUT_SLOT says.
	int output = output_setting;
	if (output < 0) {
		uint64_t best = 0;
		for (int i = 0; i < count; ++i) {
			const uint64_t area = uint64_t {slots[i].extent.width} * slots[i].extent.height;
			if (usable(i) && slots[i].written && slots[i].storage && i != depth_slot && i != motion_slot &&
			    area > best) {
				output = i;
				best   = area;
			}
		}
	}
	// Color input: a sampled color image of the render size (an HDR format first), unless KYTY_DLSS_COLOR_SLOT says.
	int color = color_setting;
	if (color < 0) {
		for (int i = 0; i < count; ++i) {
			const auto& slot = slots[i];
			if (!usable(i) || i == depth_slot || i == motion_slot || i == output || slot.written ||
			    IsDepthFormat(slot.format) || slot.extent != render) {
				continue;
			}
			if (color < 0 || (HdrFormat(slot.format) && !HdrFormat(slots[color].format))) color = i;
		}
		if (color < 0 && SrGuessColorSlot != output && SrGuessColorSlot != depth_slot &&
		    SrGuessColorSlot != motion_slot) {
			color = SrGuessColorSlot;
		}
	}
	// The choice, once per configuration (up to 8), with every image of the dispatch, to check against a capture.
	uint64_t signature = static_cast<uint64_t>(color + 1) | static_cast<uint64_t>(output + 1) << 8u;
	for (int i = 0; i < count; ++i) {
		signature = (signature * 1099511628211ull) ^
		            (static_cast<uint64_t>(slots[i].format) << 32u) ^
		            (static_cast<uint64_t>(slots[i].extent.width) << 16u) ^ slots[i].extent.height;
	}
	if (signature != g_sr_last.signature && g_sr_last.logged_configs < 8) {
		++g_sr_last.logged_configs;
		std::string list;
		for (int i = 0; i < count; ++i) {
			char item[192];
			std::snprintf(item, sizeof(item), "%s%d:fmt=%u view=%u %ux%u%s%s layout=%u", i == 0 ? "" : " | ", i,
			              static_cast<uint32_t>(slots[i].format), static_cast<uint32_t>(slots[i].view_format),
			              slots[i].extent.width, slots[i].extent.height, slots[i].written ? " written" : "",
			              slots[i].storage ? " storage" : "", static_cast<uint32_t>(slots[i].layout));
			list += item;
		}
		LOGF("DLSS: TAA 0x%016" PRIx64 " mode=%s render=%ux%u color=%d%s output=%d%s depth=%d motion=%d; images: %s\n",
		     cs_hash, g_sr_mode_name, render.width, render.height, color, color_setting >= 0 ? "(env)" : "(auto)",
		     output, output_setting >= 0 ? "(env)" : "(auto)", depth_slot, motion_slot, list.c_str());
		std::printf("DLSS: TAA 0x%016" PRIx64 " mode=%s render=%ux%u color=%d%s output=%d%s depth=%d motion=%d; "
		            "images: %s\n",
		            cs_hash, g_sr_mode_name, render.width, render.height, color, color_setting >= 0 ? "(env)" : "(auto)",
		            output, output_setting >= 0 ? "(env)" : "(auto)", depth_slot, motion_slot, list.c_str());
		std::fflush(stdout);
	}
	g_sr_last.signature = signature;
	if (!usable(output) || !usable(color) || output == color) {
		SrIssueOnce(SrIssueSlots,
		            "no color input (%d) or output (%d) slot found in the TAA dispatch (set KYTY_DLSS_COLOR_SLOT / "
		            "KYTY_DLSS_OUTPUT_SLOT)",
		            color, output);
		return false;
	}
	const auto& out            = slots[output];
	const bool  storage_format = static_cast<bool>(
        graphics.physical_device.getFormatProperties(out.format).optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eStorageImage);
	if (!(out.usage & vk::ImageUsageFlagBits::eStorage) || out.layout != vk::ImageLayout::eGeneral || !storage_format ||
	    out.view_format != out.format || out.base_level != 0 || out.layer_count != 1) {
		SrIssueOnce(SrIssueOutput,
		            "the output image (slot %d, format %u, view %u, layout %u, level %u, layers %u) cannot be a DLSS "
		            "storage output",
		            output, static_cast<uint32_t>(out.format), static_cast<uint32_t>(out.view_format),
		            static_cast<uint32_t>(out.layout), out.base_level, out.layer_count);
		return false;
	}
	const auto& in = slots[color];
	if (IsDepthFormat(in.format) || in.extent.width < render.width || in.extent.height < render.height ||
	    out.extent.width < render.width || out.extent.height < render.height) {
		SrIssueOnce(SrIssueColor,
		            "the color input (slot %d, %ux%u, format %u) or the output (%ux%u) does not fit the render size %ux%u",
		            color, in.extent.width, in.extent.height, static_cast<uint32_t>(in.format), out.extent.width,
		            out.extent.height, render.width, render.height);
		return false;
	}
	if (g_sr_last.valid && g_sr_last.frame == g_sr_frame) {
		SrIssueOnce(SrIssueTwice, "a second TAA dispatch in one frame");
		return false;
	}
	// desport: KYTY_DLSS_UPSCALE: the game's TAA ran already (its render-size output feeds the bloom chain); DLSS writes
	// an image of the output size learned from the draw that scales the TAA output up (SrPlanDraw), which that
	// draw samples instead. Until that size is known (the first frame), the game's own upscale stays.
	bool upscaled = false;
	if (upscale) {
		g_up.taa_address = out.address;
		g_up.taa_extent  = out.guest_extent;
		g_up.render      = render;
		if (out.address == 0 || g_up.output.width == 0) {
			UpscaleNoteOnce(1u << 4,
			                "waiting for the draw that samples the TAA output (0x%" PRIx64 ", %ux%u) into a larger target",
			                out.address, out.guest_extent.width, out.guest_extent.height);
			return false;
		}
		if (g_up.output.width < render.width || g_up.output.height < render.height) {
			UpscaleNoteOnce(1u << 5, "output %ux%u smaller than the render size %ux%u", g_up.output.width,
			                g_up.output.height, render.width, render.height);
			return false;
		}
		if (!PrepareUpscaled(graphics, command, out.format, g_up.output)) return false;
		upscaled = true;
	}

	std::lock_guard          api_lock(g_streamline_mutex);
	// desport: phase B: with DLSS-G the SR uses its viewport (0), so one slSetConstants serves both features.
	const bool               with_fg = SrWithFrameGeneration();
	const sl::ViewportHandle viewport {with_fg ? 0u : SrViewport};
	auto* const              native_command = static_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
	const bool               hdr            = hdr_forced ? hdr_setting : HdrFormat(in.format);
	const vk::Extent2D       output_extent  = upscaled ? g_up.output : out.extent; // desport: KYTY_DLSS_UPSCALE
	// Options (and the render size range) when the output size or the color's range changes.
	if (!g_sr_options.set || g_sr_options.output != output_extent || g_sr_options.hdr != hdr) {
		sl::DLSSOptions options {};
		options.mode                   = static_cast<sl::DLSSMode>(g_sr_mode);
		options.outputWidth            = output_extent.width;
		options.outputHeight           = output_extent.height;
		options.colorBuffersHDR        = hdr ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		options.useAutoExposure        = auto_exposure ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		options.dlaaPreset             = presets.dlaa;
		options.qualityPreset          = presets.quality;
		options.balancedPreset         = presets.balanced;
		options.performancePreset      = presets.performance;
		options.ultraPerformancePreset = presets.ultra_performance;
		sl::DLSSOptimalSettings optimal {};
		if (g_api.dlss_get_optimal_settings(options, optimal) == sl::Result::eOk) {
			g_sr_options.min_w = optimal.renderWidthMin;
			g_sr_options.min_h = optimal.renderHeightMin;
			g_sr_options.max_w = optimal.renderWidthMax;
			g_sr_options.max_h = optimal.renderHeightMax;
			LOGF("DLSS: mode %s, output %ux%u: optimal render %ux%u, range %ux%u..%ux%u\n", g_sr_mode_name,
			     output_extent.width, output_extent.height, optimal.optimalRenderWidth, optimal.optimalRenderHeight,
			     g_sr_options.min_w, g_sr_options.min_h, g_sr_options.max_w, g_sr_options.max_h);
			std::printf("DLSS: mode %s, output %ux%u: optimal render %ux%u, range %ux%u..%ux%u\n", g_sr_mode_name,
			            output_extent.width, output_extent.height, optimal.optimalRenderWidth,
			            optimal.optimalRenderHeight, g_sr_options.min_w, g_sr_options.min_h, g_sr_options.max_w,
			            g_sr_options.max_h);
		} else {
			g_sr_options.min_w = g_sr_options.min_h = g_sr_options.max_w = g_sr_options.max_h = 0;
		}
		if (const auto result = g_api.dlss_set_options(viewport, options); result != sl::Result::eOk) {
			SrIssueOnce(SrIssueOptions, "slDLSSSetOptions failed (%d)", static_cast<int>(result));
			return false;
		}
		LOGF("DLSS: options: mode %s, preset %s, output %ux%u, HDR color %d, auto exposure %d\n", g_sr_mode_name,
		     presets.name, output_extent.width, output_extent.height, hdr ? 1 : 0, auto_exposure ? 1 : 0);
		std::printf("DLSS: options: mode %s, preset %s, output %ux%u, HDR color %d, auto exposure %d\n",
		            g_sr_mode_name, presets.name, output_extent.width, output_extent.height, hdr ? 1 : 0,
		            auto_exposure ? 1 : 0);
		std::fflush(stdout);
		g_sr_options.set    = true;
		g_sr_options.output = output_extent;
		g_sr_options.hdr    = hdr;
	}
	if (g_sr_options.max_w != 0 && (render.width < g_sr_options.min_w || render.height < g_sr_options.min_h ||
	                                render.width > g_sr_options.max_w || render.height > g_sr_options.max_h)) {
		SrIssueOnce(SrIssueRange,
		            "render size %ux%u is outside DLSS %s's range %ux%u..%ux%u for %ux%u (DLAA needs the output's size "
		            "as render size)",
		            render.width, render.height, g_sr_mode_name, g_sr_options.min_w, g_sr_options.min_h,
		            g_sr_options.max_w, g_sr_options.max_h, output_extent.width, output_extent.height);
		return false;
	}

	// The camera: the frame's (a listed or a learned shader), else the last one held still (no jitter).
	Camera camera = g_camera;
	if (!camera.valid && g_learned_camera.valid) {
		camera = g_learned_camera;
		++g_sr_stats.learned;
	}
	if (!camera.valid && g_last_camera.valid) {
		camera                = g_last_camera;
		camera.prev_view_proj = camera.view_proj;
		camera.jitter_ndc     = {0.0f, 0.0f};
		++g_sr_stats.held;
	}
	const auto frame = camera.valid ? Decompose(camera.view_proj) : CameraFrame {};

	// History reset: the first frame, a gap (frames without the TAA: menus, loading, movies), a new size or slot
	// choice, no camera, or a cut (a move over KYTY_DLSS_CUT_DISTANCE world units, default 8, or a turn over ~37 degrees).
	bool reset = !g_sr_last.valid || g_sr_last.frame + 1 != g_sr_frame || g_sr_last.output != output_extent ||
	             g_sr_last.render != render || g_sr_last.color != color || g_sr_last.output_slot != output ||
	             !frame.valid;
	if (!reset && g_sr_last.camera.valid) {
		const auto& last = g_sr_last.camera;
		const float dx = frame.position.x - last.position.x, dy = frame.position.y - last.position.y,
		            dz = frame.position.z - last.position.z;
		const float turn =
		    frame.forward.x * last.forward.x + frame.forward.y * last.forward.y + frame.forward.z * last.forward.z;
		reset = std::sqrt(dx * dx + dy * dy + dz * dz) > cut_distance || turn < 0.8f;
	}

	// Constants as BeforePresent fills them for DLSS-G (reversed Z, the game's vectors in UV units along the motion,
	// negated), plus the jitter.
	sl::Constants constants {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	const float width                   = static_cast<float>(render.width);
	const float height                  = static_cast<float>(render.height);
	constants.cameraViewToClip          = identity;
	constants.clipToCameraView          = identity;
	constants.clipToLensClip            = identity;
	constants.clipToPrevClip            = identity;
	constants.prevClipToClip            = identity;
	constants.jitterOffset              = sl::float2(0.0f, 0.0f);
	constants.mvecScale = pixels_mv ? sl::float2(mv_sign / width, mv_sign / height) : sl::float2(mv_sign, mv_sign);
	constants.cameraPinholeOffset       = sl::float2(0.0f, 0.0f);
	constants.cameraPos                 = sl::float3(0, 0, 0);
	constants.cameraUp                  = sl::float3(0, 1, 0);
	constants.cameraRight               = sl::float3(1, 0, 0);
	constants.cameraFwd                 = sl::float3(0, 0, 1);
	constants.cameraNear                = 0.1f;
	constants.cameraFar                 = 100000.0f;
	constants.cameraFOV                 = 1.0f;
	constants.cameraAspectRatio         = width / height;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted             = sl::Boolean::eTrue;
	constants.cameraMotionIncluded      = sl::Boolean::eTrue;
	constants.motionVectors3D           = sl::Boolean::eFalse;
	constants.reset                     = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.orthographicProjection    = sl::Boolean::eFalse;
	constants.motionVectorsDilated      = sl::Boolean::eFalse;
	constants.motionVectorsJittered     = sl::Boolean::eFalse;
	if (frame.valid) {
		Mat4 projection;
		projection[0]              = sl::float4(frame.sx, 0, 0, 0);
		projection[1]              = sl::float4(0, frame.sy, 0, 0);
		projection[2]              = sl::float4(frame.cx, frame.cy, 0, 1);
		projection[3]              = sl::float4(0, 0, frame.z_near, 0);
		constants.cameraViewToClip = projection;
		Inverted(projection, constants.clipToCameraView);
		const auto current  = Transposed(camera.view_proj);
		const auto previous = Transposed(camera.prev_view_proj);
		Mat4       clip_to_world;
		if (Inverted(current, clip_to_world)) {
			constants.clipToPrevClip = Multiply(clip_to_world, previous);
			Inverted(constants.clipToPrevClip, constants.prevClipToClip);
		}
		constants.jitterOffset      = sl::float2(jitter_sign[0] * camera.jitter_ndc[0] * width * 0.5f,
		                                         jitter_sign[1] * camera.jitter_ndc[1] * height * 0.5f);
		constants.cameraPos         = frame.position;
		constants.cameraUp          = frame.up;
		constants.cameraRight       = frame.right;
		constants.cameraFwd         = frame.forward;
		constants.cameraNear        = frame.z_near;
		constants.cameraFOV         = 2.0f * std::atan(1.0f / std::fabs(frame.sy));
		constants.cameraAspectRatio = std::fabs(frame.sy / frame.sx);
	}

	// desport: phase B: the guest frame's index (PrepareInputs carries it to the present); without DLSS-G a new one.
	sl::FrameToken* token = nullptr;
	uint32_t        index = 0;
	if (with_fg) {
		if (g_frame_token_index == 0) g_frame_token_index = NextFrameIndex();
		index = g_frame_token_index;
	} else {
		index = NextFrameIndex();
	}
	if (g_api.get_new_frame_token(token, &index) != sl::Result::eOk || token == nullptr) {
		SrIssueOnce(SrIssueToken, "slGetNewFrameToken failed (frame %u, viewport %u)", index,
		            static_cast<uint32_t>(viewport));
		return false;
	}
	if (const auto result = g_api.set_constants(constants, *token, viewport); result != sl::Result::eOk) {
		SrIssueOnce(SrIssueEvaluate, "slSetConstants failed (%d, frame %u, viewport %u)", static_cast<int>(result), index,
		            static_cast<uint32_t>(viewport));
		return false;
	}
	if (with_fg) g_frame_constants_set = true; // desport: the present skips its own slSetConstants for this index
	auto             depth_resource  = SrResource(slots[depth_slot]);
	auto             motion_resource = SrResource(slots[motion_slot]);
	auto             color_resource  = SrResource(in);
	auto             output_resource = upscaled ? UpscaledResource() : SrResource(out); // desport: KYTY_DLSS_UPSCALE
	const sl::Extent render_rect {0, 0, render.width, render.height};
	const sl::Extent output_rect {0, 0, output_extent.width, output_extent.height};
	const sl::ResourceTag tags[] {
	    sl::ResourceTag(&depth_resource, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &render_rect),
	    sl::ResourceTag(&motion_resource, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &render_rect),
	    sl::ResourceTag(&color_resource, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &render_rect),
	    sl::ResourceTag(&output_resource, sl::kBufferTypeScalingOutputColor,
	                    sl::ResourceLifecycle::eValidUntilEvaluate, &output_rect),
	};
	if (const auto result = g_api.set_tag_for_frame(*token, viewport, tags, 4u, native_command);
	    result != sl::Result::eOk) {
		SrIssueOnce(SrIssueEvaluate, "slSetTagForFrame failed (%d)", static_cast<int>(result));
		return false;
	}
	if (!g_sr_options.allocated) {
		g_sr_options.allocated = true; // (a failure leaves the allocation to the first evaluate)
		if (const auto result = g_api.allocate_resources(native_command, sl::kFeatureDLSS, viewport);
		    result != sl::Result::eOk) {
			LOGF("DLSS: slAllocateResources failed (%d)\n", static_cast<int>(result));
		}
	}
	const sl::BaseStructure* inputs[] {&viewport};
	if (const auto result = g_api.evaluate_feature(sl::kFeatureDLSS, *token, inputs, 1u, native_command);
	    result != sl::Result::eOk) {
		SrIssueOnce(SrIssueEvaluate, "slEvaluateFeature failed (%d)", static_cast<int>(result));
		return false;
	}
	g_sr_last.valid       = true;
	g_sr_last.frame       = g_sr_frame;
	g_sr_last.output      = output_extent;
	g_sr_last.render      = render;
	g_sr_last.color       = color;
	g_sr_last.output_slot = output;
	g_sr_last.camera      = frame;
	g_up.ready            = upscaled; // desport: KYTY_DLSS_UPSCALE
	++g_sr_stats.evaluated;
	g_sr_stats.resets += reset ? 1u : 0u;
	if (g_sr_stats.evaluated % 1800u == 1u) {
		LOGF("DLSS: %u frames upscaled (%u history resets, %u learned cameras, %u held cameras), %ux%u -> %ux%u%s "
		     "(composite samples %u)\n",
		     g_sr_stats.evaluated, g_sr_stats.resets, g_sr_stats.learned, g_sr_stats.held, render.width,
		     render.height, output_extent.width, output_extent.height, upscaled ? " upscale" : "",
		     g_up.substituted); // desport: KYTY_DLSS_UPSCALE
		std::printf("DLSS: %u frames upscaled (%u history resets, %u learned cameras, %u held cameras), %ux%u -> "
		            "%ux%u%s (composite samples %u)\n",
		            g_sr_stats.evaluated, g_sr_stats.resets, g_sr_stats.learned, g_sr_stats.held, render.width,
		            render.height, output_extent.width, output_extent.height, upscaled ? " upscale" : "",
		            g_up.substituted); // desport: KYTY_DLSS_UPSCALE
		std::fflush(stdout);
	}
	return true;
#else
	(void)graphics;
	(void)command;
	(void)cs_hash;
	(void)slots;
	(void)upscale; // desport
	return false;
#endif
}

// desport: KYTY_DLSS_UPSCALE, see frame-gen.h.
// Mode 1: a draw that samples this frame's TAA output into a larger color target samples the DLSS image instead; its
// target's size is DLSS's output size (the largest seen). (In 01.007.000 no draw does: the composite runs at the render
// size, so mode 1 only renders lower and stretches.)
// Mode 2: the draw that samples the TAA output into a target of the TAA output's size is the scene composite: it is
// recorded again into an image of the output size, reading the DLSS image, and the draw that samples the composite
// target into a larger color target (the final pass) samples that image instead. The output size is the largest
// texture the final pass samples (the UI layer), at most its target. Until it is known, the game's own upscale stays.
SrDrawPlan SrPlanDraw(GraphicContext& graphics, uint64_t ps_hash, uint64_t target_address, vk::Extent2D target,
                      vk::Format target_format, bool replayable, std::span<const SrDrawTexture> textures) {
	SrDrawPlan plan;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// The 01.007.000 composite and final pass, also before the chain is known (a menu without the TAA must not store
	// them as table draws for later).
	constexpr uint64_t CompositePs = 0x95318d27a44b0dd5ull, FinalPs = 0x37db6bd367380e9cull;
	plan.involved = g_sr_upscale != 0 && (ps_hash == CompositePs || ps_hash == FinalPs);
	if (g_sr_upscale == 0 || g_up.frame != g_sr_frame) return plan;
	const uint64_t filter = UpscalePsFilter();
	int            taa = -1, composite = -1;
	for (size_t i = 0; i < textures.size(); ++i) {
		const auto& texture = textures[i];
		if (texture.address == 0) continue;
		if (texture.address == g_up.taa_address && texture.extent == g_up.taa_extent) taa = static_cast<int>(i);
		if (g_sr_upscale == 2 && texture.address == g_up.composite_address && texture.extent == g_up.composite_extent &&
		    g_up.composite_frame == g_sr_frame) {
			composite = static_cast<int>(i);
		}
	}
	if (taa >= 0 || composite >= 0) plan.involved = true;
	// The final target's writers (mode 2), once it is known.
	if (g_sr_upscale == 2 && target_address != 0 && target_address == g_up.final_address && g_up.logged_writers < 8) {
		const uint64_t writer = ps_hash ^ (uint64_t {target.width} << 32u) ^ (uint64_t {target.height} << 48u);
		if (writer != g_up.last_writer) {
			g_up.last_writer = writer;
			++g_up.logged_writers;
			LOGF("DLSS upscale: final target 0x%" PRIx64 " %ux%u written by ps 0x%016" PRIx64 " (%zu textures)\n",
			     target_address, target.width, target.height, ps_hash, textures.size());
			std::printf("DLSS upscale: final target 0x%" PRIx64 " %ux%u written by ps 0x%016" PRIx64 " (%zu textures)\n",
			            target_address, target.width, target.height, ps_hash, textures.size());
			std::fflush(stdout);
		}
	}
	if (taa >= 0 && g_sr_upscale == 1) {
		const char* verdict = "substituted";
		if (!LargerExtent(target, g_up.taa_extent)) {
			verdict = "kept (target not larger than the TAA output)";
		} else if (filter != 0 && ps_hash != filter) {
			verdict = "kept (not KYTY_DLSS_UPSCALE_PS)";
		} else if (g_up.output != target) {
			if (g_up.output.width == 0 ||
			    uint64_t {target.width} * target.height > uint64_t {g_up.output.width} * g_up.output.height) {
				g_up.output = target;
				verdict     = "kept, output size learned (DLSS writes it from the next frame)";
			} else {
				verdict = "kept (smaller than the learned output size)";
			}
		} else if (g_up.ready && g_up.dlss.current != nullptr) {
			plan.swap_slot    = taa;
			plan.swap_view    = g_up.dlss.view;
			g_up.composite_ps = ps_hash;
			++g_up.substituted;
			UpscaleAnnounce("composite_ps", ps_hash);
		} else {
			verdict = "kept (DLSS did not run for this TAA output)";
		}
		UpscaleReader("TAA output", g_up.taa_address, g_up.taa_extent, ps_hash, target, verdict);
		return plan;
	}
	if (taa >= 0 && g_sr_upscale == 2) {
		const char* verdict = "kept (target not the TAA output's size)";
		if (target == g_up.taa_extent && target_address != 0 && (filter == 0 || ps_hash == filter)) {
			// The scene composite at the render size.
			g_up.composite_address = target_address;
			g_up.composite_extent  = target;
			g_up.composite_frame   = g_sr_frame;
			g_up.composite_ps      = ps_hash;
			if (!replayable) {
				verdict = "composite seen, not replayable (needs one color target, no depth, a plain draw)";
			} else if (g_up.output.width == 0) {
				verdict = "composite seen, waiting for the final pass to learn the output size";
			} else if (!g_up.ready || g_up.dlss.current == nullptr) {
				verdict = "composite seen, DLSS did not run for this TAA output";
			} else if (!LargerExtent(g_up.output, target)) {
				verdict = "composite seen, output not larger than the render size";
			} else if (!EnsureOwned(graphics, g_up.composite, target_format, g_up.output,
			                        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
			                            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
			                        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage,
			                        "composite", 1u << 8)) {
				verdict = "composite seen, no output-size image";
			} else {
				plan.replay         = true;
				plan.replay_slot    = taa;
				plan.replay_texture = g_up.dlss.view;
				plan.replay_view    = g_up.composite.view;
				plan.replay_extent  = g_up.output;
				verdict             = "composite, recorded again at the output size from the DLSS image";
			}
		} else if (target == g_up.taa_extent) {
			verdict = "kept (not KYTY_DLSS_UPSCALE_PS)";
		}
		UpscaleReader("TAA output", g_up.taa_address, g_up.taa_extent, ps_hash, target, verdict);
	}
	if (composite >= 0) {
		const char* verdict = "kept (target not larger than the composite)";
		if (LargerExtent(target, g_up.composite_extent)) {
			// The final pass: the output size is its largest texture (the UI layer), at most its target.
			vk::Extent2D output = g_up.composite_extent;
			for (const auto& texture: textures) {
				if (uint64_t {texture.extent.width} * texture.extent.height > uint64_t {output.width} * output.height)
					output = texture.extent;
			}
			output.width  = std::min(output.width, target.width);
			output.height = std::min(output.height, target.height);
			g_up.final_address = target_address;
			g_up.final_ps      = ps_hash;
			if (!LargerExtent(output, g_up.composite_extent)) {
				verdict = "kept (no texture larger than the composite: output = render size)";
			} else if (g_up.output != output) {
				g_up.output = output;
				verdict     = "kept, output size learned (DLSS and the composite copy from the next frame)";
			} else if (g_up.composite_ready_frame == g_sr_frame && g_up.composite.current != nullptr &&
			           g_up.composite.current->extent.width == output.width &&
			           g_up.composite.current->extent.height == output.height) {
				plan.swap_slot = composite;
				plan.swap_view = g_up.composite.view;
				++g_up.substituted;
				verdict = "substituted by the output-size composite";
				UpscaleAnnounce("final_ps", ps_hash);
			} else {
				verdict = "kept (no output-size composite this frame)";
			}
		}
		UpscaleReader("composite", g_up.composite_address, g_up.composite_extent, ps_hash, target, verdict);
	}
#else
	(void)graphics; // desport
	(void)ps_hash;
	(void)target_address;
	(void)target;
	(void)target_format;
	(void)replayable;
	(void)textures;
#endif
	return plan;
}

// desport: KYTY_DLSS_UPSCALE=2, around the composite's second recording (outside any rendering): the output-size image
// to General, its previous readers (last frame's final pass) done before it is written; then its writes visible to
// the final pass.
void SrCompositeBegin(vk::CommandBuffer command) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	auto* image = g_up.composite.current;
	if (image == nullptr) return;
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentWrite;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
	barrier.oldLayout           = image->state.layout == vk::ImageLayout::eGeneral ? vk::ImageLayout::eGeneral
	                                                                                : vk::ImageLayout::eUndefined;
	barrier.newLayout           = vk::ImageLayout::eGeneral;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image->image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image->state = {vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
	                vk::ImageLayout::eGeneral};
#else
	(void)command; // desport
#endif
}

void SrCompositeEnd(vk::CommandBuffer command) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	auto* image = g_up.composite.current;
	if (image == nullptr) return;
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eColorAttachmentWrite;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
	barrier.oldLayout           = vk::ImageLayout::eGeneral;
	barrier.newLayout           = vk::ImageLayout::eGeneral;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image->image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image->state               = {vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
	                              vk::ImageLayout::eGeneral};
	g_up.composite_ready_frame = g_sr_frame;
	if (++g_up.replays % 1800u == 1u) {
		LOGF("DLSS upscale: %u composites at %ux%u (final pass samples %u)\n", g_up.replays, image->extent.width,
		     image->extent.height, g_up.substituted);
		std::printf("DLSS upscale: %u composites at %ux%u (final pass samples %u)\n", g_up.replays, image->extent.width,
		            image->extent.height, g_up.substituted);
		std::fflush(stdout);
	}
#else
	(void)command; // desport
#endif
}

uint32_t SrUpscaleMode() { // desport
	return SrActive() ? g_sr_upscale : 0u;
}
// desport: end DLSS SR/DLAA

} // namespace Libs::Graphics::FrameGen
