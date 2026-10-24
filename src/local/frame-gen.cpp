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
uint64_t         g_sr_frame    = 0;        // desport: Kyty.Gpu, guest frames (EndSrFrame / PrepareInputs)

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
	return g_generate > 0 || g_fg_runtime_toggle;
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
		// desport: phase A: the SR and DLSS-G would share Streamline's frame tokens from two threads.
		if (g_sr_mode != 0 && (requested_frames > 0 || g_fg_runtime_toggle)) {
			LOGF("DLSS: KYTY_DLSS=%s with frame generation is not supported yet; DLSS SR off\n", dlss);
			std::printf("DLSS: KYTY_DLSS=%s with frame generation is not supported yet; DLSS SR off\n", dlss);
			g_sr_mode = 0;
		}
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
		g_generate = 0;
		g_sr_mode  = 0; // desport: no Streamline, no DLSS SR
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
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
	if (g_generate == 0 && !g_fg_runtime_toggle && g_sr_mode == 0) { // desport: DLSS SR too
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
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
		reflex.mode = sl::ReflexMode::eOff;
	} else if (reflex_env != nullptr && std::strcmp(reflex_env, "boost") == 0) {
		reflex.mode = sl::ReflexMode::eLowLatencyWithBoost;
	}
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
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
                   vk::Rect2D region, vk::Format format, uint32_t image_count, bool reusing_last) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	std::lock_guard api_lock(g_streamline_mutex);
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
	++g_frame_index;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	if (g_reflex_sleep && g_api.reflex_sleep != nullptr) {
		const auto sleep_begin = std::chrono::steady_clock::now();
		g_api.reflex_sleep(token);
		const double sleep_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sleep_begin).count();
		g_sleep_ms_sum += sleep_ms;
		g_sleep_ms_max = std::max(g_sleep_ms_max, sleep_ms);
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
	if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk &&
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
#endif
}

void AfterPresent() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || g_token == nullptr) {
		return;
	}
	std::lock_guard api_lock(g_streamline_mutex);
	g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
	const auto presents = ++g_presents;
	if (presents % 300 == 0) {
		const auto& stats = g_present_tag_stats;
		LOGF("frame generation: present tags last %u presents: tagged=%u no_inputs=%u reused=%u "
		     "tag_failures=%u last_result=%d guest_frame=%" PRIu64 "..%" PRIu64 "\n",
		     stats.presents, stats.tagged_inputs, stats.no_inputs, stats.reused_frames,
		     stats.tag_failures, stats.last_tag_result, stats.first_guest_frame, stats.last_guest_frame);
		std::printf("Frame generation: present tags last %u presents: tagged=%u no_inputs=%u reused=%u "
		            "tag_failures=%u last_result=%d missing_tag_errors=%u mode=%s guest_frame=%" PRIu64 "..%" PRIu64 "\n",
		            stats.presents, stats.tagged_inputs, stats.no_inputs, stats.reused_frames,
		            stats.tag_failures, stats.last_tag_result,
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
	if (!Enabled()) {
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
uint32_t  g_sr_token_index = 0;

} // namespace
#endif

bool EvaluateSr(GraphicContext& graphics, vk::CommandBuffer command, uint64_t cs_hash, std::span<const SrSlot> slots) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!SrActive() || !IsTaaShader(cs_hash)) {
		return false;
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

	std::lock_guard          api_lock(g_streamline_mutex);
	const sl::ViewportHandle viewport {SrViewport};
	auto* const              native_command = static_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
	const bool               hdr            = hdr_forced ? hdr_setting : HdrFormat(in.format);
	const vk::Extent2D       output_extent  = out.extent;
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

	sl::FrameToken* token = nullptr;
	const uint32_t  index = ++g_sr_token_index;
	if (g_api.get_new_frame_token(token, &index) != sl::Result::eOk || token == nullptr) {
		SrIssueOnce(SrIssueToken, "slGetNewFrameToken failed");
		return false;
	}
	if (const auto result = g_api.set_constants(constants, *token, viewport); result != sl::Result::eOk) {
		SrIssueOnce(SrIssueEvaluate, "slSetConstants failed (%d)", static_cast<int>(result));
		return false;
	}
	auto             depth_resource  = SrResource(slots[depth_slot]);
	auto             motion_resource = SrResource(slots[motion_slot]);
	auto             color_resource  = SrResource(in);
	auto             output_resource = SrResource(out);
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
	++g_sr_stats.evaluated;
	g_sr_stats.resets += reset ? 1u : 0u;
	if (g_sr_stats.evaluated % 1800u == 1u) {
		LOGF("DLSS: %u frames upscaled (%u history resets, %u learned cameras, %u held cameras), %ux%u -> %ux%u\n",
		     g_sr_stats.evaluated, g_sr_stats.resets, g_sr_stats.learned, g_sr_stats.held, render.width,
		     render.height, output_extent.width, output_extent.height);
		std::printf("DLSS: %u frames upscaled (%u history resets, %u learned cameras, %u held cameras), %ux%u -> "
		            "%ux%u\n",
		            g_sr_stats.evaluated, g_sr_stats.resets, g_sr_stats.learned, g_sr_stats.held, render.width,
		            render.height, output_extent.width, output_extent.height);
		std::fflush(stdout);
	}
	return true;
#else
	(void)graphics;
	(void)command;
	(void)cs_hash;
	(void)slots;
	return false;
#endif
}
// desport: end DLSS SR/DLAA

} // namespace Libs::Graphics::FrameGen
