#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "loader/desport.h" // desport: DES_TEX

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

// desport: begin texture filtering overrides (desport-tex). All off by default (the game's S# as before).
// KYTY_TEX_ANISO=<2..16|max>: samplers that minify with a linear filter and have mips (the game's materials) get at
//   least this anisotropy (clamped to the device's maxSamplerAnisotropy); the game's own ratio when it is higher.
//   Point-filtered, unnormalized and depth-compare samplers are left alone.
// KYTY_TEX_LOD_BIAS=<float>: added to the LOD bias of every sampler with mips (negative: sharper).
// KYTY_TEX_DLSS_BIAS=auto|<float>: with the DLSS upscale (KYTY_DLSS_UPSCALE) the scene renders below the output size;
//   a pixel shader's material sampler (anisotropic, or trilinear: mip filter linear) gets an extra bias, auto:
//   log2(render width / output width) as FrameGen measured it (quality -0.58, performance -1), as NVIDIA asks for;
//   a value: that bias whenever the DLSS SR is active. UI and post-processing samplers (bilinear, mip point or none, compute
//   shaders) keep theirs.
// KYTY_TEX_LOG=1: a DES_TEX_SAMPLER line per new sampler (the first 256) and a census every 64 samplers.
namespace {

struct TexOverrides {
	float aniso          = 0.0f; // 0: off; else the minimum ratio
	bool  aniso_max      = false;
	float lod_bias       = 0.0f;
	int   dlss_bias_mode = 0; // 0: off, 1: auto, 2: fixed
	float dlss_bias      = 0.0f;
	bool  log            = false;
};

bool ParseFloat(const char* text, float& value) {
	if (text == nullptr || *text == '\0') return false;
	char*        end    = nullptr;
	const double parsed = std::strtod(text, &end);
	if (end == text || *end != '\0' || !std::isfinite(parsed)) return false;
	value = static_cast<float>(parsed);
	return true;
}

const TexOverrides& Overrides() {
	static const TexOverrides overrides = [] {
		TexOverrides o;
		if (const char* text = std::getenv("KYTY_TEX_ANISO"); text != nullptr && *text != '\0') {
			float value = 0.0f;
			if (std::strcmp(text, "max") == 0) {
				o.aniso_max = true;
				o.aniso     = 16.0f;
			} else if (ParseFloat(text, value) && value > 1.0f) {
				o.aniso = std::min(value, 16.0f);
			}
		}
		float value = 0.0f;
		if (ParseFloat(std::getenv("KYTY_TEX_LOD_BIAS"), value)) o.lod_bias = std::clamp(value, -4.0f, 4.0f);
		if (const char* text = std::getenv("KYTY_TEX_DLSS_BIAS"); text != nullptr && *text != '\0') {
			if (std::strcmp(text, "auto") == 0) {
				o.dlss_bias_mode = 1;
			} else if (ParseFloat(text, value) && value != 0.0f) {
				o.dlss_bias_mode = 2;
				o.dlss_bias      = std::clamp(value, -4.0f, 0.0f);
			}
		}
		const char* log_text = std::getenv("KYTY_TEX_LOG");
		o.log           = log_text != nullptr && *log_text != '\0' && *log_text != '0';
		char line[256];
		std::snprintf(line, sizeof(line), "DES_TEX aniso=%s%.0f lod_bias=%.2f dlss_bias=%s%.2f log=%d\n",
		              o.aniso_max ? "max:" : "", o.aniso, o.lod_bias,
		              o.dlss_bias_mode == 1 ? "auto" : (o.dlss_bias_mode == 2 ? "fixed:" : "off:"), o.dlss_bias,
		              o.log ? 1 : 0);
		Desport::EmitLine(line);
		return o;
	}();
	return overrides;
}

// Census (KYTY_TEX_LOG): what the game's samplers ask for.
struct TexCensus {
	std::atomic<uint32_t> total {0}, aniso {0}, aniso_ratio[8] {}, mip_none {0}, mip_point {0}, mip_linear {0},
	    bias_negative {0}, bias_positive {0}, perf_mip {0}, aniso_threshold {0}, aniso_bias {0}, min_lod {0},
	    forced_aniso {0}, scene_biased {0};
};
TexCensus g_census;

} // namespace

bool SamplerCache::SceneBiasEnabled() {
	return Overrides().dlss_bias_mode != 0;
}

int32_t SamplerCache::SceneBias256(int32_t dlss_bias_256) {
	const auto& o = Overrides();
	if (o.dlss_bias_mode == 1) return std::clamp(dlss_bias_256, -1024, 0);
	if (o.dlss_bias_mode == 2) return static_cast<int32_t>(std::lround(o.dlss_bias * 256.0f));
	return 0;
}
// desport: end texture filtering overrides

SamplerCache::~SamplerCache() {
	for (const auto& [key, sampler]: m_samplers) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, int32_t scene_bias_256) {
	// desport: KYTY_TEX_DLSS_BIAS only for material samplers (anisotropic, or trilinear with mips): the key keeps one
	// sampler for the others.
	if (scene_bias_256 != 0) {
		const auto filter_of = [](uint8_t filter) { return static_cast<Prospero::SamplerFilter>(filter); };
		const bool aniso_filter =
		    filter_of(r.XyMinFilter()) == Prospero::SamplerFilter::kAnisoLinear ||
		    filter_of(r.XyMagFilter()) == Prospero::SamplerFilter::kAnisoLinear;
		const bool trilinear = static_cast<Prospero::SamplerMipFilter>(r.MipFilter()) ==
		                       Prospero::SamplerMipFilter::kLinear;
		const bool mips = static_cast<Prospero::SamplerMipFilter>(r.MipFilter()) != Prospero::SamplerMipFilter::kNone;
		if (!mips || !(aniso_filter || trilinear) || r.ForceUnormCoords() || r.DepthCompareFunc() != 0) {
			scene_bias_256 = 0;
		}
	}
	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3],
	                      static_cast<uint32_t>(scene_bias_256)}; // desport: + the scene bias
	// Samplers live as long as the cache: a thread's recent lookups need neither the lock nor
	// the map.
	struct Recent {
		const SamplerCache* owner = nullptr;
		SamplerKey          key {};
		vk::Sampler         sampler;
	};
	thread_local std::array<Recent, 64> recent {};
	auto& slot = recent[SamplerKeyHash {}(key) % recent.size()];
	if (slot.owner == this && slot.key == key) return slot.sampler;

	Common::LockGuard lock(m_mutex);
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		slot = {this, key, iter->second};
		return iter->second;
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = vk::BorderColor::eIntTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack:
			border = vk::BorderColor::eIntTransparentBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = vk::BorderColor::eIntOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = vk::BorderColor::eIntOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			border = vk::BorderColor::eIntTransparentBlack;
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
    const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
    sampler_info.mipLodBias = std::clamp(
        static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
        -max_lod_bias, max_lod_bias
    );
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	// desport: begin texture filtering overrides (KYTY_TEX_ANISO / KYTY_TEX_LOD_BIAS / KYTY_TEX_DLSS_BIAS)
	const auto& overrides  = Overrides();
	const bool  has_mips   = static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone;
	const bool  plain_view = !r.ForceUnormCoords() && sampler_info.compareEnable == VK_FALSE;
	bool        forced     = false;
	if (overrides.aniso > 1.0f && has_mips && plain_view && sampler_info.minFilter == vk::Filter::eLinear) {
		const float device_max = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
		const float wanted     = std::min(overrides.aniso_max ? device_max : overrides.aniso, device_max);
		if (wanted > 1.0f && (sampler_info.anisotropyEnable == VK_FALSE || sampler_info.maxAnisotropy < wanted)) {
			sampler_info.anisotropyEnable = VK_TRUE;
			sampler_info.maxAnisotropy    = wanted;
			forced                        = true;
		}
	}
	if (has_mips && !r.ForceUnormCoords() && (overrides.lod_bias != 0.0f || scene_bias_256 != 0)) {
		sampler_info.mipLodBias =
		    std::clamp(sampler_info.mipLodBias + overrides.lod_bias + static_cast<float>(scene_bias_256) / 256.0f,
		               -max_lod_bias, max_lod_bias);
	}
	if (overrides.log) {
		auto&          census = g_census;
		const uint32_t number = census.total.fetch_add(1, std::memory_order_relaxed) + 1;
		if (aniso) {
			census.aniso.fetch_add(1, std::memory_order_relaxed);
			census.aniso_ratio[r.MaxAnisoRatio() & 7u].fetch_add(1, std::memory_order_relaxed);
		}
		const auto mip_kind = static_cast<Prospero::SamplerMipFilter>(mip_filter);
		(mip_kind == Prospero::SamplerMipFilter::kNone
		     ? census.mip_none
		     : (mip_kind == Prospero::SamplerMipFilter::kLinear ? census.mip_linear : census.mip_point))
		    .fetch_add(1, std::memory_order_relaxed);
		const auto game_bias = static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u);
		if (game_bias < 0) census.bias_negative.fetch_add(1, std::memory_order_relaxed);
		if (game_bias > 0) census.bias_positive.fetch_add(1, std::memory_order_relaxed);
		if (r.PerfMip() != 0) census.perf_mip.fetch_add(1, std::memory_order_relaxed);
		if (r.AnisoThreshold() != 0) census.aniso_threshold.fetch_add(1, std::memory_order_relaxed);
		if (r.AnisoBias() != 0) census.aniso_bias.fetch_add(1, std::memory_order_relaxed);
		if (r.MinLod() != 0 && has_mips) census.min_lod.fetch_add(1, std::memory_order_relaxed);
		if (forced) census.forced_aniso.fetch_add(1, std::memory_order_relaxed);
		if (scene_bias_256 != 0) census.scene_biased.fetch_add(1, std::memory_order_relaxed);
		char line[512];
		if (number <= 256) {
			std::snprintf(line, sizeof(line),
			              "DES_TEX_SAMPLER n=%u filter=%u/%u mip=%u aniso=%u thr=%u abias=%u bias=%.3f min_lod=%.2f "
			              "max_lod=%.2f perf_mip=%u perf_z=%u unnorm=%d cmp=%u -> vk aniso=%d/%.0f bias=%.3f%s%s "
			              "s#=%08x,%08x,%08x,%08x\n",
			              number, r.XyMinFilter(), r.XyMagFilter(), mip_filter, r.MaxAnisoRatio(), r.AnisoThreshold(),
			              r.AnisoBias(), static_cast<float>(game_bias) / 256.0f, sampler_info.minLod, sampler_info.maxLod,
			              r.PerfMip(), r.PerfZ(), r.ForceUnormCoords() ? 1 : 0, r.DepthCompareFunc(),
			              sampler_info.anisotropyEnable == VK_TRUE ? 1 : 0, sampler_info.maxAnisotropy,
			              sampler_info.mipLodBias, forced ? " forced" : "", scene_bias_256 != 0 ? " scene" : "",
			              r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
			Desport::EmitLine(line);
		}
		if (number % 64u == 0) {
			std::snprintf(line, sizeof(line),
			              "DES_TEX census samplers=%u aniso=%u ratio[1,2,4,8,16]=%u,%u,%u,%u,%u mip none/point/linear=%u/%u/%u "
			              "game_bias -/+=%u/%u perf_mip=%u aniso_thr=%u aniso_bias=%u min_lod=%u forced_aniso=%u "
			              "scene_biased=%u\n",
			              number, census.aniso.load(), census.aniso_ratio[0].load(), census.aniso_ratio[1].load(),
			              census.aniso_ratio[2].load(), census.aniso_ratio[3].load(), census.aniso_ratio[4].load(),
			              census.mip_none.load(), census.mip_point.load(), census.mip_linear.load(),
			              census.bias_negative.load(), census.bias_positive.load(), census.perf_mip.load(),
			              census.aniso_threshold.load(), census.aniso_bias.load(), census.min_lod.load(),
			              census.forced_aniso.load(), census.scene_biased.load());
			Desport::EmitLine(line);
		}
	}
	// desport: end texture filtering overrides

	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&sampler_info, nullptr, &vk_sampler);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || vk_sampler == nullptr);

	m_samplers.emplace(key, vk_sampler);
	slot = {this, key, vk_sampler};
	return vk_sampler;
}

} // namespace Libs::Graphics
