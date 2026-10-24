#pragma once

#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

struct EmbeddedFetchLoad {
	uint32_t              pc         = 0;
	int                   attrib_id  = -1;
	uint32_t              components = 0;
	std::vector<uint32_t> prolog_loads;
};

struct EmbeddedFetchPlan {
	std::vector<EmbeddedFetchLoad> loads;
	int32_t                        vertex_offset_sgpr   = -1;
	int32_t                        instance_offset_sgpr = -1;
};

struct TranslateOptions {
	ShaderType                    stage               = ShaderType::Unknown;
	uint32_t                      wave_size           = 64;
	uint64_t                      shader_hash         = 0;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string_view              fallback_reason;
	const ShaderVertexInputInfo*  vertex         = nullptr;
	const ShaderPixelInputInfo*   pixel          = nullptr;
	const ShaderComputeInputInfo* compute        = nullptr;
	const EmbeddedFetchPlan*      embedded_fetch = nullptr;
};

IR::Program TranslateProgram(const Decoder::Program& decoded, const CFG::Graph& cfg,
                             const TranslateOptions& options);

// A wave64 program whose results in each lane depend on no lane outside its 32-lane half (LaneLocal.cpp): on a host
// with 32-wide subgroups it runs one GCN lane per invocation. `reason`: why not.
bool IsWaveLaneLocal(const Decoder::Program& program, std::string* reason = nullptr);

// desport: KYTY_WAVE64_MODE. A wave64 compute program whose workgroup is one wave (64 threads) the proof above refuses
// may still run one GCN lane per invocation with the wave's two halves (its two subgroups, the whole workgroup) in
// lockstep: every branch on EXEC or VCC a vote of both halves, V_READLANE and (where a half may have no active lane)
// V_READFIRSTLANE through workgroup memory between barriers, and a workgroup barrier where an LDS access may meet
// another half's (the wave's lanes exchange data there without one).
struct WaveLaneLocalOptions {
	uint32_t mode          = 0; // 0: off; 1: lockstep halves, LDS ordered strictly; 2: also writes after writes
	                            // unordered and DS_*_ADDTID lane-private
	uint32_t group_threads = 0; // threads in the workgroup
};
struct WaveLaneLocalPlan {
	bool                  wave_uniform = false; // the lockstep halves (only with WaveLaneLocalOptions::mode)
	std::vector<uint32_t> barrier_pcs;          // a workgroup barrier before these LDS accesses (sorted)
	std::vector<uint32_t> exchange_pcs;         // V_READFIRSTLANE through workgroup memory (sorted)
};
bool IsWaveLaneLocal(const Decoder::Program& program, const WaveLaneLocalOptions& options, std::string* reason,
                     WaveLaneLocalPlan* plan);

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
