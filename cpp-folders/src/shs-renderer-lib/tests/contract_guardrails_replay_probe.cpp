#include <cstdio>
#include <cstring>
#include <memory_resource>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/render/frame/frame.gateway.hpp"

// Replay-parity probe (C1.4): a fixed recorded command span through the frame
// gateway, printed as a stable digest. The parity CTest compiles this file
// twice — SHS_CONTRACTS_ENFORCED vs release-assume — and asserts byte-identical
// stdout, proving the bridge never gates control flow (Constitution II Rule
// 4.1). Same compiler, same flags; the only difference in the pair is the
// contract mode. The test stays meaningful after the C++26 switch (C4.3).
namespace
{
    struct span_digest
    {
        long long accumulator = 0;
        float exposure = 0.0f;
    };

    span_digest run_recorded_span()
    {
        shs::FrameParams state{};
        state.w = 640;
        state.h = 480;
        state.exposure = 1.25f;
        state.technique.mode = shs::TechniqueMode::Deferred;

        const std::vector<shs::frame::FrameCommand> commands(3);
        const shs::frame::FrameContext context{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::frame::FrameEvent> events{&arena};

        span_digest result{};
        for (int frame = 0; frame < 16; ++frame)
        {
            SHS_CONTRACT_ASSERT(commands.size() == 3);
            SHS_PRE(state.w == 640 && state.h == 480);
            const auto step = shs::frame::frame_gateway(state,
                std::span<const shs::frame::FrameCommand>{commands}, context, events);
            SHS_POST(step.commands_observed == commands.size());
            result.accumulator += static_cast<long long>(step.commands_observed) * 31
                + static_cast<long long>(events.size());
            events.clear();
        }
        result.exposure = state.exposure;
        return result;
    }
} // namespace

int main()
{
    const span_digest result = run_recorded_span();
    unsigned exposure_bits = 0;
    std::memcpy(&exposure_bits, &result.exposure, sizeof(exposure_bits));
    std::printf("replay-digest %lld %08x\n", result.accumulator, exposure_bits);
    return 0;
}
