#include <shs/core/context.hpp>
#include <shs/passes/rt_types.hpp>
#include <shs/passes/pass_common.hpp>

int main()
{
    shs::Context   ctx{};
    shs::DefaultRT rt{ 800, 600, 0.1f, 1000.0f, shs::Color{0,0,0,255} };

    shs::PassContext pc{};
    pc.ctx = &ctx;
    pc.rt  = &rt;

    return 0;
}
