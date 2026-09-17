#include <cstdio>
#include <cstring>
#include <string>

#include "shs/core/contract_guardrails.hpp"

// C1.2: bridge violation-handler tests. Compiled twice by CMake:
// - shs_renderer_contract_guardrails_tests        (SHS_CONTRACTS_ENFORCED:
//   violations reach the handler with the correct kind/expression/location)
// - shs_renderer_contract_guardrails_release_tests (no define: the assume path
//   compiles and valid conditions pass through; no check runs — verified by
//   compilation, not by running)
namespace
{
    int guarded_pre(int x) { SHS_PRE(x > 0); return x * 2; }

    int guarded_post(int x)
    {
        SHS_POST(x < 100);
        return x * 2;
    }

    int guarded_assert(int x)
    {
        SHS_CONTRACT_ASSERT(x != 42);
        return x;
    }

#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::contract_kind g_kind{};
    const char* g_expr = nullptr;
    const char* g_file = nullptr;
    int g_line = 0;
    int g_calls = 0;

    void capture_handler(shs::core::contract_kind kind, const char* expr, const char* file, int line) noexcept
    {
        ++g_calls;
        g_kind = kind;
        g_expr = expr;
        g_file = file;
        g_line = line;
    }

    bool ends_with(const char* haystack, const char* needle)
    {
        const std::size_t n = std::strlen(needle);
        const std::string h{haystack};
        return h.size() >= n && h.compare(h.size() - n, n, needle) == 0;
    }

    // A violating site must report the exact kind, expression text, and
    // location (C1.2 DoD) — and nothing else observable changes.
    bool test_violation_reports_kind_expression_location()
    {
        shs::core::set_contract_violation_handler(&capture_handler);

        (void)guarded_pre(-1);
        const bool pre_ok = g_calls == 1
            && g_kind == shs::contract_kind::pre
            && std::string{g_expr} == "x > 0"
            && ends_with(g_file, "contract_guardrails_tests.cpp")
            && g_line > 0;

        (void)guarded_post(100);
        const bool post_ok = g_calls == 2
            && g_kind == shs::contract_kind::post
            && std::string{g_expr} == "x < 100";

        (void)guarded_assert(42);
        const bool assert_ok = g_calls == 3
            && g_kind == shs::contract_kind::assertion
            && std::string{g_expr} == "x != 42";

        shs::core::set_contract_violation_handler(nullptr); // restore the default
        return pre_ok && post_ok && assert_ok;
    }

    bool test_valid_conditions_stay_silent()
    {
        const int calls_before = g_calls;
        return guarded_pre(21) == 42
            && guarded_post(50) == 100
            && guarded_assert(7) == 7
            && g_calls == calls_before;
    }

    bool test_default_handler_restored()
    {
        shs::core::set_contract_violation_handler(nullptr);
        return shs::core::contract_violation_handler_instance == &shs::core::contract_violation_default;
    }
#endif
} // namespace

int main()
{
#if defined(SHS_CONTRACTS_ENFORCED)
    bool ok = true;
    ok = test_violation_reports_kind_expression_location() && ok;
    ok = test_valid_conditions_stay_silent() && ok;
    ok = test_default_handler_restored() && ok;
    if (!ok)
    {
        std::fprintf(stderr, "[contract-guardrails-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[contract-guardrails-tests] all tests passed (enforced)\n");
#else
    // Release semantics: the assume path must compile and run the same guarded
    // functions with checks discarded (C1.2: verified by compilation + run).
    if (guarded_pre(21) != 42 || guarded_post(50) != 100 || guarded_assert(7) != 7)
    {
        std::fprintf(stderr, "[contract-guardrails-tests-release] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[contract-guardrails-tests-release] all tests passed\n");
#endif
    return 0;
}
