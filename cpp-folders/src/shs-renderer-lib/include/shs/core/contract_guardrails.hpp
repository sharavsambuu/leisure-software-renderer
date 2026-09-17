#pragma once
/*
    SHS RENDERER SAN — contract_guardrails.hpp (core).
    Constitution II Rule 17 bridge: SHS_PRE / SHS_POST / SHS_CONTRACT_ASSERT
    mirror the C++26 contract keywords 1:1. Enforced builds
    (SHS_CONTRACTS_ENFORCED) check and report to the violation handler; release
    builds fold to C++23 [[assume]] and may discard the condition. Conditions
    are single-expression, side-effect-free reads (bridge rules 2-3) — they
    never gate control flow. Native C++26 keys on __cpp_contracts only (C4.3).
*/
#include <cstdio>
#include <cstdlib>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace core
    {
    enum class contract_kind { pre, post, assertion };

    // P2900-shaped handler seam: the C++26 switch replaces the handler, never its call sites.
    using contract_violation_handler =
        void (*)(contract_kind, const char*, const char*, int) noexcept;

    inline void contract_violation_default(contract_kind kind, const char* expr, const char* file, int line) noexcept
    {
        static const char* names[] = {"pre", "post", "contract_assert"};
        std::fprintf(stderr, "contract violation: %s failed: %s (%s:%d)\n",
            names[static_cast<int>(kind)], expr, file, line);
        std::abort();
    }

    inline contract_violation_handler contract_violation_handler_instance = &contract_violation_default;

    inline void set_contract_violation_handler(contract_violation_handler h) noexcept
    {
        contract_violation_handler_instance = h ? h : &contract_violation_default;
    }

    inline void contract_violation(contract_kind kind, const char* expr, const char* file, int line) noexcept
    {
        contract_violation_handler_instance(kind, expr, file, line);
    }

    } // inline namespace core
} // namespace shs

#if defined(__cpp_contracts) || defined(SHS_CONTRACTS_ENFORCED)
    #define SHS_CHECK(kind, cond) \
        do { if (!(cond)) ::shs::contract_violation(kind, #cond, __FILE__, __LINE__); } while (false)
    #define SHS_PRE(cond)             SHS_CHECK(::shs::contract_kind::pre, cond)
    #define SHS_POST(cond)            SHS_CHECK(::shs::contract_kind::post, cond)
    #define SHS_CONTRACT_ASSERT(cond) SHS_CHECK(::shs::contract_kind::assertion, cond)
#else
    #if defined(__has_cpp_attribute) && __has_cpp_attribute(assume)
        #define SHS_PRE(cond)             [[assume(!!(cond))]]
        #define SHS_POST(cond)            [[assume(!!(cond))]]
        #define SHS_CONTRACT_ASSERT(cond) [[assume(!!(cond))]]
    #endif
    #if !defined(SHS_PRE)
        #define SHS_PRE(cond)
        #define SHS_POST(cond)
        #define SHS_CONTRACT_ASSERT(cond)
    #endif
#endif
