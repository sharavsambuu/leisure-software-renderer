#include <cstdint>
#include <cstdio>
#include <expected>
#include <span>
#include <vector>

// Saga compensation spike (constitutional amendment Task 9, Rule 12).
// Minimal two-stage saga: reserve stock, then charge wallets. Facts are
// closed-enum payloads. RED phase: the compensator below uses ad hoc
// done-flags (the rejected shape) — the rollback test MUST fail on wallets.
namespace
{
    struct SagaItem
    {
        uint32_t id = 0;
        uint32_t stock = 0;
    };

    struct SagaWallet
    {
        uint32_t id = 0;
        double balance = 0.0;
    };

    struct SagaOrder
    {
        uint32_t item = 0;
        uint32_t quantity = 0;
        double price = 0.0;
    };

    enum class SagaFactKind : uint8_t
    {
        StockReserved,
        FundsDebited
    };

    struct SagaFact
    {
        SagaFactKind kind = SagaFactKind::StockReserved;
        uint32_t index = 0;
        uint32_t quantity = 0;
        double amount = 0.0;
    };

    enum class SagaError : uint8_t
    {
        InvalidQuantity,
        InsufficientStock,
        InsufficientFunds
    };

    std::expected<void, SagaError> stage_reserve(std::span<SagaItem> items,
        std::span<const SagaOrder> orders, std::vector<SagaFact>& log)
    {
        for (size_t i = 0; i < orders.size(); ++i)
        {
            if (orders[i].quantity == 0) return std::unexpected(SagaError::InvalidQuantity);
            if (items[i].stock < orders[i].quantity)
                return std::unexpected(SagaError::InsufficientStock);
            items[i].stock -= orders[i].quantity;
            log.push_back(SagaFact{SagaFactKind::StockReserved, (uint32_t)i, orders[i].quantity, 0.0});
        }
        return {};
    }

    std::expected<void, SagaError> stage_charge(std::span<SagaWallet> wallets,
        std::span<const SagaOrder> orders, std::vector<SagaFact>& log)
    {
        for (size_t i = 0; i < orders.size(); ++i)
        {
            if (wallets[i].balance < orders[i].price)
                return std::unexpected(SagaError::InsufficientFunds);
            wallets[i].balance -= orders[i].price;
            log.push_back(SagaFact{SagaFactKind::FundsDebited, (uint32_t)i, 0, orders[i].price});
        }
        return {};
    }

    // CONFORMING SHAPE (Rule 12): consumes the emitted fact log in reverse,
    // undoing every mutated stage. The rejected alternative — restoring only
    // flagged stages — leaks prior debits (RED phase pinned wallet 900.0).
    void compensate_by_log(std::span<SagaItem> items,
        std::span<SagaWallet> wallets, std::span<const SagaFact> log)
    {
        for (size_t k = log.size(); k-- > 0;)
        {
            const SagaFact& f = log[k];
            if (f.kind == SagaFactKind::StockReserved)
                items[f.index].stock += f.quantity;
            else
                wallets[f.index].balance += f.amount;
        }
    }

    bool test_commit()
    {
        std::vector<SagaItem> items{{1, 50}, {2, 10}};
        std::vector<SagaWallet> wallets{{101, 1000.0}, {102, 500.0}};
        const std::vector<SagaOrder> orders{{1, 5, 100.0}, {2, 2, 50.0}};
        std::vector<SagaFact> log{};
        if (!stage_reserve(items, orders, log)) return false;
        if (!stage_charge(wallets, orders, log)) return false;
        if (items[0].stock != 45 || items[1].stock != 8) return false;
        if (wallets[0].balance != 900.0 || wallets[1].balance != 450.0) return false;
        return log.size() == 4;
    }

    bool test_rollback_full()
    {
        std::vector<SagaItem> items{{1, 50}, {2, 10}};
        std::vector<SagaWallet> wallets{{101, 1000.0}, {102, 5.0}};
        const std::vector<SagaOrder> orders{{1, 5, 100.0}, {2, 2, 50.0}};
        std::vector<SagaFact> log{};
        if (!stage_reserve(items, orders, log)) return false;
        const auto charged = stage_charge(wallets, orders, log);
        if (charged) return false;
        if (charged.error() != SagaError::InsufficientFunds) return false;
        compensate_by_log(items, wallets, log);
        if (items[0].stock != 50 || items[1].stock != 10) return false;
        // Wallet 101 was debited $100 before order 2 failed: full rollback
        // must restore it from the fact log.
        return wallets[0].balance == 1000.0 && wallets[1].balance == 5.0;
    }
} // namespace

int main()
{
    bool ok = true;
    auto run = [&](const char* name, bool result)
    {
        std::fprintf(stderr, "[saga-tests] %s: %s\n", name, result ? "pass" : "FAIL");
        ok = result && ok;
    };

    run("commit", test_commit());
    run("rollback_full", test_rollback_full());

    if (!ok)
    {
        std::fprintf(stderr, "[saga-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[saga-tests] all tests passed\n");
    return 0;
}
