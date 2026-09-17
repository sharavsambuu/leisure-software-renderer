#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: log.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн core модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <iostream>
#include <string>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace core
    {
    inline void log_info(const std::string& msg)
    {
        std::cout << "[INFO] " << msg << std::endl;
    }

    inline void log_warn(const std::string& msg)
    {
        std::cout << "[WARN] " << msg << std::endl;
    }

    inline void log_error(const std::string& msg)
    {
        std::cerr << "[ERROR] " << msg << std::endl;
    }

    } // inline namespace core
}

