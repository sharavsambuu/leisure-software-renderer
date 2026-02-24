#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: camera_commands.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/input/command.hpp"

namespace shs
{
    class MoveCommand final : public ICommand
    {
    public:
        MoveCommand(glm::vec3 local_dir, float meters_per_sec)
            : local_dir_(local_dir), speed_mps_(meters_per_sec)
        {}

        RuntimeAction to_runtime_action() const override
        {
            return make_move_local_action(local_dir_, speed_mps_);
        }

    private:
        glm::vec3 local_dir_{};
        float speed_mps_ = 0.0f;
    };

    class LookCommand final : public ICommand
    {
    public:
        LookCommand(float dx, float dy, float sensitivity)
            : dx_(dx), dy_(dy), sensitivity_(sensitivity)
        {}

        RuntimeAction to_runtime_action() const override
        {
            return make_look_action(dx_, dy_, sensitivity_);
        }

    private:
        float dx_ = 0.0f;
        float dy_ = 0.0f;
        float sensitivity_ = 0.0f;
    };

    class ToggleLightShaftsCommand final : public ICommand
    {
    public:
        RuntimeAction to_runtime_action() const override
        {
            return make_toggle_light_shafts_action();
        }
    };

    class ToggleBotCommand final : public ICommand
    {
    public:
        RuntimeAction to_runtime_action() const override
        {
            return make_toggle_bot_action();
        }
    };

    class QuitCommand final : public ICommand
    {
    public:
        RuntimeAction to_runtime_action() const override
        {
            return make_quit_action();
        }
    };
}
