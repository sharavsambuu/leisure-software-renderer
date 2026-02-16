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

        void execute(CommandContext& ctx) override
        {
            auto& cam = ctx.state.camera;
            const glm::vec3 fwd = cam.forward();
            const glm::vec3 right = cam.right();
            const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 world_delta = right * local_dir_.x + up * local_dir_.y + fwd * local_dir_.z;
            cam.pos += world_delta * (speed_mps_ * ctx.dt);
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

        void execute(CommandContext& ctx) override
        {
            auto& cam = ctx.state.camera;
            cam.yaw += dx_ * sensitivity_;
            cam.pitch -= dy_ * sensitivity_;
            cam.pitch = glm::clamp(cam.pitch, glm::radians(-85.0f), glm::radians(85.0f));
        }

    private:
        float dx_ = 0.0f;
        float dy_ = 0.0f;
        float sensitivity_ = 0.0f;
    };

    class ToggleLightShaftsCommand final : public ICommand
    {
    public:
        void execute(CommandContext& ctx) override
        {
            ctx.state.enable_light_shafts = !ctx.state.enable_light_shafts;
        }
    };

    class ToggleBotCommand final : public ICommand
    {
    public:
        void execute(CommandContext& ctx) override
        {
            ctx.state.bot_enabled = !ctx.state.bot_enabled;
        }
    };

    class QuitCommand final : public ICommand
    {
    public:
        void execute(CommandContext& ctx) override
        {
            ctx.state.quit_requested = true;
        }
    };
}
