#pragma once
// tetris/domains/session/session.action.hpp — SESSION INTENT TOKENS
// (tetris::session) Menu/navigation intents emitted by the input edge.
// Pure tokens; the session reducer is their only consumer.
#include <variant>

namespace tetris::session {

    struct NavUpIntent        {};
    struct NavDownIntent      {};
    struct NavLeftIntent      {};
    struct NavRightIntent     {};
    struct ConfirmIntent      {};
    struct BackIntent         {};   // ESC: pause in PLAYING, back-nav elsewhere
    struct TogglePauseIntent  {};   // explicit P key
    struct ToggleSoundIntent  {};   // M key, anywhere

    using SessionCommand = std::variant<
        NavUpIntent, NavDownIntent, NavLeftIntent, NavRightIntent,
        ConfirmIntent, BackIntent, TogglePauseIntent, ToggleSoundIntent
    >;

} // namespace tetris::session