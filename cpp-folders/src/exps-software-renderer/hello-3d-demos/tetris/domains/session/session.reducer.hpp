#pragma once
// tetris/domains/session/session.reducer.hpp — PURE SESSION STATE MACHINE
// (tetris::session) Screen transitions + menu cursors from intent tokens.
// Same discipline as the other pods: (snapshot, commands, dt) -> (next, events).
// External facts (run finished, stage loaded) are plain-field writes done by
// main — never simulated here.
#include <algorithm>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include <domains/session/session.contract.hpp>
#include <domains/session/session.action.hpp>

namespace tetris::session {

    enum class SessionEventType : uint8_t {
        SCREEN_CHANGED        = 0,
        NAV_MOVED             = 1,   // cursor blip feedback
        CONFIRMED             = 2,   // confirm blip feedback
        STAGE_SELECTED        = 3,   // .stage = 0-based manifest index to load
        RUN_RESTART_REQUESTED = 4,   // reload session.current_stage
        QUIT_REQUESTED        = 5,
        SOUND_TOGGLED         = 6    // .enabled = new state
    };

    struct SessionEvent {
        SessionEventType type;
        int              stage   = -1;
        bool             enabled = false;
    };

    struct SessionStep {
        SessionSnapshot                next;
        std::pmr::vector<SessionEvent> events;

        explicit SessionStep(std::pmr::memory_resource* mr) : events(mr) {}
    };

    static constexpr int TITLE_MENU_COUNT  = 3;  // start / sound / exit
    static constexpr int PAUSE_MENU_COUNT  = 5;  // resume/restart/select/sound/exit
    static constexpr int RESULTS_MENU_BASE = 2;  // +1 contextual first row

    static inline void push_event(SessionStep& st, const SessionEvent& ev) {
        st.events.push_back(ev);
    }

    static inline SessionStep reduce_session(const SessionSnapshot& s,
                                             std::span<const SessionCommand> cmds,
                                             float dt,
                                             std::pmr::memory_resource* mr) {
        SessionStep out(mr);
        SessionSnapshot n = s;
        n.anim_time += dt;

        auto move_cursor = [&](int count, int delta) {
            if (count <= 0) return;
            const int old = n.cursor;
            n.cursor = ((n.cursor + delta) % count + count) % count;
            if (n.cursor != old) push_event(out, { SessionEventType::NAV_MOVED });
        };
        auto goto_screen = [&](Screen to) {
            if (n.screen == to) return;
            n.screen = to;
            n.cursor = 0;
            push_event(out, { SessionEventType::SCREEN_CHANGED });
        };

        for (const auto& cmd : cmds) {
            std::visit([&](const auto& c) {
                using T = std::decay_t<decltype(c)>;

                if constexpr (std::is_same_v<T, ToggleSoundIntent>) {
                    n.sound_enabled = !n.sound_enabled;
                    push_event(out, { SessionEventType::SOUND_TOGGLED, -1, n.sound_enabled });
                }
                else if constexpr (std::is_same_v<T, NavUpIntent> || std::is_same_v<T, NavDownIntent>) {
                    const int delta = std::is_same_v<T, NavDownIntent> ? +1 : -1;
                    switch (n.screen) {
                        case Screen::TITLE:
                            move_cursor(TITLE_MENU_COUNT, delta);
                            break;
                        case Screen::PAUSED:
                            move_cursor(PAUSE_MENU_COUNT, delta);
                            break;
                        case Screen::RESULTS: {
                            const int rows = RESULTS_MENU_BASE +
                                ((n.run_victory && n.current_stage + 1 < n.stage_count) ? 1 : 0);
                            move_cursor(rows, delta);
                            break;
                        }
                        default: break;   // LEVEL_SELECT uses left/right carousel
                    }
                }
                else if constexpr (std::is_same_v<T, NavLeftIntent> || std::is_same_v<T, NavRightIntent>) {
                    if (n.screen == Screen::LEVEL_SELECT && n.unlocked_stages > 0) {
                        const int delta = std::is_same_v<T, NavRightIntent> ? +1 : -1;
                        const int old   = n.stage_cursor;
                        n.stage_cursor  = ((n.stage_cursor + delta) % n.unlocked_stages
                                           + n.unlocked_stages) % n.unlocked_stages;
                        if (n.stage_cursor != old)
                            push_event(out, { SessionEventType::NAV_MOVED });
                    }
                }
                else if constexpr (std::is_same_v<T, BackIntent> || std::is_same_v<T, TogglePauseIntent>) {
                    const bool pause_key = std::is_same_v<T, TogglePauseIntent>;
                    switch (n.screen) {
                        case Screen::PLAYING:
                            goto_screen(Screen::PAUSED);      // ESC or P pauses a live run
                            break;
                        case Screen::PAUSED:
                            goto_screen(Screen::PLAYING);     // symmetric toggle resumes
                            break;
                        case Screen::LEVEL_SELECT:
                            if (!pause_key) goto_screen(Screen::TITLE);
                            break;
                        case Screen::RESULTS:
                            if (!pause_key) goto_screen(Screen::TITLE);
                            break;
                        default: break;
                    }
                }
                else if constexpr (std::is_same_v<T, ConfirmIntent>) {
                    push_event(out, { SessionEventType::CONFIRMED });
                    switch (n.screen) {
                        case Screen::TITLE: {
                            if      (n.cursor == 0) goto_screen(Screen::LEVEL_SELECT);
                            else if (n.cursor == 1) {
                                n.sound_enabled = !n.sound_enabled;
                                push_event(out, { SessionEventType::SOUND_TOGGLED, -1, n.sound_enabled });
                            }
                            else                    push_event(out, { SessionEventType::QUIT_REQUESTED });
                            break;
                        }
                        case Screen::LEVEL_SELECT: {
                            if (n.unlocked_stages <= 0) break;
                            n.stage_cursor  = ((n.stage_cursor % n.unlocked_stages)
                                               + n.unlocked_stages) % n.unlocked_stages;
                            n.current_stage = n.stage_cursor;
                            push_event(out, { SessionEventType::STAGE_SELECTED, n.current_stage });
                            goto_screen(Screen::PLAYING);
                            break;
                        }
                        case Screen::PAUSED: {
                            switch (n.cursor) {
                                case 0: goto_screen(Screen::PLAYING); break;   // resume
                                case 1:                                                        // restart stage
                                    push_event(out, { SessionEventType::RUN_RESTART_REQUESTED });
                                    goto_screen(Screen::PLAYING);
                                    break;
                                case 2: goto_screen(Screen::LEVEL_SELECT); break;
                                case 3:                                                        // sound toggle
                                    n.sound_enabled = !n.sound_enabled;
                                    push_event(out, { SessionEventType::SOUND_TOGGLED, -1, n.sound_enabled });
                                    break;
                                default: push_event(out, { SessionEventType::QUIT_REQUESTED }); break;
                            }
                            break;
                        }
                        case Screen::RESULTS: {
                            const bool has_next = n.run_victory &&
                                                  (n.current_stage + 1 < n.stage_count);
                            if (n.cursor == 0) {
                                if (has_next) {
                                    n.current_stage += 1;
                                    push_event(out, { SessionEventType::STAGE_SELECTED, n.current_stage });
                                } else {
                                    push_event(out, { SessionEventType::RUN_RESTART_REQUESTED });
                                }
                                goto_screen(Screen::PLAYING);
                            } else if (n.cursor == 1) {
                                goto_screen(Screen::LEVEL_SELECT);
                            } else {
                                goto_screen(Screen::TITLE);
                            }
                            break;
                        }
                        default: break;
                    }
                }
            }, cmd);
        }

        out.next = n;
        return out;
    }

} // namespace tetris::session
