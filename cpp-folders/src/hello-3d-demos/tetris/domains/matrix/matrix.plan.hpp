#pragma once

#include <array>
#include <memory_resource>
#include "shs_renderer.hpp"
#include "matrix.contract.hpp"

namespace tetris {
    namespace tetris {

        // ============================================================================
        // SCENE PLAN: 3D Tetris — Multi-file Domain Pod Architecture
        // ────────────────────────────────────────────────────────────────────────────
        //
        // Visual style: Low-poly 3D tetromino pieces rendered as extruded quad-based
        // prisms. The board is a dark checkerboard grid. Pieces float slightly above
        // the grid plane (z = -0.5f) for depth. Ghost piece rendered with low opacity.
        // Camera: elevated view (~50° pitch), isometric-ish angle looking down.
        // Lighting: soft ambient + directional "overhead" light, subtle shadows via
        // self-shadowing on piece faces (simplified Lambert shading).
        // ============================================================================

        static constexpr int GRID_W = 10;
        static constexpr int GRID_H = 20;
        static constexpr float CELL_SIZE = 1.0f;
        static constexpr float BOARD_OFFSET_X = -4.5f * CELL_SIZE;   // center horizontally
        static constexpr float BOARD_OFFSET_Y = 9.5f * CELL_SIZE;    // center vertically
        static constexpr float CAMERA_HEIGHT = 7.0f;                 // elevated view

        /// A single trapezoidal prism (truncated pyramid) representing one filled cell.
        struct TetrisCell {
            glm::vec3 center_pos = glm::vec3(0.0f);
            bool      active   = false;                               // is this cell occupied?
            int       piece_id = -1;                                  // which tetromino occupies it (-1 = empty)
            uint8_t   color_idx= 0;                                   // index into PieceColors array

            void generate_triangles(std::pmr::vector<shs::Triangle>& out, std::pmr::memory_resource* mr) const {
                if (!active) return;

                // Approximate a square prism by two triangles per trapezoid face (8 faces total = 16 triangles)
                // Top and bottom rings are quadrilaterals approximated as two triangles each.
                size_t num_sides = 4;
                shs::Triangle t{};

                auto res = mr->allocate(size_t(num_sides));
                std::pmr::vector<glm::vec3, shs::ArenaAllocator<glm::vec3>> vertices(mr);

                // Sample points along the top ring (slightly elevated)
                for (size_t i = 0; i < num_sides; ++i) {
                    float angle = (static_cast<float>(i) / static_cast<float>(num_sides)) * glm::two_pi<float>();
                    vertices.push_back(glm::vec3(
                        center_pos.x + CELL_SIZE * cosf(angle),
                        center_pos.y + CELL_SIZE * sinf(angle),
                        0.25f));   // top ring, slightly above z=0 plane
                }

                // Sample points along the bottom ring (slightly depressed)
                for (size_t i = 0; i < num_sides; ++i) {
                    float angle = (static_cast<float>(i) / static_cast<float>(num_sides)) * glm::two_pi<float>();
                    vertices.push_back(glm::vec3(
                        center_pos.x + CELL_SIZE * cosf(angle),
                        center_pos.y + CELL_SIZE * sinf(angle),
                        -0.25f));  // bottom ring, slightly below z=0 plane
                }

                size_t start = 0;
                for (size_t i = 0; i < num_sides - 1; ++i) {
                    t.v0[0].x = vertices[start + 0].x; t.v0[0].y = vertices[start + 0].y; t.v0[0].z = vertices[start + 0].z;
                    t.v1[0].x = vertices[start + 1].x; t.v1[0].y = vertices[start + 1].y; t.v1[0].z = vertices[start + 1].z;
                    t.v2[0].x = vertices[start + num_sides + 0].x; t.v0[0].y = vertices[start + num_sides + 0].y; t.v0[0].z = vertices[start + num_sides + 0].z;
                    out.push_back(t);

                    t.v0[1].x = vertices[start + num_sides + 0].x; t.v0[1].y = vertices[start + num_sides + 0].y; t.v0[1].z = vertices[start + num_sides + 0].z;
                    t.v1[1].x = vertices[start + num_sides + 1].x; t.v1[1].y = vertices[start + num_sides + 1].y; t.v1[1].z = vertices[start + num_sides + 1].z;
                    t.v2[1].x = vertices[start + num_sides + 2].x; t.v0[1].y = vertices[start + num_sides + 2].y; t.v0[1].z = vertices[start + num_sides + 2].z;
                    out.push_back(t);

                    start += 4; // advance by two triangles (one per side)
                }

                // closing triangle for the last side
                if (num_sides > 1) {
                    t.v0[0].x = vertices[start + 0].x; t.v0[0].y = vertices[start + 0].y; t.v0[0].z = vertices[start + 0].z;
                    t.v1[0].x = vertices[start + num_sides + 0].x; t.v0[0].y = vertices[start + num_sides + 0].y; t.v0[0].z = vertices[start + num_sides + 0].z;
                    t.v2[0].x = vertices[start + num_sides + 1].x; t.v0[0].y = vertices[start + num_sides + 1].y; t.v0[0].z = vertices[start + num_sides + 1].z;
                    out.push_back(t);

                    t.v0[1].x = vertices[start + num_sides + 1].x; t.v0[1].y = vertices[start + num_sides + 1].y; t.v0[1].z = vertices[start + num_sides + 1].z;
                    t.v1[1].x = vertices[start + num_sides + 2].x; t.v1[1].y = vertices[start + num_sides + 2].y; t.v1[1].z = vertices[start + num_sides + 2].z;
                    t.v2[1].x = vertices[start + num_sides + 3].x; t.v0[1].y = vertices[start + num_sides + 3].y; t.v0[1].z = vertices[start + num_sides + 3].z;
                    out.push_back(t);
                }

                mr->deallocate(res, sizeof(size_t));
            }
        };

        /// Ghost piece: projected position of falling tetromino onto the board.
        struct GhostPiece {
            int x = -1;      // grid column (or -1 if no ghost)
            int y = -1;      // grid row
            uint8_t color_idx = 0;

            void generate_triangles(std::pmr::vector<shs::Triangle>& out, std::pmr::memory_resource* mr, float alpha) const {
                if (x < 0 || y < 0) return;

                // Render ghost as thin wireframe-like bars using very low opacity triangles
                TetrisCell cell{};
                cell.center_pos = glm::vec3(BOARD_OFFSET_X + float(x) * CELL_SIZE, BOARD_OFFSET_Y - float(y) * CELL_SIZE, 0.0f);
                cell.active = true;
                cell.piece_id = -1;
                cell.color_idx = color_idx;

                // Create ghost with low opacity by using a separate pass or by blending; here we just emit triangles
                // For simplicity, render the same shape but with reduced alpha encoded in diffuse color.
                auto res = mr->allocate(size_t(8));
                std::pmr::vector<glm::vec3, shs::ArenaAllocator<glm::vec3>> vertices(mr);

                size_t num_sides = 4;
                for (size_t i = 0; i < num_sides; ++i) {
                    float angle = (static_cast<float>(i) / static_cast<float>(num_sides)) * glm::two_pi<float>();
                    vertices.push_back(glm::vec3(
                        cell.center_pos.x + CELL_SIZE * cosf(angle),
                        cell.center_pos.y + CELL_SIZE * sinf(angle),
                        0.15f));   // ghost is slightly elevated above board
                }

                for (size_t i = 0; i < num_sides - 1; ++i) {
                    shs::Triangle t{};
                    t.v0[0].x = vertices[i * 2 + 0].x; t.v0[0].y = vertices[i * 2 + 0].y; t.v0[0].z = vertices[i * 2 + 0].z;
                    t.v1[0].x = vertices[i * 2 + 1].x; t.v1[0].y = vertices[i * 2 + 1].y; t.v1[0].z = vertices[i * 2 + 1].z;
                    t.v2[0].x = vertices[(i + 1) * 2 + 0].x; t.v0[0].y = vertices[(i + 1) * 2 + 0].y; t.v0[0].z = vertices[(i + 1) * 2 + 0].z;
                    out.push_back(t);

                    t.v0[1].x = vertices[(i + 1) * 2 + 0].x; t.v0[1].y = vertices[(i + 1) * 2 + 0].y; t.v0[1].z = vertices[(i + 1) * 2 + 0].z;
                    t.v1[1].x = vertices[(i + 1) * 2 + 1].x; t.v1[1].y = vertices[(i + 1) * 2 + 1].y; t.v1[1].z = vertices[(i + 1) * 2 + 1].z;
                    t.v2[1].x = vertices[(i + 2) * 2 + 0].x; t.v0[1].y = vertices[(i + 2) * 2 + 0].y; t.v0[1].z = vertices[(i + 2) * 2 + 0].z;
                    out.push_back(t);

                    start += 4; // advance by two triangles
                }

                mr->deallocate(res, sizeof(size_t));
            }
        };

    } // namespace tetris
} // namespace tetris
