#pragma once

#include <cstdint>

namespace vbtstudio::backend {

enum class LoopMode {
    Once,
    Loop,
    PingPong,
};

class Timeline {
public:
    void configure(std::uint32_t frame_count, double frames_per_second);
    void update(double elapsed_seconds);

    void play() noexcept;
    void pause() noexcept;
    void toggle_playback() noexcept;
    void seek(std::uint32_t frame) noexcept;
    void step(int delta) noexcept;

    void set_fps(double value) noexcept;
    void set_loop_mode(LoopMode mode) noexcept;

    [[nodiscard]] std::uint32_t frame() const noexcept;
    [[nodiscard]] std::uint32_t frame_count() const noexcept;
    [[nodiscard]] double fps() const noexcept;
    [[nodiscard]] bool playing() const noexcept;
    [[nodiscard]] LoopMode loop_mode() const noexcept;

private:
    void advance_one();

    std::uint32_t frame_count_ = 1;
    std::uint32_t frame_ = 0;
    double fps_ = 24.0;
    double accumulator_ = 0.0;
    bool playing_ = false;
    int direction_ = 1;
    LoopMode loop_mode_ = LoopMode::Loop;
};

} // namespace vbtstudio::backend
