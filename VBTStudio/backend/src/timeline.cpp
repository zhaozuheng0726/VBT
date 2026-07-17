#include "vbtstudio/backend/timeline.h"

#include <algorithm>
#include <cmath>

namespace vbtstudio::backend {

void Timeline::configure(std::uint32_t frame_count, double frames_per_second)
{
    frame_count_ = std::max(1u, frame_count);
    frame_ = std::min(frame_, frame_count_ - 1u);
    fps_ = std::clamp(frames_per_second, 1.0, 240.0);
    accumulator_ = 0.0;
    direction_ = 1;
}

void Timeline::update(double elapsed_seconds)
{
    if (!playing_ || elapsed_seconds <= 0.0) return;
    accumulator_ += elapsed_seconds;
    const double seconds_per_frame = 1.0 / fps_;
    const auto frames_to_advance = static_cast<std::uint32_t>(std::floor(accumulator_ / seconds_per_frame));
    if (frames_to_advance == 0) return;
    accumulator_ -= static_cast<double>(frames_to_advance) * seconds_per_frame;
    for (std::uint32_t index = 0; index < frames_to_advance && playing_; ++index) {
        advance_one();
    }
}

void Timeline::play() noexcept { playing_ = true; }
void Timeline::pause() noexcept { playing_ = false; }
void Timeline::toggle_playback() noexcept { playing_ = !playing_; }

void Timeline::seek(std::uint32_t frame) noexcept
{
    frame_ = std::min(frame, frame_count_ - 1u);
    accumulator_ = 0.0;
}

void Timeline::step(int delta) noexcept
{
    if (delta == 0) return;
    pause();
    const int count = static_cast<int>(frame_count_);
    int next = static_cast<int>(frame_) + delta;
    if (loop_mode_ == LoopMode::Loop) {
        next = ((next % count) + count) % count;
    }
    else {
        next = std::clamp(next, 0, count - 1);
    }
    frame_ = static_cast<std::uint32_t>(next);
    accumulator_ = 0.0;
}

void Timeline::set_fps(double value) noexcept { fps_ = std::clamp(value, 1.0, 240.0); }
void Timeline::set_loop_mode(LoopMode mode) noexcept { loop_mode_ = mode; }
std::uint32_t Timeline::frame() const noexcept { return frame_; }
std::uint32_t Timeline::frame_count() const noexcept { return frame_count_; }
double Timeline::fps() const noexcept { return fps_; }
bool Timeline::playing() const noexcept { return playing_; }
LoopMode Timeline::loop_mode() const noexcept { return loop_mode_; }

void Timeline::advance_one()
{
    if (frame_count_ <= 1u) return;
    const int last = static_cast<int>(frame_count_ - 1u);
    int next = static_cast<int>(frame_) + direction_;
    if (next >= 0 && next <= last) {
        frame_ = static_cast<std::uint32_t>(next);
        return;
    }
    if (loop_mode_ == LoopMode::Loop) {
        frame_ = direction_ > 0 ? 0u : frame_count_ - 1u;
    }
    else if (loop_mode_ == LoopMode::PingPong) {
        direction_ *= -1;
        frame_ = static_cast<std::uint32_t>(std::clamp(static_cast<int>(frame_) + direction_, 0, last));
    }
    else {
        frame_ = direction_ > 0 ? frame_count_ - 1u : 0u;
        playing_ = false;
    }
}

} // namespace vbtstudio::backend
