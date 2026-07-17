#ifndef KEYFRAME_DETECTOR_H
#define KEYFRAME_DETECTOR_H

#include <vector>
#include "field_profile.h"

// 一维点结构 (索引, 值)
struct Point1D {
    int index;
    double value;
    Point1D(int i, double v) : index(i), value(v) {}
};

// Weighted Douglas-Peucker 关键帧检测
//
// antialiasWidth: 保留以保持调用方兼容，内部由 profile 参数控制
// profile: FieldProfile 决定误差度量和 epsilon(t) oracle
//          默认值 = SDF profile（与旧版行为一致）
//
// NOTE: values are float32 (raw field values, not quantized to uint8).
std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                  double antialiasWidth,
                                  const FieldProfile& profile = FieldProfile{});

std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                  double antialiasWidth,
                                  const FieldProfile& profile,
                                  const std::vector<float>& frameWeights);

#endif // KEYFRAME_DETECTOR_H
