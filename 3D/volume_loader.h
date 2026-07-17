#ifndef VOLUME_LOADER_H
#define VOLUME_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

enum class RawDataFormat { UINT8, FLOAT32, UNKNOWN };

/**
 * 3D体数据加载器
 * 自动识别 uint8 / float32 格式，统一输出 float32
 */
bool loadRawVolume(
    const std::string& filename,
    int& width,
    int& height,
    int& depth,
    int& frames,
    std::vector<std::vector<std::vector<std::vector<float>>>>& volumeSequence,
    RawDataFormat& detectedFormat,
    float& dataMin,
    float& dataMax,
    int hint_w = 0, int hint_h = 0, int hint_d = 0, int hint_f = 0
);

#endif // VOLUME_LOADER_H
