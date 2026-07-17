#include "volume_loader.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <limits>

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
    int hint_w, int hint_h, int hint_d, int hint_f)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    // 读头部 (int32 × 4: width, height, depth, frames)
    int header[4];
    file.read(reinterpret_cast<char*>(header), 16);

    bool dimOk = (header[0] > 0 && header[0] <= 65536 &&
                  header[1] > 0 && header[1] <= 65536 &&
                  header[2] > 0 && header[2] <= 65536 &&
                  header[3] > 0 && header[3] <= 100000);

    if (!dimOk) {
        file.seekg(0, std::ios::beg);
        if (hint_w > 0 && hint_h > 0 && hint_d > 0 && hint_f > 0) {
            // User-specified dimensions for headerless file
            width = hint_w; height = hint_h; depth = hint_d; frames = hint_f;
            const size_t voxels = (size_t)width * height * depth * frames;
            if (fileSize == voxels * 1) {
                detectedFormat = RawDataFormat::UINT8;
                std::cout << "Reading .raw file (no header, hint dims, uint8): " << filename << std::endl;
            } else if (fileSize == voxels * 4) {
                detectedFormat = RawDataFormat::FLOAT32;
                std::cout << "Reading .raw file (no header, hint dims, float32): " << filename << std::endl;
            } else {
                std::cerr << "Warning: hint dims " << width << "x" << height << "x"
                          << depth << "x" << frames << " don't match file size "
                          << fileSize << " (u8=" << voxels << " f32=" << voxels*4 << ")" << std::endl;
                detectedFormat = RawDataFormat::UINT8;
            }
        } else {
            width = 64; height = 64; depth = 64; frames = 91;
            detectedFormat = RawDataFormat::UNKNOWN;
            std::cout << "Reading .raw file (no header, defaults 64^3 x 91): " << filename << std::endl;
        }
    } else {
        width  = header[0];
        height = header[1];
        depth  = header[2];
        frames = header[3];

        const size_t voxels = (size_t)width * height * depth * frames;
        const size_t expectedF32 = 16 + voxels * 4;
        const size_t expectedU8  = 16 + voxels * 1;

        if (fileSize == expectedF32) {
            detectedFormat = RawDataFormat::FLOAT32;
            std::cout << "Reading .raw file (header + float32): " << filename << std::endl;
        } else if (fileSize == expectedU8) {
            detectedFormat = RawDataFormat::UINT8;
            std::cout << "Reading .raw file (header + uint8): " << filename << std::endl;
        } else {
            // 大小不匹配，按 float32 尝试并给出警告
            detectedFormat = RawDataFormat::FLOAT32;
            std::cerr << "Warning: File size mismatch!"
                      << "  Expected(f32)=" << expectedF32
                      << "  Expected(u8)="  << expectedU8
                      << "  Actual=" << fileSize
                      << "  -- assuming float32" << std::endl;
        }
    }

    std::cout << "  Resolution: " << width << "x" << height << "x" << depth
              << "  Frames: " << frames
              << "  Format: " << (detectedFormat == RawDataFormat::UINT8 ? "uint8" : "float32")
              << std::endl;

    // 分配输出 [frame][z][y][x]
    volumeSequence.resize(frames);
    for (int t = 0; t < frames; t++) {
        volumeSequence[t].resize(depth);
        for (int z = 0; z < depth; z++) {
            volumeSequence[t][z].resize(height);
            for (int y = 0; y < height; y++)
                volumeSequence[t][z][y].resize(width);
        }
    }

    dataMin = std::numeric_limits<float>::max();
    dataMax = std::numeric_limits<float>::lowest();
    const int voxelsPerFrame = width * height * depth;

    if (detectedFormat == RawDataFormat::UINT8) {
        std::vector<uint8_t> buf(voxelsPerFrame);
        for (int t = 0; t < frames; t++) {
            file.read(reinterpret_cast<char*>(buf.data()), voxelsPerFrame);
            if (!file) { std::cerr << "Failed to read frame " << t << std::endl; return false; }
            int idx = 0;
            for (int z = 0; z < depth; z++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++) {
                        float v = static_cast<float>(buf[idx++]);
                        volumeSequence[t][z][y][x] = v;
                        if (v < dataMin) dataMin = v;
                        if (v > dataMax) dataMax = v;
                    }
            if ((t + 1) % 10 == 0)
                std::cout << "  Loaded " << (t+1) << "/" << frames << " frames" << std::endl;
        }
    } else {
        std::vector<float> buf(voxelsPerFrame);
        for (int t = 0; t < frames; t++) {
            file.read(reinterpret_cast<char*>(buf.data()), (size_t)voxelsPerFrame * 4);
            if (!file) { std::cerr << "Failed to read frame " << t << std::endl; return false; }
            int idx = 0;
            for (int z = 0; z < depth; z++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++) {
                        float v = buf[idx++];
                        volumeSequence[t][z][y][x] = v;
                        if (v < dataMin) dataMin = v;
                        if (v > dataMax) dataMax = v;
                    }
            if ((t + 1) % 10 == 0)
                std::cout << "  Loaded " << (t+1) << "/" << frames << " frames" << std::endl;
        }
    }

    file.close();
    std::cout << "Loading complete!  dataMin=" << dataMin << "  dataMax=" << dataMax << std::endl;
    return true;
}
