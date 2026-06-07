#pragma once

#include "field_profile.h"

#include <vector>

namespace vbt {

std::vector<int> detectKeyFrames(const std::vector<float>& values,
                                 const FieldProfile& profile);

} // namespace vbt
