#pragma once

#include <map>
#include <vector>

#include "rbm/TrainingSample.hpp"

namespace DataLoader {
vector<TrainingSample> loadSamples(string inImagePath, string inLabelPath);
}
