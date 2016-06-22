#pragma once

#include <map>
#include <vector>

#include "rbm/TrainingSample.hpp"

namespace DataLoader {

// Loads the training samples from the given digits files.
vector<TrainingSample> loadSamples(string inImagePath);
}
