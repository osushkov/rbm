#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"

struct TrainingSample {
  Vector input;
  Vector label;

  TrainingSample() = default;
  TrainingSample(const Vector &input, const Vector &label) : input(input), label(label) {}
};

std::ostream &operator<<(std::ostream &stream, const TrainingSample &ts);
