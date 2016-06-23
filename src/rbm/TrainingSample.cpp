

#include "TrainingSample.hpp"
#include <iostream>

std::ostream &operator<<(std::ostream &stream, const TrainingSample &ts) {
  stream << ts.label << " : " << ts.input;
  return stream;
}
