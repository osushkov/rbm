
#include "DataLoader.hpp"

#include "image/CharImage.hpp"
#include "image/IdxImages.hpp"

#include <map>
#include <string>
#include <vector>

static TrainingSample sampleFromCharImage(const CharImage &img);

vector<TrainingSample> DataLoader::loadSamples(string imagePath) {
  IdxImages imageLoader(imagePath);
  vector<CharImage> images = imageLoader.Load();

  vector<TrainingSample> result;
  for (const auto &image : images) {
    result.push_back(sampleFromCharImage(image));
  }

  return result;
}

TrainingSample sampleFromCharImage(const CharImage &img) {
  Vector input(img.pixels.size());
  for (unsigned i = 0; i < img.pixels.size(); i++) {
    input(i) = img.pixels[i];
  }

  return TrainingSample(input);
}
