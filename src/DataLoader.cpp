
#include "DataLoader.hpp"

#include "image/CharImage.hpp"
#include "image/IdxImages.hpp"
#include "image/IdxLabels.hpp"
#include "image/ImageWriter.hpp"

#include <map>
#include <string>
#include <vector>

static constexpr float THRESOLD = 0.2f;

static map<int, vector<CharImage>> loadLabeledImages(string imagePath, string labelPath);
static TrainingSample sampleFromCharImage(int label, const CharImage &img);

// Loads the training samples from the given digits files.
vector<TrainingSample> DataLoader::loadSamples(string inImagePath, string inLabelPath) {
  auto labeledImages = loadLabeledImages(inImagePath, inLabelPath);

  vector<TrainingSample> result;
  for (const auto &entry : labeledImages) {
    for (const auto &image : entry.second) {
      result.push_back(sampleFromCharImage(entry.first, image));
    }
  }

  return result;
}

map<int, vector<CharImage>> loadLabeledImages(string imagePath, string labelPath) {
  IdxImages imageLoader(imagePath);
  IdxLabels labelLoader(labelPath);

  vector<int> labels = labelLoader.Load();
  vector<CharImage> images = imageLoader.Load();

  assert(labels.size() == images.size());

  map<int, vector<CharImage>> result;
  for (unsigned i = 0; i < labels.size(); i++) {
    if (result.find(labels[i]) == result.end()) {
      result[labels[i]] = vector<CharImage>();
    }

    result[labels[i]].push_back(images[i]);
  }

  return result;
}

TrainingSample sampleFromCharImage(int label, const CharImage &img) {
  assert(label >= 0 && label < 10);

  Vector output(10);
  output.fill(0.0f);
  output[label] = 1.0f;

  Vector input(img.pixels.size());
  for (unsigned i = 0; i < img.pixels.size(); i++) {
    input(i) = img.pixels[i] > THRESOLD ? 1.0f : 0.0f;
  }

  return TrainingSample(input, output);
}
