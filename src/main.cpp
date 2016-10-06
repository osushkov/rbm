
#include "DataLoader.hpp"
#include "common/Common.hpp"
#include "image/CharImage.hpp"
#include "image/ImageRenderer.hpp"
#include "rbm/RBM.hpp"
#include "rbm/TrainingProvider.hpp"

#include <vector>

static constexpr unsigned INPUT_SIZE = (28 * 28) + 10;
static constexpr unsigned HIDDEN_LAYER_SIZE = 256;
static constexpr unsigned BATCH_SIZE = 50;

static unsigned numCompletePasses = 0;
static unsigned curSamplesIndex = 0;
static unsigned curSamplesOffset = 0;

static Matrix getBatchMatrixWithBias(const Matrix &noBias);
static pair<Matrix, Matrix> gibbsSample(RBM *network, Matrix startVisible, unsigned iters);
static void train(RBM *network, vector<TrainingSample> samples, unsigned iters);

static Matrix makeBatch(const TrainingProvider &samplesProvider);
static TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                             unsigned totalIters);

static void renderHiddenNodes(RBM *network);

unsigned getRBMLabel(RBM *network, const TrainingSample &sample) {
  float minEnergy = 0.0f;
  unsigned minLabel = 0;

  for (unsigned label = 0; label < 10; label++) {
    Matrix visible(INPUT_SIZE, 1);
    visible.fill(0.0f);

    for (int i = 0; i < sample.input.rows(); i++) {
      visible(i, 0) = sample.input(i);
    }

    visible(label + sample.input.rows(), 0) = 1.0f;

    Matrix hidden = network->ComputeHidden(visible);
    float energy = network->Energy(visible, hidden);

    if (label == 0 || energy < minEnergy) {
      minEnergy = energy;
      minLabel = label;
    }
  }

  return minLabel;
}

unsigned getSampleLabel(const TrainingSample &sample) {
  for (unsigned i = 0; i < sample.label.rows(); i++) {
    if (sample.label(i) > 0.5f) {
      return i;
    }
  }
  return 0;
}

void testAccuracy(RBM *network, const vector<TrainingSample> &testSamples) {
  unsigned correct = 0;
  unsigned incorrect = 0;

  for (const auto &ts : testSamples) {
    if (getSampleLabel(ts) == getRBMLabel(network, ts)) {
      correct++;
    } else {
      incorrect++;
    }
  }

  cout << (correct / static_cast<float>(correct + incorrect)) << endl;
}

Matrix getBatchMatrixWithBias(const Matrix &noBias) {
  Matrix result(noBias.rows() + 1, noBias.cols());
  for (int c = 0; c < noBias.cols(); c++) {
    result(noBias.rows(), c) = 1.0f;
  }
  result.topRightCorner(noBias.rows(), noBias.cols()) = noBias;
  return result;
}

pair<Matrix, Matrix> gibbsSample(RBM *network, Matrix startVisible, unsigned iters) {
  assert(iters > 0);

  Matrix visible = startVisible;
  Matrix hidden;

  for (unsigned i = 0; i < iters; i++) {
    if (i != 0) {
      visible = network->ComputeVisible(hidden);
    }
    hidden = network->ComputeHidden(visible);
  }

  return make_pair(visible, hidden);
}

void train(RBM *network, vector<TrainingSample> samples, unsigned iters) {
  vector<TrainingSample> testSamples =
      DataLoader::loadSamples("data/test_images.idx3", "data/test_labels.idx1");

  static constexpr float INITIAL_LEARN_RATE = 0.01f;
  static constexpr float TARGET_LEARN_RATE = 0.00001f;
  float learnRateDecay =
      powf(TARGET_LEARN_RATE / INITIAL_LEARN_RATE, 1.0f / static_cast<float>(iters));

  pair<Matrix, Matrix> negative;
  for (unsigned i = 0; i < iters; i++) {
    TrainingProvider samplesProvider = getStochasticSamples(samples, i, iters);
    Matrix batchVisible = makeBatch(samplesProvider);

    auto positive = gibbsSample(network, batchVisible, 1);

    if (i == 0) {
      negative = gibbsSample(network, batchVisible, 3);
    } else {
      negative = gibbsSample(network, negative.first, 3);
    }

    Matrix positiveVisible = getBatchMatrixWithBias(positive.first);
    Matrix positiveHidden = getBatchMatrixWithBias(positive.second);

    Matrix negativeVisible = getBatchMatrixWithBias(negative.first);
    Matrix negativeHidden = getBatchMatrixWithBias(negative.second);

    Matrix gradient =
        positiveHidden * positiveVisible.transpose() - negativeHidden * negativeVisible.transpose();
    gradient *= 1.0f / BATCH_SIZE;

    float learnRate = INITIAL_LEARN_RATE * powf(learnRateDecay, i);
    network->ApplyUpdate(gradient * learnRate);

    if (i % 1000 == 0) {
//      cout << "iter: " << i << endl;
      testAccuracy(network, testSamples);
    }
  }
}

Matrix makeBatch(const TrainingProvider &samplesProvider) {
  Matrix result(INPUT_SIZE, samplesProvider.NumSamples());
  for (unsigned c = 0; c < samplesProvider.NumSamples(); c++) {
    TrainingSample sample = samplesProvider.GetSample(c);

    unsigned r = 0;
    for (int i = 0; i < sample.input.rows(); i++) {
      result(r++, c) = sample.input(i);
    }
    for (int i = 0; i < sample.label.rows(); i++) {
      result(r++, c) = sample.label(i);
    }
  }
  return result;
}

TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                      unsigned totalIters) {

  unsigned numSamples = min<unsigned>(allSamples.size(), BATCH_SIZE);

  if ((curSamplesIndex + numSamples) > allSamples.size()) {
    if (numCompletePasses % 10 == 0) {
      random_shuffle(allSamples.begin(), allSamples.end());
    } else {
      curSamplesOffset = rand() % allSamples.size();
    }
    curSamplesIndex = 0;
    numCompletePasses++;
  }

  auto result = TrainingProvider(allSamples, numSamples, curSamplesIndex + curSamplesOffset);
  curSamplesIndex += numSamples;

  return result;
}

void renderHiddenNodes(RBM *network) {
  Matrix weights = network->GetWeights();
  cout << weights << endl;

  for (unsigned i = 0; i < 100; i++) {
    unsigned node = rand() % weights.rows();

    vector<float> pixels(28 * 28);
    for (unsigned p = 0; p < 28 * 28; p++) {
      pixels[p] = weights(node, p) / 4.0f + 0.5f;
    }

    CharImage img(28, 28, pixels);
    ImageRenderer::RenderImage(img);
  }
}



int main(int argc, char **argv) {
  cout << "loading training data" << endl;
  vector<TrainingSample> trainingSamples =
      DataLoader::loadSamples("data/train_images.idx3", "data/train_labels.idx1");
  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  cout << "training data size: " << trainingSamples.size() << endl;

  vector<TrainingSample> testSamples =
      DataLoader::loadSamples("data/test_images.idx3", "data/test_labels.idx1");
  cout << "test data size: " << testSamples.size() << endl;

  auto network = make_unique<RBM>(INPUT_SIZE, HIDDEN_LAYER_SIZE);
//  testAccuracy(network.get(), testSamples);

  train(network.get(), trainingSamples, 100000);
  // renderHiddenNodes(network.get());
  testAccuracy(network.get(), testSamples);

  cout << "hello world" << endl;
  return 0;
}
