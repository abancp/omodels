#include "activations.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

std::vector<double> softmax(std::vector<double> x) {
  std::vector<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(),
                 [](double val) { return exp(val); });
  double sum = std::accumulate(result.begin(), result.end(), 0.0);
  std::transform(result.begin(), result.end(), result.begin(),
                 [sum](double val) { return val / sum; });
  return result;
}

double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

std::vector<double> sigmoid(std::vector<double> x) {
  std::vector<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(),
                 [](double val) { return sigmoid(val); });
  return result;
}

double relu(double x) { return x > 0 ? x : 0; }

std::vector<double> relu(std::vector<double> x) {
  std::vector<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(),
                 [](double val) { return relu(val); });
  return result;
}

double relu_derivative(double x) { return x < 0 ? 0 : 1; }

std::vector<double> relu_derivative(std::vector<double> x) {
  std::vector<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(),
                 [](double val) { return relu_derivative(val); });
  return result;
}
