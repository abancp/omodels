#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
std::vector<double> softmax(const std::vector<double>);
double sigmoid(const double x);
std::vector<double> sigmoid(const std::vector<double> x);
double relu(const double x);
std::vector<double> relu(const std::vector<double> x);

#endif
