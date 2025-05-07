#include "functions.h"
#include <iostream>
#include <ostream>
#include <vector>

void print(std::vector<double> vec) {
  for (int i = 0; i < vec.size(); i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

void print(std::vector<std::vector<double>> vec) {
  for (int i = 0; i < vec.size(); i++) {
    print(vec[i]);
  }
  std::cout << std::endl;
}

/*std::vector<std::vector<double>> mul(std::vector<double> x,
                                     std::vector<double> y) {
  std::vector<std::vector<double>> z(x.size(), std::vector<double>(x.size()));
  for()
}*/
