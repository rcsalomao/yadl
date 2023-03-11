#include "yadl.hpp"
#include <memory>

int main() {
  std::vector<std::unique_ptr<yadl::RV>> vec;
  vec.push_back(std::make_unique<yadl::UniformInt>(23, 32));
  vec.push_back(std::make_unique<yadl::Poisson>(4.7));
  vec.push_back(std::make_unique<yadl::Normal>(5, 1));
  vec.push_back(std::make_unique<yadl::Normal>(gsl_rng_mt19937, 5, 1));
  vec.push_back(std::make_unique<yadl::Weibull>());
  for (auto &random_var : vec) {
    printf("sample: %.3f\n", random_var->sample());
    printf("mean: %.3f\n", random_var->mean());
    printf("stdv: %.3f\n", random_var->stdv());
  }

  yadl::Normal rv = yadl::Normal(gsl_rng_default, 15, 5, 1);
  double samp = rv.sample();
  printf("%s\n", gsl_rng_name(rv.m_rng));
  printf("%.2f\n", rv.mean());
  printf("%.2f\n", rv.stdv());
  printf("%.2f\n", rv.pdf(rv.mean()));
  printf("%.2f\n", rv.cdf_P(99));
  printf("%.2f\n", rv.cdf_P(rv.mean()));
  printf("%.2f\n", rv.cdf_P_inv(0.9999));
  printf("%.3f\n", samp);
  printf("%.3f\n", rv.pdf(samp));
  printf("%.3f\n", rv.cdf_P(samp));
  printf("%.3f\n", rv.cdf_P_inv(rv.cdf_P(samp)));

  return 0;
};
