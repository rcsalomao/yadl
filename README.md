# yadl

Yet Another Distribution Library -- A random distribution library that wraps Gnu Scientific Library (GSL) random distribution functions and aims to create a uniform interface between those different random variables.

## Installation

This is basically a header only library.
Therefore you just need to include the file `./src/yadl.hpp` into your project and use it.

### Requirements

As already stated, this project is a wrapper around GSL and thus it expects the library to already be installed on the system.
In the case of an Ubuntu linux distribution, it's possible to use the default repo:

```bash
sudo apt install gsl-dev
```

Do not forget to pass linker flags `-lgsl -lgslcblas -lm` to link to the gsl library.

## yadl interface

All random variables objects derive from RV parent class.
The RV parent class has the following structure:

```cpp
struct RV {
  virtual ~RV() = default;
  virtual void set_rng(gsl_rng *) = 0;
  virtual void set_seed(unsigned long int) = 0;
  virtual double sample() = 0;
  virtual double mean() = 0;
  virtual double stdv() { return std::numeric_limits<double>::quiet_NaN(); }
  virtual double pdf(double) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  virtual double cdf_P(double) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  virtual double cdf_P_inv(double) {
    return std::numeric_limits<double>::quiet_NaN();
  }
};
```

This way, any random variable derived has at least the most used (in my experience) methods as `sample()`, `mean()`, `stdv()`, `pdf(double)`, `cdf_P(double)`, `cdf_P_inv(double)`.

## Examples

Next is shown some examples on how to use this lib:

```cpp
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
```

This same example can be found in `./example/main.cpp`.
