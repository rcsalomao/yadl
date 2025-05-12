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
class RV {
   public:
    virtual double sample(const RNG&) = 0;
    virtual double mean() = 0;
    virtual double stdv() = 0;
    virtual double pdf(double) = 0;
    virtual double cdf_P(double) = 0;
    virtual double cdf_P_inv(double) = 0;
    virtual ~RV() = default;
};
```

This way, any random variable derived has at least the most used (in my experience) methods as `sample()`, `mean()`, `stdv()`, `pdf(double)`, `cdf_P(double)`, `cdf_P_inv(double)`.


For the sampling process, a random number generator (`rng`) is necessary.
Therefore, this library provides a class to encapsulate the `rng` objects provided by GSL.
The `RNG` class has the following structure, with both moving operations deleted:

```cpp
class RNG {
    gsl_rng* m_rng;

   public:
    // ctor
    RNG(unsigned long int seed = std::random_device()()) {
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng, seed);
    };
    RNG(const gsl_rng_type* rng_type,
        unsigned long int seed = std::random_device()()) {
        m_rng = gsl_rng_alloc(rng_type);
        gsl_rng_set(m_rng, seed);
    };
    // copy
    RNG(const RNG& other_rng) { m_rng = gsl_rng_clone(other_rng.get()); };
    RNG& operator=(const RNG& other_rng) {
        gsl_rng_memcpy(m_rng, other_rng.get());
        return *this;
    };
    // move
    RNG(RNG&& other_rng) = delete;
    RNG& operator=(RNG&& other_rng) = delete;
    // dtor
    ~RNG() { gsl_rng_free(m_rng); };
    // getter
    gsl_rng* get() const { return m_rng; };
};
```

## Examples

Next is shown some examples on how to use this lib:

```cpp
int main() {
    yadl::RNG rng{};
    {
        std::cout << std::endl;
        printf("rng name: %s\n", gsl_rng_name(rng.get()));
    }

    {
        std::cout << std::endl;
        std::vector v{1, 2, 3, 4, 5};
        std::cout << "v: " << stringify(v) << std::endl;
        yadl::shuffle(rng, v);
        std::cout << "shuffled v: " << stringify(v) << std::endl;
    }

    {
        std::cout << std::endl;
        printf("yadl::Dice(12).sample(rng): %i\n", yadl::Dice(12).sample(rng));
    }

    {
        std::cout << std::endl;
        std::vector<std::unique_ptr<yadl::RV>> v_rv;
        v_rv.push_back(std::make_unique<yadl::UniformInt>(23, 32));
        v_rv.push_back(std::make_unique<yadl::Poisson>(4.7));
        v_rv.push_back(std::make_unique<yadl::Normal>(5, 1));
        v_rv.push_back(std::make_unique<yadl::Weibull>());
        for (auto [i, rv] : v_rv | std::ranges::views::enumerate) {
            printf("rv[%i]->sample(rng): %.3f\n", static_cast<int>(i),
                   rv->sample(rng));
            printf("rv[%i]->mean(): %.3f\n", static_cast<int>(i), rv->mean());
            printf("rv[%i]->stdv(): %.3f\n", static_cast<int>(i), rv->stdv());
        }
    }

    {
        std::cout << std::endl;
        printf("yadl::Normal rv = yadl::Normal(15, 5);\n");
        yadl::Normal rv = yadl::Normal(15, 5);
        printf("double sample = rv.sample(rng);\n");
        double sample = rv.sample(rng);
        printf("rv.mean(): %.2f\n", rv.mean());
        printf("rv.stdv(): %.2f\n", rv.stdv());
        printf("rv.pdf(rv.mean()): %.2f\n", rv.pdf(rv.mean()));
        printf("rv.cdf_P(33.60): %.4f\n", rv.cdf_P(33.60));
        printf("rv.cdf_P(rv.mean()): %.2f\n", rv.cdf_P(rv.mean()));
        printf("rv.cdf_P_inv(0.9999): %.2f\n", rv.cdf_P_inv(0.9999));
        printf("sample: %.3f\n", sample);
        printf("rv.pdf(): %.3f\n", rv.pdf(sample));
        printf("rv.cdf_P(sample): %.3f\n", rv.cdf_P(sample));
        printf("rv.cdf_P_inv(rv.cdf_P(sample)): %.3f\n",
               rv.cdf_P_inv(rv.cdf_P(sample)));
    }

    return 0;
};
```

This same example can be found in `./example/main.cpp`.
