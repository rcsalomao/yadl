// #include <iostream>
#include <memory>
#include <print>

#include "yadl.hpp"

using std::to_string;

template <std::ranges::forward_range Rng>
auto stringify(Rng&& seq) {
    auto b = seq | std::ranges::views::transform([](const auto& a) {
                 return to_string(a);
             }) |
             std::ranges::views::join_with(',') | std::ranges::views::common;
    return "[" + std::string(std::begin(b), std::end(b)) + "]";
};

int main() {
    yadl::RNG rng{};
    {
        std::println("");
        std::println("rng name: {}", gsl_rng_name(rng.get()));
    }

    {
        std::println("");
        std::vector v{1, 2, 3, 4, 5};
        std::println("v: {}", stringify(v));
        yadl::shuffle(rng, v);
        std::println("shuffled v: {}", stringify(v));
    }

    {
        std::println("");
        std::println("yadl::Dice(12).sample(rng): {}",
                     yadl::Dice(12).sample(rng));
    }

    {
        std::println("");
        std::vector<std::unique_ptr<yadl::RV>> v_rv;
        v_rv.push_back(std::make_unique<yadl::UniformInt>(23, 32));
        v_rv.push_back(std::make_unique<yadl::Poisson>(4.7));
        v_rv.push_back(std::make_unique<yadl::Normal>(5, 1));
        v_rv.push_back(std::make_unique<yadl::Weibull>());
        for (auto [i, rv] : v_rv | std::ranges::views::enumerate) {
            std::println("rv[{}]->sample(rng): {:.4f}", i, rv->sample(rng));
            std::println("rv[{}]->mean(): {:.4f}", i, rv->mean());
            std::println("rv[{}]->stdv(): {:.4f}", i, rv->stdv());
        }
    }

    {
        std::println("");
        std::println("yadl::Normal rv = yadl::Normal(15, 5);");
        yadl::Normal rv = yadl::Normal(15, 5);
        std::println("double sample = rv.sample(rng);");
        double sample = rv.sample(rng);
        std::println("rv.mean(): {:.4f}", rv.mean());
        std::println("rv.stdv(): {:.4f}", rv.stdv());
        std::println("rv.pdf(rv.mean()): {:.4f}", rv.pdf(rv.mean()));
        std::println("rv.cdf_P(33.60): {:.4f}", rv.cdf_P(33.60));
        std::println("rv.cdf_P(rv.mean()): {:.4f}", rv.cdf_P(rv.mean()));
        std::println("rv.cdf_P_inv(0.9999): {:.4f}", rv.cdf_P_inv(0.9999));
        std::println("random sample: {:.4f}", sample);
        std::println("rv.pdf(): {:.4f}", rv.pdf(sample));
        std::println("rv.cdf_P(random sample): {:.4f}", rv.cdf_P(sample));
        std::println("rv.cdf_P_inv(rv.cdf_P(random sample)): {:.4f}",
                     rv.cdf_P_inv(rv.cdf_P(sample)));
    }

    return 0;
};
