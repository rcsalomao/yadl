#include <iostream>
#include <memory>

#include "yadl.hpp"

using std::to_string;

template <std::ranges::forward_range Rng>
auto stringify(Rng&& seq) {
    auto b = seq | std::ranges::views::transform([](const auto& a) {
                 return to_string(a);
             }) |
             std::ranges::views::join_with(',') | std::ranges::views::common;
    return "{" + std::string(std::begin(b), std::end(b)) + "}";
};

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
        auto f = [rng](int n_side) {
            std::cout << "From f(12): " << yadl::Dice(n_side).sample(rng)
                      << std::endl;
        };
        f(12);
        std::cout << "yadl::Dice(12).sample(rng): "
                  << yadl::Dice(12).sample(rng) << std::endl;
        std::cout << "yadl::Dice(12).sample(rng): "
                  << yadl::Dice(12).sample(rng) << std::endl;
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
        printf("double samp = rv.sample(rng);\n");
        double samp = rv.sample(rng);
        printf("rv.mean(): %.2f\n", rv.mean());
        printf("rv.stdv(): %.2f\n", rv.stdv());
        printf("rv.pdf(rv.mean()): %.2f\n", rv.pdf(rv.mean()));
        printf("rv.cdf_P(33.60): %.4f\n", rv.cdf_P(33.60));
        printf("rv.cdf_P(rv.mean()): %.2f\n", rv.cdf_P(rv.mean()));
        printf("rv.cdf_P_inv(0.9999): %.2f\n", rv.cdf_P_inv(0.9999));
        printf("samp: %.3f\n", samp);
        printf("rv.pdf(): %.3f\n", rv.pdf(samp));
        printf("rv.cdf_P(samp): %.3f\n", rv.cdf_P(samp));
        printf("rv.cdf_P_inv(rv.cdf_P(samp)): %.3f\n",
               rv.cdf_P_inv(rv.cdf_P(samp)));
    }

    return 0;
};
