#include <memory>
#include "yadl.hpp"

int main() {
    std::vector<std::unique_ptr<yadl::RV>> vec;

    std::unique_ptr<yadl::RV> rv = std::make_unique<yadl::Poisson>(5.7);
    // std::unique_ptr<yadl::Normal> rv = std::make_unique<yadl::Normal>(gsl_rng_default, 5, 1);
    // std::unique_ptr<yadl::RV> rv = std::make_unique<yadl::UniformInt>();

    double samp = rv->sample();
    printf("%.2f\n", samp);
    printf("%.2f\n", rv->mean());
    printf("%f\n", rv->stdv());
    // printf("%s\n", gsl_rng_name(rv->m_rng));
    printf("%f\n", rv->pdf(rv->mean()));
    printf("%f\n", rv->pdf(samp));
    printf("%f\n", rv->cdfP(9999));

    return 0;
};
