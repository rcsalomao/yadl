#pragma once
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

#include <cassert>
#include <random>
#include <ranges>
#include <type_traits>

namespace yadl {

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

class Deterministic : public RV {
    double m_value;

   public:
    Deterministic(double value) : m_value(value) {}
    ~Deterministic() = default;
    void set_values(double value) { m_value = value; }
    double sample(const RNG&) override { return m_value; }
    double mean() override { return m_value; }
    double stdv() override { return 0.0; }
    double pdf(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class Normal : public RV {
    double m_mu;
    double m_sigma;

   public:
    Normal(double mu = 0.0, double sigma = 1.0) : m_mu(mu), m_sigma(sigma) {}
    ~Normal() = default;
    void set_values(double mu, double sigma) {
        m_mu = mu;
        m_sigma = sigma;
    }
    double sample(const RNG& rng) override {
        return m_mu + gsl_ran_gaussian(rng.get(), m_sigma);
    }
    double mean() override { return m_mu; }
    double stdv() override { return m_sigma; }
    double pdf(double x) override {
        return gsl_ran_gaussian_pdf((x - m_mu), m_sigma);
    }
    double cdf_P(double x) override {
        return gsl_cdf_gaussian_P((x - m_mu), m_sigma);
    }
    double cdf_P_inv(double P) override {
        return m_mu + gsl_cdf_gaussian_Pinv(P, m_sigma);
    }
};

class Lognormal : public RV {
    double m_zeta;
    double m_sigma;

   public:
    Lognormal(double zeta = 0.0, double sigma = 1.0)
        : m_zeta(zeta), m_sigma(sigma) {}
    ~Lognormal() = default;
    void set_values(double zeta, double sigma) {
        m_zeta = zeta;
        m_sigma = sigma;
    }
    double sample(const RNG& rng) override {
        return gsl_ran_lognormal(rng.get(), m_zeta, m_sigma);
    }
    double mean() override { return exp(m_zeta + pow(m_sigma, 2) / 2.0); }
    double stdv() override {
        return sqrt((exp(pow(m_sigma, 2)) - 1.0) *
                    (exp(2 * m_zeta + pow(m_sigma, 2))));
    }
    double pdf(double x) override {
        return gsl_ran_lognormal_pdf(x, m_zeta, m_sigma);
    }
    double cdf_P(double x) override {
        return gsl_cdf_lognormal_P(x, m_zeta, m_sigma);
    }
    double cdf_P_inv(double P) override {
        return gsl_cdf_lognormal_Pinv(P, m_zeta, m_sigma);
    }
};

class Gumbel : public RV {
    double m_a;
    double m_b;

   public:
    Gumbel(double a = 0.0, double b = 1.0) : m_a(a), m_b(b) {}
    ~Gumbel() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
    }
    double sample(const RNG& rng) override {
        return gsl_ran_gumbel1(rng.get(), m_a, m_b);
    }
    double mean() override {
        double euler_mascheroni = 0.577215664901532;
        return m_a + m_b * euler_mascheroni;
    }
    double stdv() override { return sqrt(pow(M_PI, 2) / 6.0 * pow(m_b, 2)); }
    double pdf(double x) override { return gsl_ran_gumbel1_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_gumbel1_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_gumbel1_Pinv(P, m_a, m_b);
    }
};

class Weibull : public RV {
    double m_a;
    double m_b;

   public:
    Weibull(double a = 1.0, double b = 1.0) : m_a(a), m_b(b) {}
    ~Weibull() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
    }
    double sample(const RNG& rng) override {
        return gsl_ran_weibull(rng.get(), m_a, m_b);
    }
    double mean() override { return m_a * gsl_sf_gamma(1.0 + 1.0 / m_b); }
    double stdv() override {
        return sqrt(pow(m_a, 2) * (gsl_sf_gamma(1.0 + 2.0 / m_b) -
                                   pow(gsl_sf_gamma(1.0 + 1.0 / m_b), 2)));
    }
    double pdf(double x) override { return gsl_ran_weibull_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_weibull_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_weibull_Pinv(P, m_a, m_b);
    }
};

class LevySaS : public RV {
    double m_mu;
    double m_c;
    double m_alpha;

   public:
    LevySaS(double mu = 0.0, double c = 1.0, double alpha = 1.0)
        : m_mu(mu), m_c(c), m_alpha(alpha) {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
    }
    ~LevySaS() = default;
    void set_values(double mu, double c, double alpha) {
        m_mu = mu;
        m_c = c;
        m_alpha = alpha;
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
    }
    double sample(const RNG& rng) override {
        return m_mu + gsl_ran_levy(rng.get(), m_c, m_alpha);
    }
    double mean() override {
        if (m_alpha > 1.0) {
            return m_mu;
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override {
        if (abs(m_c - 2.0) < 1e-8) {
            return sqrt(2.0 * pow(m_c, 2));
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    double pdf(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class LevySkew : public RV {
    double m_mu;
    double m_c;
    double m_alpha;
    double m_beta;

   public:
    LevySkew(double mu = 0.0, double c = 1.0, double alpha = 0.5,
             double beta = 1.0)
        : m_mu(mu), m_c(c), m_alpha(alpha), m_beta(beta) {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        assert((m_beta >= -1.0) && (m_beta <= 1.0));
    }
    ~LevySkew() = default;
    void set_values(double mu, double c, double alpha, double beta) {
        m_mu = mu;
        m_c = c;
        m_alpha = alpha;
        m_beta = beta;
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        assert((m_beta >= -1.0) && (m_beta <= 1.0));
    }
    double sample(const RNG& rng) override {
        return m_mu + gsl_ran_levy_skew(rng.get(), m_c, m_alpha, m_beta);
    }
    double mean() override {
        if (m_alpha > 1.0) {
            return m_mu;
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override {
        if (abs(m_c - 2.0) < 1e-8) {
            return sqrt(2.0 * pow(m_c, 2));
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    double pdf(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class UniformReal : public RV {
    double m_a;
    double m_b;

   public:
    UniformReal(double a = 0.0, double b = 1.0) : m_a(a), m_b(b) {
        assert(m_a < m_b);
    }
    ~UniformReal() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a < m_b);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_flat(rng.get(), m_a, m_b);
    }
    double mean() override { return (m_a + m_b) / 2.0; }
    double stdv() override { return sqrt(pow(m_b - m_a, 2) / 12.0); }
    double pdf(double x) override { return gsl_ran_flat_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_flat_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_flat_Pinv(P, m_a, m_b);
    }
};

class GeneralDiscrete : public RV {
    std::vector<double> m_P;
    gsl_ran_discrete_t* m_table;

   public:
    // ctor
    GeneralDiscrete(std::vector<double> P) : m_P(P) {
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    }
    // copy
    GeneralDiscrete(const GeneralDiscrete& other_rv) {
        m_P = other_rv.m_P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    };
    GeneralDiscrete& operator=(const GeneralDiscrete& other_rv) {
        gsl_ran_discrete_free(m_table);
        m_P = other_rv.m_P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
        return *this;
    };
    // move
    GeneralDiscrete(GeneralDiscrete&& other_rng) = delete;
    GeneralDiscrete& operator=(GeneralDiscrete&& other_rng) = delete;
    // dtor
    ~GeneralDiscrete() { gsl_ran_discrete_free(m_table); }
    double sample(const RNG& rng) override {
        return gsl_ran_discrete(rng.get(), m_table);
    }
    double mean() override {
        double num{0.0};
        double den{0.0};
        for (auto [i, v] : m_P | std::ranges::views::enumerate) {
            num += v * i;
            den += v;
        }
        return num / den;
    }
    double stdv() override { return std::numeric_limits<double>::quiet_NaN(); }
    double pdf(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class UniformInt : public RV {
    int m_a;
    int m_b;
    std::vector<double> m_P;
    gsl_ran_discrete_t* m_table;

   public:
    UniformInt(int a = 1, int b = 10) : m_a(a), m_b(b) {
        assert(m_a < m_b);
        int end = b + 1;
        int length = end - a;
        std::vector<double> P(length, 1.0);
        m_P = P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    }
    UniformInt(const UniformInt& other_rv) {
        m_P = other_rv.m_P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    };
    UniformInt& operator=(const UniformInt& other_rv) {
        gsl_ran_discrete_free(m_table);
        m_P = other_rv.m_P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
        return *this;
    };
    // move
    UniformInt(UniformInt&& other_rng) = delete;
    UniformInt& operator=(UniformInt&& other_rng) = delete;
    // dtor
    ~UniformInt() { gsl_ran_discrete_free(m_table); }
    double sample(const RNG& rng) override {
        return m_a + gsl_ran_discrete(rng.get(), m_table);
    }
    double mean() override { return (m_b + m_a) / 2.0; }
    double stdv() override { return sqrt((pow(m_P.size(), 2) - 1) / 12.0); }
    double pdf(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class Exponential : public RV {
    double m_mu;

   public:
    Exponential(double mu = 1.0) : m_mu(mu) { assert(m_mu > 0.0); }
    ~Exponential() = default;
    void set_values(double mu) {
        m_mu = mu;
        assert(m_mu > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_exponential(rng.get(), m_mu);
    }
    double mean() override { return m_mu; }
    double stdv() override { return m_mu; }
    double pdf(double x) override { return gsl_ran_exponential_pdf(x, m_mu); }
    double cdf_P(double x) override { return gsl_cdf_exponential_P(x, m_mu); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_exponential_Pinv(P, m_mu);
    }
};

class Gamma : public RV {
    double m_a;
    double m_b;

   public:
    Gamma(double a = 1.0, double b = 1.0) : m_a(a), m_b(b) {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    ~Gamma() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_gamma(rng.get(), m_a, m_b);
    }
    double mean() override { return m_a * m_b; }
    double stdv() override { return sqrt(m_a * pow(m_b, 2)); }
    double pdf(double x) override { return gsl_ran_gamma_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_gamma_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_gamma_Pinv(P, m_a, m_b);
    }
};

class Beta : public RV {
    double m_a;
    double m_b;

   public:
    Beta(double a = 1.0, double b = 1.0) : m_a(a), m_b(b) {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    ~Beta() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_beta(rng.get(), m_a, m_b);
    }
    double mean() override { return m_a / (m_a + m_b); }
    double stdv() override {
        return sqrt(m_a * m_b / (pow(m_a + m_b, 2) * (m_a + m_b + 1.0)));
    }
    double pdf(double x) override { return gsl_ran_beta_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_beta_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_beta_Pinv(P, m_a, m_b);
    }
};

class ScaledBeta : public RV {
    double m_a;
    double m_b;
    double m_infLimit;
    double m_supLimit;

   public:
    ScaledBeta(double a = 1.0, double b = 1.0, double infLimit = 0.0,
               double supLimit = 1.0)
        : m_a(a), m_b(b), m_infLimit(infLimit), m_supLimit(supLimit) {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        assert(m_supLimit > m_infLimit);
    }
    ~ScaledBeta() = default;
    void set_values(double a, double b, double infLimit, double supLimit) {
        m_a = a;
        m_b = b;
        m_infLimit = infLimit;
        m_supLimit = supLimit;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        assert(m_supLimit > m_infLimit);
    }
    double sample(const RNG& rng) override {
        return m_infLimit +
               (m_supLimit - m_infLimit) * gsl_ran_beta(rng.get(), m_a, m_b);
    }
    double mean() override {
        return m_infLimit + (m_supLimit - m_infLimit) * m_a / (m_a + m_b);
    }
    double stdv() override {
        return (m_supLimit - m_infLimit) *
               sqrt(m_a * m_b / (pow(m_a + m_b, 2) * (m_a + m_b + 1.0)));
    }
    double pdf(double xScaled) override {
        double x = (xScaled - m_infLimit) / (m_supLimit - m_infLimit);
        assert(x > 0.0);
        assert(x < 1.0);
        return gsl_ran_beta_pdf(x, m_a, m_b);
    }
    double cdf_P(double xScaled) override {
        double x = (xScaled - m_infLimit) / (m_supLimit - m_infLimit);
        assert(x > 0.0);
        assert(x < 1.0);
        return gsl_cdf_beta_P(x, m_a, m_b);
    }
    double cdf_P_inv(double P) override {
        return m_infLimit +
               (m_supLimit - m_infLimit) * gsl_cdf_beta_Pinv(P, m_a, m_b);
    }
};

class Poisson : public RV {
    double m_mu;

   public:
    Poisson(double mu = 1.0) : m_mu(mu) { assert(m_mu > 0.0); }
    ~Poisson() = default;
    void set_values(double mu) {
        m_mu = mu;
        assert(m_mu > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_poisson(rng.get(), m_mu);
    }
    double mean() override { return m_mu; }
    double stdv() override { return sqrt(m_mu); }
    double pdf(double x) override { return gsl_ran_poisson_pdf(x, m_mu); }
    double cdf_P(double x) override { return gsl_cdf_poisson_P(x, m_mu); }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class Bernoulli : public RV {
    double m_p;

   public:
    Bernoulli(double p = 0.5) : m_p(p) {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
    }
    ~Bernoulli() = default;
    void set_values(double p) {
        m_p = p;
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_bernoulli(rng.get(), m_p);
    }
    double mean() override { return m_p; }
    double stdv() override { return sqrt(m_p * (1.0 - m_p)); }
    double pdf(double x) override { return gsl_ran_bernoulli_pdf(x, m_p); }
    double cdf_P(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class Binomial : public RV {
    double m_p;
    int m_n;

   public:
    Binomial(double p = 0.5, int n = 1) : m_p(p), m_n(n) {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        assert(m_n >= 0);
    }
    ~Binomial() = default;
    void set_values(double p, int n) {
        m_p = p;
        m_n = n;
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        assert(m_n >= 0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_binomial(rng.get(), m_p, m_n);
    }
    double mean() override { return m_n * m_p; }
    double stdv() override { return sqrt(m_n * m_p * (1.0 - m_p)); }
    double pdf(double x) override { return gsl_ran_binomial_pdf(x, m_p, m_n); }
    double cdf_P(double x) override { return gsl_cdf_binomial_P(x, m_p, m_n); }
    double cdf_P_inv(double) override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class Pareto : public RV {
    double m_a;
    double m_b;

   public:
    Pareto(double a = 1.0, double b = 1.0) : m_a(a), m_b(b) {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    ~Pareto() = default;
    void set_values(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_pareto(rng.get(), m_a, m_b);
    }
    double mean() override {
        if (m_a <= 1.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return (m_a * m_b) / (m_a - 1.0);
        }
    }
    double stdv() override {
        if (m_a <= 2.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return sqrt((pow(m_b, 2) * m_a) /
                        (pow(m_a - 1.0, 2) * (m_a - 2.0)));
        }
    }
    double pdf(double x) override { return gsl_ran_pareto_pdf(x, m_a, m_b); }
    double cdf_P(double x) override { return gsl_cdf_pareto_P(x, m_a, m_b); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_pareto_Pinv(P, m_a, m_b);
    }
};

class ChiSquared : public RV {
    double m_nu;

   public:
    ChiSquared(double nu = 1.0) : m_nu(nu) { assert(m_nu > 0.0); }
    ~ChiSquared() = default;
    void set_values(double nu) {
        m_nu = nu;
        assert(m_nu > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_chisq(rng.get(), m_nu);
    }
    double mean() override { return m_nu; }
    double stdv() override { return sqrt(2 * m_nu); }
    double pdf(double x) override { return gsl_ran_chisq_pdf(x, m_nu); }
    double cdf_P(double x) override { return gsl_cdf_chisq_P(x, m_nu); }
    double cdf_P_inv(double P) override { return gsl_cdf_chisq_Pinv(P, m_nu); }
};

class TStudent : public RV {
    double m_nu;

   public:
    TStudent(double nu = 1.0) : m_nu(nu) { assert(m_nu > 0.0); }
    ~TStudent() = default;
    void set_values(double nu) {
        m_nu = nu;
        assert(m_nu > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_tdist(rng.get(), m_nu);
    }
    double mean() override { return 0.0; }
    double stdv() override {
        if (m_nu > 2.0) {
            return m_nu / (m_nu - 2.0);
        } else if (m_nu > 1.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double pdf(double x) override { return gsl_ran_tdist_pdf(x, m_nu); }
    double cdf_P(double x) override { return gsl_cdf_tdist_P(x, m_nu); }
    double cdf_P_inv(double P) override { return gsl_cdf_tdist_Pinv(P, m_nu); }
};

class FDistribution : public RV {
    double m_nu1;
    double m_nu2;

   public:
    FDistribution(double nu1 = 1.0, double nu2 = 1.0) : m_nu1(nu1), m_nu2(nu2) {
        assert(m_nu1 > 0.0);
        assert(m_nu2 > 0.0);
    }
    ~FDistribution() = default;
    void set_values(double nu1, double nu2) {
        m_nu1 = nu1;
        m_nu2 = nu2;
        assert(m_nu1 > 0.0);
        assert(m_nu2 > 0.0);
    }
    double sample(const RNG& rng) override {
        return gsl_ran_fdist(rng.get(), m_nu1, m_nu2);
    }
    double mean() override {
        if (m_nu1 > 2.0) {
            return (m_nu1 - 2.0) / m_nu1 * (m_nu2) / (m_nu2 + 2.0);
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override {
        if (m_nu2 > 4.0) {
            return sqrt(2.0 * pow(m_nu2, 2) * (m_nu1 + m_nu2 - 2.0) /
                        (m_nu1 * pow(m_nu2 - 2.0, 2) * (m_nu2 - 4.0)));
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double pdf(double x) override { return gsl_ran_fdist_pdf(x, m_nu1, m_nu2); }
    double cdf_P(double x) override { return gsl_cdf_fdist_P(x, m_nu1, m_nu2); }
    double cdf_P_inv(double P) override {
        return gsl_cdf_fdist_Pinv(P, m_nu1, m_nu2);
    }
};

struct Dice {
    int m_nSides;
    UniformInt m_unifInt;

    Dice(int nSides) : m_nSides{nSides}, m_unifInt{1, m_nSides} {};
    double sample(const RNG& rng) { return m_unifInt.sample(rng); }
    double mean() { return m_unifInt.mean(); }
};

class SphericalVectors {
    size_t m_nd;

   public:
    SphericalVectors(size_t nd = 2) : m_nd(nd) { assert(m_nd >= 1); }
    ~SphericalVectors() = default;
    void set_values(size_t nd) { m_nd = nd; }
    std::vector<double> sample(RNG& rng) {
        std::vector<double> v(m_nd);
        gsl_ran_dir_nd(rng.get(), m_nd, v.data());
        return v;
    }
};

template <std::ranges::random_access_range Rng>
auto shuffle(RNG& rng, Rng& seq) {
    gsl_ran_shuffle(
        rng.get(), seq.data(), seq.size(),
        sizeof(typename std::remove_reference_t<decltype(seq)>::value_type));
};

}  // namespace yadl
