#pragma once
#include <random>
#include <cassert>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

namespace yadl
{

struct RV
{
    virtual ~RV() = default;
    virtual void setRNG(gsl_rng*) = 0;
    virtual void setSeed(unsigned long int) = 0;
    virtual double sample() = 0;
    virtual double mean() = 0;
    virtual double stdv()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    virtual double pdf(double)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    virtual double cdfP(double)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    virtual double cdfPinv(double)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

struct Deterministic : RV
{
    double m_value;

    Deterministic(double value=42.0): m_value(value) { }
    ~Deterministic() { }
    void setRNG(gsl_rng*) override { }
    void setSeed(unsigned long int) override { }
    void setValues(double value)
    {
        m_value = value;
    }
    double sample() override
    {
        return m_value;
    }
    double mean() override
    {
        return m_value;
    }
    double stdv() override
    {
        return 0.0;
    }
};

struct Normal : RV
{
    double m_mu;
    double m_sigma;
    gsl_rng* m_rng;

    Normal(const gsl_rng_type* rngType, double mu=0.0, double sigma=1.0, unsigned long int seed=std::random_device()()): m_mu(mu), m_sigma(sigma)
    {
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Normal(double mu=0.0, double sigma=1.0, unsigned long int seed=std::random_device()()): m_mu(mu), m_sigma(sigma)
    {
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Normal()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double mu, double sigma)
    {
        m_mu = mu;
        m_sigma = sigma;
    }
    double sample() override
    {
        return m_mu + gsl_ran_gaussian(m_rng, m_sigma);
    }
    double mean() override
    {
        return m_mu;
    }
    double stdv() override
    {
        return m_sigma;
    }
    double pdf(double x) override
    {
        return gsl_ran_gaussian_pdf((x-m_mu),m_sigma);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_gaussian_P((x-m_mu),m_sigma);
    }
    double cdfPinv(double P) override
    {
        return m_mu + gsl_cdf_gaussian_Pinv(P,m_sigma);
    }
};

struct Lognormal : RV
{
    double m_zeta;
    double m_sigma;
    gsl_rng* m_rng;

    Lognormal(const gsl_rng_type* rngType, double zeta=0.0, double sigma=1.0, unsigned long int seed=std::random_device()()) : m_zeta(zeta), m_sigma(sigma)
    {
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Lognormal(double zeta=0.0, double sigma=1.0, unsigned long int seed=std::random_device()()) : m_zeta(zeta), m_sigma(sigma)
    {
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Lognormal()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double zeta, double sigma)
    {
        m_zeta = zeta;
        m_sigma = sigma;
    }
    double sample() override
    {
        return gsl_ran_lognormal(m_rng, m_zeta, m_sigma);
    }
    double mean() override
    {
        return exp(m_zeta + pow(m_sigma,2)/2.0);
    }
    double stdv() override
    {
        return sqrt((exp(pow(m_sigma,2))-1.0)*(exp(2*m_zeta+pow(m_sigma,2))));
    }
    double pdf(double x) override
    {
        return gsl_ran_lognormal_pdf(x,m_zeta,m_sigma);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_lognormal_P(x,m_zeta,m_sigma);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_lognormal_Pinv(P,m_zeta,m_sigma);
    }
};

struct Gumbel : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    Gumbel(const gsl_rng_type* rngType, double a=0.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Gumbel(double a=0.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Gumbel()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b)
    {
        m_a = a;
        m_b = b;
    }
    double sample() override
    {
        return gsl_ran_gumbel1(m_rng, m_a, m_b);
    }
    double mean() override
    {
        double euler_mascheroni = 0.577215664901532;
        return m_a + m_b*euler_mascheroni;
    }
    double stdv() override
    {
        return sqrt(pow(M_PI,2)/6.0 * pow(m_b,2));
    }
    double pdf(double x) override
    {
        return gsl_ran_gumbel1_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_gumbel1_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_gumbel1_Pinv(P,m_a,m_b);
    }
};

struct Weibull : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    Weibull(const gsl_rng_type* rngType, double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Weibull(double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Weibull()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b)
    {
        m_a = a;
        m_b = b;
    }
    double sample() override
    {
        return gsl_ran_weibull(m_rng, m_a, m_b);
    }
    double mean() override
    {
        return m_a*gsl_sf_gamma(1.0 + 1.0/m_b);
    }
    double stdv() override
    {
        return sqrt(pow(m_a,2)*(gsl_sf_gamma(1.0+2.0/m_b) - pow(gsl_sf_gamma(1.0+1.0/m_b),2)));
    }
    double pdf(double x) override
    {
        return gsl_ran_weibull_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_weibull_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_weibull_Pinv(P,m_a,m_b);
    }
};

struct LevySaS : RV
{
    double m_mu;
    double m_c;
    double m_alpha;
    gsl_rng* m_rng;

    LevySaS(const gsl_rng_type* rngType, double mu=0.0, double c=1.0, double alpha=1.0, unsigned long int seed=std::random_device()()) : m_mu(mu), m_c(c), m_alpha(alpha)
    {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    LevySaS(double mu=0.0, double c=1.0, double alpha=1.0, unsigned long int seed=std::random_device()()) : m_mu(mu), m_c(c), m_alpha(alpha)
    {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~LevySaS()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double mu, double c, double alpha)
    {
        m_mu = mu;
        m_c = c;
        m_alpha = alpha;
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
    }
    double sample() override
    {
        return m_mu + gsl_ran_levy(m_rng, m_c, m_alpha);
    }
    double mean() override
    {
        if (m_alpha > 1.0) {
            return m_mu;
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override
    {
        if (abs(m_c-2.0) < 1e-8) {
            return sqrt(2.0*pow(m_c,2));
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
};

struct LevySkew : RV
{
    double m_mu;
    double m_c;
    double m_alpha;
    double m_beta;
    gsl_rng* m_rng;

    LevySkew(const gsl_rng_type* rngType, double mu=0.0, double c=1.0, double alpha=0.5, double beta=1.0, unsigned long int seed=std::random_device()()) : m_mu(mu), m_c(c), m_alpha(alpha), m_beta(beta)
    {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        assert((m_beta >= -1.0) && (m_beta <= 1.0));
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    LevySkew(double mu=0.0, double c=1.0, double alpha=0.5, double beta=1.0, unsigned long int seed=std::random_device()()) : m_mu(mu), m_c(c), m_alpha(alpha), m_beta(beta)
    {
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        assert((m_beta >= -1.0) && (m_beta <= 1.0));
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~LevySkew()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double mu, double c, double alpha, double beta)
    {
        m_mu = mu;
        m_c = c;
        m_alpha = alpha;
        m_beta = beta;
        assert((m_alpha > 0.0) && (m_alpha <= 2.0));
        assert((m_beta >= -1.0) && (m_beta <= 1.0));
    }
    double sample() override
    {
        return m_mu + gsl_ran_levy_skew(m_rng, m_c, m_alpha, m_beta);
    }
    double mean() override
    {
        if (m_alpha > 1.0) {
            return m_mu;
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override
    {
        if (abs(m_c-2.0) < 1e-8) {
            return sqrt(2.0*pow(m_c,2));
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
};

struct UniformReal : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    UniformReal(const gsl_rng_type* rngType, double a=0.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a < m_b);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    UniformReal(double a=0.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a < m_b);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~UniformReal()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b)
    {
        m_a = a;
        m_b = b;
        assert(m_a < m_b);
    }
    double sample() override
    {
        return gsl_ran_flat(m_rng, m_a, m_b);
    }
    double mean() override
    {
        return (m_a+m_b)/2.0;
    }
    double stdv() override
    {
        return sqrt(pow(m_b-m_a,2)/12.0);
    }
    double pdf(double x) override
    {
        return gsl_ran_flat_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_flat_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_flat_Pinv(P,m_a,m_b);
    }
};

struct GeneralDiscrete : RV
{
    std::vector<double> m_P;
    gsl_ran_discrete_t* m_table;
    gsl_rng* m_rng;

    GeneralDiscrete(const gsl_rng_type* rngType, std::vector<double> P, unsigned long int seed=std::random_device()()) : m_P(P)
    {
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    GeneralDiscrete(std::vector<double> P, unsigned long int seed=std::random_device()()) : m_P(P)
    {
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~GeneralDiscrete()
    {
        gsl_rng_free(m_rng);
        gsl_ran_discrete_free(m_table);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    double sample() override
    {
        return gsl_ran_discrete(m_rng, m_table);
    }
    double mean() override
    {
        double num{0.0};
        double den{0.0};
        for (size_t i=0; i<m_P.size(); i++) {
            num += m_P[i]*i;
            den += m_P[i];
        }
        return num/den;
    }
};

struct UniformInt : RV
{
    int m_a;
    int m_b;
    std::vector<double> m_P;
    gsl_ran_discrete_t* m_table;
    gsl_rng* m_rng;

    UniformInt(const gsl_rng_type* rngType, int a=0, int b=10, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a < m_b);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
        int range = b-a;
        range += 1;
        std::vector<double> P;
        P.reserve(range);
        for (int i=0; i<range; i++) {
            P.push_back(1.0);
        }
        m_P = P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    }
    UniformInt(int a=0, int b=10, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a < m_b);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
        int range = b-a;
        range += 1;
        std::vector<double> P;
        P.reserve(range);
        for (int i=0; i<range; i++) {
            P.push_back(1.0);
        }
        m_P = P;
        m_table = gsl_ran_discrete_preproc(size(m_P), m_P.data());
    }
    ~UniformInt()
    {
        gsl_rng_free(m_rng);
        gsl_ran_discrete_free(m_table);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    double sample() override
    {
        return m_a + gsl_ran_discrete(m_rng, m_table);
    }
    double mean() override
    {
        return (m_b-m_a)/2.0;
    }
};

struct Exponential : RV
{
    double m_mu;
    gsl_rng* m_rng;

    Exponential(const gsl_rng_type* rngType, double mu=1.0, unsigned long int seed=std::random_device()()): m_mu(mu)
    {
        assert(m_mu > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Exponential(double mu=1.0, unsigned long int seed=std::random_device()()): m_mu(mu)
    {
        assert(m_mu > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Exponential()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double mu)
    {
        m_mu = mu;
        assert(m_mu > 0.0);
    }
    double sample() override
    {
        return gsl_ran_exponential(m_rng, m_mu);
    }
    double mean() override
    {
        return m_mu;
    }
    double stdv() override
    {
        return m_mu;
    }
    double pdf(double x) override
    {
        return gsl_ran_exponential_pdf(x,m_mu);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_exponential_P(x,m_mu);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_exponential_Pinv(P,m_mu);
    }
};

struct Gamma : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    Gamma(const gsl_rng_type* rngType, double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Gamma(double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Gamma()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample() override
    {
        return gsl_ran_gamma(m_rng, m_a, m_b);
    }
    double mean() override
    {
        return m_a*m_b;
    }
    double stdv() override
    {
        return sqrt(m_a*pow(m_b,2));
    }
    double pdf(double x) override
    {
        return gsl_ran_gamma_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_gamma_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_gamma_Pinv(P,m_a,m_b);
    }
};

struct Beta : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    Beta(const gsl_rng_type* rngType, double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Beta(double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Beta()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b) {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample() override
    {
        return gsl_ran_beta(m_rng, m_a, m_b);
    }
    double mean() override
    {
        return m_a/(m_a+m_b);
    }
    double stdv() override
    {
        return sqrt(m_a*m_b/(pow(m_a+m_b,2)*(m_a+m_b+1.0)));
    }
    double pdf(double x) override
    {
        return gsl_ran_beta_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_beta_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_beta_Pinv(P,m_a,m_b);
    }
};

struct ScaledBeta : RV
{
    double m_a;
    double m_b;
    double m_infLimit;
    double m_supLimit;
    gsl_rng* m_rng;

    ScaledBeta(const gsl_rng_type* rngType, double a=1.0, double b=1.0, double infLimit=0.0, double supLimit=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b), m_infLimit(infLimit), m_supLimit(supLimit)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        assert(m_supLimit > m_infLimit);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    ScaledBeta(double a=1.0, double b=1.0, double infLimit=0.0, double supLimit=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b), m_infLimit(infLimit), m_supLimit(supLimit)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        assert(m_supLimit > m_infLimit);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~ScaledBeta()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b, double infLimit, double supLimit)
    {
        m_a = a;
        m_b = b;
        m_infLimit = infLimit;
        m_supLimit = supLimit;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        assert(m_supLimit > m_infLimit);
    }
    double sample() override
    {
        return m_infLimit + (m_supLimit-m_infLimit)*gsl_ran_beta(m_rng, m_a, m_b);
    }
    double mean() override
    {
        return m_infLimit + (m_supLimit-m_infLimit)*m_a/(m_a+m_b);
    }
    double stdv() override
    {
        return (m_supLimit-m_infLimit)*sqrt(m_a*m_b/(pow(m_a+m_b,2)*(m_a+m_b+1.0)));
    }
    double pdf(double xScaled) override
    {
        double x = (xScaled-m_infLimit)/(m_supLimit-m_infLimit);
        assert(x>0.0);
        assert(x<1.0);
        return gsl_ran_beta_pdf(x,m_a,m_b);
    }
    double cdfP(double xScaled) override
    {
        double x = (xScaled-m_infLimit)/(m_supLimit-m_infLimit);
        assert(x>0.0);
        assert(x<1.0);
        return gsl_cdf_beta_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return m_infLimit + (m_supLimit-m_infLimit)*gsl_cdf_beta_Pinv(P,m_a,m_b);
    }
};

struct Poisson : RV
{
    double m_mu;
    gsl_rng* m_rng;

    Poisson(const gsl_rng_type* rngType, double mu=1.0, unsigned long int seed=std::random_device()()): m_mu(mu)
    {
        assert(m_mu > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Poisson(double mu=1.0, unsigned long int seed=std::random_device()()): m_mu(mu)
    {
        assert(m_mu > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Poisson()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double mu)
    {
        m_mu = mu;
        assert(m_mu > 0.0);
    }
    double sample() override
    {
        return gsl_ran_poisson(m_rng, m_mu);
    }
    double mean() override
    {
        return m_mu;
    }
    double stdv() override
    {
        return sqrt(m_mu);
    }
    double pdf(double x) override
    {
        return gsl_ran_poisson_pdf(x,m_mu);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_poisson_P(x,m_mu);
    }
};

struct Bernoulli : RV
{
    double m_p;
    gsl_rng* m_rng;

    Bernoulli(const gsl_rng_type* rngType, double p=0.5, unsigned long int seed=std::random_device()()): m_p(p)
    {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Bernoulli(double p=0.5, unsigned long int seed=std::random_device()()): m_p(p)
    {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Bernoulli()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double p)
    {
        m_p = p;
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
    }
    double sample() override
    {
        return gsl_ran_bernoulli(m_rng, m_p);
    }
    double mean() override
    {
        return m_p;
    }
    double stdv() override
    {
        return sqrt(m_p*(1.0-m_p));
    }
    double pdf(double x) override
    {
        return gsl_ran_bernoulli_pdf(x,m_p);
    }
};

struct Binomial : RV
{
    double m_p;
    int m_n;
    gsl_rng* m_rng;

    Binomial(const gsl_rng_type* rngType, double p=0.5, int n=1, unsigned long int seed=std::random_device()()): m_p(p), m_n(n)
    {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        assert(m_n >= 0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Binomial(double p=0.5, int n=1, unsigned long int seed=std::random_device()()): m_p(p), m_n(n)
    {
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        assert(m_n >= 0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Binomial()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double p, int n)
    {
        m_p = p;
        m_n = n;
        assert(m_p >= 0.0);
        assert(m_p <= 1.0);
        assert(m_n >= 0);
    }
    double sample() override
    {
        return gsl_ran_binomial(m_rng,m_p,m_n);
    }
    double mean() override
    {
        return m_n*m_p;
    }
    double stdv() override
    {
        return sqrt(m_n*m_p*(1.0-m_p));
    }
    double pdf(double x) override
    {
        return gsl_ran_binomial_pdf(x,m_p,m_n);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_binomial_P(x,m_p,m_n);
    }
};

struct Pareto : RV
{
    double m_a;
    double m_b;
    gsl_rng* m_rng;

    Pareto(const gsl_rng_type* rngType, double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    Pareto(double a=1.0, double b=1.0, unsigned long int seed=std::random_device()()): m_a(a), m_b(b)
    {
        assert(m_a > 0.0);
        assert(m_b > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~Pareto()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double a, double b)
    {
        m_a = a;
        m_b = b;
        assert(m_a > 0.0);
        assert(m_b > 0.0);
    }
    double sample() override
    {
        return gsl_ran_pareto(m_rng,m_a,m_b);
    }
    double mean() override
    {
        if (m_a <= 1.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return (m_a*m_b)/(m_a-1.0);
        }
    }
    double stdv() override
    {
        if (m_a <= 2.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return sqrt(
                (pow(m_b,2)*m_a)/(pow(m_a-1.0,2)*(m_a-2.0))
            );
        }
    }
    double pdf(double x) override
    {
        return gsl_ran_pareto_pdf(x,m_a,m_b);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_pareto_P(x,m_a,m_b);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_pareto_Pinv(P,m_a,m_b);
    }
};

struct ChiSquared : RV
{
    double m_nu;
    gsl_rng* m_rng;

    ChiSquared(const gsl_rng_type* rngType, double nu=1.0, unsigned long int seed=std::random_device()()): m_nu(nu)
    {
        assert(m_nu > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    ChiSquared(double nu=1.0, unsigned long int seed=std::random_device()()): m_nu(nu)
    {
        assert(m_nu > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~ChiSquared()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double nu)
    {
        m_nu = nu;
        assert(m_nu > 0.0);
    }
    double sample() override
    {
        return gsl_ran_chisq(m_rng,m_nu);
    }
    double mean() override
    {
        return m_nu;
    }
    double stdv() override
    {
        return sqrt(2*m_nu);
    }
    double pdf(double x) override
    {
        return gsl_ran_chisq_pdf(x,m_nu);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_chisq_P(x,m_nu);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_chisq_Pinv(P,m_nu);
    }
};

struct TStudent : RV
{
    double m_nu;
    gsl_rng* m_rng;

    TStudent(const gsl_rng_type* rngType, double nu=1.0, unsigned long int seed=std::random_device()()): m_nu(nu)
    {
        assert(m_nu > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    TStudent(double nu=1.0, unsigned long int seed=std::random_device()()): m_nu(nu)
    {
        assert(m_nu > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~TStudent()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double nu)
    {
        m_nu = nu;
        assert(m_nu > 0.0);
    }
    double sample() override
    {
        return gsl_ran_tdist(m_rng,m_nu);
    }
    double mean() override
    {
        return 0.0;
    }
    double stdv() override
    {
        if (m_nu > 2.0) {
            return m_nu/(m_nu-2.0);
        } else if (m_nu > 1.0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double pdf(double x) override
    {
        return gsl_ran_tdist_pdf(x,m_nu);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_tdist_P(x,m_nu);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_tdist_Pinv(P,m_nu);
    }
};

struct FDistribution : RV
{
    double m_nu1;
    double m_nu2;
    gsl_rng* m_rng;

    FDistribution(const gsl_rng_type* rngType, double nu1=1.0, double nu2=1.0, unsigned long int seed=std::random_device()()): m_nu1(nu1), m_nu2(nu2)
    {
        assert(m_nu1 > 0.0);
        assert(m_nu2 > 0.0);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    FDistribution(double nu1=1.0, double nu2=1.0, unsigned long int seed=std::random_device()()): m_nu1(nu1), m_nu2(nu2)
    {
        assert(m_nu1 > 0.0);
        assert(m_nu2 > 0.0);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~FDistribution()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) override
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed) override
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(double nu1, double nu2)
    {
        m_nu1 = nu1;
        m_nu2 = nu2;
        assert(m_nu1 > 0.0);
        assert(m_nu2 > 0.0);
    }
    double sample() override
    {
        return gsl_ran_fdist(m_rng,m_nu1,m_nu2);
    }
    double mean() override
    {
        if (m_nu1 > 2.0) {
            return (m_nu1-2.0)/m_nu1*(m_nu2)/(m_nu2+2.0);
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double stdv() override
    {
        if (m_nu2 > 4.0) {
            return sqrt(
                2.0*pow(m_nu2,2)*(m_nu1+m_nu2-2.0)/(m_nu1*pow(m_nu2-2.0,2)*(m_nu2-4.0))
            );
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    double pdf(double x) override
    {
        return gsl_ran_fdist_pdf(x,m_nu1,m_nu2);
    }
    double cdfP(double x) override
    {
        return gsl_cdf_fdist_P(x,m_nu1,m_nu2);
    }
    double cdfPinv(double P) override
    {
        return gsl_cdf_fdist_Pinv(P,m_nu1,m_nu2);
    }
};

struct Shuffler
{
    gsl_rng* m_rng;

    Shuffler(const gsl_rng_type* rngType=gsl_rng_ranlxd2, unsigned long int seed=std::random_device()())
    {
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    ~Shuffler()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG)
    {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed)
    {
        gsl_rng_set(m_rng, seed);
    }

    template<typename Vector>
    void shuffle(Vector& vec) {
        gsl_ran_shuffle(m_rng,vec.data(),vec.size(),sizeof(vec[0]));
    }
};

struct Dice
{
    int m_nSides;
    UniformInt m_unifInt;

    Dice(const gsl_rng_type* rngType, int nSides, unsigned long int seed=std::random_device()()): m_nSides(nSides)
    {
        m_unifInt = UniformInt(rngType,1,m_nSides,seed);
    }
    Dice(int nSides): m_nSides(nSides)
    {
        m_unifInt = UniformInt(1,m_nSides);
    }
    double sample()
    {
        return m_unifInt.sample();
    }
    double mean()
    {
        return m_unifInt.mean();
    }
};

struct SphericalVectors
{
    size_t m_nd;
    gsl_rng* m_rng;

    SphericalVectors(const gsl_rng_type* rngType, size_t nd=2, unsigned long int seed=std::random_device()()): m_nd(nd)
    {
        assert(m_nd >= 1);
        m_rng = gsl_rng_alloc(rngType);
        gsl_rng_set(m_rng,seed);
    }
    SphericalVectors(size_t nd=2, unsigned long int seed=std::random_device()()): m_nd(nd)
    {
        assert(m_nd >= 1);
        m_rng = gsl_rng_alloc(gsl_rng_ranlxd2);
        gsl_rng_set(m_rng,seed);
    }
    ~SphericalVectors()
    {
        gsl_rng_free(m_rng);
    }
    void setRNG(gsl_rng* newRNG) {
        gsl_rng_free(m_rng);
        m_rng = newRNG;
    }
    void setSeed(unsigned long int seed)
    {
        gsl_rng_set(m_rng, seed);
    }
    void setValues(size_t nd) {
        m_nd = nd;
    }
    std::vector<double> sample()
    {
        std::vector<double> v(m_nd);
        gsl_ran_dir_nd(m_rng, m_nd, v.data());
        return v;
    }
};

} // ns yadl
