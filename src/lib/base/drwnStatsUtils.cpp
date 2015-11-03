/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStatsUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(__LINUX__)
#include <sys/time.h>
#else
#include <time.h>
#endif

#include "drwnLogger.h"
#include "drwnCompatibility.h"
#include "drwnStatsUtils.h"
#include "drwnConstants.h"

using namespace std;

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

void drwnInitializeRand()
{
#if defined(__LINUX__)
    timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec * t.tv_sec);
#else
    srand((unsigned)time(NULL));
#endif
}

bool drwn::containsInvalidEntries(const vector<double> & v)
{
    DRWN_ASSERT(v.size() > 0);

    for (unsigned i = 0; i < v.size(); i++) {
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
        if (isnan(v[i]) || isinf(v[i]))
#else
		if (::isnan(v[i]) || ::isinf(v[i]))
#endif
            return true;
    }

    return false;
}

double drwn::logistic(const vector<double>& theta, const vector<double>& data)
{
    DRWN_ASSERT(theta.size() == data.size());

    double sigma;
    sigma = 0.0;
    for (unsigned i = 0; i < theta.size(); i++) {
        sigma -= theta[i] * data[i];
    }

    return 1.0 / (1.0 + exp(sigma));
}

double drwn::logistic(const double *theta, const double *data, int n)
{
    double sigma;
    sigma = 0.0;
    for (int i = 0; i < n; i++) {
        sigma -= theta[i] * data[i];
    }

    return 1.0 / (1.0 + exp(sigma));
}

// computes the entropy of a possibly unnormalized distribution
double drwn::entropy(const std::vector<double>& p)
{
    double Z = 0.0;
    double H = 0.0;

    for (unsigned i = 0; i < p.size(); i++) {
        if (p[i] > 0.0)
            H += p[i] * log(p[i]);
        Z += p[i];
    }

    return (log(Z) - H / Z) / M_LN2;
}

double drwn::entropy(const std::vector<int>& counts)
{
    int Z = 0;
    double H = 0.0;

    for (unsigned i = 0; i < counts.size(); i++) {
        if (counts[i] > 0)
            H += counts[i] * log((double)counts[i]);
        Z += counts[i];
    }

    return (log((double)Z) - H / Z) / M_LN2;
}

// computes the gini impurity of a possibly unnormalized distribution
double drwn::gini(const std::vector<double>& p)
{
    double Z = 0.0;
    double G = 0.0;

    for (unsigned i = 0; i < p.size(); i++) {
        G += p[i] * p[i];
        Z += p[i];
    }

    return 1.0 - G / (Z * Z);
}

double drwn::gini(const std::vector<int>& p)
{
    int Z = 0;
    double G = 0.0;

    for (unsigned i = 0; i < p.size(); i++) {
        G += (double)(p[i] * p[i]);
        Z += p[i];
    }

    return 1.0 - G / ((double)Z * (double)Z);
}


// exponentiates and normalizes a vector
double drwn::expAndNormalize(VectorXd& v)
{
    if (v.size() == 0) return 0.0;

    double maxValue = v[0];
    for (int i = 1; i < v.size(); i++) {
        maxValue = std::max(maxValue, v[i]);
    }

    v = (v.array() - maxValue).exp();
    double Z = v.sum();
    v /= Z;

    return log(Z) + maxValue;
}

double drwn::expAndNormalize(std::vector<double>& v)
{
    if (v.empty()) return 0.0;

    double maxValue = v[0];
    for (unsigned i = 1; i < v.size(); i++) {
        maxValue = std::max(maxValue, v[i]);
    }

    double Z = 0.0;
    for (unsigned i = 0; i < v.size(); i++) {
        v[i] = exp(v[i] - maxValue);
        Z += v[i];
    }

    double *v_ptr = &v[0];
    for (int i = (int)v.size() / 2; i != 0; i--) {
        v_ptr[0] /= Z;
        v_ptr[1] /= Z;
        v_ptr += 2;
    }
    if (v.size() % 2 != 0) {
        *v_ptr /= Z;
    }

    return log(Z) + maxValue;
}

// computes a uniform random permutation of of integers 0..(n-1)
vector<int> drwn::randomPermutation(int n)
{
    DRWN_ASSERT(n > 0);
    vector<int> v;
    int i, j, k;

    // fill vector with 0 to (n-1)
    v.resize(n);
    for (i = 0; i < n; i++) {
        v[i] = i;
    }

    // randomly swap entries
    for (i = 0; i < n - 1; i++) {
        j = rand() % (n - i);
        k = v[i]; v[i] = v[i + j]; v[i + j] = k;    // swap
    }

    return v;
}

vector<double> drwn::linSpaceVector(double startValue, double endValue, unsigned n)
{
    DRWN_ASSERT(startValue <= endValue);
    if (n == 0) return vector<double>();
    if (n == 1) return vector<double>(1, 0.5 * (startValue + endValue));

    vector<double> v(n);   
    const double delta = (endValue - startValue) / (double)(n - 1);

    v.front() = startValue;
    for (unsigned i = 1; i < n - 1; i++) {
        v[i] = startValue + (double)(i * delta);
    }
    v.back() = endValue;

    return v;
}

vector<double> drwn::logSpaceVector(double startValue, double endValue, unsigned n)
{
    DRWN_ASSERT((startValue <= endValue) && (startValue > 0.0));
    if (n == 0) return vector<double>();
    if (n == 1) return vector<double>(1, sqrt(startValue * endValue));

    vector<double> v(n);   
    const double delta = (log(endValue) - log(startValue)) / (double)(n - 1);

    v.front() = startValue;
    for (unsigned i = 1; i < n - 1; i++) {
        v[i] = startValue * exp((double)(i * delta));
    }
    v.back() = endValue;

    return v;
}

// vector successor and predecessor functions
void drwn::predecessor(std::vector<int>& array, int limit)
{
    DRWN_ASSERT(limit > 0);

    for (unsigned i = 0; i < array.size(); i++) {
        array[i] -= 1;
        if (array[i] < 0) {
            array[i] = limit - 1;
        } else {
            break;
        }
    }
}

void drwn::successor(std::vector<int>& array, int limit)
{
    for (unsigned i = 0; i < array.size(); i++) {
        array[i] += 1;
        if (array[i] >= limit) {
            array[i] = 0;
        } else {
            break;
        }
    }
}

void drwn::predecessor(std::vector<int>& array, const std::vector<int>& limits)
{
    DRWN_ASSERT(array.size() == limits.size());

    for (unsigned i = 0; i < array.size(); i++) {
        array[i] -= 1;
        if (array[i] < 0) {
            DRWN_ASSERT(limits[i] > 0);
            array[i] = limits[i] - 1;
        } else {
            break;
        }
    }
}

void drwn::successor(std::vector<int>& array, const std::vector<int>& limits)
{
    DRWN_ASSERT(array.size() == limits.size());

    for (unsigned i = 0; i < array.size(); i++) {
        array[i] += 1;
        if (array[i] >= limits[i]) {
            array[i] = 0;
        } else {
            break;
        }
    }
}

// distance metrics
double drwn::bhattacharyyaDistance(std::vector<double>& p, std::vector<double>& q)
{
    DRWN_ASSERT(p.size() == q.size());

    double d = 0.0;
    double Zp = 0.0; // normalization constant for p
    double Zq = 0.0; // normalization constant for q
    for (unsigned i = 0; i < p.size(); i++) {
	d += sqrt(p[i] * q[i]);
	Zp += p[i];
	Zq += q[i];
    }

    DRWN_ASSERT((Zp > 0.0) && (Zq > 0.0));

    return -log(d / sqrt(Zp * Zq));
}

double drwn::euclideanDistanceSq(std::vector<double>& p, std::vector<double>& q)
{
    DRWN_ASSERT(p.size() == q.size());

    double dist = 0.0;
    for (unsigned i = 0; i < p.size(); i++)
        dist += (p[i] - q[i]) * (p[i] - q[i]);
    return dist;
}

double drwn::sum(const std::vector<double> &v)
{
    double s = 0.0;
    for (unsigned i = 0; i < v.size(); i++)
        s += v[i];
    return s;
}

double drwn::sum(const double *v, size_t length)
{
    double s = 0.0;
    for (unsigned i = 0; i < length; i++)
        s += v[i];
    return s;
}

double drwn::dot(const double *x, const double *y, size_t length)
{
    double d = 0.0;
#if 0
    for (unsigned i = 0; i < length; i++)
        d += x[i] * y[i];
#else
    const unsigned nDiv4 = length / 4;
    const unsigned nMod4 = length % 4;

    for (unsigned i = nDiv4; i != 0; i--) {
        d += x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
        x += 4; y += 4;
    }

    switch (nMod4) {
    case 3:
        d += x[2] * y[2];
    case 2:
        d += x[1] * y[1];
    case 1:
        d += x[0] * y[0];
    }
#endif

    return d;
}

double drwn::dot(const vector<double>& x, const vector<double>& y)
{
    const unsigned nDiv4 = x.size() / 4;
    const unsigned nMod4 = x.size() % 4;

    double d = 0.0;
    const double *ix = &x[0];
    const double *iy = &y[0];
    for (unsigned i = nDiv4; i != 0; i--) {
        d += ix[0] * iy[0] + ix[1] * iy[1] + ix[2] * iy[2] + ix[3] * iy[3];
        ix += 4; iy += 4;
    }

    switch (nMod4) {
    case 3:
        d += ix[2] * iy[2];
    case 2:
        d += ix[1] * iy[1];
    case 1:
        d += ix[0] * iy[0];
    }

    return d;
}

int drwn::argmin(const VectorXd &v)
{
    if (v.size() == 0)
        return -1;

    int minInd = 0;
    for (int i = 1; i < (int)v.size(); i++) {
        if (v[i] < v[minInd])
            minInd = i;
    }

    return minInd;
}

int drwn::argmax(const VectorXd &v)
{
    if (v.size() == 0)
        return -1;

    int maxInd = 0;
    for (int i = 1; i < (int)v.size(); i++) {
        if (v[i] > v[maxInd])
            maxInd = i;
    }

    return maxInd;
}

int drwn::argrand(const vector<double>& v)
{
    return argrand(Eigen::Map<const VectorXd>(&v[0], v.size()));
}

int drwn::argrand(const VectorXd &v)
{
    double cutoff = v.sum() * rand() / (double)RAND_MAX;
    double cumSum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        cumSum += v[i];
        if (cumSum >= cutoff) {
            return i;
        }
    }

    DRWN_LOG_FATAL("bug");
    return -1;
}

bool drwn::eq(const double x, const double y)
{
	return fabs(x - y) < DRWN_EPSILON;
}

bool drwn::lt(const double x, const double y)
{
	return (x + DRWN_EPSILON) < y;
}
