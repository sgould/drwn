/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStatsUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnStatsUtils.h
** \anchor drwnStatsUtils
** \brief Generic statistical utilities.
*/

#pragma once

#include <cassert>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <math.h>

#include "Eigen/Core"
using namespace Eigen;

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
#undef min
#undef max
#endif

using namespace std;

//! initialize the standard C library random number generator
//! with a time-of-day seed
void drwnInitializeRand();

namespace drwn {

    //! returns the minimum element in a vector of objects
    template <typename T>
    inline T minElem(const vector<T>& v);

    //! returns the maximum element in a vector of objects
    template <typename T>
    inline T maxElem(const vector<T>& v);

    //! returns the mean of all elements in a vector of objects
    template <typename T>
    T mean(const vector<T>& v);

    //! returns the median element in a vector of objects
    template <typename T>
    T median(const vector<T>& v);

    //! returns the median element in a vector of objects (but may
    //! modify the vector's contents)
    template <typename T>
    T destructive_median(vector<T>& w);

    //! returns the most frequent element in a vector of objects
    template <typename T>
    T mode(const vector<T>& v);

    //! returns the variance (second moment about the mean) of all elements in a vector of objects
    template <typename T>
    T variance(const vector<T>& v);

    //! returns the standard deviation of all elements in a vector of objects
    template <typename T>
    T stdev(const vector<T>& v);

    //! returns the index of the smallest element in a vector of objects
    template <typename T>
    int argmin(const vector<T>& v);

    //! returns the index of the smallest element in a vector of objects
    int argmin(const VectorXd &v);

    //! returns the index for the smallest element in each of vector of vector of objects
    template <typename T>
    vector<int> argmins(const vector<vector<T> >& v);

    //! returns the index of the largest element in a vector of objects
    template <typename T>
    int argmax(const vector<T>& v);

    //! returns the index of the largest element in a vector of objects
    int argmax(const VectorXd &v);

    //! returns the index for the largest element in each of vector of vector of objects
    template <typename T>
    vector<int> argmaxs(const vector<vector<T> >& v);

    //! returns the index for a random element sampled in proportion to the
    //! size of the element from a vector of positive entries
    int argrand(const vector<double>& v);
    //! returns the index for a random element sampled in proportion to the
    //! size of the element from a vector of positive entries
    int argrand(const VectorXd &v);

    //! returns the kurtosis for a vector of objects
    template <typename T>
    T excessKurtosis(const vector<T> &v);

    template <typename T>
    vector<float> percentiles(const vector<T> & v);

    //! returns the minimum and maximum values in a vector of objects
    template <typename T>
    pair<T, T> range(const vector<T>& v);

    //! returns the minimum and maximum values in a vector of vector of objects
    template <typename T>
    pair<T, T> range(const vector<vector<T> >& v);

    //! select an ordered subvector from a vector
    template <typename T>
    vector<T> extractSubVector(const vector<T>& v, const vector<int>& indx);

    //! removes (v.size() - keepSize)/2 minimum and maximum entries
    template <typename T>
    vector<T> removeOutliers(const vector<T>& v,
        const vector<double>& scores, int keepSize);

    //! generate powerset of a set
    template <typename T>
    set<set<T> > powerset(const set<T>& s);

    //! rounds (away from zero) to nearest discretization
    int roundUp(int n, int d);

    //! returns true if the vector contains NaN or Inf values
    bool containsInvalidEntries(const vector<double> &v);

    //! logistic function \f$y = \frac{1}{1 + \exp\left\{- \theta^T x\right\}}\f$
    double logistic(const vector<double>& theta, const vector<double>& data);
    //! logistic function \f$y = \frac{1}{1 + \exp\left\{- \theta^T x\right\}}\f$
    double logistic(const double *theta, const double *data, int n);

    //! computes the entropy of a possibly unnormalized distribution
    double entropy(const std::vector<double>& p);
    //! computes the entropy of a frequency histogram
    double entropy(const std::vector<int>& counts);

    //! computes the gini impurity of a possibly unnormalized distribution
    double gini(const std::vector<double>& p);
    //! computes the gini impurity of a frequency histogram
    double gini(const std::vector<int>& p);

    //! exponentiates and normalizes a vector in-place; returns log of the normalization constant
    double expAndNormalize(std::vector<double>& v);
    //! exponentiates and normalizes a vector in-place; returns log of the normalization constant
    double expAndNormalize(VectorXd& v);

    //! fast exponentiation
    inline double fastexp(double x);

    //! compute a random permutation of the numbers [0..n-1]
    vector<int> randomPermutation(int n);

    //! randomly permutes the entries of a vector inline
    template <typename T>
    void shuffle(vector<T>& v);

    //! extract a subsample from a vector of size \p n
    template <typename T>
    vector<T> subSample(const vector<T>& v, size_t n);

    //! generate a vector of linearly-spaced values from \p startValue to \p endValue
    vector<double> linSpaceVector(double startValue, double endValue, unsigned n = 10);
    //! generate a vector of logarithmically-spaced values from \p startValue to \p endValue
    vector<double> logSpaceVector(double startValue, double endValue, unsigned n = 10);

    //! Computes the predecessor of a discrete vector,
    //! for example, predecessor([1 0 0], 2) produces [0 0 0].
    //! Each position must have the same cardinality.
    void predecessor(std::vector<int>& array, int limit);
    //! Computes the successor of a discrete vector,
    //! for example, successor([1 0 0], 2) produces [0 1 0].
    //! Each position must have the same cardinality.
    void successor(std::vector<int>& array, int limit);
    //! Computes the predecessor of a discrete vector,
    //! for example, predecessor([1 0 0], [2 2 2]) produces [0 0 0].
    //! Each position can have different cardinality.
    void predecessor(std::vector<int>& array, const std::vector<int>& limits);
    //! Computes the successor of a discrete vector,
    //! for example, successor([1 0 0], [2 2 2]) produces [0 1 0].
    //! Each position can have different cardinality.
    void successor(std::vector<int>& array, const std::vector<int>& limits);

    //! huber penalty function, \f$y = x^2\f$ for \f$|x| \leq m\f$ and
    //! \f$\textrm{sgn}(x) m (2x - m)\f$ otherwise
    inline double huberFunction(double x, double m = 1.0);
    //! derivative of huberFunction at \b x
    inline double huberDerivative(double x, double m = 1.0);
    //! huber penalty function and derivative at \b x
    inline double huberFunctionAndDerivative(double x, double *df, double m = 1.0);

    //! Computes the Bhattacharyya distance between two discrete probability
    //! distributions. The distributions do not need to be normalized.
    double bhattacharyyaDistance(std::vector<double>& p, std::vector<double>& q);
    //! Computes the Euclidean norm between two discrete probability
    //! distributions. The distributions do not need to be normalized.
    double euclideanDistanceSq(std::vector<double>& p, std::vector<double>& q);

    //! sum the elements in a vector
    double sum(const vector<double> &v);
    //! sum the elements in a vector
    double sum(const double *v, size_t length);

    //! dot product between elements in two vectors
    double dot(const double *x, const double *y, size_t length);
    //! dot product between elements in two vectors
    double dot(const vector<double>& x, const vector<double>& y);
};

// Implementation -----------------------------------------------------------

template <typename T>
T drwn::minElem(const vector<T> & v)
{
    switch (v.size()) {
    case 0: DRWN_LOG_FATAL("invalid size"); break;
    case 1: return v.front(); break;
    case 2: return std::min(v.front(), v.back()); break;
    }

    T minObj(v.front());
    for (typename vector<T>::const_iterator i = v.begin() + 1; i != v.end(); ++i) {
        minObj = std::min(minObj, *i);
    }

    return minObj;
}

template <typename T>
T drwn::maxElem(const vector<T> & v)
{
    switch (v.size()) {
    case 0: DRWN_LOG_FATAL("invalid size"); break;
    case 1: return v.front(); break;
    case 2: return std::max(v.front(), v.back()); break;
    }

    T maxObj(v.front());
    for (typename vector<T>::const_iterator i = v.begin() + 1; i != v.end(); ++i) {
        maxObj = std::max(maxObj, *i);
    }

    return maxObj;
}

template <typename T>
T drwn::mean(const vector<T>& v)
{
    DRWN_ASSERT(v.size() > 0);

    T sum(0);

    for (typename vector<T>::const_iterator i = v.begin();  i != v.end(); ++i) {
        sum += *i;
    }

    return sum / T(v.size());
}

template <typename T>
T drwn::median(const vector<T>& v)
{
    DRWN_ASSERT(v.size() > 0);

    vector<T> w(v);
    if (w.size() % 2 == 1) {
        int ix = w.size() / 2;
        nth_element(w.begin(), w.begin()+ix, w.end());
        return w[ix];
    } else {
        // Get superior and inferior middle elements.
        int ix_sup = w.size()/2;
        nth_element(w.begin(), w.begin() + ix_sup, w.end());
        nth_element(w.begin(), w.begin() + ix_sup - 1, w.begin()+ ix_sup);
        return T(0.5 * ( w[ix_sup] + w[ix_sup-1] ));
    }
}

template <typename T>
T drwn::destructive_median(vector<T> &w)
{
    DRWN_ASSERT(w.size() > 0);
    if (w.size() % 2 == 1) {
        int ix = w.size() / 2;
        nth_element(w.begin(), w.begin()+ix, w.end());
        return w[ix];
    } else {
        // Get superior and inferior middle elements.
        int ix_sup = w.size()/2;
        nth_element(w.begin(), w.begin() + ix_sup, w.end());
        nth_element(w.begin(), w.begin() + ix_sup - 1, w.begin()+ ix_sup);
        return T(0.5 * ( w[ix_sup] + w[ix_sup-1] ));
    }
}

template <typename T>
T drwn::mode (const vector<T>& v)
{
    DRWN_ASSERT(v.size() > 0);
    map<T, int> w;
    int maxCount = -1;
    typename vector<T>::const_iterator modeElement = v.begin();
    for (typename vector<T>::const_iterator it = v.begin(); it != v.end(); it++) {
        typename map<T, int>::iterator jt = w.find(*it);
        if (jt == w.end()) {
            jt = w.insert(w.end(), make_pair(*it, 0));
        } else {
            jt->second += 1;
        }

        if (jt->second > maxCount) {
            modeElement = it;
        }
    }

    return *modeElement;
}

template <typename T>
T drwn::variance(const vector<T> & v)
{
  DRWN_ASSERT(v.size() > 0);

  T mu = mean(v);
  T sum(0);

  for (typename vector<T>::const_iterator i = v.begin(), last = v.end(); i != last; ++i) {
      double dev =  *i - mu;
      sum += dev * dev;
  }

  return sum / T(v.size());
}

template <typename T>
T drwn::stdev(const vector<T> &v)
{
    T std2 = variance(v);
    return (std2 > 0.0 ? sqrt(std2) : 0.0);
}

template <typename T>
int drwn::argmin(const vector<T> & v)
{
    int minIndx;

    switch (v.size()) {
        case 0: minIndx = -1; break;
        case 1: minIndx = 0; break;
        case 2: minIndx = (v[0] <= v[1]) ? 0 : 1; break;
        default:
        {
            minIndx = 0;
            for (int i = 1; i < (int)v.size(); i++) {
                if (v[i] < v[minIndx]) {
                    minIndx = i;
                }
            }
        }
    }

    return minIndx;
}

template <typename T>
vector<int> drwn::argmins(const vector<vector<T> >& v)
{
    vector<int> minIndx(v.size(), -1);
    for (int i = 0; i < (int)v.size(); i++) {
        minIndx[i] = argmin(v[i]);
    }

    return minIndx;
}

template <typename T>
int drwn::argmax(const vector<T> & v)
{
    int maxIndx;

    switch (v.size()) {
        case 0: maxIndx = -1; break;
        case 1: maxIndx = 0; break;
        case 2: maxIndx = (v[0] >= v[1]) ? 0 : 1; break;
        default:
        {
            maxIndx = 0;
            for (int i = 1; i < (int)v.size(); i++) {
                if (v[i] > v[maxIndx]) {
                    maxIndx = i;
                }
            }
        }
    }

    return maxIndx;
}

template <typename T>
vector<int> drwn::argmaxs(const vector<vector<T> >& v)
{
    vector<int> maxIndx(v.size(), -1);
    for (int i = 0; i < (int)v.size(); i++) {
        maxIndx[i] = argmax(v[i]);
    }

    return maxIndx;
}

template <typename T>
T drwn::excessKurtosis(const vector<T> & v)
{
  DRWN_ASSERT(!v.empty());

  T mu = mean(v);
  T sigma_squared = variance(v);

  T sum(0);
  for (typename vector<T>::const_iterator i = v.begin(), last = v.end(); i != last; ++i) {
      double dev = *i - mu;
      double sqDev = dev * dev;
      sum += sqDev * sqDev;
  }

  return sum / ( T(v.size() * sigma_squared * sigma_squared)) - 3.0;
}

template <typename T>
vector<float> drwn::percentiles(const vector<T> &v)
{
  //! \todo can change from O(n^2) to O(n log n) by using a sorting implementation
  vector<float> rval;
  for (int i = 0; i < v.size(); i++) {
      int sum = 0;
      for (int j = 0; j < v.size(); j++) {
	  if (v[j] < v[i])
              sum++;
      }
      rval.push_back(float(sum)/float(v.size()));
  }
  return rval;
}

template <typename T>
pair<T, T> drwn::range(const vector<T>& v)
{
    DRWN_ASSERT(v.size() > 0);

    typename vector<T>::const_iterator minObj(v.begin());
    typename vector<T>::const_iterator maxObj(v.begin());
    for (typename vector<T>::const_iterator i = v.begin() + 1;
         i != v.end(); ++i) {
        if (*i < *minObj) minObj = i;
        if (*i > *maxObj) maxObj = i;
    }

    return make_pair(*minObj, *maxObj);
}

template <typename T>
pair<T, T> drwn::range(const vector<vector<T> >& v)
{
    DRWN_ASSERT(v.size() > 0);

    pair<T, T> r = range(*v.begin());
    for (typename vector<vector<T> >::const_iterator i = v.begin() + 1;
         i != v.end(); ++i) {
        pair<T, T> ri = range(*i);
        if (ri.first < r.first)
            r.first = ri.first;
        if (ri.second > r.second)
            r.second = ri.second;
    }

    return r;
}

template <typename T>
vector<T> drwn::extractSubVector(const vector<T>& v, const vector<int>& indx)
{
    vector<T> w;

    w.reserve(indx.size());
    for (vector<int>::const_iterator it = indx.begin(); it != indx.end(); ++it) {
        w.push_back(v[*it]);
    }

    return w;
}

template <typename T>
vector<T> drwn::removeOutliers(const vector<T>& v,
    const vector<double>& scores, int keepSize)
{
    DRWN_ASSERT(scores.size() == v.size());
    if (keepSize >= (int)v.size()) {
        return v;
    }

    // sort scores
    vector<pair<double, int> > indx(v.size());
    for (unsigned i = 0; i < v.size(); i++) {
        indx[i] = make_pair(scores[i], i);
    }
    sort(indx.begin(), indx.end());

    vector<T> w(keepSize);
    unsigned startIndx = (v.size() - keepSize) / 2;
    unsigned endIndx = startIndx + keepSize;
    for (unsigned i = startIndx; i < endIndx; i++) {
        w[i - startIndx] = v[indx[i].second];
    }

    return w;
}

template <typename T>
set<set<T> > drwn::powerset(const set<T>& s)
{
    set<set<T> > result;

    if (s.empty()) {
        result.insert(set<T>());
    } else {
        for (typename set<T>::const_iterator it = s.begin(); it != s.end(); ++it) {
            T elem = *it;

            // copy the original set, and delete one element from it
            set<T> smallS(s);
            smallS.erase(elem);

            // compute the power set of this smaller set
            set<set<T> > smallP = powerset(smallS);
            result.insert(smallP.begin(), smallP.end());

            // add the deleted element to each member of this power set,
            // and insert each new set
            for (typename set<set<T> >::const_iterator jt = smallP.begin();
                 jt != smallP.end(); ++jt) {
                set<T> next = *jt;
                next.insert(elem);
                result.insert(next);
            }
        }
    }

    return result;
}

// fastexp -------------------------------------------------------------------
// Nicol N. Schraudolph, "A Fast, Compact Approximation of the Exponential Function",
// in Neural Computation, 11,853-862 (1999).

#define EXP_A (1048576.0 / M_LN2)
#define EXP_C 60801

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

inline double drwn::fastexp(double y)
{
    if (y < -700.0) return 0.0;

    union
    {
        double d;
#ifdef LITTLE_ENDIAN
        struct { int j, i; } n;
#else
        struct { int i, j; } n;
#endif
    } _eco;
    _eco.n.i = (int)(EXP_A * (y)) + (1072693248 - EXP_C);
    _eco.n.j = 0;
    return _eco.d;
}

template <typename T>
void drwn::shuffle(vector<T>& v)
{
    const size_t n = v.size();
    if (n < 2) return;
    for (size_t i = 0; i < n - 1; i++) {
        size_t j = rand() % (n - i);
        std::swap(v[i], v[i + j]);
    }
}

template <typename T>
vector<T> drwn::subSample(const vector<T>& v, size_t n)
{
    if (n >= v.size()) return v;
    if (n == 0) return vector<T>();

    // make a copy of v
    vector<T> w(v);

    // randomly swap entries
    for (size_t i = 0; i < n; i++) {
        size_t j = rand() % (w.size() - i);
        std::swap(w[i], w[i + j]);
    }

    // resize w to only keep first n and return
    w.resize(n);
    return w;
}

inline double drwn::huberFunction(double x, double m)
{
    if (x < -m) return (m * (-2.0 * x - m));
    if (x > m) return (m * (2.0 * x - m));

    return x * x;
}

inline double drwn::huberDerivative(double x, double m)
{
    if (x < -m) return -2.0 * m;
    if (x > m) return 2.0 * m;

    return 2.0 * x;
}

inline double drwn::huberFunctionAndDerivative(double x, double *df, double m)
{
    if (x < -m) {
	*df = -2.0 * m;
	return (m * (-2.0 * x - m));
    } else if (x > m) {
	*df = 2.0 * m;
	return (m * (2.0 * x - m));
    } else {
	*df = 2.0 * x;
	return x * x;
    }
}

inline int drwn::roundUp(int n, int d) {
    return (n % d == 0) ? n : n + d - (n % d);
}
