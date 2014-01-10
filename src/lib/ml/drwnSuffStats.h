/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFeatureWhitener.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <limits>

#include "Eigen/Core"

#include "drwnBase.h"

using namespace std;
using namespace Eigen;

// drwnPairSuffStatsType -----------------------------------------------------
//! Type of pairwise statistics to accumulate

typedef enum _drwnPairSuffStatsType {
    DRWN_PSS_FULL,             //!< full pairwise sufficient statistics (default)
    DRWN_PSS_DIAG,             //!< only diagonal pairwise terms
    DRWN_PSS_NONE,             //!< no pairwise statistics
} drwnPairSuffStatsType;

// drwnSuffStats class -------------------------------------------------------
//! Implements a class for accumulating first- and second-order sufficient
//! statistics (moments).
//!
//! When not maintaining full pairwise statistics, only keeps a vector of
//! second-order statistics, which saves a lot of space.
//!
//! \sa drwnCondSuffStats, \ref drwnTutorialML "drwnML Tutorial"

class drwnSuffStats : public drwnStdObjIface {
 private:
    int _n;                   //!< size of feature vector
    drwnPairSuffStatsType _pairStats; //!< type of pairwise statistics to accumulate

    double _count;            //!< /f$ \sum_i w_i /f$
    VectorXd _sum;            //!< /f$ \sum_i w_i x_i /f$
    MatrixXd _sum2;           //!< /f$ \sum_i w_i x_i x_i^T /f$

 public:
    //! construct a sufficient statistic object of dimensionality \p n
    drwnSuffStats(int n = 1, drwnPairSuffStatsType pairStats = DRWN_PSS_FULL);
    //! construct a sufficient statistic object with initial counts
    drwnSuffStats(double count, VectorXd& sum, MatrixXd& sum2);
    //! copy constructor
    drwnSuffStats(const drwnSuffStats& stats);
    ~drwnSuffStats();

    //! clear the accumulated counts
    void clear();
    //! clear the accumulated counts and re-initialize the dimensionality
    //! of the sufficient statistics
    void clear(int n, drwnPairSuffStatsType pairStats = DRWN_PSS_FULL);
    //! diagonalize the second-order sufficient statistics
    void diagonalize();

    //! return the dimensionality of the sufficient statistics
    inline int size() const { return _n; }
    //! return true if only accumulating diagonalized statistics (i.e., no cross terms)
    inline bool isDiagonal() const { return (_pairStats == DRWN_PSS_DIAG); }
    //! return true if accumulating both first- and second-order statistics
    inline bool hasPairs() const { return (_pairStats != DRWN_PSS_NONE); }
    //! return the weighted count of samples accumulated
    inline double count() const { return _count; }
    //! return the weighted sum of samples accumulated for a given dimension
    inline double sum(int i = 0) const { return _sum(i); }
    //! return the weighted sum-of-squares of samples accumulated a given dimension
    inline double sum2(int i = 0, int j = 0) const {
        switch (_pairStats) {
        case DRWN_PSS_FULL: return _sum2(i, j);
        case DRWN_PSS_DIAG: return (i == j) ? _sum2(i, 0) : 0.0;
        case DRWN_PSS_NONE: return 0.0;
        }
        return 0.0;
    }

    //! return the first moment
    //! return the weighted sum of samples accumulated
    inline const VectorXd& firstMoments() const { return _sum; }
    //! return the weighted sum-of-squares of samples accumulated
    inline MatrixXd secondMoments() const {
        switch (_pairStats) {
        case DRWN_PSS_FULL: return _sum2;
        case DRWN_PSS_DIAG: return _sum2.col(0).asDiagonal();
        case DRWN_PSS_NONE: return MatrixXd::Zero(_n, _n);
        }
        return _sum2;
    }

    // i/o
    const char *type() const { return "drwnSuffStats"; }
    drwnSuffStats *clone() const { return new drwnSuffStats(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // modification
    //! add a single sample to the sufficient statistics
    void accumulate(const vector<double>& x);
    //! add a single weighted sample to the sufficient statistics
    void accumulate(const vector<double>& x, double w);
    //! add a number of samples to the sufficient statistics
    void accumulate(const vector<vector<double> >& x, double w = 1.0);
    //! add previously accumulated sufficient statistics
    void accumulate(const drwnSuffStats& stats, double w = 1.0);
    //! remove a single sample from the sufficient statistics
    void subtract(const vector<double>& x);
    //! remove a single weighted sample from the sufficient statistics
    void subtract(const vector<double>& x, double w);
    //! remove a number of samples from the sufficient statistics
    void subtract(const vector<vector<double> >& x, double w = 1.0);
    //! remove previously accumulated sufficient statistics
    void subtract(const drwnSuffStats& stats, double w = 1.0);

    // standard operators
    //! assignment operator
    drwnSuffStats& operator=(const drwnSuffStats& stats);
};

// drwnCondSuffStats class --------------------------------------------------
//! Implements a class for accumulating conditional first- and second-order
//! sufficient statistics.
//!
//! Calling the function \p accumulate member function will perform the
//! following updates
//!    \li \p count = count + \f$\sum_{i : y_i = k} w_i\f$
//!    \li \p sum = sum + \f$\sum_{i : y_i = k} w_i x_i\f$
//!    \li \p sum2 = sum2 + \f$\sum_{i : y_i = k} w_i x_i x_i^T\f$
//!
//! Calling the function \p subtract member function will perform the
//! following updates
//!    \li \p count = count - \f$\sum_{i : y_i = k} w_i\f$
//!    \li \p sum = sum - \f$\sum_{i : y_i = k} w_i x_i\f$
//!    \li \p sum2 = sum2 - \f$\sum_{i : y_i = k} w_i x_i x_i^T\f$
//! The statistic \p count must remain non-negative.
//!
//! \sa drwnSuffStats, \ref drwnTutorialML "drwnML Tutorial"

class drwnCondSuffStats : public drwnStdObjIface {
 private:
    int _n;                 //!< size of feature vector
    int _k;                 //!< number of states for conditioning

    vector<drwnSuffStats> _stats;

 public:
    //! construct a sufficient statistic object of dimensionality \p n
    //! with \p k conditional states
    drwnCondSuffStats(int n = 1, int k = 2, drwnPairSuffStatsType pairStats = DRWN_PSS_FULL);
    //! copy constructor
    drwnCondSuffStats(const drwnCondSuffStats& condStats);
    ~drwnCondSuffStats();

    //! clear the conditional sufficient statistics
    void clear();
    //! clear the conditional sufficient statistics and re-initialize to dimensionality \p n
    //! with \p k conditional states
    void clear(int n, int k, drwnPairSuffStatsType pairStats = DRWN_PSS_FULL);

    //! return the number of dimensions
    inline int size() const { return _n; }
    //! return the number of conditional states
    inline int states() const { return _k; }
    //! return the total weight of samples accumulated
    inline double count() const;
    //! return the weight of samples accumulated for the \p k-th state
    inline double count(int k) const { return _stats[k].count(); }

    //! return the sufficient statistics for the \p k-th state
    inline drwnSuffStats const& suffStats(int k) const {
        return _stats[k];
    }

    // i/o
    const char *type() const { return "drwnCondSuffStats"; }
    drwnCondSuffStats *clone() const { return new drwnCondSuffStats(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // modification
    //! add a single sample to the sufficient statistics for state \p y
    inline void accumulate(const vector<double>& x, int y);
    //! add a single weighted sample to the sufficient statistics for state \p y
    inline void accumulate(const vector<double>& x, int y, double w);
    //! add a number of samples to the sufficient statistics for state \p y
    void accumulate(const vector<vector<double> >& x, int y, double w = 1.0);
    //! add a number of samples to the sufficient statistics for states \p y
    void accumulate(const vector<vector<double> >& x, const vector<int>& y);
    //! add a number of weighted samples to the sufficient statistics for states \p y
    void accumulate(const vector<vector<double> >& x, const vector<int>& y, const vector<double>& w);
    //! add previously accumulated sufficient statistics to the sufficient statistics for state \p y
    void accumulate(const drwnSuffStats& stats, int y, double w = 1.0);
    //! remove a single sample from the sufficient statistics for state \p y
    inline void subtract(const vector<double>& x, int y);
    //! remove a single weighted sample from the sufficient statistics for state \p y
    inline void subtract(const vector<double>& x, int y, double w);
    //! remove a number of samples from the sufficient statistics for state \p y
    void subtract(const vector<vector<double> >& x, int y, double w = 1.0);
    //! remove a number of samples from the sufficient statistics for states \p y
    void subtract(const vector<vector<double> >& x, const vector<int>& y);
    //! remove a number of weighted samples from the sufficient statistics for states \p y
    void subtract(const vector<vector<double> >& x, const vector<int>& y, const vector<double>& w);
    //! remove previously accumulated sufficient statistics from the sufficient statistics for state \p y
    void subtract(const drwnSuffStats& stats, int y, double w = 1.0);
    //! redistribute the counts from state \p y to state \p k
    void redistribute(int y, int k);

    // operators
    //! copy constructor
    inline const drwnSuffStats& operator[](unsigned k) const { return _stats[k]; }
};

// inline functions ---------------------------------------------------------

inline double drwnCondSuffStats::count() const {
    double c = 0;
    for (int y = 0; y < _k; y++) {
        c += _stats[y].count();
    }

    return c;
}

inline void drwnCondSuffStats::accumulate(const vector<double>& x, int y) {
    DRWN_ASSERT_MSG((y >= 0) && (y < _k), y);
    _stats[y].accumulate(x);
}

inline void drwnCondSuffStats::accumulate(const vector<double>& x, int y, double w) {
    DRWN_ASSERT_MSG((y >= 0) && (y < _k), y);
    _stats[y].accumulate(x, w);
}

inline void drwnCondSuffStats::subtract(const vector<double>& x, int y) {
    DRWN_ASSERT_MSG((y >= 0) && (y < _k), y);
    _stats[y].subtract(x);
}

inline void drwnCondSuffStats::subtract(const vector<double>& x, int y, double w) {
    DRWN_ASSERT_MSG((y >= 0) && (y < _k), y);
    _stats[y].subtract(x, w);
}
