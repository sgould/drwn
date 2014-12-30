/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSuffStats.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>

// drwn libraries
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnSuffStats.h"

using namespace std;

// drwnSuffStats class -------------------------------------------------------

drwnSuffStats::drwnSuffStats(int n, drwnPairSuffStatsType pairStats) :
    _n(n), _pairStats(pairStats), _count(0.0)
{
    if (_n > 0) {
        clear(_n, _pairStats);
    }
}

drwnSuffStats::drwnSuffStats(double count, VectorXd& sum, MatrixXd& sum2) :
    _pairStats(DRWN_PSS_FULL), _count(count), _sum(sum), _sum2(sum2)
{
    DRWN_ASSERT(_sum2.rows() == _sum2.cols());
    DRWN_ASSERT(_sum2.rows() == _sum.rows());
    _n = sum.rows();
}

drwnSuffStats::drwnSuffStats(const drwnSuffStats& stats) :
    _n(stats._n), _pairStats(stats._pairStats), _count(stats._count),
    _sum(stats._sum), _sum2(stats._sum2)
{
    // do nothing
}

drwnSuffStats::~drwnSuffStats()
{
    // do nothing
}

void drwnSuffStats::clear()
{
    _count = 0.0;
    _sum.setZero();
    _sum2.setZero();
}

void drwnSuffStats::clear(int n, drwnPairSuffStatsType pairStats)
{
    DRWN_ASSERT(n > 0);

    _n = n;
    _pairStats = pairStats;
    _count = 0.0;
    _sum = VectorXd::Zero(_n);
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        _sum2 = MatrixXd::Zero(_n, _n);
        break;
    case DRWN_PSS_DIAG:
        _sum2 = MatrixXd::Zero(_n, 1);
        break;
    case DRWN_PSS_NONE:
        _sum2 = MatrixXd::Zero(0, 0);
        break;
    }
}

void drwnSuffStats::diagonalize()
{
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        {
            MatrixXd t = _sum2.diagonal();
            _sum2 = t;
        }
        break;
    case DRWN_PSS_DIAG:
        // do nothing
        break;
    case DRWN_PSS_NONE:
        _sum2 = MatrixXd::Zero(_n, 1);
        break;
    }
    _pairStats = DRWN_PSS_DIAG;
}

bool drwnSuffStats::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "n", toString(_n).c_str(), false);
    drwnAddXMLAttribute(xml, "count", toString(_count).c_str(), false);

    drwnXMLNode *node = drwnAddXMLChildNode(xml, "sum", NULL, false);
    drwnXMLUtils::serialize(*node, _sum);

    node = drwnAddXMLChildNode(xml, "sumSq", NULL, false);
    drwnXMLUtils::serialize(*node, secondMoments());

    return true;
}

bool drwnSuffStats::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "n") != NULL);
    clear(atoi(drwnGetXMLAttribute(xml, "n")));

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "count") != NULL);
    _count = atof(drwnGetXMLAttribute(xml, "count"));

    drwnXMLNode *node = xml.first_node("sum");
    DRWN_ASSERT(node != NULL);
    drwnXMLUtils::deserialize(*node, _sum);
    DRWN_ASSERT(_sum.rows() == _n);

    MatrixXd tmp(_n, _n);
    node = xml.first_node("sumSq");
    DRWN_ASSERT(node != NULL);
    drwnXMLUtils::deserialize(*node, tmp);

    switch (_pairStats) {
    case DRWN_PSS_FULL:
        _sum2 = tmp;
        break;
    case DRWN_PSS_DIAG:
        _sum2 = tmp.diagonal();
        break;
    default:
        // do nothing
        break;
    }

    return true;
}

void drwnSuffStats::accumulate(const vector<double>& x)
{
    DRWN_ASSERT_MSG((int)x.size() == _n, x.size() << " != " << _n);

    _sum += Eigen::Map<const VectorXd>(&x[0], x.size());
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        //_sum2 += Eigen::Map<const VectorXd>(&x[0], x.size()) *
        //    Eigen::Map<const VectorXd>(&x[0], x.size()).transpose();
        _sum2.selfadjointView<Eigen::Lower>().rankUpdate(Eigen::Map<const VectorXd>(&x[0], x.size()));
        _sum2.triangularView<Eigen::StrictlyUpper>() = _sum2.transpose();
        break;
    case DRWN_PSS_DIAG:
        _sum2 += Eigen::Map<const ArrayXd>(&x[0], x.size(), 1).square().matrix();
        break;
    default:
        // do nothing
        break;
    }
    _count += 1.0;
}

void drwnSuffStats::accumulate(const vector<double>& x, double w)
{
    DRWN_ASSERT_MSG((int)x.size() == _n, x.size() << " != " << _n);

    const VectorXd t(w * Eigen::Map<const VectorXd>(&x[0], x.size()));
    _sum += t;
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        //_sum2 += Eigen::Map<const VectorXd>(&x[0], x.size()) * t.transpose();
        _sum2.selfadjointView<Eigen::Lower>().rankUpdate(Eigen::Map<const VectorXd>(&x[0], x.size()), w);
        _sum2.triangularView<Eigen::StrictlyUpper>() = _sum2.transpose();
        break;
    case DRWN_PSS_DIAG:
        _sum2 += w * Eigen::Map<const ArrayXd>(&x[0], x.size(), 1).square().matrix();
        break;
    default:
        // do nothing
        break;
    }
    _count += w;
}

void drwnSuffStats::accumulate(const vector<vector<double> >& x, double w)
{
    drwnSuffStats stats(_n, _pairStats);
    for (vector<vector<double> >::const_iterator it = x.begin(); it != x.end(); ++it) {
        stats.accumulate(*it);
    }
    accumulate(stats, w);
}

void drwnSuffStats::accumulate(const drwnSuffStats& stats, double w)
{
    DRWN_ASSERT(stats._n == _n);

    _count += w * stats._count;
    _sum += w * stats._sum;
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        switch (stats._pairStats) {
        case DRWN_PSS_FULL:
            _sum2 += w * stats._sum2;
            break;
        case DRWN_PSS_DIAG:
            _sum2 += w * stats._sum2.col(0).diagonal();
            break;
        default:
            // do nothing
            break;
        }
        break;
    case DRWN_PSS_DIAG:
        switch (stats._pairStats) {
        case DRWN_PSS_FULL:
            _sum2 += w * stats._sum2.diagonal();
            break;
        case DRWN_PSS_DIAG:
            _sum2 += w * stats._sum2;
            break;
        default:
            // do nothing
            break;
        }
        break;
    default:
        // do nothing
        break;
    }
}

void drwnSuffStats::subtract(const vector<double>& x)
{
    DRWN_ASSERT_MSG((int)x.size() == _n, x.size() << " != " << _n);
    DRWN_ASSERT(_count >= 1.0);

    _sum -= Eigen::Map<const VectorXd>(&x[0], x.size());
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        //_sum2 -= Eigen::Map<const VectorXd>(&x[0], x.size()) *
        //    Eigen::Map<const VectorXd>(&x[0], x.size()).transpose();
        _sum2.selfadjointView<Eigen::Lower>().rankUpdate(Eigen::Map<const VectorXd>(&x[0], x.size()), -1.0);
        _sum2.triangularView<Eigen::StrictlyUpper>() = _sum2.transpose();
        break;
    case DRWN_PSS_DIAG:
        _sum2 -= Eigen::Map<const ArrayXd>(&x[0], x.size(), 1).square().matrix();
        break;
    case DRWN_PSS_NONE:
    default:
        // do nothing
        break;
    }
    _count -= 1.0;
}

void drwnSuffStats::subtract(const vector<double>& x, double w)
{
    DRWN_ASSERT((int)x.size() == _n);
    DRWN_ASSERT(_count >= w);

    const VectorXd t(w * Eigen::Map<const VectorXd>(&x[0], x.size()));
    _sum -= t;
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        //_sum2 -= Eigen::Map<const VectorXd>(&x[0], x.size()) * t.transpose();
        _sum2.selfadjointView<Eigen::Lower>().rankUpdate(Eigen::Map<const VectorXd>(&x[0], x.size()), -1.0 * w);
        _sum2.triangularView<Eigen::StrictlyUpper>() = _sum2.transpose();
        break;
    case DRWN_PSS_DIAG:
        _sum2 -= w * Eigen::Map<const ArrayXd>(&x[0], x.size(), 1).square().matrix();
        break;
    default:
        // do nothing
        break;
    }
    _count -= w;
}

void drwnSuffStats::subtract(const vector<vector<double> >& x, double w)
{
    DRWN_ASSERT(w * x.size() <= _count);

    drwnSuffStats stats(_n, _pairStats);
    for (vector<vector<double> >::const_iterator it = x.begin(); it != x.end(); ++it) {
        stats.accumulate(*it);
    }
    subtract(stats, w);
}

void drwnSuffStats::subtract(const drwnSuffStats& stats, double w)
{
    DRWN_ASSERT(stats._n == _n);
    DRWN_ASSERT(w * stats._count <= _count);

    _count -= w * stats._count;
    _sum -= w * stats._sum;
    switch (_pairStats) {
    case DRWN_PSS_FULL:
        switch (stats._pairStats) {
        case DRWN_PSS_FULL:
            _sum2 -= w * stats._sum2;
            break;
        case DRWN_PSS_DIAG:
            _sum2 -= w * stats._sum2.col(0).diagonal();
            break;
        default:
            // do nothing
            break;
        }
        break;
    case DRWN_PSS_DIAG:
        switch (stats._pairStats) {
        case DRWN_PSS_FULL:
            _sum2 -= w * stats._sum2.diagonal();
            break;
        case DRWN_PSS_DIAG:
            _sum2 -= w * stats._sum2;
            break;
        default:
            // do nothing
            break;
        }
        break;
    default:
        // do nothing
        break;
    }
}

// standard operators
drwnSuffStats& drwnSuffStats::operator=(const drwnSuffStats& stats)
{
    _n = stats._n;
    _pairStats = stats._pairStats;
    _count = stats._count;
    _sum = stats._sum;
    _sum2 = stats._sum2;

    return *this;
}

// drwnCondSuffStats class ---------------------------------------------------

drwnCondSuffStats::drwnCondSuffStats(int n, int k, drwnPairSuffStatsType pairStats) :
    _n(n), _k(k) {
    if (_k > 0) {
        _stats.resize(_k, drwnSuffStats(_n, pairStats));
    }
}

drwnCondSuffStats::drwnCondSuffStats(const drwnCondSuffStats& condStats) :
    _n(condStats._n), _k(condStats._k), _stats(condStats._stats)
{
    // do nothing
}

drwnCondSuffStats::~drwnCondSuffStats()
{
    // do nothing
}

void drwnCondSuffStats::clear()
{
    for (int y = 0; y < _k; y++) {
        _stats[y].clear();
    }
}

void drwnCondSuffStats::clear(int n, int k, drwnPairSuffStatsType pairStats)
{
    DRWN_ASSERT((n > 0) && (k > 0));

    _n = n;
    _k = k;
    _stats.resize(_k);
    for (int y = 0; y < _k; y++) {
        _stats[y].clear(_n, pairStats);
    }
}

bool drwnCondSuffStats::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "n", toString(_n).c_str(), false);
    drwnAddXMLAttribute(xml, "k", toString(_k).c_str(), false);

    bool bSuccess = true;
    for (int i = 0; i < _k; i++) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnSuffStats", NULL, false);
        drwnAddXMLAttribute(*node, "k", toString(i).c_str(), false);
        bSuccess = bSuccess && _stats[i].save(*node);
    }

    return bSuccess;
}

bool drwnCondSuffStats::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "n") != NULL);
    int n = atoi(drwnGetXMLAttribute(xml, "n"));

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "k") != NULL);
    int k = atoi(drwnGetXMLAttribute(xml, "k"));

    DRWN_ASSERT(drwnCountXMLChildren(xml, "drwnSuffStats") == k);
    clear(n, k);

    bool bSuccess = true;
    drwnXMLNode *node = NULL;
    for (int i = 0; i < k; i++) {
        node = (i == 0) ? xml.first_node("drwnSuffStats") : node->next_sibling("drwnSuffStats");
        bSuccess = bSuccess && _stats[i].load(*node);
    }

    return bSuccess;
}

void drwnCondSuffStats::accumulate(const vector<vector<double> >& x, int y, double w)
{
    DRWN_ASSERT((y >= 0) && (y < _k));
    _stats[y].accumulate(x, w);
}

void drwnCondSuffStats::accumulate(const vector<vector<double> >& x, const vector<int>& y)
{
    DRWN_ASSERT(x.size() == y.size());
    for (int i = 0; i < (int)x.size(); i++) {
        accumulate(x[i], y[i]);
    }
}

void drwnCondSuffStats::accumulate(const vector<vector<double> >& x, const vector<int>& y,
    const vector<double>& w)
{
    DRWN_ASSERT((x.size() == y.size()) && (x.size() == w.size()));
    for (int i = 0; i < (int)x.size(); i++) {
        accumulate(x[i], y[i], w[i]);
    }
}

void drwnCondSuffStats::accumulate(const drwnSuffStats& stats, int y, double w)
{
    DRWN_ASSERT((y >= 0) && (y < _k));
    _stats[y].accumulate(stats, w);
}

void drwnCondSuffStats::subtract(const vector<vector<double> >& x, int y, double w)
{
    DRWN_ASSERT((y >= 0) && (y < _k));
    _stats[y].subtract(x, w);
}

void drwnCondSuffStats::subtract(const vector<vector<double> >& x, const vector<int>& y)
{
    DRWN_ASSERT(x.size() == y.size());
    for (int i = 0; i < (int)x.size(); i++) {
        subtract(x[i], y[i]);
    }
}

void drwnCondSuffStats::subtract(const vector<vector<double> >& x, const vector<int>& y,
    const vector<double>& w)
{
    DRWN_ASSERT((x.size() == y.size()) && (x.size() == w.size()));
    for (int i = 0; i < (int)x.size(); i++) {
        subtract(x[i], y[i], w[i]);
    }
}

void drwnCondSuffStats::subtract(const drwnSuffStats& stats, int y, double w)
{
    DRWN_ASSERT((y >= 0) && (y < _k));
    _stats[y].subtract(stats, w);
}

void drwnCondSuffStats::redistribute(int y, int k)
{
    DRWN_ASSERT((y >= 0) && (y < _k))
    DRWN_ASSERT((k >= 0) && (k < _k));

    _stats[k].accumulate(_stats[y]);
    _stats[y].clear();
}
