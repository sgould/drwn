/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFeatureWhitener.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <vector>
#include <limits>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "drwnBase.h"
#include "drwnSuffStats.h"
#include "drwnFeatureWhitener.h"

using namespace std;
using namespace Eigen;

// drwnFeatureWhitener class --------------------------------------------------

drwnFeatureWhitener::drwnFeatureWhitener() : drwnUnsupervisedTransform()
{
    // do nothing
}

drwnFeatureWhitener::drwnFeatureWhitener(const drwnSuffStats& stats) :
    drwnUnsupervisedTransform()
{
    train(stats);
}

drwnFeatureWhitener::drwnFeatureWhitener(const drwnFeatureWhitener& fw) :
    drwnUnsupervisedTransform(fw), _mu(fw._mu), _beta(fw._beta)
{
    // do nothing
}

drwnFeatureWhitener::~drwnFeatureWhitener()
{
    // do nothing
}

// i/o
void drwnFeatureWhitener::clear()
{
    drwnUnsupervisedTransform::clear();
    _mu = VectorXd::Zero(0);
    _beta = VectorXd::Zero(0);
}

bool drwnFeatureWhitener::save(drwnXMLNode& node) const
{
    drwnUnsupervisedTransform::save(node);

    drwnXMLNode *child = drwnAddXMLChildNode(node, "mu", NULL, false);
    drwnXMLUtils::serialize(*child, _mu);

    child = drwnAddXMLChildNode(node, "beta", NULL, false);
    drwnXMLUtils::serialize(*child, _beta);

    return true;
}

bool drwnFeatureWhitener::load(drwnXMLNode& node)
{
    drwnUnsupervisedTransform::load(node);

    drwnXMLNode *child = node.first_node("mu");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _mu);

    child = node.first_node("beta");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _beta);

    return true;
}

// training
double drwnFeatureWhitener::train(const drwnSuffStats& stats)
{
    _nFeatures = stats.size();
    _mu = VectorXd::Zero(_nFeatures);
    _beta = VectorXd::Zero(_nFeatures);

    for (int i = 0; i < _nFeatures; i++) {
        _mu[i] = stats.sum(i) / (stats.count() + DRWN_DBL_MIN);
        double sigma = stats.sum2(i, i) / (stats.count() + DRWN_DBL_MIN) -
            _mu[i] * _mu[i];
        if (sigma < DRWN_DBL_MIN) {
            _beta[i] = (fabs(_mu[i]) > DRWN_DBL_MIN) ? (1.0 / _mu[i]) : 1.0;
            _mu[i] = 0.0;
        } else {
            _beta[i] = 1.0 / sqrt(sigma);
        }
    }

    DRWN_LOG_DEBUG("drwnFeatureWhitener::_mu = " << _mu.transpose());
    DRWN_LOG_DEBUG("drwnFeatureWhitener::_beta = " << _beta.transpose());

    _bValid = true;
    return _beta.sum();
}

double drwnFeatureWhitener::train(const vector<vector<double> >& features)
{
    DRWN_ASSERT(!features.empty());
    _nFeatures = features[0].size();

    drwnSuffStats stats(_nFeatures, DRWN_PSS_DIAG);
    stats.accumulate(features, 1.0);
    return this->train(stats);
}

// in-place evaluation
void drwnFeatureWhitener::transform(vector<double> &x) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

#if 0
    for (int i = 0; i < _nFeatures; i++) {
        x[i] = _beta[i] * (x[i] - _mu[i]);
    }
#else
    for (int i = 0; i < _nFeatures - 1; i += 2) {
        x[i] = _beta[i] * (x[i] - _mu[i]);
        x[i + 1] = _beta[i + 1] * (x[i + 1] - _mu[i + 1]);
    }

    if (_nFeatures % 2 == 1) {
        x[_nFeatures - 1] = _beta[_nFeatures - 1] * (x[_nFeatures - 1] - _mu[_nFeatures - 1]);
    }
#endif
}

void drwnFeatureWhitener::transform(const vector<double>& x, vector<double>& y) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

    y.resize(_nFeatures);
    for (int i = 0; i < _nFeatures - 1; i += 2) {
        y[i] = _beta[i] * (x[i] - _mu[i]);
        y[i + 1] = _beta[i + 1] * (x[i + 1] - _mu[i + 1]);
    }

    if (_nFeatures % 2 == 1) {
        y[_nFeatures - 1] = _beta[_nFeatures - 1] * (x[_nFeatures - 1] - _mu[_nFeatures - 1]);
    }
}
