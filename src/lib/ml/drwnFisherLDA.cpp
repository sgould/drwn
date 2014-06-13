/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFisherLDA.cpp
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

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/QR"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "drwnBase.h"
#include "drwnFisherLDA.h"

using namespace std;
using namespace Eigen;

// drwnFisherLDA class --------------------------------------------------------

drwnFisherLDA::drwnFisherLDA() : drwnSupervisedTransform()
{
    // do nothing
}

drwnFisherLDA::drwnFisherLDA(const drwnFisherLDA& lda) :
    drwnSupervisedTransform(lda), _translation(lda._translation), _projection(lda._projection)
{
    // do nothing
}

drwnFisherLDA::~drwnFisherLDA()
{
    // do nothing
}

// i/o
void drwnFisherLDA::clear()
{
    drwnSupervisedTransform::clear();
    _translation = VectorXd::Zero(0);
    _projection = VectorXd::Zero(0);
}

bool drwnFisherLDA::save(drwnXMLNode& node) const
{
    drwnSupervisedTransform::save(node);

    drwnXMLNode *child = drwnAddXMLChildNode(node, "translation", NULL, false);
    drwnXMLUtils::serialize(*child, _translation);

    child = drwnAddXMLChildNode(node, "projection", NULL, false);
    drwnXMLUtils::serialize(*child, _projection);

    return true;
}

bool drwnFisherLDA::load(drwnXMLNode& node)
{
    drwnSupervisedTransform::load(node);

    drwnXMLNode *child = node.first_node("translation");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _translation);

    child = node.first_node("projection");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _projection);

    return true;
}

// training
double drwnFisherLDA::train(const drwnCondSuffStats& stats)
{
    _nFeatures = stats.size();
    const int numClasses = stats.states();
    const double totalCount = stats.count();
    DRWN_ASSERT(totalCount > 0.0);

    // compute projections
    if (numClasses < 2) {
        DRWN_LOG_ERROR("LDA requires more than two classes");
        _bValid = false;
        return 0.0;
    }

    if (numClasses > _nFeatures) {
        DRWN_LOG_ERROR("LDA requires more features than classes");
        _bValid = false;
        return 0.0;
    }

    // compute global mean and covariance
    _translation = VectorXd::Zero(_nFeatures);
    MatrixXd sigma = MatrixXd::Zero(_nFeatures, _nFeatures);
    for (int i = 0; i < numClasses; i++) {
        const drwnSuffStats& s = stats.suffStats(i);
        _translation += s.firstMoments();
        sigma += s.secondMoments();
    }

    _translation /= totalCount;
    sigma = sigma / totalCount - _translation * _translation.transpose();

    // between class covariance
    MatrixXd sigmaB = MatrixXd::Zero(_nFeatures, _nFeatures);
    for (int i = 0; i < numClasses; i++) {
        if (stats.count(i) > 0.0) {
            const drwnSuffStats& s = stats.suffStats(i);
            const VectorXd mu = s.firstMoments() / s.count();
            sigmaB += (_translation - mu) * (_translation - mu).transpose();
        } else {
            DRWN_LOG_WARNING("no examples of class " << i);
        }
    }
    sigmaB /= (double)numClasses;

    // TODO: regularization
    Eigen::EigenSolver<MatrixXd> solver(sigma.inverse() * sigmaB);

    // construct transformation
    if (solver.eigenvalues()[0].real() > solver.eigenvalues()[_nFeatures - 1].real()) {
        _projection = solver.eigenvectors().real().topLeftCorner(_nFeatures, numClasses - 1).transpose();
        for (int i = 0; i < numClasses - 1; i++) {
            _projection.row(i) /= sqrt(solver.eigenvalues()[i].real());
        }
    } else {
        _projection = solver.eigenvectors().real().topRightCorner(_nFeatures, numClasses - 1).transpose();
        for (int i = 0; i < numClasses - 1; i++) {
            _projection.row(i) /= sqrt(solver.eigenvalues()[_nFeatures - numClasses + 1 + i].real());
        }
    }

    DRWN_LOG_DEBUG("eigenvalues: " << solver.eigenvalues().transpose());

    _bValid = true;
    return _projection.trace();
}

double drwnFisherLDA::train(const vector<vector<double> >& features, const vector<int>& labels)
{
    DRWN_ASSERT(!features.empty() && (labels.size() == features.size()));

    drwnCondSuffStats stats(features[0].size(), drwn::maxElem(labels) + 1);
    stats.accumulate(features, labels);
    return this->train(stats);
}

double drwnFisherLDA::train(const vector<vector<double> >& features,
    const vector<int>& labels, const vector<double>& weights)
{
    DRWN_ASSERT(!features.empty());
    DRWN_ASSERT((labels.size() == features.size()) && (labels.size() == weights.size()));

    drwnCondSuffStats stats(features[0].size(), drwn::maxElem(labels) + 1);
    stats.accumulate(features, labels, weights);
    return this->train(stats);
}

double drwnFisherLDA::train(const vector<vector<double> >& features, const vector<int>& labels,
    const drwnFeatureTransform& xform)
{
    DRWN_ASSERT(!features.empty() && (labels.size() == features.size()));

    drwnCondSuffStats stats;

    vector<double> z;
    for (unsigned i = 0; i < features.size(); i++) {
        xform.transform(features[i], z);
        if (i == 0) {
            stats.clear(z.size(), drwn::maxElem(labels) + 1, DRWN_PSS_FULL);
        }
        stats.accumulate(z, labels[i], 1.0);
    }

    return this->train(stats);
}

double drwnFisherLDA::train(const vector<vector<double> >& features, const vector<int>& labels,
    const vector<double>& weights, const drwnFeatureTransform& xform)
{
    DRWN_ASSERT(!features.empty());
    DRWN_ASSERT((labels.size() == features.size()) && (labels.size() == weights.size()));

    drwnCondSuffStats stats;

    vector<double> z;
    for (unsigned i = 0; i < features.size(); i++) {
        xform.transform(features[i], z);
        if (i == 0) {
            stats.clear(z.size(), drwn::maxElem(labels) + 1, DRWN_PSS_FULL);
        }
        stats.accumulate(z, labels[i], weights[i]);
    }

    return this->train(stats);
}

void drwnFisherLDA::transform(const vector<double>& x, vector<double>& y) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

    y.resize(_projection.rows());
    Eigen::Map<VectorXd>(&y[0], y.size()) = _projection *
        (Eigen::Map<const VectorXd>(&x[0], x.size()) - _translation);
}
