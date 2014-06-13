/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPCA.cpp
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
#include "drwnSuffStats.h"
#include "drwnPCA.h"

using namespace std;
using namespace Eigen;

// drwnPCA class --------------------------------------------------------------

drwnPCA::drwnPCA() : drwnUnsupervisedTransform(), _numOutputDims(0), _energyThreshold(1.0),
                     _doNormalization(true)
{
    // define properties
    declareProperty("outputDim", new drwnIntegerProperty(&_numOutputDims));
    declareProperty("energyThreshold", new drwnDoubleRangeProperty(&_energyThreshold, 0.0, 1.0));
    declareProperty("normalizeOutput", new drwnBooleanProperty(&_doNormalization));
}

drwnPCA::drwnPCA(const drwnSuffStats& stats, double energyThreshold, bool doNormalization) :
    drwnUnsupervisedTransform(), _numOutputDims(0), _energyThreshold(energyThreshold),
    _doNormalization(doNormalization)
{
    // define properties
    declareProperty("outputDim", new drwnIntegerProperty(&_numOutputDims));
    declareProperty("energyThreshold", new drwnDoubleRangeProperty(&_energyThreshold, 0.0, 1.0));
    declareProperty("normalizeOutput", new drwnBooleanProperty(&_doNormalization));

    // train
    train(stats);
}

drwnPCA::drwnPCA(const drwnPCA& fw) :
    drwnUnsupervisedTransform(fw), _numOutputDims(fw._numOutputDims),
    _energyThreshold(fw._energyThreshold), _doNormalization(fw._doNormalization),
    _translation(fw._translation), _projection(fw._projection)
{
    // define properties
    declareProperty("outputDim", new drwnIntegerProperty(&_numOutputDims));
    declareProperty("energyThreshold", new drwnDoubleRangeProperty(&_energyThreshold, 0.0, 1.0));
    declareProperty("normalizeOutput", new drwnBooleanProperty(&_doNormalization));
}

drwnPCA::~drwnPCA()
{
    // do nothing
}

// i/o
void drwnPCA::clear()
{
    drwnUnsupervisedTransform::clear();
    _translation = VectorXd::Zero(0);
    _projection = MatrixXd::Zero(0,0);
}

bool drwnPCA::save(drwnXMLNode& node) const
{
    drwnUnsupervisedTransform::save(node);

    drwnAddXMLAttribute(node, "numOutputDims", toString(_numOutputDims).c_str(), false);

    drwnXMLNode *child = drwnAddXMLChildNode(node, "translation", NULL, false);
    drwnXMLUtils::serialize(*child, _translation);

    child = drwnAddXMLChildNode(node, "projection", NULL, false);
    drwnXMLUtils::serialize(*child, _projection);

    return true;
}

bool drwnPCA::load(drwnXMLNode& node)
{
    drwnUnsupervisedTransform::load(node);

    if (drwnGetXMLAttribute(node, "numOutputDims") != NULL) {
        _numOutputDims = atoi(drwnGetXMLAttribute(node, "numOutputDims"));
    } else {
        _numOutputDims = 0;
    }

    drwnXMLNode *child = node.first_node("translation");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _translation);

    child = node.first_node("projection");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _projection);

    return true;
}

// training
double drwnPCA::train(const drwnSuffStats& stats)
{
    _nFeatures = stats.size();
    _translation = stats.firstMoments() / stats.count();

    MatrixXd sigma = stats.secondMoments() / stats.count() -
        _translation * _translation.transpose();
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(sigma);

    int outputDims = (_numOutputDims <= 0) ?
        _nFeatures : std::min(_nFeatures, (int)_numOutputDims);

    // TODO: check order of eigenvalues. Is it consistent?
    DRWN_LOG_DEBUG("...drwnPCA::train() eigenvalues " << solver.eigenvalues().transpose());
    const double totalEnergy = solver.eigenvalues().cwiseMax(VectorXd::Zero(_nFeatures)).sum();
    DRWN_LOG_DEBUG("...drwnPCA::train() totalEnergy " << totalEnergy);

    if (solver.eigenvalues()[0] > solver.eigenvalues()[_nFeatures - 1]) {
        // check for negative eigenvalues
        if (solver.eigenvalues()[0] <= 0.0) {
            DRWN_LOG_WARNING("all eigenvalues are negative in \"" << type() << "\"");
        } else {
            double energy = solver.eigenvalues()[0];
            for (int i = 1; i < outputDims; i++) {
                if (energy > _energyThreshold * totalEnergy) {
                    outputDims = i;
                    break;
                }
                if (solver.eigenvalues()[i] <= 0.0) {
                    outputDims = i;
                    break;
                }
                energy += solver.eigenvalues()[i];
            }

            // construct transformation
            _projection = solver.eigenvectors().topLeftCorner(_nFeatures, outputDims).transpose();
            if (_doNormalization) {
                for (int i = 0; i < outputDims; i++) {
                    _projection.row(i) /= sqrt(solver.eigenvalues()[i]);
                }
            }
        }
    } else {
        // check for negative eigenvalues
        if (solver.eigenvalues()[_nFeatures - 1] <= 0.0) {
            DRWN_LOG_WARNING("all eigenvalues are negative in \"" << type() << "\"");
        } else {
            double energy = solver.eigenvalues()[_nFeatures - 1];
            for (int i = 1; i < outputDims; i++) {
                if (energy > _energyThreshold * totalEnergy) {
                    outputDims = i;
                    break;
                }
                if (solver.eigenvalues()[_nFeatures - i - 1] <= 0.0) {
                    outputDims = i;
                    break;
                }
                energy += solver.eigenvalues()[_nFeatures - i - 1];
            }

            // construct transformation
            _projection = solver.eigenvectors().topRightCorner(_nFeatures, outputDims).transpose();
            if (_doNormalization) {
                for (int i = 0; i < outputDims; i++) {
                    _projection.row(i) /= sqrt(solver.eigenvalues()[_nFeatures - outputDims + i]);
                }
            }
        }
    }

    _bValid = true;
    return _projection.trace();
}

double drwnPCA::train(const vector<vector<double> >& features)
{
    DRWN_ASSERT(!features.empty());
    drwnSuffStats stats(features[0].size(), DRWN_PSS_FULL);
    stats.accumulate(features, 1.0);
    return this->train(stats);
}

double drwnPCA::train(const vector<vector<double> >& features, const drwnFeatureTransform& xform)
{
    DRWN_ASSERT(!features.empty());
    drwnSuffStats stats;

    vector<double> z;
    for (unsigned i = 0; i < features.size(); i++) {
        xform.transform(features[i], z);
        if (i == 0) {
            stats.clear(z.size(), DRWN_PSS_FULL);
        }
        stats.accumulate(z, 1.0);
    }
    return this->train(stats);
}

void drwnPCA::transform(const vector<double>& x, vector<double>& y) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

    y.resize(_projection.rows());
    Eigen::Map<VectorXd>(&y[0], y.size()) = _projection *
        (Eigen::Map<const VectorXd>(&x[0], x.size()) - _translation);
}
