/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnKMeans.cpp
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

#include "drwnBase.h"
#include "drwnKMeans.h"

using namespace std;
using namespace Eigen;

// drwnKMeans class -----------------------------------------------------------

int drwnKMeans::DEFAULT_K = 2;
int drwnKMeans::MAX_ITERATIONS = DRWN_INT_MAX;

drwnKMeans::drwnKMeans(unsigned k) : drwnUnsupervisedTransform(), _numClusters(k)
{
    // do nothing
}

drwnKMeans::drwnKMeans(const drwnKMeans& ft) : drwnUnsupervisedTransform(ft),
    _numClusters(ft._numClusters), _centroids(ft._centroids), _cSqNorm(ft._cSqNorm)
{
    // do nothing
}

drwnKMeans::~drwnKMeans()
{
    // do nothing
}

// i/o
void drwnKMeans::clear()
{
    drwnUnsupervisedTransform::clear();
    _centroids = MatrixXd::Zero(0,0);
    _cSqNorm = VectorXd::Zero(0);
}

bool drwnKMeans::save(drwnXMLNode& node) const
{
    drwnUnsupervisedTransform::save(node);

    drwnAddXMLAttribute(node, "k", toString(_numClusters).c_str(), false);

    drwnXMLNode *child = drwnAddXMLChildNode(node, "centroids", NULL, false);
    drwnXMLUtils::serialize(*child, _centroids);

    return true;
}

bool drwnKMeans::load(drwnXMLNode& node)
{
    drwnUnsupervisedTransform::load(node);

    DRWN_ASSERT(drwnGetXMLAttribute(node, "k") != NULL);
    _numClusters = atoi(drwnGetXMLAttribute(node, "k"));

    drwnXMLNode *child = node.first_node("centroids");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _centroids);

    DRWN_ASSERT((_centroids.rows() == (int)_numClusters) || (_centroids.rows() == 0));
    DRWN_ASSERT((_centroids.cols() == _nFeatures) || (_centroids.cols() == 0));

    // compute centroid norms
    _cSqNorm = VectorXd::Zero(_numClusters);
    for (int k = 0; k < (int)_numClusters; k++) {
      _cSqNorm[k] = _centroids.row(k).squaredNorm();
    }

    return true;
}

// training
double drwnKMeans::train(const vector<vector<double> >& features)
{
    DRWN_ASSERT(!features.empty());
    return this->train(features, vector<double>());
}

double drwnKMeans::train(const vector<vector<double> >& features, const vector<double>& weights)
{
    DRWN_ASSERT(!features.empty() && (weights.empty() || (features.size() == weights.size())));

    // initialize centroids
    _nFeatures = features[0].size();
    if (features.size() < _numClusters) {
        DRWN_LOG_WARNING("number of training examples " << features.size()
            << " is less than number of clusters " << _numClusters);
    }

    _centroids = MatrixXd::Zero(_numClusters, _nFeatures);
    _cSqNorm = VectorXd::Zero(_numClusters);
    vector<int> index = drwn::randomPermutation(features.size());
    for (unsigned i = 0; i < std::min((size_t)_numClusters, features.size()); i++) {
        _centroids.row(i) = Eigen::Map<const VectorXd>(&features[i][0], _nFeatures).transpose();
	_cSqNorm[i] = _centroids.row(i).squaredNorm();
    }

    // compute feature vector norms
    double fSqNormSum = 0.0;
    for (unsigned i = 0; i < features.size(); i++) {
        fSqNormSum += Eigen::Map<const VectorXd>(&features[i][0], _nFeatures).squaredNorm();
    }

    // iterate until convergence
    Eigen::Map<VectorXi>(&index[0], index.size()) = VectorXi::Zero(index.size());
    VectorXd counts;
    double dTotal = 0.0;

    for (int t = 0; t < MAX_ITERATIONS; t++) {
        bool bChanged = false;
        dTotal = fSqNormSum;

        // compute assignments
        const MatrixXd twiceCentroids = 2.0 * _centroids;
        for (unsigned i = 0; i < features.size(); i++) {
            const int lastIndex = index[i];
            dTotal += (_cSqNorm - twiceCentroids * Eigen::Map<const VectorXd>(&features[i][0], _nFeatures)).minCoeff(&index[i]);
            bChanged = bChanged || (lastIndex != index[i]);
        }

        // check for convergence
        DRWN_LOG_DEBUG("k-means objective after " << t << " iterations is " << (dTotal / (double)features.size()));
        if (!bChanged) {
            DRWN_LOG_VERBOSE("k-means converged after " << t << " iterations");
            break;
        }

        // compute new centroids
        _centroids = MatrixXd::Zero(_numClusters, _nFeatures);
        counts = VectorXd::Zero(_numClusters);
        if (weights.empty()) {
            for (unsigned i = 0; i < features.size(); i++) {
                _centroids.row(index[i]) += Eigen::Map<const VectorXd>(&features[i][0], _nFeatures);
                counts[index[i]] += 1.0;
            }
        } else {
            for (unsigned i = 0; i < features.size(); i++) {
                _centroids.row(index[i]) += weights[i] * Eigen::Map<const VectorXd>(&features[i][0], _nFeatures);
                counts[index[i]] += weights[i];
            }
        }

        for (unsigned k = 0; k < _numClusters; k++) {
            if (counts[k] == 0.0) continue;
            _centroids.row(k) /= counts[k];
	    _cSqNorm[k] = _centroids.row(k).squaredNorm();
        }
    }

    _bValid = true;
    return dTotal;
}

void drwnKMeans::transform(const vector<double>& x, vector<double>& y) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

    const double fSqNorm = Eigen::Map<const VectorXd>(&x[0], x.size()).squaredNorm();

    y.resize(_centroids.rows());
    Eigen::Map<VectorXd>(&y[0], y.size()) = VectorXd::Constant(y.size(), fSqNorm) + _cSqNorm -
      2.0 * _centroids * Eigen::Map<const VectorXd>(&x[0], x.size());
}

// drwnKMeansConfig ---------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnKMeans
//! \b K :: default number of clusters\n
//! \b maxIterations :: maximum number of training iterations

class drwnKMeansConfig : public drwnConfigurableModule {
public:
    drwnKMeansConfig() : drwnConfigurableModule("drwnKMeans") { }
    ~drwnKMeansConfig() { }

    void usage(ostream &os) const {
        os << "      K             :: default number of clusters (default: " << drwnKMeans::DEFAULT_K << ")\n";
        os << "      maxIterations :: maximum number of training iterations (default: "
           << drwnKMeans::MAX_ITERATIONS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "K")) {
            drwnKMeans::DEFAULT_K = std::max(2, atoi(value));
        } else if (!strcmp(name, "maxIterations")) {
            drwnKMeans::MAX_ITERATIONS = std::max(0, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnKMeansConfig gKMeansConfig;
