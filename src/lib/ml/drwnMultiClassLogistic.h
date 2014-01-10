/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiClassLogistic.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnFeatureMaps.h"
#include "drwnClassifier.h"
#include "drwnOptimizer.h"

using namespace std;

//! \file

// drwnMultiClassLogisticBase -----------------------------------------------
//! Common functionality for drwnMultiClassLogistic.

class drwnMultiClassLogisticBase : public drwnClassifier, protected drwnOptimizer {
 public:
    static double REG_STRENGTH; //!< default strength of regularizer (used during construction)
    static int MAX_ITERATIONS;  //!< maximum number of training iterations

 protected:
    VectorXd _theta;            //!< joint feature map weights
    int _regularizer;           //!< regularization option
    double _lambda;             //!< regularization strength

    // cached data for parameter estimation
    // TODO: change to drwnDataset when ownership flag is implemented
    const vector<vector<double> > *_features;
    const vector<int> *_targets;
    const vector<double> *_weights;

 public:
    //! default constructor
    drwnMultiClassLogisticBase();
    //! construct a \p k-class logistic classifier for data of dimension \p n
    drwnMultiClassLogisticBase(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnMultiClassLogisticBase(const drwnMultiClassLogisticBase &c);
    ~drwnMultiClassLogisticBase();

    // access functions
    virtual const char *type() const { return "drwnMultiClassLogistic"; }

    // i/o
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // training
    using drwnClassifier::train;
    virtual double train(const drwnClassifierDataset& dataset);
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets);
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets, const vector<double>& weights);

    // evaluation (log-probability)
    using drwnClassifier::getClassScores;
    virtual void getClassScores(const vector<double>& features,
        vector<double>& outputScores) const = 0;

 protected:
    // drwnOptimizer interface
    double objective(const double *x) const;
    void gradient(const double *x, double *df) const;
    virtual double objectiveAndGradient(const double *x, double *df) const = 0;
};

// drwnTMultiClassLogistic -----------------------------------------------------
//! Implements a multi-class logistic classifier templated on a drwnJointFeatureMap.
//!
//! Parameters are learned using the negative log-likelihood loss and can be
//! regularized with L2 or huber penalty.
//!
//! \sa drwnJointFeatureMap
//! \sa \ref drwnFeatureMapDoc for example usage

template<class FeatureMap = drwnBiasJointFeatureMap>
class drwnTMultiClassLogistic : public drwnMultiClassLogisticBase {
 public:
    //! default constructor
    drwnTMultiClassLogistic() : drwnMultiClassLogisticBase() { /* do nothing */ }
    //! construct a \p k-class logistic classifier for data of dimension \p n
    drwnTMultiClassLogistic(unsigned n, unsigned k = 2) : 
        drwnMultiClassLogisticBase(n, k) { initialize(n, k); }
    //! copy constructor
    drwnTMultiClassLogistic(const drwnTMultiClassLogistic<FeatureMap> &c) :
       drwnMultiClassLogisticBase(c) { /* do nothing */ }

    ~drwnTMultiClassLogistic() { /* do nothing */ }

    // access
    virtual drwnTMultiClassLogistic<FeatureMap> *clone() const {
        return new drwnTMultiClassLogistic<FeatureMap>(*this);
    }

    // initialization
    virtual void initialize(unsigned n, unsigned k = 2);

    // evaluation (log-probability)
    using drwnMultiClassLogisticBase::getClassScores;
    virtual void getClassScores(const vector<double>& features,
        vector<double>& outputScores) const;

 protected:
    virtual double objectiveAndGradient(const double *x, double *df) const;
};

// drwnMultiClassLogistic ------------------------------------------------------
//! \typedef drwnMultiClassLogistic 
//! Conveinience type declaration for multi-class logistic classifier with
//! default feature mapping.

typedef drwnTMultiClassLogistic<> drwnMultiClassLogistic;

// drwnTMultiClassLogistic implementation --------------------------------------

template<class FeatureMap>
void drwnTMultiClassLogistic<FeatureMap>::initialize(unsigned n, unsigned k)
{
    drwnClassifier::initialize(n, k);
    const FeatureMap phi(_nFeatures, _nClasses);
    const int m = phi.numParameters();
    if (m == 0) {
        _theta = VectorXd();
    } else {
        _theta = VectorXd::Zero(phi.numParameters());
    }
}

template<class FeatureMap>
void drwnTMultiClassLogistic<FeatureMap>::getClassScores(const vector<double>& features,
    vector<double>& outputScores) const
{
    DRWN_ASSERT((int)features.size() == _nFeatures);

    //! \todo define feature mapping functions in terms of Eigen::VectorXd
    vector<double> t(_theta.rows());
    Eigen::Map<VectorXd>(&t[0], t.size()) = _theta;

    const FeatureMap phi(_nFeatures, _nClasses);
    outputScores.resize(_nClasses);
    for (int k = 0; k < _nClasses; k++) {
        outputScores[k] = phi.dot(t, features, k);
    }
}

template<class FeatureMap>
double drwnTMultiClassLogistic<FeatureMap>::objectiveAndGradient(const double *x, double *df) const
{
    double negLogL = 0.0;
    int numTerms = 0;

    const FeatureMap phi(_nFeatures, _nClasses);
    vector<double> p(_nClasses);

    const vector<double> vx(x, x + _n);
    vector<double> vdf(_n, 0.0);

    for (unsigned n = 0; n < _features->size(); n++) {
        if ((*_targets)[n] < 0) continue; // skip missing labels
        double alpha = (_weights == NULL) ? 1.0 : (*_weights)[n];

	// compute marginal for training sample
	double maxValue = 0.0;
        for (int k = 0; k < _nClasses; k++) {
            p[k] = phi.dot(vx, (*_features)[n], k);
            maxValue = std::max(maxValue, p[k]);
        }

	// exponentiate and normalize
	double Z = 0.0;
	for (vector<double>::iterator it = p.begin(); it != p.end(); ++it) {
	    Z += (*it = exp(*it - maxValue));
	}

	// increment log-likelihood
	negLogL -= alpha * log(p[(*_targets)[n]] / Z);
	numTerms += 1;

	// increment derivative
        p[(*_targets)[n]] -= Z;
        for (int k = 0; k < _nClasses; k++) {
            phi.mac(vdf, (*_features)[n], alpha * p[k] / Z, k);
        }
    }

    memcpy((void *)df, (void *)&vdf[0], _n * sizeof(double));

    if (numTerms == 0) return 0.0;
    negLogL /= (double)numTerms;
    Eigen::Map<VectorXd>(df, _n) /= (double)numTerms;

    // regularization
    switch (_regularizer) {
    case 0: // sum-of-squares
        {
            double weightNorm = 0.0;
            for (unsigned i = 0; i < _n; i++) {
                weightNorm += x[i] * x[i];
                df[i] += _lambda * x[i];
            }

            negLogL += 0.5 * _lambda * weightNorm;
        }
        break;

    case 1: // huber
        {
            double dh;
            for (unsigned i = 0; i < _n; i++) {
                negLogL += _lambda * drwn::huberFunctionAndDerivative(x[i], &dh, 1.0e-3);
                df[i] += _lambda * dh;
            }
        }
        break;

    default:
        DRWN_LOG_ERROR("unsupported regularizer " << _regularizer);
    }

    return negLogL;
}
