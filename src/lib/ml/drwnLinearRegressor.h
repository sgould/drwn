/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearRegressor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnFeatureMaps.h"
#include "drwnRegression.h"
#include "drwnOptimizer.h"

using namespace std;

//! \file

// drwnLinearRegressorBase --------------------------------------------------
//! Common functionality for drwnLinearRegressor.

class drwnLinearRegressorBase : public drwnRegression, protected drwnOptimizer {
 public:
    static double HUBER_BETA;    //!< beta parameter for huber penalty
    static double REG_STRENGTH;  //!< regularization strength
    static int MAX_ITERATIONS;   //!< maximum training iterations

 protected:
    VectorXd _theta;    //!< regression weights
    int _penalty;       //!< regression penalty option
    double _beta;       //!< huber penalty threshold
    int _regularizer;   //!< regularization option
    double _lambda;     //!< regularization strength

    // cached data for parameter estimation
    // \todo change to drwnDataset when ownership flag is implemented
    const vector<vector<double> > *_features;
    const vector<double> *_targets;
    const vector<double> *_weights;

 public:
    //! default constructor
    drwnLinearRegressorBase();
    //! construct a linear regressor for data of dimension \p n
    drwnLinearRegressorBase(unsigned n);
    //! copy constructor
    drwnLinearRegressorBase(const drwnLinearRegressorBase &r);
    ~drwnLinearRegressorBase();

    // access functions
    virtual const char *type() const { return "drwnLinearRegressor"; }

    // i/o
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // training
    using drwnRegression::train;
    virtual double train(const drwnRegressionDataset& dataset);

    // evaluation (regression)
    virtual double getRegression(const vector<double>& features) const = 0;

 protected:
    // drwnOptimizer interface
    double objective(const double *x) const;
    void gradient(const double *x, double *df) const;
    virtual double objectiveAndGradient(const double *x, double *df) const = 0;
};

// drwnTLinearRegressor -----------------------------------------------------
//! Implements linear regression optimization templated on a drwnFeatureMap.
//!
//! Parameters are learned using either L2 or Huber penalty on the prediction
//! error and can be regularized with L2 or huber penalty.
//!
//! The following code snippet shows example usage:
//! \code
//!   // load a dataset
//!   drwnRegressionDataset dataset("training.bin");
//!   const int numFeatures = dataset.numFeatures();
//!
//!   // train the regressor
//!   drwnLinearRegressor regressor(numFeatures);
//!   regressor.train(dataset);
//!
//!   // compute mean-square-error
//!   double sse = 0.0;
//!   for (size_t i = 0; i < dataset.size(); i++) {
//!       const double y = regressor.getRegression(dataset.features[i]);
//!       const double err = dataset.targets[i] - y;
//!       sse += err * err;
//!   }
//!   DRWN_LOG_MESSAGE("mse is " << sse / (double)dataset.size());
//! \endcode
//!
//! \sa drwnFeatureMap, \ref drwnFeatureMapDoc

template<class FeatureMap = drwnBiasFeatureMap>
class drwnTLinearRegressor : public drwnLinearRegressorBase {
 public:
    //! default constructor
    drwnTLinearRegressor() : drwnLinearRegressorBase() { /* do nothing */ }
    //! construct a linear regressor for data of dimension \p n
    drwnTLinearRegressor(unsigned n) :
        drwnLinearRegressorBase(n) { initialize(n); }
    //! copy constructor
    drwnTLinearRegressor(const drwnTLinearRegressor<FeatureMap> &c) :
       drwnLinearRegressorBase(c) { /* do nothing */ }

    ~drwnTLinearRegressor() { /* do nothing */ }

    // access
    virtual drwnTLinearRegressor<FeatureMap> *clone() const {
        return new drwnTLinearRegressor<FeatureMap>(*this);
    }

    // initialization
    virtual void initialize(unsigned n);

    // evaluation (regression)
    virtual double getRegression(const vector<double>& features) const;

 protected:
    virtual double objectiveAndGradient(const double *x, double *df) const;
};

// drwnLinearRegressor ------------------------------------------------------
//! \typedef drwnLinearRegressor 
//! Conveinience type declaration for linear regression with default feature
//! mapping.

typedef drwnTLinearRegressor<> drwnLinearRegressor;

// drwnTLinearRegressor implementation --------------------------------------

template<class FeatureMap>
void drwnTLinearRegressor<FeatureMap>::initialize(unsigned n)
{
    drwnRegression::initialize(n);
    const FeatureMap phi(_nFeatures);
    _theta = VectorXd::Zero(phi.numParameters());
}

template<class FeatureMap>
double drwnTLinearRegressor<FeatureMap>::getRegression(const vector<double>& features) const
{
    DRWN_ASSERT(features.size() == (unsigned)_nFeatures);

    //! \todo define feature mapping functions in terms of Eigen::VectorXd
    vector<double> t(_theta.rows());
    Eigen::Map<VectorXd>(&t[0], t.size()) = _theta;

    const FeatureMap phi(_nFeatures);
    return phi.dot(t, features);
}

template<class FeatureMap>
double drwnTLinearRegressor<FeatureMap>::objectiveAndGradient(const double *x, double *df) const
{
    // compute gradient and objective
    double obj = 0.0;

    const unsigned m = _targets->size();
    const FeatureMap phi(_nFeatures);
    const vector<double> vx(x, x + _n);
    vector<double> vdf(_n, 0.0);

    if (_penalty == 0) {
        // L2 penalty
        for (unsigned i = 0; i < m; i++) {
            double predicted = phi.dot(vx, (*_features)[i]);
            double dist =  predicted - (*_targets)[i];
            double wdist = (_weights == NULL) ? dist : dist * (*_weights)[i];
            obj += dist * wdist;
            phi.mac(vdf, (*_features)[i], wdist);
        }

        obj *= 0.5;
    } else {
        // huber penalty
        double dh;
        for (unsigned i = 0; i < m; i++) {
            double predicted = phi.dot(vx, (*_features)[i]);
            double u = predicted - (*_targets)[i];
            if (_weights == NULL) {
                obj += drwn::huberFunctionAndDerivative(u, &dh, _beta);
            } else {
                obj += (*_weights)[i] * drwn::huberFunctionAndDerivative(u, &dh, _beta);
                dh *= (*_weights)[i];
            }
            phi.mac(vdf, (*_features)[i], dh);
        }
    }

    memcpy((void *)df, (void *)&vdf[0], _n * sizeof(double));
    if (m == 0.0) return 0.0;

    obj /= (double)m;
    Eigen::Map<VectorXd>(df, _n) /= (double)m;

    // regularization
    switch (_regularizer) {
    case 0: // sum-of-squares
        {
            double weightNorm = 0.0;
            for (unsigned i = 0; i < _n; i++) {
                weightNorm += x[i] * x[i];
                df[i] += _lambda * x[i];
            }

            obj += 0.5 * _lambda * weightNorm;
        }
        break;

    case 1: // huber
        {
            double dh;
            for (unsigned i = 0; i < _n; i++) {
                obj += _lambda * drwn::huberFunctionAndDerivative(x[i], &dh, 1.0e-3);
                df[i] += _lambda * dh;
            }
        }
        break;

    default:
        DRWN_LOG_ERROR("unsupported regularizer " << _regularizer);
    }

    return obj;
}
