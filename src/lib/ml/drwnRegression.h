/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRegression.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnDataset.h"

using namespace std;

// drwnRegression -----------------------------------------------------------
//! Implements the interface for a generic machine learning regression, e.g.
//! see drwnLinearRegressor.

class drwnRegression : public drwnStdObjIface, public drwnProperties {
 protected:
    int _nFeatures;     //!< number of features
    bool _bValid;       //!< true if regression parameters are trained or loaded

 public:
    //! default constructor
    drwnRegression();
    //! construct a regression object for data of dimension \p n
    drwnRegression(unsigned n);
    //! copy constructor
    drwnRegression(const drwnRegression &r);
    virtual ~drwnRegression() {
        // do nothing
    }

    // access functions
    //! return the dimensionality of the feature space
    int numFeatures() const { return _nFeatures; }
    //! return true if the regressor has valid parameters (i.e., has been trained)
    virtual bool valid() const { return _bValid; }

    // initialization
    //! initialize the regressor to accept data of dimensionality \p n
    virtual void initialize(unsigned n);

    // i/o
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // training
    //! estimate the regression parameters a drwnRegressionDataset
    virtual double train(const drwnRegressionDataset& dataset) = 0;
    //! estimate the regression parameters a set of training examples
    virtual double train(const vector<vector<double> >& features,
        const vector<double>& targets);
    //! estimate the regression parameters a set of weighted training examples
    virtual double train(const vector<vector<double> >& features,
        const vector<double>& targets, const vector<double>& weights);
    //! estimate the regression parameters from a drwnRegressionDataset file
    virtual double train(const char *filename);

    // evaluation (regression)
    //! return the estimated value for a given feature vector
    virtual double getRegression(const vector<double>& features) const = 0;
    //! compute the estimated values for a set of feature vector and return the
    //! estimates in \p outputTargets
    virtual void getRegressions(const vector<vector<double> >& features,
        vector<double>& outputTargets) const;
};
