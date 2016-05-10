/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
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
#include "drwnFeatureTransform.h"
#include "drwnSuffStats.h"

using namespace std;
using namespace Eigen;

// drwnFeatureWhitener class --------------------------------------------------
//! Whitens (zero mean, unit variance) feature vector (see also \ref drwnPCA).
//!
//! Each features is whitened independently by subtracting the sample mean and
//! dividing by sample standard-deviation. Constant features (i.e., those with
//! near zero-variance) are not whitened.
//!
//! The following code snippet demonstrates example usage:
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training.bin");
//!
//!   // whiten features
//!   drwnFeatureWhitener whitener;
//!   whitener.train(dataset.features);
//!   whitener.transform(dataset.features);
//!
//!   // save whitened features
//!   dataset.write("training.bin");
//!
//!   // apply same whitening to test dataset
//!   dataset.clear();
//!   dataset.read("testing.bin");
//!   whitener.transform(dataset.features);
//!   dataset.write("testing.bin");
//! \endcode

class drwnFeatureWhitener : public drwnUnsupervisedTransform {
 private:
    VectorXd _mu;       //!< mean vector
    VectorXd _beta;     //!< 1.0 / standard deviation

 public:
    //! default constructor
    drwnFeatureWhitener();
    //! construct a feature whitener from sufficient statistics
    drwnFeatureWhitener(const drwnSuffStats& stats);
    //! copy constructor
    drwnFeatureWhitener(const drwnFeatureWhitener& fw);
    ~drwnFeatureWhitener();

    // access functions
    const char *type() const { return "drwnFeatureWhitener"; }
    drwnFeatureWhitener *clone() const { return new drwnFeatureWhitener(*this); }

    // i/o
    void clear();
    bool save(drwnXMLNode& node) const;
    bool load(drwnXMLNode& node);

    // training
    using drwnUnsupervisedTransform::train;
    //! Estimate the parameters of the feature transformation from
    //! previously accumulated second-order sufficient statistics.
    double train(const drwnSuffStats& stats);
    double train(const vector<vector<double> >& features);

    // evaluation
    using drwnFeatureTransform::transform;
    void transform(vector<double>& x) const;
    void transform(const vector<double>& x, vector<double>& y) const;
};
