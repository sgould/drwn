/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPCA.h
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

// drwnPCA class --------------------------------------------------------------
//!  Principal component analysis feature transformation.
//!
//! The following code snippet demonstrates example usage:
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training.bin");
//!
//!   // extract top 5 principal components
//!   drwnPCA pca;
//!   pca.setProperty("outputDim", 5);
//!   pca.train(dataset.features);
//!
//!   // save the pca parameters
//!   pca.write("model.xml");
//!
//!   // project features onto PCA dimensions
//!   pca.transform(dataset.features);
//!   dataset.write("training_pca.bin");
//!
//!   // apply same projection to the test dataset
//!   dataset.clear();
//!   dataset.read("testing.bin");
//!   pca.transform(dataset.features);
//!   dataset.write("testing_pca.bin");
//! \endcode

class drwnPCA : public drwnUnsupervisedTransform {
 private:
    int _numOutputDims;       //!< number of output dimensions (set with "outputDim" property)
    double _energyThreshold;  //!< keep dimensions where sum of variance is above this (set with "energyThreshold" property)
    bool _doNormalization;    //!< apply output normalization (set with "normalizeOutput" property)

    VectorXd _translation;    //!< mean vector
    MatrixXd _projection;     //!< projection matrix

 public:
    //! default constructor
    drwnPCA();
    //! construct a PCA object from previously accumulated second-order
    //! sufficient statistics
    drwnPCA(const drwnSuffStats& stats, double energyThreshold = 1.0, bool doNormalization = true);
    //! copy constructor
    drwnPCA(const drwnPCA& fw);
    ~drwnPCA();

    // access functions
    const char *type() const { return "drwnPCA"; }
    drwnPCA *clone() const { return new drwnPCA(*this); }

    // i/o
    void clear();
    bool save(drwnXMLNode& node) const;
    bool load(drwnXMLNode& node);

    // input/output size
    //! feature vector size for the input space
    int numInputs() const { return _projection.cols(); }
    //! feature vector size for the output space
    int numOutputs() const { return _projection.rows(); }

    // training
    using drwnUnsupervisedTransform::train;
    //! Estimate the parameters of the feature transformation from
    //! previously accumulated second-order sufficient statistics.
    double train(const drwnSuffStats& stats);
    double train(const vector<vector<double> >& features);
    double train(const vector<vector<double> >& features, const drwnFeatureTransform& xform);

    // evaluation
    using drwnFeatureTransform::transform;
    void transform(const vector<double>& x, vector<double>& y) const;
};
