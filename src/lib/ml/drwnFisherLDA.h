/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFisherLDA.h
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

// drwnFisherLDA class --------------------------------------------------------
//! Fisher's linear discriminant analysis (LDA).
//!
//! The following code snippet shows example usage:
//!
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training.bin");
//!
//!   // compute LDA features
//!   drwnFisherLDA lda;
//!   lda.train(dataset.features, dataset.labels);
//!   lda.transform(dataset.features);
//!
//!   // save LDA features
//!   dataset.write("training_lda.bin");
//!
//!   // apply same transformation to test dataset
//!   dataset.clear();
//!   dataset.read("testing.bin");
//!   lda.transform(dataset.features);
//!   dataset.write("testing_lda.bin");
//! \endcode

class drwnFisherLDA : public drwnSupervisedTransform {
 private:
    VectorXd _translation;    //!< mean vector
    MatrixXd _projection;     //!< projection matrix

 public:
    //! default constructor
    drwnFisherLDA();
    //! copy constructor
    drwnFisherLDA(const drwnFisherLDA& lda);
    ~drwnFisherLDA();

    // access functions
    const char *type() const { return "drwnFisherLDA"; }
    drwnFisherLDA *clone() const { return new drwnFisherLDA(*this); }

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
    using drwnSupervisedTransform::train;
    //! Estimate the parameters of the feature transformation from
    //! previously accumulated second-order conditional sufficient statistics.
    double train(const drwnCondSuffStats& stats);
    double train(const vector<vector<double> >& features, const vector<int>& labels);
    double train(const vector<vector<double> >& features,
        const vector<int>& labels, const vector<double>& weights);
    double train(const vector<vector<double> >& features, const vector<int>& labels,
        const drwnFeatureTransform& xform);
    double train(const vector<vector<double> >& features, const vector<int>& labels,
        const vector<double>& weights, const drwnFeatureTransform& xform);

    // evaluation
    using drwnFeatureTransform::transform;
    void transform(const vector<double>& x, vector<double>& y) const;
};
