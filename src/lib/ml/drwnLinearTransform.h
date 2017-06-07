/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearTransform.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Jiecheng Zhao <u5143437@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <limits>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnML.h"

using namespace std;
using namespace Eigen;

// drwnLinearTransform class ------------------------------------------------------
//! Implements a linear feature transform with externally settable parameters.
//!
//! The following code snippet demonstrates example usage:
//! \code
//!   // define a linear transform
//!   VectorXd t = VectorXd::Zero(n);
//!   MatrixXd P = MatrixXd::Identity(n, n);
//!
//!   drwnLinearTransform xform;
//!   xform.set(t, P);
//!
//!   // apply to some dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("dataset.bin");
//!   xform.transform(dataset.features);
//!   dataset.write("transformed_dataset.bin");
//! \endcode

class drwnLinearTransform : public drwnFeatureTransform {
 private:
    VectorXd _mu;             //!< translation vector
    MatrixXd _projection;     //!< projection matrix

 public:
    //! default constructor
    drwnLinearTransform();
    //! construct with some transform
    drwnLinearTransform(const VectorXd& mu, const MatrixXd& projection);
    //! copy constructor
    drwnLinearTransform(const drwnLinearTransform& xform);
    ~drwnLinearTransform();

    // access functions
    const char *type() const { return "drwnLinearTransform"; }
    drwnLinearTransform *clone() const { return new drwnLinearTransform(*this); }

    //! set the translation vector and projection matrix
    void set(const VectorXd& mu, const MatrixXd& projection);
    //! set the projection matrix (zero translation)
    void set(const MatrixXd& projection);
    //! get the translation vector
    const VectorXd &translation() const { return _mu; }
    //! get the projection matrix
    const MatrixXd &projection() const { return _projection; }

    // i/o
    void clear();
    bool save(drwnXMLNode& node) const;
    bool load(drwnXMLNode& node);

    // input/output size
    //! feature vector size for the input space
    int numInputs() const { return _projection.cols(); }
    //! feature vector size for the output space
    int numOutputs() const { return _projection.rows(); }

    // evaluation
    using drwnFeatureTransform::transform;
    void transform(const vector<double>& x, vector<double>& y) const;
};
