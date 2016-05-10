/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnKMeans.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <limits>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnFeatureTransform.h"

using namespace std;
using namespace Eigen;

// drwnKMeans class -----------------------------------------------------------
//! Implements k-means clustering. Outputs the squared-distance to each of the
//! cluster centroids. The nearest cluster can be found by passing the output
//! to the drwn::argmin function. Supports weighted training examples.
//!
//! The following code snippet shows example usage:
//! \code
//!   // load dataset
//!   vector<vector<double> > features;
//!
//!   ... // code to load the dataset
//!
//!   // cluster into 5 components
//!   drwnKMeans clusters(5);
//!   clusters.train(features);
//!
//!   // save the learned cluster centroids
//!   kmeans.write("model.xml");
//!
//!   // show which clusters each feature vector belongs to
//!   vector<double> y;
//!   for (size_t i = 0; i < features.size(); i++) {
//!      kmeans.transform(features[i], y);
//!      int k = drwn::argmin(y);
//!      DRWN_LOG_MESSAGE("sample " << (i + 1) << " belongs to cluster " << k);
//!   }
//! \endcode

class drwnKMeans : public drwnUnsupervisedTransform {
 public:
    static int DEFAULT_K;      //!< default number of clusters
    static int MAX_ITERATIONS; //!< maximum number of training iterations

 private:
    unsigned _numClusters;    //!< number of clusters, k
    MatrixXd _centroids;      //!< centroids for each cluster
    VectorXd _cSqNorm;        //!< norm for each centroid

 public:
    //! construct a k-means object with \p k clusters
    drwnKMeans(unsigned k = DEFAULT_K);
    //! copy constructor
    drwnKMeans(const drwnKMeans& ft);
    ~drwnKMeans();

    // access functions
    const char *type() const { return "drwnKMeans"; }
    drwnKMeans *clone() const { return new drwnKMeans(*this); }

    //! returns the k centroids
    const MatrixXd& getCentroids() const { return _centroids; }

    // i/o
    void clear();
    bool save(drwnXMLNode& node) const;
    bool load(drwnXMLNode& node);

    // training
    using drwnUnsupervisedTransform::train;
    double train(const vector<vector<double> >& features);
    double train(const vector<vector<double> >& features, const vector<double>& weights);

    // evaluation
    using drwnFeatureTransform::transform;
    void transform(const vector<double>& x, vector<double>& y) const;
};
