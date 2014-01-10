/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGaussianMixture.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <limits>

#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "drwnBase.h"
#include "drwnSuffStats.h"

using namespace std;
using namespace Eigen;

// drwnGaussianMixture ------------------------------------------------------
//! Implements a multi-variant Gaussian mixture model.
//!
//! \f[ p(x; \left\{\lambda_k, \mu_k, \Sigma\right\}_k) =
//!   \sum_k \lambda_k \frac{1}{\sqrt{|2 \pi \Sigma_k|}}
//!   \exp\{-\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\}\f]
//!
//! The following code snippet shows example usage:
//! \code
//!   // load dataset
//!   vector<vector<double> > features;
//!
//!   ... // code to load the dataset
//!
//!   // estimate parameters for a 5-component mixture model
//!   drwnGaussianMixture gmm(features[0].size(), 5);
//!   gmm.train(features);
//!
//!   // save the learned model
//!   gmm.write("gmm.xml");
//!
//!   // generate 10 samples from the model
//!   vector<double> s;
//!   for (int i = 0; i < 10; i++) {
//!      gmm.sample(s);
//!      DRWN_LOG_MESSAGE("sample " << (i + 1) << " is " << toString(s));
//!   }
//! \endcode

class drwnGaussianMixture : public drwnStdObjIface {
public:
    static int MAX_ITERATIONS;

protected:
    unsigned _n;               //!< feature dimension
    vector<drwnGaussian> _g;   //!< gaussian components
    VectorXd _logLambda;       //!< mixture weights (log space)

public:
    //! construct a gaussian mixture model over \p n dimensional features with
    //! \p k mixture components
    drwnGaussianMixture(unsigned n = 1, unsigned k = 1);
    virtual ~drwnGaussianMixture();

    // initialization
    //! initialize the gaussian mixture model to be over \p n dimensional
    //! features with \p k mixture components
    void initialize(unsigned n, unsigned k);

    // i/o
    const char *type() const { return "drwnGaussianMixture"; }
    drwnGaussianMixture *clone() const { return new drwnGaussianMixture(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // evaluate (log-likelihood)
    //! compute the log-likelihood of each row of \p x and put the results in \p p
    void evaluate(const MatrixXd& x, VectorXd& p) const;
    //! compute the log-likelihood of each vector in \p x and put the results in \p p
    void evaluate(const vector<vector<double> >& x, vector<double>& p) const;
    //! compute the log-likelihood of a given vector
    double evaluateSingle(const VectorXd& x) const;
    //! compute the log-likelihood of a given vector
    double evaluateSingle(const vector<double>& x) const;

    // sampling
    //! generate a random sample from the mixture of gaussian distribution
    void sample(VectorXd& x) const;
    //! generate a random sample from the mixture of gaussian distribution
    void sample(vector<double>& x) const;

    // learn parameters
    //! Estimate the parameters of the mixture of gaussians from a matrix of
    //! training examples using the EM algorithm. The parameter \p lambda
    //! regularizes the component covariance matrices towards the global
    //! covariance matrix.
    void train(const MatrixXd& x, double lambda = 1.0e-3);
    //! See above.
    void train(const vector<vector<double> >& x, double lambda = 1.0e-3);

    // access
    //! returns the number of mixture components
    unsigned mixtures() const { return _g.size(); }
    //! returns the dimensionality of the features space
    unsigned dimension() const { return _n; }
    //! returns the mixture weight of the \p k-th component
    double weight(int k) const { return exp(_logLambda[k]); }
    //! returns the mean of the \p k-th component
    const VectorXd& mean(int k) const { return _g[k].mean(); }
    //! returns the covariance matrix for the \p k-th component
    const MatrixXd& covariance(int k) const { return _g[k].covariance(); }
    //! returns the \p k-th component as a gaussian distribution
    const drwnGaussian& component(int k) const { return _g[k]; }

    // standard operators
    //! assignment operator
    drwnGaussianMixture& operator=(const drwnGaussianMixture& model);
};

