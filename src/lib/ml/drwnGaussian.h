/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGaussian.h
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

class drwnConditionalGaussian;

// drwnGaussian class --------------------------------------------------------
//! Implements a multi-variate gaussian distribution.
//!
//! \f[ \log p(x; \mu, \Sigma) = -\frac{n}{2} \log(2 \pi) -\frac{1}{2} \log |\Sigma| -
//! \frac{1}{2}\left(x - \mu\right)^T \Sigma^{-1} \left(x - \mu\right) \f]
//!
//! \sa \ref drwnTutorialML "drwnML Tutorial"

class drwnGaussian : public drwnStdObjIface {
 public:
    static bool AUTO_RIDGE;

 private:
    int _n;                   //!< gaussian dimension
    VectorXd _mu;             //!< n element mean vector
    MatrixXd _mSigma;         //!< n-by-n element covariance matrix

    mutable MatrixXd *_invSigma; //!< n-by-n inverse covariance matrix
    mutable double _logZ;        //!< log partition function (\sqrt(2*\pi*det(\Sigma)^n))
    mutable MatrixXd *_mL;       //!< n-by-n sqrt matrix for sampling

 public:
    //! construct an \p n dimensional zero-mean identity-covariance gaussian
    drwnGaussian(int n = 1);
    //! construct a gaussian with given mean and isotropic covariance
    drwnGaussian(const VectorXd& mu, double sigma2);
    //! construct a gaussian with given mean and covariance
    drwnGaussian(const VectorXd& mu, const MatrixXd& sigma2);
    //! construct a gaussian with given mean and isotropic covariance
    drwnGaussian(const vector<double>& mu, double sigma2);
    //! construct a gaussian from given second-order sufficient statistics
    drwnGaussian(const drwnSuffStats& stats);
    //! copy constructor
    drwnGaussian(const drwnGaussian& model);
    ~drwnGaussian();

    // initialization
    //! initialize the gaussian to be \p n dimensional zero-mean and identity-covariance
    void initialize(int n);
    //! initialize the gaussian to have mean \p mu and isotropic variance \p sigma2
    void initialize(const VectorXd& mu, double sigma2);
    //! initialize the gaussian to the given mean and covariance
    void initialize(const VectorXd& mu, const MatrixXd& sigma2);

    // marginalization
    //! Generate a gaussian with the variables not in \p indx marginalized out. The
    //! calling function is responsible for deleting the returned object.
    drwnGaussian *marginalize(const vector<int> & indx) const;

    // conditioning
    //! Returns a gaussian conditioned on \p x. Repeated calls to reduce
    //! on the same set of variables should use the conditionOn function.
    drwnGaussian reduce(const vector<double>& x, const vector<int>& indx) const;
    //! Returns a gaussian conditioned on \p x.
    drwnGaussian reduce(const map<int, double>& x) const;
    //! Returns a temporary objects for conditioning on a set of variables.
    drwnConditionalGaussian conditionOn(const vector<int>& indx) const;

    // i/o
    const char *type() const { return "drwnGaussian"; }
    drwnGaussian *clone() const { return new drwnGaussian(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // evaluate (log-likelihood)
    //! compute the log-likelihood of each row of \p X and put the results in \p p
    void evaluate(const MatrixXd& x, VectorXd& p) const;
    //! compute the log-likelihood of each vector in \p x and put the results in \p p
    void evaluate(const vector<vector<double> >& x, vector<double>& p) const;
    //! compute the log-likelihood of a given vector
    double evaluateSingle(const VectorXd& x) const;
    //! compute the log-likelihood of a given vector
    double evaluateSingle(const vector<double>& x) const;
    //! compute the log-likelihood of a scalar (for one dimensional gaussians only)
    double evaluateSingle(double x) const;

    // sampling
    //! generate a random sample from the gaussian
    void sample(VectorXd& x) const;
    //! generate a random sample from the gaussian
    void sample(vector<double>& x) const;

    // learn parameters
    //! Estimate the mean and covariance of the gaussian from a matrix of training examples.
    //! Data should be arranged row-wise. The parameter \p lambda can be used to regularize
    //! the covariance matrix with an additive isotropic component.
    void train(const MatrixXd& x, double lambda = 0.0);
    //! See above.
    void train(const vector<vector<double> >& x, double lambda = 0.0);
    //! See above, but for one dimensional gaussians.
    void train(const vector<double> &x, double lambda = 0.0);
    //! See above, but using given second-order sufficient statistics.    
    void train(const drwnSuffStats& stats, double lambda = 0.0);

    // access
    //! returns the dimensionality of the gaussian
    unsigned dimension() const { return _n; }
    //! returns the mean of the gaussian
    const VectorXd& mean() const { return _mu; }
    //! returns the covariance matrix for the gaussian
    const MatrixXd& covariance() const { return _mSigma; }
    //! computes the log partition function of the gaussian
    double logPartitionFunction() const;

    //! computes the KL divergence between the gaussian and the given model
    double klDivergence(const drwnGaussian& model) const;
    //! computes the KL divergence between the gaussian and distribution induced
    //! by the given sufficient statistics
    double klDivergence(const drwnSuffStats& stats) const;

    // standard operators
    //! assignment operator
    drwnGaussian& operator=(const drwnGaussian& model);

 protected:
    inline void guaranteeInvSigma() const;
    void freeCachedParameters();
    void updateCachedParameters();
};

// drwnConditionalGaussian class ---------------------------------------------
//! Utility class for generating conditonal gaussian distribution.
//!
//! The conditional gaussian is given by
//! \f[ 
//!   {\cal N}(X, Y; \mu, \Sigma \mid Y = y) = 
//!      {\cal N}(X; \mu_X - \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mu_Y),
//!          \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX})
//! \f]
//!
//! \sa drwnGaussian, \ref drwnTutorialML "drwnML Tutorial"

class drwnConditionalGaussian : public drwnStdObjIface {
 protected:
    int _n;               //!< dimensionality of unobserved features
    int _m;               //!< dimensionality of observed features (i.e., those conditioned on)
    VectorXd _mu;         //!< n element mean vector \f$ \mu_1 - \Sigma_{12} \Sigma_{22}^{-1} \mu_2 \f$
    MatrixXd _mSigma;     //!< n-by-n element covariance matrix \f$ \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \f$
    MatrixXd _mSigmaGain; //!< m-by-n gain matrix \f$ \Sigma_{12} \Sigma_{22}^{-1} \f$

 public:
    //! construct a conditional gaussian object with given mean, covariance and gain matrix
    drwnConditionalGaussian(const VectorXd& mu, const MatrixXd& Sigma,
        const MatrixXd& SigmaGain);
    //! copy constructor
    drwnConditionalGaussian(const drwnConditionalGaussian& model);
    ~drwnConditionalGaussian();

    // i/o
    const char *type() const { return "drwnConditionalGaussian"; }
    drwnConditionalGaussian *clone() const { return new drwnConditionalGaussian(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // construct gaussian given observation
    //! construct a gaussian conditioned on obeserving \p x
    drwnGaussian reduce(const VectorXd& x);
    //! construct a gaussian conditioned on obeserving \p x
    drwnGaussian reduce(const vector<double>& x);
};



