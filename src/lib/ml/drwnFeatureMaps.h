/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFeatureMaps.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>
#include <cmath>

using namespace std;

#define DRWN_UNROLL_FEATURE_MAP_LOOPS

//
// classless feature mappings ----------------------------------------------
//

// drwnFeatureMap ----------------------------------------------------------
//! Defines the interface for a feature mapping \f$\phi : \mathbb{R}^n
//! \rightarrow \mathbb{R}^m\f$
//!
//! \warning The default behaviour of all virtual functions are very
//! inefficient and should be replaced in the derived classes. They
//! are provided for testing during development, requiring the
//! programmer to only implement the operator()(const vector<double>&
//! x) function.
//!
//! \sa drwnJointFeatureMap
//! \sa \ref drwnFeatureMapDoc

class drwnFeatureMap : public drwnTypeable, public drwnCloneable {
 protected:
    int _nFeatures; //!< number of input features

 public:
    //! default constructor
    drwnFeatureMap() : _nFeatures(0) { /* do nothing */ }
    //! construct with known number of input features
    drwnFeatureMap(int nFeatures) : _nFeatures(nFeatures) { /* do nothing */ }
    //! destructor
    virtual ~drwnFeatureMap() { /* do nothing */ }

    virtual drwnFeatureMap *clone() const = 0;

    //! returns the number of features in the input space
    int numFeatures() const { return _nFeatures; }
    //! returns the number of features in the output space
    virtual int numParameters() const = 0;

    //! initialize number of (intput) features
    virtual void initialize(int nFeatures) {
        _nFeatures = nFeatures;
    }

    //! returns \f$\phi(x) \in \mathbb{R}^m\f$ where \f$x \in \mathbb{R}^n\f$
    virtual vector<double> operator()(const vector<double>& x) const = 0;

    //! returns the dot product \f$\theta^T \phi(x)\f$ (default behaviour is very inefficient)
    virtual double dot(const vector<double>& theta, const vector<double>& x) const {
        vector<double> phi((*this)(x));
        double v = 0.0;
        for (unsigned i = 0; i < phi.size(); i++) {
            v += phi[i] * theta[i];
        }
        return v;
    }

    //! provides multiply-accumulate operation \f$\theta = \theta + \alpha \phi(x)\f$ (default behaviour is very inefficient)
    virtual void mac(vector<double>& theta, const vector<double>& x, double alpha) const {
        const vector<double> phi((*this)(x));
        for (unsigned i = 0; i < phi.size(); i++) {
            theta[i] += alpha * phi[i];
        }
    }
};

// drwnIdentityFeatureMap ---------------------------------------------------
//! Copies input feature space to output feature space.

class drwnIdentityFeatureMap : public drwnFeatureMap {
 public:
    drwnIdentityFeatureMap() : drwnFeatureMap() { /* do nothing */ }
    drwnIdentityFeatureMap(int nFeatures) : drwnFeatureMap(nFeatures) { /* do nothing */ }
    ~drwnIdentityFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnIdentityFeatureMap"; }
    drwnIdentityFeatureMap *clone() const { return new drwnIdentityFeatureMap(*this); }

    //! returns the number of features in the output space
    int numParameters() const { return _nFeatures; }

    //! feature vector
    vector<double> operator()(const vector<double>& x) const {
        return x;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        return Eigen::Map<const VectorXd>(&theta[0], 
            theta.size()).dot(Eigen::Map<const VectorXd>(&x[0], x.size()));
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha) const {
        Eigen::Map<VectorXd>(&theta[0], theta.size()) += alpha *
            Eigen::Map<const VectorXd>(&x[0], x.size());
    }
};

// drwnBiasFeatureMap ------------------------------------------------------
//! Augments input feature vector with 1 (i.e., to allow for a bias weight)

class drwnBiasFeatureMap : public drwnFeatureMap {
 public:
    drwnBiasFeatureMap() : drwnFeatureMap() { /* do nothing */ }
    drwnBiasFeatureMap(int nFeatures) : drwnFeatureMap(nFeatures) { /* do nothing */ }
    ~drwnBiasFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnBiasFeatureMap"; }
    drwnBiasFeatureMap *clone() const { return new drwnBiasFeatureMap(*this); }

    //! returns the number of features in the output space
    int numParameters() const { return _nFeatures + 1; }

    //! feature vector
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(x.size() + 1);
        copy(x.begin(), x.end(), phi.begin());
        phi.back() = 1.0;
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        return theta.back() +
            Eigen::Map<const VectorXd>(&theta[0], 
                theta.size() - 1).dot(Eigen::Map<const VectorXd>(&x[0], x.size()));
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha) const {
        Eigen::Map<VectorXd>(&theta[0], theta.size() - 1) += alpha *
            Eigen::Map<const VectorXd>(&x[0], x.size());
        theta.back() += alpha;
    }
};

// drwnSquareFeatureMap ----------------------------------------------------
//! Augments input feature vector with square of each feature (normalized so
//! that if input is zero mean and unit variance so will output) and 1
//! (i.e., to allow for a bias weight)

class drwnSquareFeatureMap : public drwnFeatureMap {
 public:
    drwnSquareFeatureMap() : drwnFeatureMap() { /* do nothing */ }
    drwnSquareFeatureMap(int nFeatures) : drwnFeatureMap(nFeatures) { /* do nothing */ }
    ~drwnSquareFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnSquareFeatureMap"; }
    drwnSquareFeatureMap *clone() const { return new drwnSquareFeatureMap(*this); }

    //! returns the number of features in the output space
    int numParameters() const { return 2 * _nFeatures + 1; }

    //! feature vector
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(2 * x.size() + 1);
        copy(x.begin(), x.end(), phi.begin());
        for (unsigned i = 0; i < x.size(); i++) {
            phi[x.size() + i] = M_SQRT1_2 * (x[i] * x[i] - 1.0);
        }
        phi.back() = 1.0;
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = theta.back();
        vector<double>::const_iterator it = theta.begin();
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha) const {
        Eigen::Map<ArrayXd>(&theta[0], _nFeatures) += alpha *
            Eigen::Map<const ArrayXd>(&x[0], x.size());
        Eigen::Map<ArrayXd>(&theta[_nFeatures], _nFeatures) += M_SQRT1_2 * alpha *
            (Eigen::Map<const ArrayXd>(&x[0], x.size()).array().square() - 1.0);
        theta.back() += alpha;
    }
};

// drwnQuadraticFeatureMap -------------------------------------------------
//! Augments input feature vector with square of each feature (normalized so
//! that if input is zero mean and unit variance so will output) as
//! well as cross-terms and constant one (i.e., to allow for a bias weight)

class drwnQuadraticFeatureMap : public drwnFeatureMap {
 public:
    drwnQuadraticFeatureMap() : drwnFeatureMap() { /* do nothing */ }
    drwnQuadraticFeatureMap(int nFeatures) : drwnFeatureMap(nFeatures) { /* do nothing */ }
    ~drwnQuadraticFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnQuadraticFeatureMap"; }
    drwnQuadraticFeatureMap *clone() const { return new drwnQuadraticFeatureMap(*this); }

    //! returns the number of features in the output space
    int numParameters() const { return (_nFeatures + 3) * _nFeatures / 2 + 1; }

    //! feature vector
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(2 * x.size() + 1);
        copy(x.begin(), x.end(), phi.begin());
        int indx = x.size();
        for (unsigned i = 0; i < x.size(); i++) {
            for (unsigned j = 0; j < i; j++) {
                phi[indx++] = x[i] * x[j];
            }
            phi[indx++] = M_SQRT1_2 * (x[i] * x[i] - 1.0);
        }
        phi.back() = 1.0;
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = theta.back();
        vector<double>::const_iterator it = theta.begin();
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                v += (*it) * (*ix) * (*jx);
            }
            v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha) const {
        vector<double>::iterator it = theta.begin();
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += alpha * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                (*it) += alpha * (*ix) * (*jx);
            }
            (*it) += alpha * M_SQRT1_2 * ((*ix) * (*ix) - 1.0);
        }
        theta.back() += alpha;
    }
};

//
// classful feature mappings -----------------------------------------------
//

// drwnJointFeatureMap -----------------------------------------------------
//! Defines the interface for a joint feature mapping \f$\phi :
//! \mathbb{R} \times \mathbb{Z} \rightarrow \mathbb{R}^m\f$
//!
//! \warning The default behaviour of all virtual functions are very
//! inefficient and should be replaced in the derived classes. They
//! are provided for testing during development, requiring the
//! programmer to only implement the operator()(const
//! vector<double>&x, double y) function.
//!
//! \sa drwnFeatureMap
//! \sa \ref drwnFeatureMapDoc

class drwnJointFeatureMap : public drwnTypeable, public drwnCloneable {
 protected:
    int _nFeatures; //!< number of input features
    int _nClasses;  //!< number of class labels

 public:
    //! default constructor
    drwnJointFeatureMap() : _nFeatures(0), _nClasses(0) { /* do nothing */ }
    //! construct with known number of input features and classes
    drwnJointFeatureMap(int nFeatures, int nClasses) :
        _nFeatures(nFeatures), _nClasses(nClasses) { /* do nothing */ }
    //! destructor
    virtual ~drwnJointFeatureMap() { /* do nothing */ }

    virtual drwnJointFeatureMap *clone() const = 0;

    //! returns the number of features in the input space
    int numFeatures() const { return _nFeatures; }
    //! returns the number of classes
    int numClasses() const { return _nClasses; }
    //! returns the number of features in the (joint) output space
    virtual int numParameters() const = 0;

    //! initialize number of classes and number of features
    virtual void initialize(int nFeatures, int nClasses) {
        _nFeatures = nFeatures; _nClasses = nClasses;
    }

    //! returns \f$\phi(x, y) \in \mathbb{R}^m\f$ where \f$x \in \mathbb{R}^n\f$
    virtual vector<double> operator()(const vector<double>& x, int y) const = 0;

    //! returns \f$ \sum_y \phi(x, y) \in \mathbb{R}^m\f$ (default behaviour is very inefficient)
    virtual vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(numParameters());
        for (int y = 0; y < _nClasses; y++) {
            vector<double> phi_y((*this)(x, y));
            for (unsigned i = 0; i < phi.size(); i++) {
                phi[i] += phi_y[i];
            }
        }
        return phi;
    }

    //! returns the dot product \f$\theta^T \phi(x, y)\f$ (default behaviour is very inefficient)
    virtual double dot(const vector<double>& theta, const vector<double>& x, int y) const {
        vector<double> phi((*this)(x, y));
        double v = 0.0;
        for (unsigned i = 0; i < phi.size(); i++) {
            v += phi[i] * theta[i];
        }
        return v;
    }

    //! returns the dot product \f$\sum_y \theta^T \phi(x, y)\f$ (default behaviour is very inefficient)
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = 0.0;
        for (int y = 0; y < _nClasses; y++) {
            v += dot(theta, x, y);
        }
        return v;
    }

    //! provides multiply-accumulate operation \f$\theta = \theta + \alpha \phi(x, y)\f$ (default behaviour is very inefficient)
    virtual void mac(vector<double>& theta, const vector<double>& x, double alpha, int y) const {
        const vector<double> phi((*this)(x, y));
        for (unsigned i = 0; i < phi.size(); i++) {
            theta[i] += alpha * phi[i];
        }
    }

    //! provides multiply-accumulate operation \f$\theta = \theta + \sum_y \alpha_y \phi(x, y)\f$ (default behaviour is very inefficient)
    virtual void mac(vector<double>& theta, const vector<double>& x, const vector<double>& alpha) const {
        for (int y = 0; y < _nClasses; y++) {
            const vector<double> phi((*this)(x, y));
            for (unsigned i = 0; i < phi.size(); i++) {
                theta[i] += alpha[y] * phi[i];
            }
        }
    }
};

// drwnIdentityJointFeatureMap ----------------------------------------------
//! Includes a copy of each feature from the input space for each class
//! other than the last, i.e., \f$\phi(x, y) = \left(\delta\!\left\{y = 0\right\} x^T,
//! \ldots, \delta\!\left\{y = K - 2\right\}x^T\right) \in \mathbb{R}^{(K - 1) n}\f$.
//! This is the standard feature mapping for multi-class logistic models.

class drwnIdentityJointFeatureMap : public drwnJointFeatureMap {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
 protected:
    int _nFeaturesDiv4;
    int _nFeaturesMod4;
#endif

 public:
    drwnIdentityJointFeatureMap() : drwnJointFeatureMap() {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        _nFeaturesDiv4 = 0;
        _nFeaturesMod4 = 0;
#endif
    }
    drwnIdentityJointFeatureMap(int nFeatures, int nClasses) :
        drwnJointFeatureMap(nFeatures, nClasses) {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
            _nFeaturesDiv4 = nFeatures / 4;
            _nFeaturesMod4 = nFeatures % 4;
#endif
    }
    ~drwnIdentityJointFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnIdentityJointFeatureMap"; }
    drwnIdentityJointFeatureMap *clone() const { return new drwnIdentityJointFeatureMap(*this); }

    //! returns the number of features in the (joint) output space
    int numParameters() const { return std::max(_nFeatures * (_nClasses - 1), 0); }

#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
    //! initialize number of classes and number of features
    void initialize(int nFeatures, int nClasses) {
        _nFeatures = nFeatures; _nClasses = nClasses;
        _nFeaturesDiv4 = nFeatures / 4;
        _nFeaturesMod4 = nFeatures % 4;
    }
#endif

    //! joint feature vector for given class
    vector<double> operator()(const vector<double>& x, int y) const {
        vector<double> phi(numParameters(), 0.0);
        if (y != _nClasses - 1) {
            copy(x.begin(), x.end(), phi.begin() + y * _nFeatures);
        }
        return phi;
    }

    //! joint feature vector summed over classes
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(numParameters());
        for (int y = 0; y < _nClasses - 1; y++) {
            copy(x.begin(), x.end(), phi.begin() + y * _nFeatures);
        }
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x, int y) const {
        if (y == _nClasses - 1) return 0.0;
        double v = 0.0;
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        const double *it = &theta[y * _nFeatures];
        const double *ix = &x[0];
        for (int i = _nFeaturesDiv4; i != 0; i--) {
            v += it[0] * ix[0] + it[1] * ix[1] + it[2] * ix[2] + it[3] * ix[3];
            it += 4; ix += 4;
        }
        for (int i = 0; i < _nFeaturesMod4; i++, ++it, ++ix) {
            v += (*it) * (*ix);
        }
#else
        vector<double>::const_iterator it = theta.begin() + y * _nFeatures;
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
#endif
        return v;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                v += (*it) * (*ix);
            }
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha, int y) const {
        if (y == _nClasses - 1) return;
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        double *it = &theta[y * _nFeatures];
        const double *ix = &x[0];
        for (int i = _nFeaturesDiv4; i != 0; i--) {
            it[0] += alpha * ix[0];
            it[1] += alpha * ix[1];
            it[2] += alpha * ix[2];
            it[3] += alpha * ix[3];
            it += 4; ix += 4;
        }
        for (int i = 0; i < _nFeaturesMod4; i++, ++it, ++ix) {
            (*it) += alpha * (*ix);
        }
#else
        vector<double>::iterator it = theta.begin() + y * _nFeatures;
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += alpha * (*ix);
        }
#endif
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, const vector<double>& alpha) const {
        vector<double>::iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) += alpha[y] * (*ix);
            }
        }
    }
};

// drwnBiasJointFeatureMap -------------------------------------------------
//! Same as drwnIdentityJointFeatureMap but adds a bias term for each class
//! i.e., \f$\phi(x, y) = \left(\delta\!\left\{y = 0\right\} (x^T, 1), \ldots,
//! \delta\!\left\{y = K - 2\right\} (x^T, 1)\right) \in \mathbb{R}^{(K - 1)(n + 1)}\f$.

class drwnBiasJointFeatureMap : public drwnJointFeatureMap {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
 protected:
    int _nFeaturesDiv4;
    int _nFeaturesMod4;
#endif

 public:
    drwnBiasJointFeatureMap() : drwnJointFeatureMap() {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        _nFeaturesDiv4 = 0;
        _nFeaturesMod4 = 0;
#endif
    }
    drwnBiasJointFeatureMap(int nFeatures, int nClasses) :
        drwnJointFeatureMap(nFeatures, nClasses) {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
            _nFeaturesDiv4 = nFeatures / 4;
            _nFeaturesMod4 = nFeatures % 4;
#endif
    }
    ~drwnBiasJointFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnBiasJointFeatureMap"; }
    drwnBiasJointFeatureMap *clone() const { return new drwnBiasJointFeatureMap(*this); }

    //! returns the number of features in the (joint) output space
    int numParameters() const { return std::max((_nFeatures + 1) * (_nClasses - 1), 0); }

#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
    //! initialize number of classes and number of features
    void initialize(int nFeatures, int nClasses) {
        _nFeatures = nFeatures; _nClasses = nClasses;
        _nFeaturesDiv4 = nFeatures / 4;
        _nFeaturesMod4 = nFeatures % 4;
    }
#endif

    //! feature vector for given y
    vector<double> operator()(const vector<double>& x, int y) const {
        vector<double> phi(numParameters(), 0.0);
        if (y != _nClasses - 1) {
            copy(x.begin(), x.end(), phi.begin() + y * (_nFeatures + 1));
            phi[y * (_nFeatures + 1) + _nFeatures] = 1.0;
        }
        return phi;
    }

    //! feature vector summed over y
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(numParameters());
        vector<double>::iterator it = phi.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            copy(x.begin(), x.end(), it);
            *(it += _nFeatures)++ = 1.0;
        }
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x, int y) const {
        if (y == _nClasses - 1) return 0.0;
        double v = 0.0;
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        const double *it = &theta[y * (_nFeatures + 1)];
        const double *ix = &x[0];
        for (int i = _nFeaturesDiv4; i != 0; i--) {
            v += it[0] * ix[0] + it[1] * ix[1] + it[2] * ix[2] + it[3] * ix[3];
            it += 4; ix += 4;
        }
        for (int i = 0; i < _nFeaturesMod4; i++, ++it, ++ix) {
            v += (*it) * (*ix);
        }
#else
        vector<double>::const_iterator it = theta.begin() + y * (_nFeatures + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
#endif
        v += (*it);
        return v;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                v += (*it) * (*ix);
            }
            v += (*it++);
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha, int y) const {
        if (y == _nClasses - 1) return;
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        double *it = &theta[y * (_nFeatures + 1)];
        const double *ix = &x[0];
        for (int i = _nFeaturesDiv4; i != 0; i--) {
            it[0] += alpha * ix[0];
            it[1] += alpha * ix[1];
            it[2] += alpha * ix[2];
            it[3] += alpha * ix[3];
            it += 4; ix += 4;
        }
        for (int i = 0; i < _nFeaturesMod4; i++, ++it, ++ix) {
            (*it) += alpha * (*ix);
        }
#else
        vector<double>::iterator it = theta.begin() + y * (_nFeatures + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += alpha * (*ix);
        }
#endif
        (*it) += alpha;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, const vector<double>& alpha) const {
        vector<double>::iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) += alpha[y] * (*ix);
            }
            (*it++) += alpha[y];
        }
    }
};

// drwnSquareJointFeatureMap -----------------------------------------------
//! Same as drwnIdentityJointFeatureMap but adds a square term for each feature
//! i.e., \f$\phi(x, y) = \left(\delta\!\left\{y = 0\right\} (x^T, \frac{1}{\sqrt{2}} diag(xx^T - I), 1), \ldots,
//! \delta\!\left\{y = K - 2\right\} (x^T, \frac{1}{\sqrt{2}} diag(xx^T - I), 1)\right) \in
//! \mathbb{R}^{(K - 1)(2n + 1)}\f$.

class drwnSquareJointFeatureMap : public drwnJointFeatureMap {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
 protected:
    int _nFeaturesDiv4;
    int _nFeaturesMod4;
#endif

 public:
    drwnSquareJointFeatureMap() : drwnJointFeatureMap() {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
        _nFeaturesDiv4 = 0;
        _nFeaturesMod4 = 0;
#endif
    }
    drwnSquareJointFeatureMap(int nFeatures, int nClasses) :
        drwnJointFeatureMap(nFeatures, nClasses) {
#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
            _nFeaturesDiv4 = nFeatures / 4;
            _nFeaturesMod4 = nFeatures % 4;
#endif
    }
    ~drwnSquareJointFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnSquareJointFeatureMap"; }
    drwnSquareJointFeatureMap *clone() const { return new drwnSquareJointFeatureMap(*this); }

    //! returns the number of features in the (joint) output space
    int numParameters() const { return std::max((2 * _nFeatures + 1) * (_nClasses - 1), 0); }

#ifdef DRWN_UNROLL_FEATURE_MAP_LOOPS
    //! initialize number of classes and number of features
    void initialize(int nFeatures, int nClasses) {
        _nFeatures = nFeatures; _nClasses = nClasses;
        _nFeaturesDiv4 = nFeatures / 4;
        _nFeaturesMod4 = nFeatures % 4;
    }
#endif

    //! feature vector for given y
    vector<double> operator()(const vector<double>& x, int y) const {
        vector<double> phi(numParameters(), 0.0);
        if (y != _nClasses - 1) {
            vector<double>::iterator jt = phi.begin() + y * (2 * _nFeatures + 1);
            for (vector<double>::const_iterator it = x.begin(); it != x.end(); ++it, ++jt) {
                (*jt) = (*it);
            }
            for (vector<double>::const_iterator it = x.begin(); it != x.end(); ++it, ++jt) {
                (*jt) = M_SQRT1_2 * ((*it) * (*it) - 1.0);
            }
            (*jt) = 1.0;
        }
        return phi;
    }

    //! feature vector summed over y
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(numParameters());
        vector<double>::iterator jt = phi.begin();
        for (vector<double>::const_iterator it = x.begin(); it != x.end(); ++it, ++jt) {
            (*jt) = (*it);
        }
        for (vector<double>::const_iterator it = x.begin(); it != x.end(); ++it, ++jt) {
            (*jt) = M_SQRT1_2 * ((*it) * (*it) - 1.0);
        }
        (*jt) = 1.0;

        for (int y = 1; y < _nClasses - 1; y++) {
            copy(jt - (2 * _nFeatures + 1), jt, jt + 1);
            jt += (2 * _nFeatures + 1);
        }
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x, int y) const {
        if (y == _nClasses - 1) return 0.0;
#if 1
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin() + y * (2 * _nFeatures + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
        }
        v += (*it);

        return v;
#else
        return Eigen::Map<const VectorXd>(&theta[y * (2 * _nFeatures + 1)],
            _nFeatures).dot(Eigen::Map<const VectorXd>(&x[0], _nFeatures)) +
            Eigen::Map<const VectorXd>(&theta[y * (2 * _nFeatures + 1) + _nFeatures],
                _nFeatures).dot(M_SQRT1_2 *
                    (Eigen::Map<const VectorXd>(&x[0], x.size()).array().square().array() - 1.0)) +
            theta[y * (2 * _nFeatures + 1) + 2 * _nFeatures];
#endif
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                v += (*it) * (*ix);
            }
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
            }
            v += (*it++);
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha, int y) const {
        if (y == _nClasses - 1) return;
        vector<double>::iterator it = theta.begin() + y * (2 * _nFeatures + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += alpha * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += M_SQRT1_2 * alpha * ((*ix) * (*ix) - 1.0);
        }
        (*it) += alpha;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, const vector<double>& alpha) const {
        vector<double>::iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
#if 1
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) += alpha[y] * (*ix);
            }
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) += M_SQRT1_2 * alpha[y] * ((*ix) * (*ix) - 1.0);
            }
            (*it++) += alpha[y];
#else
            Eigen::Map<VectorXd>(&theta[y * (2 * _nFeatures + 1)], _nFeatures) += alpha[y] *
                Eigen::Map<const VectorXd>(&x[0], x.size());
            Eigen::Map<VectorXd>(&theta[y * (2 * _nFeatures + 1) + _nFeatures], _nFeatures) += M_SQRT1_2 * alpha[y] *
                (Eigen::Map<const VectorXd>(&x[0], x.size()).array().square().array() - 1.0);
            theta[y * (2 * _nFeatures + 1) + 2 * _nFeatures] += alpha[y];
#endif
        }
    }
};

// drwnQuadraticJointFeatureMap --------------------------------------------
//! Same as drwnSquareJointFeatureMap but adds cross-terms.

class drwnQuadraticJointFeatureMap : public drwnJointFeatureMap {
 public:
    drwnQuadraticJointFeatureMap() : drwnJointFeatureMap() { /* do nothing */ }
    drwnQuadraticJointFeatureMap(int nFeatures, int nClasses) :
        drwnJointFeatureMap(nFeatures, nClasses) { /* do nothing */ }
    ~drwnQuadraticJointFeatureMap() { /* do nothing */ }

    // type and cloning
    const char *type () const { return "drwnQuadraticJointFeatureMap"; }
    drwnQuadraticJointFeatureMap *clone() const { return new drwnQuadraticJointFeatureMap(*this); }

    //! returns the number of features in the (joint) output space
    int numParameters() const { return std::max(((_nFeatures + 3) * _nFeatures / 2 + 1) *
            (_nClasses - 1), 0); }

    //! feature vector for given y
    vector<double> operator()(const vector<double>& x, int y) const {
        vector<double> phi(numParameters(), 0.0);
        if (y != _nClasses - 1) {
            vector<double>::iterator it = phi.begin() + y * ((_nFeatures + 3) * _nFeatures / 2 + 1);
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) = (*ix);
            }
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                    (*it) = (*ix) * (*jx);
                }
                (*it) = M_SQRT1_2 * ((*ix) * (*ix) - 1.0);
            }
            (*it) = 1.0;
        }
        return phi;
    }

    //! feature vector summed over y
    vector<double> operator()(const vector<double>& x) const {
        vector<double> phi(numParameters());
        vector<double>::iterator it = phi.begin();
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) = (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                (*it) = (*ix) * (*jx);
            }
            (*it) = M_SQRT1_2 * ((*ix) * (*ix) - 1.0);
        }
        (*it) = 1.0;

        vector<double>::iterator jt = ++it;
        while (jt != phi.end()) {
            copy(phi.begin(), it, jt);
            jt += it - phi.begin();
        }
        return phi;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x, int y) const {
        if (y == _nClasses - 1) return 0.0;
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin() +
            y * ((_nFeatures + 3) * _nFeatures / 2 + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            v += (*it) * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                v += (*it) * (*ix) * (*jx);
            }
            v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
        }
        v += (*it);

        return v;
    }

    //! dot product
    double dot(const vector<double>& theta, const vector<double>& x) const {
        double v = 0.0;
        vector<double>::const_iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                v += (*it) * (*ix);
            }
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                    v += (*it) * (*ix) * (*jx);
                }
                v += M_SQRT1_2 * (*it) * ((*ix) * (*ix) - 1.0);
            }
            v += (*it++);
        }
        return v;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, double alpha, int y) const {
        if (y == _nClasses - 1) return;
        vector<double>::iterator it = theta.begin() + y * ((_nFeatures + 3) * _nFeatures / 2 + 1);
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            (*it) += alpha * (*ix);
        }
        for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
            for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                (*it) += alpha * (*ix) * (*jx);
            }
            (*it) += M_SQRT1_2 * alpha * ((*ix) * (*ix) - 1.0);
        }
        (*it) += alpha;
    }

    //! multiply-accumulate
    void mac(vector<double>& theta, const vector<double>& x, const vector<double>& alpha) const {
        vector<double>::iterator it = theta.begin();
        for (int y = 0; y < _nClasses - 1; y++) {
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                (*it) += alpha[y] * (*ix);
            }
            for (vector<double>::const_iterator ix = x.begin(); ix != x.end(); ++ix, ++it) {
                for (vector<double>::const_iterator jx = x.begin(); jx != ix; ++jx, ++it) {
                    (*it) += alpha[y] * (*ix) * (*jx);
                }
                (*it) += M_SQRT1_2 * alpha[y] * ((*ix) * (*ix) - 1.0);
            }
            (*it++) += alpha[y];
        }
    }
};

