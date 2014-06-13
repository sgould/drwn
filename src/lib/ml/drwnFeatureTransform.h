/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFeatureTransform.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnFeatureMaps.h"

using namespace std;
using namespace Eigen;

// drwnFeatureTransform -----------------------------------------------------
//! Implements the interface for a generic feature transforms possibly with
//! learned parameters, e.g., PCA (unsupervised) or LDA (supervised).
//!
//! \sa drwnPCA
//! \sa drwnFisherLDA

class drwnFeatureTransform : public drwnStdObjIface, public drwnProperties {
 protected:
    int _nFeatures;     //!< number of (input) features
    bool _bValid;       //!< true if transform parameters are trained or loaded

 public:
    //! default constructor
    drwnFeatureTransform();
    //! copy constructor
    drwnFeatureTransform(const drwnFeatureTransform& t);
    virtual ~drwnFeatureTransform() {
        // do nothing
    }

    // access functions
    //! returns the length of the feature vector expected by the feature transform
    //! object (or zero for arbitrary)
    int numFeatures() const { return _nFeatures; }
    //! returns true if the feature transform object is initialized (and trained)
    virtual bool valid() const { return _bValid; }

    // clone operation from (drwnStdObjIface)
    virtual drwnFeatureTransform *clone() const = 0;

    // i/o
    //! clears the parameters of the feature transform object
    virtual void clear() { _nFeatures = 0; _bValid = false; }
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // training methods (implemented in derived classes)

    // evaluation (in-place and copy)
    //! transforms a feature vector in-place
    virtual void transform(vector<double>& x) const;
    //! transforms feature vector \p x into feature vector \p y
    virtual void transform(const vector<double>& x, vector<double>& y) const = 0;
    //! transforms a set of feature vectors in-place
    virtual void transform(vector<vector<double> >& x) const;
    //! transforms a set of feature vectors from \p x to corresponding feature vectors \p y
    virtual void transform(const vector<vector<double> >& x,
        vector<vector<double> >& y) const;

    // evaluation with pre-processing (in-place and copy)
    //! transforms a feature vector in-place first applying another transform
    virtual void transform(vector<double>& x, const drwnFeatureTransform& xform) const;
    //! transforms feature vector \p x into feature vector \p y first applying another transform
    virtual void transform(const vector<double>& x, vector<double>& y,
        const drwnFeatureTransform& xform) const;
    //! transforms a set of feature vectors in-place first applying another transform
    virtual void transform(vector<vector<double> >& x, const drwnFeatureTransform& xform) const;
    //! transforms a set of feature vectors from \p x to corresponding feature vectors \p y
    //! first applying another transform
    virtual void transform(const vector<vector<double> >& x,
        vector<vector<double> >& y, const drwnFeatureTransform& xform) const;
};

// drwnTFeatureMapTranform --------------------------------------------------
//! Helper feature transformation based on a drwnFeatureMap.

template<class FeatureMap>
class drwnTFeatureMapTransform : public drwnFeatureTransform {
 public:
    drwnTFeatureMapTransform() : drwnFeatureTransform() { /* do nothing */ }
    drwnTFeatureMapTransform(const drwnTFeatureMapTransform<FeatureMap> &t) :
       drwnFeatureTransform(t) { /* do nothing */ }
    ~drwnTFeatureMapTransform() { /* do nothing */ }

    // access
    const char *type() const { return "drwnTFeatureMapTransform"; }
    drwnTFeatureMapTransform<FeatureMap> *clone() const { 
        return new drwnTFeatureMapTransform<FeatureMap>(*this);
    }

    bool valid() const { return true; }

    // evaluation
    void transform(const vector<double>& x, vector<double>& y) const {
        const FeatureMap phi(x.size());
        y = phi(x);
    }
};

// drwnUnsupervisedTransform ------------------------------------------------
//! Implements interface for unsupervised feature transforms (i.e,
//! without class labels).

class drwnUnsupervisedTransform : public drwnFeatureTransform {
 public:
    //! default constructor
    drwnUnsupervisedTransform();
    //! copy constructor
    drwnUnsupervisedTransform(const drwnUnsupervisedTransform& t);
    virtual ~drwnUnsupervisedTransform() {
        // do nothing
    }

    // clone operation from (drwnStdObjIface)
    virtual drwnUnsupervisedTransform *clone() const = 0;

    // training
    //! Estimate the parameters of the features transformation. This function
    //! must be implemented in the derived class.
    virtual double train(const vector<vector<double> >& features) = 0;
    //! Estimate the parameters of the feature transformation using weighted
    //! training examples.
    virtual double train(const vector<vector<double> >& features,
        const vector<double>& weights);

    //! Estimate the parameters of the features transformation first applying
    //! another transform. The default implementation of this function naively
    //! transforms the data and passes the transformed data onto the relevant
    //! training code.
    virtual double train(const vector<vector<double> >& features,
        const drwnFeatureTransform& xform);
    //! Estimate the parameters of the feature transformation using weighted
    //! training examples first applying another transform. The default
    //! implementation of this function naively transforms the data and passes
    //! the transformed data onto the relevant training code.
    virtual double train(const vector<vector<double> >& features,
        const vector<double>& weights, const drwnFeatureTransform& xform);
};

// drwnSupervisedTransform --------------------------------------------------
//! Implements interface for supervised feature transforms (i.e., with
//! class labels).

class drwnSupervisedTransform : public drwnFeatureTransform {
 public:
    //! default constructor
    drwnSupervisedTransform();
    //! copy constructor
    drwnSupervisedTransform(const drwnSupervisedTransform& t);
    virtual ~drwnSupervisedTransform() {
        // do nothing
    }

    // clone operation from (drwnStdObjIface)
    virtual drwnSupervisedTransform *clone() const = 0;

    // training
    //! Estimate the parameters of the features transformation. This function
    //! must be implemented in the derived class.
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& labels) = 0;
    //! Estimate the parameters of the feature transformation using weighted
    //! training examples.
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& labels, const vector<double>& weights);

    //! Estimate the parameters of the features transformation first applying
    //! another transform. The default implementation of this function naively
    //! transforms the data and passes the transformed data onto the relevant
    //! training code.
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& labels, const drwnFeatureTransform& xform);
    //! Estimate the parameters of the feature transformation using weighted
    //! training examples first applying another transform. The default
    //! implementation of this function naively transforms the data and passes
    //! the transformed data onto the relevant training code.
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& labels, const vector<double>& weights,
        const drwnFeatureTransform& xform);
};

// drwnFeatureTransformFactory ----------------------------------------------
//! Implements factory for classes derived from drwnFeatureTransform with
//! automatic registration of built-in classes.

template <>
struct drwnFactoryTraits<drwnFeatureTransform> {
    static void staticRegistration();
};

typedef drwnFactory<drwnFeatureTransform> drwnFeatureTransformFactory;
