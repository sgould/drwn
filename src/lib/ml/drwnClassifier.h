/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnClassifier.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnDataset.h"

using namespace std;

// drwnClassifier -----------------------------------------------------------
//! Implements the interface for a generic machine learning classifier.
//!
//! Derived classes must implement training and evaluation methods and should
//! also override XML save and load methods, and optionally the initialize method.
//!
//! \sa \ref drwnTutorialML "drwnML Tutorial"

class drwnClassifier : public drwnStdObjIface, public drwnProperties {
 protected:
    int _nFeatures;     //!< number of features
    int _nClasses;      //!< number of classes
    bool _bValid;       //!< true if the classifier has been trained or loaded

 public:
    //! default constructor
    drwnClassifier();
    //! construct a classifer with \p n features and \p k classes
    drwnClassifier(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnClassifier(const drwnClassifier &c);
    virtual ~drwnClassifier() { /* do nothing */ }

    // access functions
    //! returns the number of features expected by the classifier object
    int numFeatures() const { return _nFeatures; }
    //! returns the number of classes predicted by the classifier object
    int numClasses() const { return _nClasses; }
    //! returns true if the classifier is valid (has been initialized and trained)
    virtual bool valid() const { return _bValid; }

    // initialization
    //! initialize the classifier object for \p n features and \p k classes
    virtual void initialize(unsigned n, unsigned k = 2);

    // i/o
    virtual drwnClassifier *clone() const = 0;
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // training
    //! train the parameters of the classifier from a drwnClassifierDataset object
    virtual double train(const drwnClassifierDataset& dataset) = 0;
    //! train the parameters of the classifier from a set of features and corresponding labels
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets);
    //! train the parameters of the classifier from a weighted set of features and
    //! corresponding labels
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets, const vector<double>& weights);
    //! train the parameters of the classifier from data stored in \p filename
    virtual double train(const char *filename);

    // evaluation (log-probability unnormalized)
    //! compute the unnormalized log-probability for a single feature vector
    virtual void getClassScores(const vector<double>& features,
        vector<double>& outputScores) const = 0;
    //! compute the unnormalized log-probability for a set of feature vectors
    virtual void getClassScores(const vector<vector<double> >& features,
        vector<vector<double> >& outputScores) const;

    // evaluation (normalized marginals)
    //! compute the class marginal probabilities for a single feature vector
    virtual void getClassMarginals(const vector<double>& features,
        vector<double>& outputMarginals) const;
    //! compute the class marginal probabilities for a set of feature vectors
    virtual void getClassMarginals(const vector<vector<double> >& features,
        vector<vector<double> >& outputMarginals) const;

    // evaluation (classification)
    //! return the most likely class label for a single feature vector
    virtual int getClassification(const vector<double>& features) const;
    //! compute the most likely class labels for a set of feature vector
    virtual void getClassifications(const vector<vector<double> >& features,
        vector<int>& outputLabels) const;
};

// drwnClassifierFactory ----------------------------------------------------
//! Implements factory for classes derived from drwnClassifier with automatic
//! registration of built-in classes.

template <>
struct drwnFactoryTraits<drwnClassifier> {
    static void staticRegistration();
};

typedef drwnFactory<drwnClassifier> drwnClassifierFactory;
