/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCompositeClassifier.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnClassifier.h"
#include "drwnMultiClassLogistic.h"
#include "drwnFeatureWhitener.h"

using namespace std;

// drwnCompositeClassifierMethod ---------------------------------------------

typedef enum _drwnCompositeClassifierMethod {
    DRWN_ONE_VS_ALL = 0, //!< one-versus-all binary classifiers
    DRWN_ONE_VS_ONE      //!< one-versus-one binary classifiers 
} drwnCompositeClassifierMethod;

// drwnCompositeClassifier ---------------------------------------------------
//! Implements a multi-class classifier by combining binary classifiers
//!
//! A common approach to multi-class classification is to learn a bank of
//! one-versus-one or one-versus-all classifiers and combine their output
//! via multi-class logistic regression. This class provides this functionality
//! where the type of classifier used by the one-versus-one or ones-versus-all
//! bank is controlled by the \ref BASE_CLASSIFIER parameters.
//!
//! The following code snippet shows example learning a composite classifier
//! (with boosted base classifier) on a training dataset and then testing it
//! on a hold out evaluation dataset.
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training_data.bin");
//!
//!   // train the classifier
//!   const int nFeatures = dataset.numFeatures();
//!   const int nClasses = dataset.maxTarget() + 1;
//!   drwnCompositeClassifier::BASE_CLASSIFIER = string("drwnBoostedClassifier");
//!   drwnCompositeClassifier::METHOD = DRWN_ONE_VS_ONE;
//!   drwnCompositeClassifier model(nFeatures, nClasses);
//!   model.train(dataset);
//!
//!   // load evaluation set
//!   dataset.read("testing_data.bin", false);
//!   
//!   // predict labels
//!   vector<int> predictions;
//!   model.getClassifications(dataset.features, predictions);
//! \endcode
//!
//! \sa drwnClassifier, \ref drwnTutorialML "drwnML Tutorial"

class drwnCompositeClassifier : public drwnClassifier {
 public:
    // default meta-parameters
    static string BASE_CLASSIFIER; //!< the base classifier (e.g., drwnBoostedClassifier)
    static drwnCompositeClassifierMethod METHOD; //!< composition method

 protected:
    // actual meta-parameters
    string _baseClassifier;        //!< the base classifier (e.g., drwnBoostedClassifier)
    drwnCompositeClassifierMethod _method; //!< composition method

    //! binary classifiers
    vector<drwnClassifier *> _binaryClassifiers;
    //! feature whitener for output of binary classifiers
    drwnFeatureWhitener _featureWhitener;
    //! calibration weights
    drwnTMultiClassLogistic<drwnBiasJointFeatureMap> _calibrationWeights;

 public:
    //! default constructor
    drwnCompositeClassifier();
    //! construct a classifer with \p n features and \p k classes
    drwnCompositeClassifier(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnCompositeClassifier(const drwnCompositeClassifier &c);
    ~drwnCompositeClassifier();

    // access functions
    virtual const char *type() const { return "drwnCompositeClassifier"; }
    virtual drwnCompositeClassifier *clone() const { return new drwnCompositeClassifier(*this); }

    // initialization
    virtual void initialize(unsigned n, unsigned k = 2);

    // i/o
    virtual bool save(drwnXMLNode& node) const;
    virtual bool load(drwnXMLNode& node);

    // training
    using drwnClassifier::train;
    virtual double train(const drwnClassifierDataset& dataset);

    // evaluation (log-probability)
    using drwnClassifier::getClassScores;
    virtual void getClassScores(const vector<double>& features,
        vector<double>& outputScores) const;

 protected:

};

