/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRandomForest.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnClassifier.h"
#include "drwnDecisionTree.h"

using namespace std;

// drwnRandomForest ----------------------------------------------------------
//! Implements a Random forest ensemble of decision trees classifier. See
//! L. Breiman, "Random Forests", Machine Learning, 2001.
//!
//! The following code snippet shows example learning a random forest classifier
//! on a training dataset and then testing it on a hold out evaluation dataset.
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training_data.bin");
//!
//!   // train the classifier
//!   const int nFeatures = dataset.numFeatures();
//!   const int nClasses = dataset.maxTarget() + 1;
//!   drwnRandomForest model(nFeatures, nClasses);
//!   model.train(dataset);
//!
//!   // load evaluation set
//!   dataset.read("testing_data.bin", false);
//!   
//!   // predict labels
//!   vector<int> predictions;
//!   model.getClassifications(dataset.features, predictions);
//! \endcode

class drwnRandomForest : public drwnClassifier {
 public:
    // default training parameters
    static int NUM_TREES;    //!< default number of trees in the forest (used during construction)
    static int MAX_DEPTH;    //!< default depth of each tree (used during construction)
    static int MAX_FEATURES; //!< maximum number of features to use at each iteration (used during construction)

 protected:
    // actual training parameters
    int _numTrees;                 //!< number of trees to learn
    int _maxDepth;                 //!< maximum depth of each decision tree
    int _maxFeatures;              //!< maximum number of features to use at each iteration

    //! forest
    vector<drwnDecisionTree *> _forest;
    //! weight for each weak tree
    vector<double> _alphas;

 public:
    //! default constructor
    drwnRandomForest();
    //! construct a random forest classifier for \p k classes and \p n features
    drwnRandomForest(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnRandomForest(const drwnRandomForest &c);
    ~drwnRandomForest();

    // access functions
    virtual const char *type() const { return "drwnRandomForest"; }
    virtual drwnRandomForest *clone() const { return new drwnRandomForest(*this); }

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
};

