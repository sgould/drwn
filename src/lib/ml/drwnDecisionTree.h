/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDecisionTree.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "drwnBase.h"
#include "drwnClassifier.h"

using namespace std;

// drwnTreeSplitCriterion ----------------------------------------------------

typedef enum _drwnTreeSplitCriterion {
    DRWN_DT_SPLIT_ENTROPY, DRWN_DT_SPLIT_MISCLASS, DRWN_DT_SPLIT_GINI
} drwnTreeSplitCriterion;

// drwnDecisionTree ---------------------------------------------------------
//! Implements a (binary-split) decision tree classifier of arbitrary depth.
//!
//! The following code snippet shows example learning a decision tree classifier
//! on a training dataset and then testing it on a hold out evaluation dataset.
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training_data.bin");
//!
//!   // train the classifier
//!   const int nFeatures = dataset.numFeatures();
//!   const int nClasses = dataset.maxTarget() + 1;
//!   drwnDecisionTree model(nFeatures, nClasses);
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
//! The decision classifier has a number of parameters for controlling
//! it's operation during training. See \ref drwnDecisionTree::MAX_DEPTH, 
//! \ref drwnDecisionTree::MAX_FEATURE_THRESHOLDS, \ref drwnDecisionTree::MIN_SAMPLES,
//! and \ref drwnDecisionTree::SPLIT_CRITERION for details.
//!
//! \sa drwnClassifier, \ref drwnTutorialML "drwnML Tutorial"

class drwnDecisionTree : public drwnClassifier {
 public:
    friend class drwnDecisionTreeThread;
    friend class drwnDecisionTreeConfig;
    friend class drwnBoostedClassifier;
    friend class drwnRandomForest;

 public:
    static int MAX_DEPTH;          //!< default maximum tree depth
    static int MAX_FEATURE_THRESHOLDS; //!< maximum number of thresholds to try during learning
    static int MIN_SAMPLES;       //!< minimum number of samples (after first split)
    static drwnTreeSplitCriterion SPLIT_CRITERION; //!< tree split criteria during learning

 protected:
    int _splitIndx;                //!< variable index on which to split
    double _splitValue;            //!< split value (go left if less than)
    drwnDecisionTree *_leftChild;  //!< left child (or NULL)
    drwnDecisionTree *_rightChild; //!< right child (or NULL)
    Eigen::VectorXd _scores;       //!< log-marginal for each class at this node
    int _predictedClass;           //!< argmax of _scores
    
    int _maxDepth;                 //!< maximum depth of decision tree

 public:
    //! default constructor
    drwnDecisionTree();
    //! construct a classifier object for \p n features and \p k classes
    drwnDecisionTree(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnDecisionTree(const drwnDecisionTree &c);
    ~drwnDecisionTree();

    // access functions
    virtual const char *type() const { return "drwnDecisionTree"; }
    virtual drwnDecisionTree *clone() const { return new drwnDecisionTree(*this); }

    // initialization
    virtual void initialize(unsigned n, unsigned k = 2);

    // i/o
    virtual bool save(drwnXMLNode& node) const;
    virtual bool load(drwnXMLNode& node);

    // training
    using drwnClassifier::train;
    virtual double train(const drwnClassifierDataset& dataset);
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets);
    virtual double train(const vector<vector<double> >& features,
        const vector<int>& targets, const vector<double>& weights);

    // evaluation (log-probability)
    using drwnClassifier::getClassScores;
    virtual void getClassScores(const vector<double>& features,
        vector<double>& outputScores) const;

    // evaluation (classification)
    using drwnClassifier::getClassification;
    virtual int getClassification(const vector<double>& features) const;

 protected:
    // evaluation
    const Eigen::VectorXd &evaluate(const Eigen::VectorXd& x) const;

    // training
    static void computeSortedFeatureIndex(const vector<vector<double> >& x,
        vector<vector<int> >& sortIndex);
    void learnDecisionTree(const vector<vector<double> >& x, const vector<int>& y,
        const vector<double>& w, const vector<vector<int> >& sortIndex,
        const drwnBitArray& sampleIndex);    
};

