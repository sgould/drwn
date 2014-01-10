/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBoostedClassifier.h
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

// drwnBoostingMethod --------------------------------------------------------

typedef enum _drwnBoostingMethod {
    DRWN_BOOST_DISCRETE, DRWN_BOOST_GENTLE, DRWN_BOOST_REAL
} drwnBoostingMethod;

// drwnBoostedClassifier -----------------------------------------------------
//! Implements a mult-class boosted decision-tree classifier. See Zhu et al., 
//! Multi-class AdaBoost, 2006.
//!
//! The following code snippet shows example learning a boosted classifier on
//! a training dataset and then testing it on a hold out evaluation dataset.
//! \code
//!   // load training dataset
//!   drwnClassifierDataset dataset;
//!   dataset.read("training_data.bin");
//!
//!   // train the classifier
//!   const int nFeatures = dataset.numFeatures();
//!   const int nClasses = dataset.maxTarget() + 1;
//!   drwnBoostedClassifier model(nFeatures, nClasses);
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
//! The boosted classifier has a number of parameters for controlling
//! it's operation during training. See \ref METHOD, \ref NUM_ROUNDS,
//! \ref MAX_DEPTH and \ref SHRINKAGE for details.
//!
//! \sa drwnClassifier, \ref drwnTutorialML "drwnML Tutorial"

class drwnBoostedClassifier : public drwnClassifier {
 public:
    // default training parameters
    //! controls the re-weighting of data samples at the end of each training iteration
    static drwnBoostingMethod METHOD;
    static int NUM_ROUNDS;   //!< maximum number of boosting rounds
    static int MAX_DEPTH;    //!< maximum depth of each decision tree
    static double SHRINKAGE; //!< boosting shrinkage

 protected:
    // actual training parameters
    drwnBoostingMethod _method;    //!< boosting method
    int _numRounds;                //!< number of rounds of boosting
    int _maxDepth;                 //!< maximum depth of each decision tree
    double _shrinkage;             //!< boosting shrinkage

    //! weak learners
    vector<drwnDecisionTree *> _weakLearners;
    //! weight for each weak learner
    vector<double> _alphas;

 public:
    //! default constructor
    drwnBoostedClassifier();
    //! construct a classifer with \p n features and \p k classes
    drwnBoostedClassifier(unsigned n, unsigned k = 2);
    //! copy constructor
    drwnBoostedClassifier(const drwnBoostedClassifier &c);
    ~drwnBoostedClassifier();

    // access functions
    virtual const char *type() const { return "drwnBoostedClassifier"; }
    virtual drwnBoostedClassifier *clone() const { return new drwnBoostedClassifier(*this); }

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
