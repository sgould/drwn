/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiClassLogisticNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements a multi-class logistic classifier with parameters learnded
**  using a log-likelihood penalty.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

using namespace std;

// drwnMultiClassLogisticNode ------------------------------------------------

class drwnMultiClassLogisticNode : public drwnAdaptiveNode, protected drwnOptimizer {
 protected:
    MatrixXd _theta;        // class weights
    bool _bOutputScores;    // output scores instead of normalized marginals

    // cached data for parameter estimation
    vector<vector<double> > _features;
    vector<int> _target;
    vector<double> _weights;

 public:
    drwnMultiClassLogisticNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnMultiClassLogisticNode(const drwnMultiClassLogisticNode& node);
    virtual ~drwnMultiClassLogisticNode();

    // i/o
    const char *type() const { return "drwnMultiClassLogisticNode"; }
    drwnMultiClassLogisticNode *clone() const { return new drwnMultiClassLogisticNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();

    // learning
    void resetParameters();
    void initializeParameters();
    
 protected:
    // drwnOptimizer interface
    double objective(const double *x) const;
    void gradient(const double *x, double *df) const;
    double objectiveAndGradient(const double *x, double *df) const;
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnMultiClassLogisticNode);


