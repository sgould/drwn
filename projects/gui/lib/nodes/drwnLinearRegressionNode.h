/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearRegressionNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements a linear regression node with sum-of-squares (L2) or huber
**  (smooth L1) penalty.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

using namespace std;

// drwnLinearRegressionNode --------------------------------------------------

class drwnLinearRegressionNode : public drwnAdaptiveNode, protected drwnOptimizer {
 protected:
    static vector<string> _penaltyProperties;
    int _penalty;           // huber or sum-of-squares
    double _argument;       // argument for huber penalty
    
    VectorXd _theta;        // regression parameters

    // cached data for parameter estimation
    vector<vector<double> > _features;
    vector<double> _target;
    vector<double> _weights;

 public:
    drwnLinearRegressionNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnLinearRegressionNode(const drwnLinearRegressionNode& node);
    virtual ~drwnLinearRegressionNode();

    // i/o
    const char *type() const { return "drwnLinearRegressionNode"; }
    drwnLinearRegressionNode *clone() const { return new drwnLinearRegressionNode(*this); }

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

DRWN_DECLARE_AUTOREGISTERNODE(drwnLinearRegressionNode);


