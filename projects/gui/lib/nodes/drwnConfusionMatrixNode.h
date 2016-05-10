/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfusionMatrixNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements a confusion matrix performance evaluation sink node. The node
**  can be used to back-propagate a weighted performance metric.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnConfusionMatrixNode ---------------------------------------------------

class drwnConfusionMatrixNode : public drwnNode {
 protected:
    int _colour;                // colour of records for evaluating
    string _filename;           // output confusion matrix

    Eigen::MatrixXd _confusion; // confusion matrix

 public:
    drwnConfusionMatrixNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnConfusionMatrixNode(const drwnConfusionMatrixNode& node);
    virtual ~drwnConfusionMatrixNode();

    // i/o
    const char *type() const { return "drwnConfusionMatrixNode"; }
    drwnConfusionMatrixNode *clone() const { return new drwnConfusionMatrixNode(*this); }

    // gui
    void showWindow();
    void updateWindow();

    // processing
    void evaluateForwards();

    // learning
    void resetParameters();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnConfusionMatrixNode);
