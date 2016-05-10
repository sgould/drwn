/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDecisionTreeNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements a multi-class decision tree.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

using namespace std;

// drwnDecisionTreeNode ------------------------------------------------------

class drwnDecisionTree;

class drwnDecisionTreeNode : public drwnNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)
    //int _subSamplingRate;  // sub-sampling rate for training (e.g., every n-th)
    drwnDecisionTree _classifier; // boosted classifier

 public:
    drwnDecisionTreeNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnDecisionTreeNode(const drwnDecisionTreeNode& node);
    virtual ~drwnDecisionTreeNode();

    // i/o
    const char *type() const { return "drwnDecisionTreeNode"; }
    drwnDecisionTreeNode *clone() const { return new drwnDecisionTreeNode(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // processing
    void evaluateForwards();
    void updateForwards();

    // learning
    void resetParameters();
    void initializeParameters();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnDecisionTreeNode);


