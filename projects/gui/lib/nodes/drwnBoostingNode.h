/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBoostingNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements a boosted decision tree classifier.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

using namespace std;

// drwnBoostedClassifierNode -------------------------------------------------

class drwnBoostedClassifierNode : public drwnNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)
    int _subSamplingRate;  // sub-sampling rate for training (e.g., every n-th)
    drwnBoostedClassifier _classifier; // boosted classifier

 public:
    drwnBoostedClassifierNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnBoostedClassifierNode(const drwnBoostedClassifierNode& node);
    virtual ~drwnBoostedClassifierNode();

    // i/o
    const char *type() const { return "drwnBoostedClassifierNode"; }
    drwnBoostedClassifierNode *clone() const { return new drwnBoostedClassifierNode(*this); }
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

DRWN_DECLARE_AUTOREGISTERNODE(drwnBoostedClassifierNode);
