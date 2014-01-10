/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPRCurveNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Node for plotting a precision-recall curve.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnPRCurveNode -----------------------------------------------------------

class drwnPRCurveNode : public drwnNode {
 protected:
    int _colour;                 // colour of records for plotting
    bool _bIgnoreMissing;        // ignore missing labels

    vector<drwnPRCurve> _curves; // vector of results for each class

 public:
    drwnPRCurveNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnPRCurveNode(const drwnPRCurveNode& node);
    virtual ~drwnPRCurveNode();

    // i/o
    const char *type() const { return "drwnPRCurveNode"; }
    drwnPRCurveNode *clone() const { return new drwnPRCurveNode(*this); }

    // gui
    void showWindow();
    void updateWindow();

    // processing
    void evaluateForwards();

    // learning
    void resetParameters();

 protected:
    // property callback
    virtual void propertyChanged(const string& name);    
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnPRCurveNode);
