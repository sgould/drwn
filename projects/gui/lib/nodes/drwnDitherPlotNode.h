/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDitherPlotNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Node for feature visualization.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnDitherPlotNode --------------------------------------------------------

class drwnDitherPlotNode : public drwnNode {
 protected:
    int _colour;                // colour of records for exploring
    int _subSamplingRate;       // subsampling rate
    bool _bIgnoreMissing;       // ignore missing labels

    vector<vector<double> > _features; // data features
    vector<int> _labels;        // data labels (markers)

 public:
    drwnDitherPlotNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnDitherPlotNode(const drwnDitherPlotNode& node);
    virtual ~drwnDitherPlotNode();

    // i/o
    const char *type() const { return "drwnDitherPlotNode"; }
    drwnDitherPlotNode *clone() const { return new drwnDitherPlotNode(*this); }

    // gui
    void showWindow();
    void updateWindow();

    // processing
    void evaluateForwards();

    // learning
    void resetParameters();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnDitherPlotNode);
