/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnScatterPlotNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Node for 2-dimensional data plotting. For higher dimensional visualization
**  use the drwnDataExplorerNode. Set feature index to negative for random
**  dither on that axis.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnScatterPlotNode -------------------------------------------------------

class drwnScatterPlotNode : public drwnNode {
 protected:
    int _colour;                // colour of records for exploring
    int _subSamplingRate;       // subsampling rate
    bool _bIgnoreMissing;       // ignore missing labels
    int _xAxisIndex;            // data dimension for x-axis
    int _yAxisIndex;            // data dimension for y-axis

    vector<vector<double> > _features; // data features
    vector<int> _labels;        // data labels (markers)

 public:
    drwnScatterPlotNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnScatterPlotNode(const drwnScatterPlotNode& node);
    virtual ~drwnScatterPlotNode();

    // i/o
    const char *type() const { return "drwnScatterPlotNode"; }
    drwnScatterPlotNode *clone() const { return new drwnScatterPlotNode(*this); }

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

DRWN_DECLARE_AUTOREGISTERNODE(drwnScatterPlotNode);
