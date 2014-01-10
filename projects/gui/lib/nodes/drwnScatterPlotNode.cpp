/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnScatterPlotNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnScatterPlotNode.h"

using namespace std;
using namespace Eigen;

// drwnScatterPlotNode -------------------------------------------------------

drwnScatterPlotNode::drwnScatterPlotNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _colour(-1), _subSamplingRate(1), _bIgnoreMissing(false),
    _xAxisIndex(0), _yAxisIndex(1)
{
    _nVersion = 100;
    _desc = "2-dimensional data visualization";

    _inputPorts.push_back(new drwnInputPort(this, "dataIn",
            "N-by-K matrix of feature vectors"));
    _inputPorts.push_back(new drwnInputPort(this, "labelsIn",
            "Propertyal N-by-K or N-by-1 matrix of class labels"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
    declareProperty("x-axis", new drwnIntegerProperty(&_xAxisIndex));
    declareProperty("y-axis", new drwnIntegerProperty(&_yAxisIndex));
}

drwnScatterPlotNode::drwnScatterPlotNode(const drwnScatterPlotNode& node) :
    drwnNode(node), _colour(node._colour), _subSamplingRate(node._subSamplingRate),
    _bIgnoreMissing(node._bIgnoreMissing), _xAxisIndex(node._xAxisIndex), _yAxisIndex(node._yAxisIndex)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
    declareProperty("x-axis", new drwnIntegerProperty(&_xAxisIndex));
    declareProperty("y-axis", new drwnIntegerProperty(&_yAxisIndex));
}

drwnScatterPlotNode::~drwnScatterPlotNode()
{
    // do nothing
}

// gui
void drwnScatterPlotNode::showWindow()
{
    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnGUIScatterPlot(this);
        _window->Show();
    }

    updateWindow();
}

void drwnScatterPlotNode::updateWindow()
{
    if ((_window == NULL) || (!_window->IsShown())) return;

    if (_features.empty() || (_xAxisIndex >= (int)_features[0].size()) ||
        (_yAxisIndex >= (int)_features[0].size())) {
        ((drwnGUIScatterPlot *)_window)->setData(vector<double>(), vector<double>());
        return;
    }

    vector<double> x(_features.size());
    vector<double> y(_features.size());
    for (int i = 0; i < (int)_features.size(); i++) {
        x[i] = _xAxisIndex < 0 ? drand48() : _features[i][_xAxisIndex];
        y[i] = _yAxisIndex < 0 ? drand48() : _features[i][_yAxisIndex];
    }

    ((drwnGUIScatterPlot *)_window)->setData(x, y, _labels);
}

// processing
void drwnScatterPlotNode::evaluateForwards()
{
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblTarget = _inputPorts[1]->getTable();
    if (tblData == NULL) {
        DRWN_LOG_WARNING("node \"" << _name << "\" is missing data input");
        return;
    }

    _features.clear();
    _labels.clear();

    // iterate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // skip missing targets or non-matching colour
        if ((tblTarget != NULL) && !tblTarget->hasKey(*it)) continue;
        if (!getOwner()->getDatabase()->matchColour(*it, _colour)) continue;

        // get data
        const drwnDataRecord *recData = tblData->lockRecord(*it);
        const drwnDataRecord *recTarget = 
            (tblTarget == NULL) ? NULL : tblTarget->lockRecord(*it);

        // TODO: check data is the right format
        // target columns must be 1; rows must be 1 or data rows
        // (if exists) weight columns must be 1; rows must be 1 or data rows
        // data columns must match accumulated features

        // accumulate data (with sampling)
        for (int i = 0; i < recData->numObservations(); i++) {
            // ignore "unknown" class labels
            if (_bIgnoreMissing && (recTarget != NULL) && (recTarget->data()(i) < 0))
                continue;

            // random sample
            if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
                continue;

            // add features
            _features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&_features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            if (recTarget != NULL) {
                _labels.push_back((int)recTarget->data()(i));
            } else {
                _labels.push_back(-1);
            }
        }

        // release records
        tblData->unlockRecord(*it);
        if (recTarget != NULL) tblTarget->unlockRecord(*it);
    }

    DRWN_END_PROGRESS;
    updateWindow();
}

void drwnScatterPlotNode::resetParameters()
{
    _features.clear();
    _labels.clear();
    updateWindow();
}

// property callback
void drwnScatterPlotNode::propertyChanged(const string& name)
{
    if ((name == string("x-axis")) || (name == string("y-axis"))) {
        updateWindow();        
    }
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Visualization", drwnScatterPlotNode);
