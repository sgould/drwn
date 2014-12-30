/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDitherPlotNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnDitherPlotNode.h"

using namespace std;
using namespace Eigen;

// drwnDitherPlotNode -------------------------------------------------------

drwnDitherPlotNode::drwnDitherPlotNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _colour(-1), _subSamplingRate(1), _bIgnoreMissing(false)
{
    _nVersion = 100;
    _desc = "feature vs. class visualization";

    _inputPorts.push_back(new drwnInputPort(this, "dataIn",
            "N-by-K matrix of feature vectors"));
    _inputPorts.push_back(new drwnInputPort(this, "labelsIn",
            "Propertyal N-by-K or N-by-1 matrix of class labels"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
}

drwnDitherPlotNode::drwnDitherPlotNode(const drwnDitherPlotNode& node) :
    drwnNode(node), _colour(node._colour), _subSamplingRate(node._subSamplingRate),
    _bIgnoreMissing(node._bIgnoreMissing)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
}

drwnDitherPlotNode::~drwnDitherPlotNode()
{
    // do nothing
}

// gui
void drwnDitherPlotNode::showWindow()
{
    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnGUIScatterPlot(this);
        _window->Show();
    }

    updateWindow();
}

void drwnDitherPlotNode::updateWindow()
{
    if ((_window == NULL) || (!_window->IsShown())) return;

    if (_features.empty()) {
        ((drwnGUIScatterPlot *)_window)->setData(vector<double>(), vector<double>());
        return;
    }

    const int numLabels = drwn::maxElem(_labels);
    vector<double> x(_features.size() * _features[0].size());
    vector<double> y(_features.size() * _features[0].size());
    vector<int> labels(_features.size() * _features[0].size());
    int k = 0;
    for (int i = 0; i < (int)_features[0].size(); i++) {
        for (int j = 0; j < (int)_features.size(); j++) {
            x[k] = i + (0.8 * drand48() + _labels[j]) / (numLabels + 2);
            y[k] = _features[j][i];
            labels[k] = _labels[j];
            k += 1;
        }
    }

    ((drwnGUIScatterPlot *)_window)->setData(x, y, labels);
}

// processing
void drwnDitherPlotNode::evaluateForwards()
{
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblLabel = _inputPorts[1]->getTable();
    if ((tblData == NULL) || (tblLabel == NULL)) {
        DRWN_LOG_WARNING("node \"" << _name << "\" is missing data or label input");
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
        if (!tblLabel->hasKey(*it)) continue;
        if (!getOwner()->getDatabase()->matchColour(*it, _colour)) continue;

        // get data
        const drwnDataRecord *recData = tblData->lockRecord(*it);
        const drwnDataRecord *recLabel = tblLabel->lockRecord(*it);

        // TODO: check data is the right format
        // target columns must be 1; rows must be 1 or data rows
        // (if exists) weight columns must be 1; rows must be 1 or data rows
        // data columns must match accumulated features

        // accumulate data (with sampling)
        for (int i = 0; i < recData->numObservations(); i++) {
            // ignore "unknown" class labels
            if (_bIgnoreMissing && (recLabel->data()(i) < 0))
                continue;

            // random sample
            if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
                continue;

            // add features
            _features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&_features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            _labels.push_back((int)recLabel->data()(i));
        }

        // release records
        tblData->unlockRecord(*it);
        tblLabel->unlockRecord(*it);
    }

    DRWN_END_PROGRESS;
    updateWindow();
}

void drwnDitherPlotNode::resetParameters()
{
    _features.clear();
    _labels.clear();
    updateWindow();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Visualization", drwnDitherPlotNode);
