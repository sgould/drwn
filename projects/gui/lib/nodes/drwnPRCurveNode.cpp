/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPRCurveNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

#include "drwnPRCurveNode.h"

using namespace std;
using namespace Eigen;

// drwnPRCurveNode -------------------------------------------------------

drwnPRCurveNode::drwnPRCurveNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _colour(-1), _bIgnoreMissing(false)
{
    _nVersion = 100;
    _desc = "precision-recall curve";

    _inputPorts.push_back(new drwnInputPort(this, "actual",
            "N-by-K or N-by-1 matrix of classifications"));
    _inputPorts.push_back(new drwnInputPort(this, "predicted",
            "N-by-K matrix of classifications scores"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
}

drwnPRCurveNode::drwnPRCurveNode(const drwnPRCurveNode& node) :
    drwnNode(node), _colour(node._colour), _bIgnoreMissing(node._bIgnoreMissing),
    _curves(node._curves)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
}

drwnPRCurveNode::~drwnPRCurveNode()
{
    // do nothing
}

// gui
void drwnPRCurveNode::showWindow()
{
    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnGUILinePlot(this);
        ((drwnGUILinePlot *)_window)->setXRange(0.0, 1.0);
        ((drwnGUILinePlot *)_window)->setYRange(0.0, 1.0);
        _window->Show();
    }

    updateWindow();
}

void drwnPRCurveNode::updateWindow()
{
    if ((_window == NULL) || (!_window->IsShown())) return;

    ((drwnGUILinePlot *)_window)->clear();
    for (int i = 0; i < (int)_curves.size(); i++) {
        ((drwnGUILinePlot *)_window)->addCurve(_curves[i].getCurve(), i);
    }
}

// processing
void drwnPRCurveNode::evaluateForwards()
{
    _curves.clear();
    drwnDataTable *tblActual = _inputPorts[0]->getTable();
    drwnDataTable *tblPredicted = _inputPorts[1]->getTable();
    if ((tblActual == NULL) || (tblPredicted == NULL)) {
        DRWN_LOG_WARNING("node \"" << _name << "\" requires two inputs");
        return;
    }

    // interate over records
    vector<string> keys = tblActual->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        if (!getOwner()->getDatabase()->matchColour(*it, _colour)) continue;

        const drwnDataRecord *recActual = tblActual->lockRecord(*it);
        DRWN_ASSERT(recActual != NULL);

        const drwnDataRecord *recPredicted = tblPredicted->lockRecord(*it);
        if (recPredicted == NULL) {
            DRWN_LOG_WARNING("missing predictions for record \"" << *it << "\"");
        } else {

            // check dimensions
            if (recPredicted->numObservations() == 0) {
                DRWN_LOG_WARNING("missing predictions for \"" << *it << "\"");

            } else if (recActual->numObservations() != recPredicted->numObservations()) {
                DRWN_LOG_ERROR("size mismatch between actual (" << recActual->numObservations()
                    << ") and predicted observations (" << recPredicted->numObservations() << ")");

            } else {
                VectorXi actualClass;
                if (recActual->numFeatures() == 1) {
                    actualClass = recActual->data().col(0).cast<int>();
                } else {
                    actualClass = VectorXi::Zero(recActual->numObservations());
                    for (int i = 0; i < recActual->numObservations(); i++) {
                        recActual->data().row(i).maxCoeff(&actualClass[i]);
                    }
                }

                VectorXd predictedClass = recPredicted->data().col(0);
                if (_curves.empty()) {
                    _curves.resize(recPredicted->numFeatures());
                }
                DRWN_ASSERT((int)_curves.size() == recPredicted->numFeatures());

                for (int i = 0; i < actualClass.rows(); i++) {
                    if (actualClass[i] == -1) continue;
                    if (actualClass[i] >= (int)_curves.size()) {
                        DRWN_LOG_ERROR("mismatch between actual and predicted labels in \""
                            << getName() << "\" for record \"" << *it << "\"");
                        break;
                    }
                    for (int j = 0; j < (int)_curves.size(); j++) {
                        if (actualClass[i] == j) {
                            _curves[j].accumulatePositives(recPredicted->data()(i, j));
                        } else {
                            _curves[j].accumulateNegatives(recPredicted->data()(i, j));
                        }
                    }
                }
            }
        }

        // release records
        tblActual->unlockRecord(*it);
        tblPredicted->unlockRecord(*it);
    }

    // show results (average precision)
    for (int i = 0; i < (int)_curves.size(); i++) {
        DRWN_LOG_VERBOSE("11-pt average precision for \"" << getName()
            << "\" class " << i << " is " << toString(_curves[i].averagePrecision()));
    }

    DRWN_END_PROGRESS;
    updateWindow();
}

void drwnPRCurveNode::resetParameters()
{
    _curves.clear();
    updateWindow();
}

// property callback
void drwnPRCurveNode::propertyChanged(const string& name)
{
    //updateWindow();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Visualization", drwnPRCurveNode);
