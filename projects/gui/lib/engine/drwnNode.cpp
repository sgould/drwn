/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnDatabase.h"
#include "drwnGraph.h"
#include "drwnNode.h"
#include "drwnPort.h"

using namespace std;
using namespace Eigen;

// drwnNode ------------------------------------------------------------------

drwnNode::drwnNode(const char *name, drwnGraph *owner) :
    drwnProperties(), _nVersion(0), _owner(owner), _name(""), _x(0), _y(0), _window(NULL)
{
    if (name != NULL) _name = string(name);
    declareProperty("name", new SetNameInterface(this));
    declareProperty("version", new drwnIntegerProperty((int *)&_nVersion, true));
}

drwnNode::drwnNode(const drwnNode& node) :
    drwnProperties(), _nVersion(node._nVersion), _owner(node._owner),
    _name(node._name), _x(node._x), _y(node._y), _window(NULL)
{
    // declare properties
    declareProperty("name", new SetNameInterface(this));
    declareProperty("version", new drwnIntegerProperty((int *)&_nVersion, true));

    // copy ports
    for (int i = 0; i < node.numInputPorts(); i++) {
        _inputPorts.push_back(new drwnInputPort(this,
            node._inputPorts[i]->getName(), node._inputPorts[i]->getDescription()));
    }

    for (int i = 0; i < node.numOutputPorts(); i++) {
        _outputPorts.push_back(new drwnOutputPort(this,
            node._outputPorts[i]->getName(), node._outputPorts[i]->getDescription()));
    }
}

drwnNode::~drwnNode()
{
    // delete ports
    for (vector<drwnInputPort *>::iterator it = _inputPorts.begin();
        it != _inputPorts.end(); it++) {
        delete *it;
    }
    for (vector<drwnOutputPort *>::iterator it = _outputPorts.begin();
        it != _outputPorts.end(); it++) {
        delete *it;
    }

    // delete GUI window
    delete _window;
}

void drwnNode::setName(const string& name)
{
    // already has this name
    if (name == _name) return;

    // no owner
    if (_owner == NULL) {
        _name = name;
        return;
    }

    // otherwise check that node with this name doesn't already exist
    string newName = _owner->getNewName(name.c_str());
    if (newName != name) {
        DRWN_LOG_WARNING("node with name \"" << name << "\" already exists in the network");
    }

    // rename database tables
    for (vector<drwnOutputPort *>::iterator it = _outputPorts.begin(); it != _outputPorts.end(); it++) {
        DRWN_ASSERT((*it)->getOwner() == this);
        drwnDataTable *tbl = (*it)->getTable();
        if (tbl != NULL) {
            tbl->getOwner()->renameTable(tbl->name(),
                (newName + string(".") + (*it)->getName()).c_str());
        }
    }

    // rename node
    _name = newName;
}

// i/o
bool drwnNode::save(drwnXMLNode& xml) const
{
    // write header
    drwnAddXMLAttribute(xml, "type", this->type(), false);
    drwnAddXMLAttribute(xml, "version", toString(_nVersion).c_str(), false);

    // write standard (non-read-only) propertys
    writeProperties(xml);

    // write additional parameters
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "location", NULL, false);
    drwnAddXMLAttribute(*node, "x", toString(_x).c_str(), false);
    drwnAddXMLAttribute(*node, "y", toString(_y).c_str(), false);

    return true;
}

bool drwnNode::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(!drwnIsXMLEmpty(xml));

    // check version number
    if (drwnGetXMLAttribute(xml, "version") != NULL) {
        unsigned v = atoi(drwnGetXMLAttribute(xml, "version"));
        if (v > _nVersion) {
            DRWN_LOG_WARNING("node " << _name << " has higher version than recognized by this build");
        } else {
            _nVersion = v;
        }
    }

    // read standard (non-read-only) propertys
    readProperties(xml);

    // read additional configuration
    drwnXMLNode *node = xml.first_node("location");
    if (node != NULL) {
        DRWN_ASSERT((drwnGetXMLAttribute(*node, "x") != NULL) && 
            (drwnGetXMLAttribute(*node, "y") != NULL));
        _x = atoi(drwnGetXMLAttribute(*node, "x"));
        _y = atoi(drwnGetXMLAttribute(*node, "y"));
    }

    return true;
}

// gui
void drwnNode::showWindow()
{
    // do nothing
}

void drwnNode::hideWindow()
{
    if (_window != NULL) {
        delete _window;
        _window = NULL;
    }
}

bool drwnNode::isShowingWindow() const
{
    return ((_window != NULL) && (_window->IsShown()));
}

void drwnNode::updateWindow()
{
    // do nothing
}

// connectivity
drwnInputPort *drwnNode::getInputPort(const char *name) const
{
    for (vector<drwnInputPort *>::const_iterator it = _inputPorts.begin();
        it != _inputPorts.end(); it++) {
        if (!strcmp((*it)->getName(), name)) {
            return *it;
        }
    }

    DRWN_LOG_WARNING("could not find input port " << name << " on node " << _name);
    return NULL;
}

drwnOutputPort *drwnNode::getOutputPort(const char *name) const
{
    for (vector<drwnOutputPort *>::const_iterator it = _outputPorts.begin();
        it != _outputPorts.end(); it++) {
        if (!strcmp((*it)->getName(), name)) {
            return *it;
        }
    }

    DRWN_LOG_WARNING("could not find output port " << name << " on node " << _name);
    return NULL;
}

// clear data tables
void drwnNode::clearOutput()
{
    for (vector<drwnOutputPort *>::iterator it = _outputPorts.begin();
         it != _outputPorts.end(); it++) {
        drwnDataTable *tbl = (*it)->getTable();
        if (tbl != NULL) tbl->clear();
    }
}

// processing
void drwnNode::initializeForwards(bool bClearOutput)
{
    // clear output tables
    if (bClearOutput) {
        clearOutput();
    }

    // derived classes will implement specific initializations
}

void drwnNode::evaluateForwards()
{
    // should be overridden by derived classes with output ports
    if (!_outputPorts.empty()) {
        DRWN_LOG_ERROR("no forward evalution function defined for node " << type());
    }
}

void drwnNode::updateForwards()
{
    // evaluate forwards if updateForwards not implemented in derived class
    if (!_outputPorts.empty()) {
        DRWN_LOG_WARNING("update not implemented for this block; performing full forward evaluation");
        evaluateForwards();
    }
}

void drwnNode::finalizeForwards()
{
    // do nothing: derived classes will implement specific finalizations
}

void drwnNode::initializeBackwards()
{
    DRWN_NOT_IMPLEMENTED_YET;
}

void drwnNode::propagateBackwards()
{
    // should be overridden by derived classes with input ports
    if (_inputPorts.empty()) {
        DRWN_LOG_ERROR("no backward propagation function defined for node " << type());
    }
}

void drwnNode::finalizeBackwards()
{
    // do nothing: derived classes will implement specific finalizations
}

void drwnNode::resetParameters()
{
    // do nothing (no parameters)
}

void drwnNode::initializeParameters()
{
    // do nothing (no parameters)
}

// drwnSimpleNode ------------------------------------------------------------

drwnSimpleNode::drwnSimpleNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner)
{
    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "generic data matrix"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "generic data matrix"));
}

drwnSimpleNode::~drwnSimpleNode()
{
    // do nothing
}

// processing
void drwnSimpleNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnSimpleNode::updateForwards()
{
   // get input and output tables
    drwnDataTable *tblIn = _inputPorts[0]->getTable();
    if (tblIn == NULL) {
        DRWN_LOG_WARNING("node " << getName() << " has no input");
        return;
    }

    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    // interate over input records
    vector<string> keys = tblIn->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        if (tblOut->hasKey(*it)) continue;

        // copy structure and evaluate forward function
        const drwnDataRecord *recordIn = tblIn->lockRecord(*it);
        drwnDataRecord *recordOut = tblOut->lockRecord(*it);
        recordOut->structure() = recordIn->structure();
        forwardFunction(*it, recordIn, recordOut);

        // unlock records
        tblOut->unlockRecord(*it);
        tblIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnSimpleNode::propagateBackwards()
{
    DRWN_NOT_IMPLEMENTED_YET;
}

bool drwnSimpleNode::forwardFunction(const string& key, const drwnDataRecord *src,
    drwnDataRecord *dst)
{
    DRWN_LOG_WARNING("function not implemented for " << type());
    return false;
}

bool drwnSimpleNode::backwardGradient(const string& key, drwnDataRecord *src,
    const drwnDataRecord *dst)
{
    DRWN_LOG_WARNING("gradient not implemented for " << type());
    return false;
}

// drwnMultiIONode -----------------------------------------------------------

drwnMultiIONode::drwnMultiIONode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner)
{
    // do nothing
}

drwnMultiIONode::~drwnMultiIONode()
{
    // do nothing
}

// processing
void drwnMultiIONode::evaluateForwards()
{
    // clear output tables and then update forwards
    for (int i = 0; i < (int)_outputPorts.size(); i++) {
        drwnDataTable *tblOut = _outputPorts[i]->getTable();
        DRWN_ASSERT(tblOut != NULL);

        tblOut->clear();
    }

    updateForwards();
}

void drwnMultiIONode::updateForwards()
{
   // get input and output tables
    vector<drwnDataTable *> tblsIn(_inputPorts.size(), NULL);
    for (int i = 0; i < (int)_inputPorts.size(); i++) {
        tblsIn[i] = _inputPorts[i]->getTable();
    }

    vector<drwnDataTable *> tblsOut(_outputPorts.size(), NULL);
    for (int i = 0; i < (int)_outputPorts.size(); i++) {
        tblsOut[i] = _outputPorts[i]->getTable();
        DRWN_ASSERT(tblsOut[i] != NULL);
    }

    // interate over all input records
    set<string> keys;
    for (int i = 0; i < (int)tblsIn.size(); i++) {
        if (tblsIn[i] == NULL) continue;
        vector<string> tblKeys = tblsIn[i]->getKeys();
        keys.insert(tblKeys.begin(), tblKeys.end());
    }

    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (set<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        bool bExistingOutput = true;
        for (int i = 0; i < (int)tblsOut.size(); i++) {
            if (!tblsOut[i]->hasKey(*it)) {
                bExistingOutput = false;
                break;
            }
        }
        if (bExistingOutput) continue;

        // lock records
        vector<const drwnDataRecord *> recordsIn(tblsIn.size(), NULL);
        vector<drwnDataRecord *> recordsOut(tblsOut.size(), NULL);
        for (int i = 0; i < (int)tblsIn.size(); i++) {
            if (tblsIn[i] != NULL) {
                recordsIn[i] = tblsIn[i]->lockRecord(*it);
            }
        }
        for (int i = 0; i < (int)tblsOut.size(); i++) {
            recordsOut[i] = tblsOut[i]->lockRecord(*it);
        }

        // evaluate forward function
        forwardFunction(*it, recordsIn, recordsOut);

        // unlock records
        for (int i = 0; i < (int)tblsOut.size(); i++) {
            tblsOut[i]->unlockRecord(*it);
        }
        for (int i = 0; i < (int)tblsIn.size(); i++) {
            if (tblsIn[i] != NULL) {
                tblsIn[i]->unlockRecord(*it);
            }
        }
    }
    DRWN_END_PROGRESS;
}

void drwnMultiIONode::propagateBackwards()
{
    DRWN_NOT_IMPLEMENTED_YET;
}

bool drwnMultiIONode::forwardFunction(const string& key,
    const vector<const drwnDataRecord *>& src,
    const vector<drwnDataRecord *>& dst)
{
    DRWN_LOG_WARNING("function not implemented for " << type());
    return false;
}

bool drwnMultiIONode::backwardGradient(const string& key,
    const vector<drwnDataRecord *>& src,
    const vector<const drwnDataRecord *>& dst)
{
    DRWN_LOG_WARNING("gradient not implemented for " << type());
    return false;
}

// drwnAdaptiveNode ----------------------------------------------------------

vector<string> drwnAdaptiveNode::_regularizationChoices;

drwnAdaptiveNode::drwnAdaptiveNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _trainingColour(-1), _subSamplingRate(0),
    _maxIterations(1000), _regularizer(0), _lambda(1.0e-9)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("subSample", new drwnIntegerProperty(&_subSamplingRate));
    declareProperty("maxIterations", new drwnIntegerProperty(&_maxIterations));

    // define regularization propertys if not already done
    if (_regularizationChoices.empty()) {
        _regularizationChoices.push_back(string("sum-of-squares (L2)"));
        _regularizationChoices.push_back(string("huber (soft L1)"));
    }

    declareProperty("regularizer", new drwnSelectionProperty(&_regularizer, &_regularizationChoices));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnAdaptiveNode::drwnAdaptiveNode(const drwnAdaptiveNode& node) :
    drwnNode(node), _trainingColour(node._trainingColour), _subSamplingRate(node._subSamplingRate),
    _maxIterations(node._maxIterations), _regularizer(node._regularizer), _lambda(node._lambda)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("subSample", new drwnIntegerProperty(&_subSamplingRate));
    declareProperty("maxIterations", new drwnIntegerProperty(&_maxIterations));
    declareProperty("regularizer", new drwnSelectionProperty(&_regularizer, &_regularizationChoices));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnAdaptiveNode::~drwnAdaptiveNode()
{
    // do nothing
}

/*
double drwnAdaptiveNode::addRegularization(const double *x, double *df)
{
    double regObj = 0.0;

    switch (_regularizer.nSelection) {
    case 0: // sum-of-squares
        {
            double weightNorm = 0.0;
            for (unsigned i = 0; i < _n; i++) {
                weightNorm += x[i] * x[i];
                if (df != NULL) {
                    df[i] += _lambda * x[i];
                }
            }

            regObj = 0.5 * _lambda * weightNorm;
        }
        break;

    case 1: // huber
        {
            double dh;
            for (unsigned i = 0; i < _n; i++) {
                regObj += _lambda *
                    huberFunctionAndDerivative(x[i], &dh, 1.0e-3);
                if (df != NULL) {
                    df[i] += _lambda * dh;
                }
            }
        }
        break;

    default:
        DRWN_LOG_ERROR("unsupported regularizer \"" << _regularizer.name() << "\"");
    }

    return regObj;
}
*/
