/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConcatenationNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnConcatenationNode.h"

using namespace std;
using namespace Eigen;

// drwnConcatenationNode -----------------------------------------------------

drwnConcatenationNode::drwnConcatenationNode(const char *name, drwnGraph *owner) :
    drwnMultiIONode(name, owner), _nInputs(2), _bAddOnes(false)
{
    _nVersion = 100;
    _desc = "Combines features from two or more records";

    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("appendOnes", new drwnBooleanProperty(&_bAddOnes));

    // declare ports
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-(D1+D2+...) data matrix"));
    updatePorts();
}

drwnConcatenationNode::drwnConcatenationNode(const drwnConcatenationNode& node) :
    drwnMultiIONode(node), _nInputs(node._nInputs), _bAddOnes(node._bAddOnes)
{
    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("appendOnes", new drwnBooleanProperty(&_bAddOnes));
}

drwnConcatenationNode::~drwnConcatenationNode()
{
    // do nothing
}

// i/o
bool drwnConcatenationNode::load(drwnXMLNode& xml)
{
    drwnMultiIONode::load(xml);
    updatePorts();
    return true;
}

bool drwnConcatenationNode::forwardFunction(const string& key,
    const vector<const drwnDataRecord *>& src,
    const vector<drwnDataRecord *>& dst)
{
    // get matrix dimensions
    int N = 1;
    int D = _bAddOnes ? 1 : 0;
    for (unsigned i = 0; i < src.size(); i++) {
        if ((N == 1) && (src[i]->numObservations() > 1)) {
            N = src[i]->numObservations();
        }
        D += src[i]->numFeatures();

        if ((src[i]->numObservations() != 1) && (src[i]->numObservations() != N)) {
            DRWN_LOG_ERROR("size (observations) mismatch for record \"" << key << "\"");
            return false;
        }
    }

    // create output
    dst[0]->data() = MatrixXd::Ones(N, D);

    int currentCol = 0;
    for (unsigned i = 0; i < src.size(); i++) {
        // copy structure for first record that has one
        if (!dst[0]->hasStructure() && src[i]->hasStructure()) {
            dst[0]->structure() = src[i]->structure();
        }

        // append data
        if (src[i]->numObservations() == N) {
            dst[0]->data().block(0, currentCol, N, src[i]->numFeatures()) = src[i]->data();
        } else {
            dst[0]->data().block(0, currentCol, N, src[i]->numFeatures()) =
                MatrixXd::Constant(N, src[i]->numFeatures(), src[i]->data()(0, 0));
        }
        currentCol += src[i]->numFeatures();
    }
    DRWN_ASSERT(currentCol == D - (_bAddOnes ? 1 : 0));

    return true;
}

bool drwnConcatenationNode::backwardGradient(const string& key,
    const vector<drwnDataRecord *>& src,
    const vector<const drwnDataRecord *>& dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

void drwnConcatenationNode::propertyChanged(const string& name)
{
    if (name == string("numInputs")) {
        _nInputs = std::max(1, _nInputs);
        updatePorts();
    } else {
        drwnMultiIONode::propertyChanged(name);
    }
}

void drwnConcatenationNode::updatePorts()
{
    // re-assign input ports
    if (_nInputs != (int)_inputPorts.size()) {
        for (vector<drwnInputPort *>::iterator it = _inputPorts.begin();
             it != _inputPorts.end(); it++) {
            delete *it;
        }
        _inputPorts.clear();

        if (_nInputs == 1) {
            _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
        } else {
            for (int i = 0; i < _nInputs; i++) {
                string portName = string("dataIn") + toString(i);
                _inputPorts.push_back(new drwnInputPort(this, portName.c_str(), "N-by-D or 1-by-D data matrix"));
            }
        }
    }
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnConcatenationNode);

