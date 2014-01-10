/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBinaryOpNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnBinaryOpNode.h"

using namespace std;
using namespace Eigen;

// drwnBinaryOpNode ---------------------------------------------------------

vector<string> drwnBinaryOpNode::_operations;

drwnBinaryOpNode::drwnBinaryOpNode(const char *name, drwnGraph *owner) :
    drwnMultiIONode(name, owner), _selection(0)
{
    _nVersion = 100;
    _desc = "Computes binary operation on two records";

    // define operations if not already done
    if (_operations.empty()) {
        _operations.push_back(string("add"));
        _operations.push_back(string("subtract"));
        _operations.push_back(string("multiply"));
        _operations.push_back(string("divide"));
        _operations.push_back(string("min"));
        _operations.push_back(string("max"));
    }

    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataInA", "N-by-D, N-by-1, 1-by-D or 1-by-1 data matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "dataInB", "N-by-D, N-by-1, 1-by-D or 1-by-1 data matrix"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "K-by-1 data matrix"));
}

drwnBinaryOpNode::drwnBinaryOpNode(const drwnBinaryOpNode& node) :
    drwnMultiIONode(node), _selection(node._selection)
{
    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));
}

drwnBinaryOpNode::~drwnBinaryOpNode()
{
    // do nothing
}

bool drwnBinaryOpNode::forwardFunction(const string& key,
    const vector<const drwnDataRecord *>& src,
    const vector<drwnDataRecord *>& dst)
{
    // get matrix dimensions equal
    // TODO: this is inefficient, is there a better way without explicitly coding for each operation?
    MatrixXd A = src[0]->data();
    MatrixXd B = src[1]->data();
    if (A.rows() == B.rows()) {
        if ((A.cols() == 1) && (B.cols() != 1)) {
            A = VectorXd::Ones(B.cols()).transpose() * A;
        } else if ((A.cols() != 1) && (B.cols() == 1)) {
            B = VectorXd::Ones(A.cols()).transpose() * B;
        } else if (A.cols() != B.cols()) {
            DRWN_LOG_ERROR("invalid combination of sizes for binary operation");
            return false;
        }
    } else if (A.cols() == B.cols()) {
        if ((A.rows() == 1) && (B.rows() != 1)) {
            A = A * VectorXd::Ones(B.rows());
        } else if ((A.rows() != 1) && (B.rows() == 1)) {
            B = B * VectorXd::Ones(B.rows());
        } else if (A.rows() != B.rows()) {
            DRWN_LOG_ERROR("invalid combination of sizes for binary operation");
            return false;
        }
    } else if ((A.rows() == 1) && (A.cols() == 1)) {
        A = MatrixXd::Constant(B.rows(), B.cols(), A(0,0));
    } else if ((B.rows() == 1) && (B.cols() == 1)) {
        B = MatrixXd::Constant(A.rows(), A.cols(), B(0,0));
    } else {
        DRWN_LOG_ERROR("invalid combination of sizes for binary operation");
        return false;
    }

    // execute operation
    switch (_selection) {
    case 0: // add
        dst[0]->data() = A + B;
        break;
    case 1: // subtract
        dst[0]->data() = A - B;
        break;
    case 2: // multiply
        dst[0]->data() = A.cwiseProduct(B);
        break;
    case 3: // divide
        dst[0]->data() = A.cwiseQuotient(B);
        break;
    case 4: // min
        dst[0]->data() = A.cwiseMin(B);
        break;
    case 5: // max
        dst[0]->data() = A.cwiseMax(B);
        break;
    default:
        DRWN_LOG_FATAL("unsupported operation " << _selection);
        return false;
    }

    return true;
}

bool drwnBinaryOpNode::backwardGradient(const string& key,
    const vector<drwnDataRecord *>& src,
    const vector<const drwnDataRecord *>& dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnBinaryOpNode);

