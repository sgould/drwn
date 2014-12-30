/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnUnaryOpNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnUnaryOpNode.h"

using namespace std;
using namespace Eigen;

// drwnUnaryOpNode -----------------------------------------------------------

vector<string> drwnUnaryOpNode::_operations;

drwnUnaryOpNode::drwnUnaryOpNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner), _selection(0), _argument(0.0)
{
    _nVersion = 100;
    _desc = "Implements basic (component-wise) mathematical transformations";

    // define operations if not already done
    if (_operations.empty()) {
        _operations.push_back(string("exp-normalize"));
        _operations.push_back(string("exp"));
        _operations.push_back(string("log"));
        _operations.push_back(string("pow"));
        _operations.push_back(string("pow-normalize"));
        _operations.push_back(string("normalize"));
        _operations.push_back(string("sum"));
        _operations.push_back(string("L1-norm"));
        _operations.push_back(string("L2-norm"));
        _operations.push_back(string("L2-norm-squared"));
    }

    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));
    declareProperty("argument", new drwnDoubleProperty(&_argument));
}

drwnUnaryOpNode::drwnUnaryOpNode(const drwnUnaryOpNode& node) :
    drwnSimpleNode(node), _selection(node._selection), _argument(node._argument)
{
    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));
    declareProperty("argument", new drwnDoubleProperty(&_argument));
}

drwnUnaryOpNode::~drwnUnaryOpNode()
{
    // do nothing
}

bool drwnUnaryOpNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    switch (_selection) {
    case 0:  // exp-normalize
        dst->data() = Eigen::MatrixXd(src->data().rows(), src->data().cols());
        for (int i = 0; i < dst->data().rows(); i++) {
            dst->data().row(i) = (src->data().row(i).array() -
                src->data().row(i).maxCoeff()).array().exp();
            dst->data().row(i) /= dst->data().row(i).sum();
        }
        break;
    case 1:  // exp
        dst->data() = src->data().array().exp();
        break;
    case 2:  // log
        dst->data() = src->data().array().log();
        break;
    case 3:  // pow
        dst->data() = src->data().array().pow(_argument);
        break;
    case 4:  // pow-normalize
        dst->data() = Eigen::MatrixXd(src->data().rows(), src->data().cols());
        for (int i = 0; i < dst->data().rows(); i++) {
            dst->data().row(i) = (src->data().row(i) /
                (fabs(src->data().row(i).maxCoeff()) + DRWN_EPSILON)).array().pow(_argument);
            dst->data().row(i) /= dst->data().row(i).sum();
        }
        break;
    case 5: // normalize
        dst->data() = Eigen::MatrixXd(src->data().rows(), 1);
        for (int i = 0; i < dst->data().rows(); i++) {
            dst->data().row(i) = src->data().row(i).normalized();
        }
        break;
    case 6: // sum
        dst->data() = src->data().rowwise().sum();
        break;
    case 7: // L1-norm
        dst->data() = src->data().array().abs().rowwise().sum();
        break;
    case 8: // L2-norm
        dst->data() = src->data().array().square().matrix().rowwise().sum().cwiseSqrt();
        break;
    case 9: // L2-norm-squared
        dst->data() = src->data().array().square().matrix().rowwise().sum();
        break;
    default:
        DRWN_LOG_ERROR("unknown operation " << _selection); 
        return false;
    }

    return true;
}

bool drwnUnaryOpNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnUnaryOpNode);

