/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCodebookNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnCodebookNodes.h"

using namespace std;
using namespace Eigen;

// drwnLUTDecoderNode --------------------------------------------------------

drwnLUTDecoderNode::drwnLUTDecoderNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner)
{
    _nVersion = 100;
    _desc = "Implements simple lookup table decoding";

    // declare propertys
    declareProperty("LUT", new drwnMatrixProperty(&_lut));
}

drwnLUTDecoderNode::drwnLUTDecoderNode(const drwnLUTDecoderNode& node) :
    drwnSimpleNode(node), _lut(node._lut)
{
    // declare propertys
    declareProperty("LUT", new drwnMatrixProperty(&_lut));
}

drwnLUTDecoderNode::~drwnLUTDecoderNode()
{
    // do nothing
}

bool drwnLUTDecoderNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    if (_lut.cols() == 0) {
        return false;
    }

    dst->data() = Eigen::MatrixXd(src->data().rows(), _lut.cols());
    for (int i = 0; i < src->data().rows(); i++) {
        int indx;
        if (src->numFeatures() == 1) {
            indx = (int)src->data().row(i)[0];
        } else {
            src->data().row(i).maxCoeff(&indx);
        }

        // valid index
        if ((indx < 0) || (indx > _lut.rows())) {
            indx = _lut.rows() -1;
        }

        // encode data
        dst->data().row(i) = _lut.row(indx);
    }

    return true;
}

bool drwnLUTDecoderNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// drwnLUTEncoderNode --------------------------------------------------------

vector<string> drwnLUTEncoderNode::_modeProperties;

drwnLUTEncoderNode::drwnLUTEncoderNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner), _mode(0)
{
    _nVersion = 100;
    _desc = "Implements simple lookup table encoding";

    // define propertys if not already done
    if (_modeProperties.empty()) {
        _modeProperties.push_back(string("exact"));
        _modeProperties.push_back(string("nearest (L2)"));
        _modeProperties.push_back(string("nearest (L1)"));
    }

    // declare propertys
    declareProperty("mode", new drwnSelectionProperty(&_mode, &_modeProperties));
    declareProperty("LUT", new drwnMatrixProperty(&_lut));
}

drwnLUTEncoderNode::drwnLUTEncoderNode(const drwnLUTEncoderNode& node) :
    drwnSimpleNode(node), _mode(node._mode), _lut(node._lut)
{
    // declare propertys
    declareProperty("mode", new drwnSelectionProperty(&_mode, &_modeProperties));
    declareProperty("LUT", new drwnMatrixProperty(&_lut));
}

drwnLUTEncoderNode::~drwnLUTEncoderNode()
{
    // do nothing
}

bool drwnLUTEncoderNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    if (_lut.cols() == 0) {
        return false;
    }

    if (src->numFeatures() != _lut.cols()) {
        DRWN_LOG_ERROR("mismatch between input feature length and lookup table");
        return false;
    }

    // encode features
    dst->data() = Eigen::MatrixXd(src->data().rows(), 1);
    for (int i = 0; i < src->data().rows(); i++) {
        switch (_mode) {
        case 0: // exact
            dst->data().row(i)[0] = -1;
            for (int j = 0; j < _lut.rows(); j++) {
                if (src->data().row(i) == _lut.row(j)) {
                    dst->data().row(i)[0] = (double)j;
                    break;
                }
            }
            break;
        case 1: // nearest (L2)
            {
                int indx;
                (VectorXd::Ones(_lut.rows()) * src->data().row(i) - 
                    _lut).array().square().rowwise().sum().minCoeff(&indx);
                dst->data().row(i)[0] = (double)indx;
            }
            break;

        case 2: // nearest (L1)
            {
                int indx;
                (VectorXd::Ones(_lut.rows()) * src->data().row(i) - 
                    _lut).array().abs().rowwise().sum().minCoeff(&indx);
                dst->data().row(i)[0] = (double)indx;
            }
            break;
            
        default:
            DRWN_LOG_ERROR("unknown operation " << _mode);
            return false;
        }
    }

    return true;
}

bool drwnLUTEncoderNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnLUTDecoderNode);
DRWN_AUTOREGISTERNODE("Static", drwnLUTEncoderNode);
