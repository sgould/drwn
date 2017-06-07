/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRollupNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnRollupNode.h"

using namespace std;
using namespace Eigen;

// drwnRollupNode ---------------------------------------------------------

vector<string> drwnRollupNode::_operations;

drwnRollupNode::drwnRollupNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _selection(0)
{
    _nVersion = 100;
    _desc = "Computes summary statistics over disjoint sets of observations";

    // define operations if not already done
    if (_operations.empty()) {
        _operations.push_back(string("mean"));
        _operations.push_back(string("variance"));
        _operations.push_back(string("skewness"));
        _operations.push_back(string("kurtosis"));
        _operations.push_back(string("stdev"));
        _operations.push_back(string("sum"));
        _operations.push_back(string("sum-squared"));
        _operations.push_back(string("max"));
        _operations.push_back(string("min"));
        _operations.push_back(string("count"));
    }

    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "indexIn", "N-by-1 index matrix (0 .. K)"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "K-by-1 data matrix"));
}

drwnRollupNode::drwnRollupNode(const drwnRollupNode& node) :
    drwnNode(node), _selection(node._selection)
{
    // declare propertys
    declareProperty("operation", new drwnSelectionProperty(&_selection, &_operations));
}

drwnRollupNode::~drwnRollupNode()
{
    // do nothing
}

void drwnRollupNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnRollupNode::updateForwards()
{
   // get input and output tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblIndx = _inputPorts[1]->getTable();
    if (tblData == NULL) {
        DRWN_LOG_WARNING("node " << getName() << " has no input");
        return;        
    }

    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    // interate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        if (tblOut->hasKey(*it)) continue;

        // evaluate forward function
        const drwnDataRecord *recordData = tblData->lockRecord(*it);
        const drwnDataRecord *recordIndx = (tblIndx == NULL) ? NULL : tblIndx->lockRecord(*it);

        // check sizes
        if ((recordIndx != NULL) && (recordIndx->numObservations() != 0) &&
            (recordData->numObservations() != recordIndx->numObservations())) {
            DRWN_LOG_ERROR("size mismatch between dataIn and indexIn");
            if (tblIndx != NULL) tblIndx->unlockRecord(*it);
            tblData->unlockRecord(*it);            
        }

        drwnDataRecord *recordOut = tblOut->lockRecord(*it);

        // create output index sets
        bool bFullRollup = (recordIndx == NULL) || (recordIndx->numObservations() == 0); 
        int nOutputObs = bFullRollup ? 1 : (int)recordIndx->data().maxCoeff() + 1;
        DRWN_ASSERT(nOutputObs > 0);

        vector<set<int> > indexSets(nOutputObs);
        if (bFullRollup) {
            for (int i = 0; i < recordData->numObservations(); i++) {
                indexSets[0].insert(i);
            }
        } else {
            for (int i = 0; i < recordIndx->numObservations(); i++) {
                indexSets[(int)recordIndx->data()(i, 0)].insert(i);
            }
        }

        // generate output
        recordOut->data() = MatrixXd::Zero(nOutputObs, recordData->numFeatures());
        for (int i = 0; i < nOutputObs; i++) {
            switch (_selection) {
            case 0: // mean
                recordOut->data().row(i) = fcnMean(recordData->data(), indexSets[i]);
                break;
            case 1: // variance
                recordOut->data().row(i) = fcnVariance(recordData->data(), indexSets[i]);
                break;
            case 2: // skewness
                DRWN_NOT_IMPLEMENTED_YET;
                break;
            case 3: // kurtosis
                DRWN_NOT_IMPLEMENTED_YET;
                break;
            case 4: // stdev
                recordOut->data().row(i) = fcnStdev(recordData->data(), indexSets[i]);                
                break;
                
            default:
                DRWN_NOT_IMPLEMENTED_YET;
            }
        }

        // unlock records
        tblOut->unlockRecord(*it);
        if (tblIndx != NULL) tblIndx->unlockRecord(*it);
        tblData->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnRollupNode::propagateBackwards()
{
    DRWN_NOT_IMPLEMENTED_YET;
}

// operations ----------------------------------------------------------------

MatrixXd drwnRollupNode::fcnMean(const MatrixXd& data, const set<int>& index) const
{
    MatrixXd m = MatrixXd::Zero(1, data.cols());
    for (set<int>::const_iterator it = index.begin(); it != index.end(); it++) {
        m += data.row(*it);
    }

    if (!index.empty()) {
        m /= (double)index.size();
    }

    return m;
}

MatrixXd drwnRollupNode::fcnVariance(const MatrixXd& data, const set<int>& index) const
{
    MatrixXd m = MatrixXd::Zero(1, data.cols());
    MatrixXd m2 = MatrixXd::Zero(1, data.cols());

    for (set<int>::const_iterator it = index.begin(); it != index.end(); it++) {
        m += data.row(*it);
        m2 += data.row(*it).array().square().matrix();
    }

    // the component-wise max below prevents numberical underflow problems
    if (!index.empty()) {
        m /= (double)index.size();
        m2 = (m2 / (double)index.size() - m.cwiseProduct(m)).cwiseMax(MatrixXd::Zero(1, data.cols()));
    }

    return m2;
}

MatrixXd drwnRollupNode::fcnStdev(const MatrixXd& data, const set<int>& index) const
{
    MatrixXd m = fcnVariance(data, index);
    return m.cwiseSqrt();
}


// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnRollupNode);

