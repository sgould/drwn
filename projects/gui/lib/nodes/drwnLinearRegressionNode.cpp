/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearRegressionNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnLinearRegressionNode.h"

using namespace std;
using namespace Eigen;

// drwnLinearRegressionNode --------------------------------------------------

vector<string> drwnLinearRegressionNode::_penaltyProperties;

drwnLinearRegressionNode::drwnLinearRegressionNode(const char *name, drwnGraph *owner) :
    drwnAdaptiveNode(name, owner), drwnOptimizer(), _penalty(0),  _argument(1.0e-3)
{
    _nVersion = 100;
    _desc = "Linear regression";

    // define operations if not already done
    if (_penaltyProperties.empty()) {
        _penaltyProperties.push_back(string("sum-of-squares (L2)"));
        _penaltyProperties.push_back(string("huber (soft L1)"));
    }

    // declare propertys
    declareProperty("theta", new drwnVectorProperty(&_theta));
    declareProperty("penalty", new drwnSelectionProperty(&_penalty, &_penaltyProperties));
    declareProperty("argument", new drwnDoubleProperty(&_argument));

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D feature matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "targetIn", "N-by-1 target vector"));
    _inputPorts.push_back(new drwnInputPort(this, "weightsIn", "N-by-1 sample weights (for training)"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-1 linear combination of features"));
}

drwnLinearRegressionNode::drwnLinearRegressionNode(const drwnLinearRegressionNode& node) :
    drwnAdaptiveNode(node), drwnOptimizer(), _penalty(node._penalty),
    _argument(node._argument), _theta(node._theta)
{
    // declare propertys
    declareProperty("theta", new drwnVectorProperty(&_theta));
    declareProperty("penalty", new drwnSelectionProperty(&_penalty, &_penaltyProperties));
    declareProperty("argument", new drwnDoubleProperty(&_argument));
}

drwnLinearRegressionNode::~drwnLinearRegressionNode()
{
    // do nothing
}

void drwnLinearRegressionNode::evaluateForwards()
{
    // check that model has been trained
    if (_theta.size() == 0) {
        DRWN_LOG_WARNING("model parameters have not been learned");
        return;
    }

    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnLinearRegressionNode::updateForwards()
{
    // check that model has been trained
    if (_theta.size() == 0) {
        DRWN_LOG_WARNING("model parameters have not been learned");
        return;
    }

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

        // evaluate forward function
        const drwnDataRecord *recordIn = tblIn->lockRecord(*it);
        drwnDataRecord *recordOut = tblOut->lockRecord(*it);

        // check feature dimensions
        if (recordIn->data().cols() == _theta.rows()) {
            recordOut->structure() = recordIn->structure();
            recordOut->data() = recordIn->data() * _theta;
        } else {
            DRWN_LOG_ERROR("feature dimension mismatch for record \"" << *it << "\"");
        }

        // unlock records
        tblOut->unlockRecord(*it);
        tblIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnLinearRegressionNode::resetParameters()
{
    // clear parameters
    _theta = VectorXd::Zero(0);
}

void drwnLinearRegressionNode::initializeParameters()
{
   // get input tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblTarget = _inputPorts[1]->getTable();
    drwnDataTable *tblWeight = _inputPorts[2]->getTable();
    if ((tblData == NULL) || (tblTarget == NULL)) {
        DRWN_LOG_ERROR("node \"" << getName() << "\" has no data and/or target input needed for parameter estimation");
        return;
    }

    // accumulate data
    _features.clear();
    _target.clear();
    _weights.clear();

    // interate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // skip missing targets or non-matching colour
        if (!tblTarget->hasKey(*it)) continue;
        if (!getOwner()->getDatabase()->matchColour(*it, _trainingColour)) continue;

        // get data
        const drwnDataRecord *recData = tblData->lockRecord(*it);
        const drwnDataRecord *recTarget = tblTarget->lockRecord(*it);
        const drwnDataRecord *recWeight = tblWeight == NULL ? NULL :
            tblWeight->lockRecord(*it);

        // TODO: check data is the right format
        // target columns must be 1; rows must be 1 or data rows
        // (if exists) weight columns must be 1; rows must be 1 or data rows
        // data columns must match accumulated features

        // accumulate data (with sampling)
        for (int i = 0; i < recData->numObservations(); i++) {
            // random sample
            if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
                continue;

            // add features
            _features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&_features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            _target.push_back(recTarget->data()(i));
            if (recWeight != NULL) {
                _weights.push_back(recWeight->data()(i));
            } else {
                _weights.push_back(1.0);
            }
        }

        // release records
        tblData->unlockRecord(*it);
        tblTarget->unlockRecord(*it);
        if (recWeight != NULL) tblWeight->unlockRecord(*it);
    }

    // check for data
    if (_features.empty()) {
        DRWN_LOG_ERROR("no training data for parameter estimation");
    } else {
        DRWN_LOG_VERBOSE("training \"" << getName() << "\" with " << _features.size() << " training examples");

        // estimate parameters
        initialize((int)_features[0].size());
        solve(_maxIterations, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
        _theta = Eigen::Map<VectorXd>(drwnOptimizer::_x, _n);
    }

    // clear cached training data
    _features.clear();
    _target.clear();
    _weights.clear();
    DRWN_END_PROGRESS;
}

// drwnOptimizer interface
double drwnLinearRegressionNode::objective(const double *x) const
{
    double *df = new double[_n];
    double f = objectiveAndGradient(x, df);
    delete[] df;
    return f;
}

void drwnLinearRegressionNode::gradient(const double *x, double *df) const
{
    objectiveAndGradient(x, df);
}

double drwnLinearRegressionNode::objectiveAndGradient(const double *x, double *df) const
{
    // predict output
    unsigned m = _target.size();
    vector<double> predicted(m, 0.0);
    for (unsigned i = 0; i < m; i++) {
	for (unsigned j = 0; j < _n; j++) {
	    predicted[i] += _features[i][j] * x[j];
	}
    }

    // compute gradient and objective
    double obj = 0.0;
    memset((void *)df, (int)0.0, _n * sizeof(double));

    if (_penalty == 0) {
        // L2 penalty
        for (unsigned i = 0; i < m; i++) {
            double dist =  predicted[i] - _target[i];
            double wdist =  dist * _weights[i];
            obj += dist * wdist;
            for (unsigned j = 0; j < _n; j++) {
                df[j] += wdist * _features[i][j];
            }
        }

        obj *= 0.5;
    } else {
        // huber penalty
        double dh;
        for (unsigned i = 0; i < m; i++) {
            double u = predicted[i] - _target[i];
            obj += _weights[i] * drwn::huberFunctionAndDerivative(u, &dh, _argument);
            dh *= _weights[i];
            for (unsigned j = 0; j < _n; j++) {
                df[j] += dh * _features[i][j];
            }
        }
    }

    return obj;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Adaptive", drwnLinearRegressionNode);

