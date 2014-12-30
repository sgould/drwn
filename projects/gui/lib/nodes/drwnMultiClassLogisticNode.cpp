/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiClassLogisticNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnMultiClassLogisticNode.h"

using namespace std;
using namespace Eigen;

// drwnMultiClassLogisticNode ------------------------------------------------

drwnMultiClassLogisticNode::drwnMultiClassLogisticNode(const char *name, drwnGraph *owner) :
    drwnAdaptiveNode(name, owner), drwnOptimizer(), _bOutputScores(false)
{
    _nVersion = 100;
    _desc = "Multi-class logistic classifier with weight regularization";

    // declare propertys
    declareProperty("theta", new drwnMatrixProperty(&_theta));
    declareProperty("outputScores", new drwnBooleanProperty(&_bOutputScores));

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D feature matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "targetIn", "N-by-1 target vector"));
    _inputPorts.push_back(new drwnInputPort(this, "weightsIn", "N-by-1 sample weights (for training)"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-1 output marginals or log-marginals"));
}

drwnMultiClassLogisticNode::drwnMultiClassLogisticNode(const drwnMultiClassLogisticNode& node) :
    drwnAdaptiveNode(node),  _theta(node._theta), _bOutputScores(node._bOutputScores)
{
    // declare propertys
    declareProperty("theta", new drwnMatrixProperty(&_theta));
    declareProperty("outputScores", new drwnBooleanProperty(&_bOutputScores));
}

drwnMultiClassLogisticNode::~drwnMultiClassLogisticNode()
{
    // do nothing
}

void drwnMultiClassLogisticNode::evaluateForwards()
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

void drwnMultiClassLogisticNode::updateForwards()
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
        if (recordIn->data().cols() == _theta.cols()) {
            recordOut->structure() = recordIn->structure();
            recordOut->data() = MatrixXd::Zero(recordIn->data().rows(), _theta.rows() + 1);
            recordOut->data().block(0, 0, recordIn->data().rows(), _theta.rows()) =
                recordIn->data() * _theta.transpose();
            // exp-normalize
            if (!_bOutputScores) {
                for (int i = 0; i < recordOut->data().rows(); i++) {
                    recordOut->data().row(i) = (recordOut->data().row(i).array() -
                        recordOut->data().row(i).maxCoeff()).array().exp();
                    recordOut->data().row(i) /= recordOut->data().row(i).sum();
                }
            }
        } else {
            DRWN_LOG_ERROR("feature dimension mismatch for record \"" << *it << "\"");
        }

        // unlock records
        tblOut->unlockRecord(*it);
        tblIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnMultiClassLogisticNode::resetParameters()
{
    // clear parameters
    _theta = MatrixXd::Zero(0, 0);
}

void drwnMultiClassLogisticNode::initializeParameters()
{
    DRWN_FCN_TIC;

    // get input tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblTarget = _inputPorts[1]->getTable();
    drwnDataTable *tblWeight = _inputPorts[2]->getTable();
    if ((tblData == NULL) || (tblTarget == NULL)) {
        DRWN_LOG_ERROR("node \"" << getName() << "\" has no data and/or target input needed for parameter estimation");
        DRWN_FCN_TOC;
        return;
    }

    // accumulate data
    _features.clear();
    _target.clear();
    _weights.clear();

    // iterate over input records
    int nClasses = 0;
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
        const drwnDataRecord *recWeight = (tblWeight == NULL) ? NULL :
            tblWeight->lockRecord(*it);

        // TODO: check data is the right format
        // target columns must be 1; rows must be 1 or data rows
        // (if exists) weight columns must be 1; rows must be 1 or data rows
        // data columns must match accumulated features

        // accumulate data (with sampling)
        for (int i = 0; i < recData->numObservations(); i++) {
            // ignore "unknown" class labels
            if (recTarget->data()(i) < 0)
                continue;

            // random sample
            if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
                continue;

            // add features
            _features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&_features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            _target.push_back((int)recTarget->data()(i));
            nClasses = std::max(_target.back() + 1, nClasses);
            if (tblWeight != NULL) {
                if (recWeight != NULL) {
                    _weights.push_back(recWeight->data()(i));
                } else {
                    _weights.push_back(1.0);
                }
            }
        }

        // release records
        tblData->unlockRecord(*it);
        tblTarget->unlockRecord(*it);
        if (recWeight != NULL) tblWeight->unlockRecord(*it);
    }

    // check for data
    if (_features.empty() || (nClasses < 2)) {
        DRWN_LOG_ERROR("insufficient training data for parameter estimation");
    } else {
        DRWN_LOG_VERBOSE("training \"" << getName() << "\" with " << _features.size() << " training examples");

        // estimate parameters
        _theta = MatrixXd::Zero(nClasses - 1, _features[0].size());
        initialize(_theta.rows() * _theta.cols());
        solve(_maxIterations, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
        _theta = Eigen::Map<MatrixXd>(drwnOptimizer::_x, _theta.cols(), _theta.rows()).transpose();
    }

    // clear cached training data
    _features.clear();
    _target.clear();
    _weights.clear();
    DRWN_END_PROGRESS;
    DRWN_FCN_TOC;
}

// drwnOptimizer interface
double drwnMultiClassLogisticNode::objective(const double *x) const
{
    double *df = new double[_n];
    double f = objectiveAndGradient(x, df);
    delete[] df;
    return f;
}

void drwnMultiClassLogisticNode::gradient(const double *x, double *df) const
{
    objectiveAndGradient(x, df);
}

double drwnMultiClassLogisticNode::objectiveAndGradient(const double *x, double *df) const
{
    const int nFeatures = _theta.cols();
    const int nClasses = _theta.rows() + 1;
    const int nFeaturesDiv4 = nFeatures / 4;
    const int nFeaturesMod4 = nFeatures % 4;

    double negLogL = 0.0;
    int numTerms = 0;
    memset(df, (int)0.0, _n * sizeof(double));

    vector<double> p(nClasses);
    for (unsigned n = 0; n < _features.size(); n++) {
        double alpha = _weights.empty() ? 1.0 : _weights[n];
        fill(p.begin(), p.end(), 0.0);

	// compute marginal for training sample
	double maxValue = 0.0;
        const double *x_ptr = x;
	for (int k = 0; k < nClasses - 1; k++) {
            const double *f_ptr = &_features[n][0];
            for (int i = nFeaturesDiv4; i != 0; i--) {
                p[k] += x_ptr[0] * f_ptr[0] + x_ptr[1] * f_ptr[1] +
                    x_ptr[2] * f_ptr[2] + x_ptr[3] * f_ptr[3];
                x_ptr += 4; f_ptr += 4;
            }
            for (int i = 0; i < nFeaturesMod4; i++) {
                p[k] += (*x_ptr++) * (*f_ptr++);
            }

	    if (p[k] > maxValue)
		maxValue = p[k];
	}

	// exponentiate and normalize
	double Z = 0.0;
	for (vector<double>::iterator it = p.begin(); it != p.end(); ++it) {
	    Z += (*it = exp(*it - maxValue));
	}

        double *p_ptr = &p[0];
        for (int i = nClasses / 2; i != 0; i--) {
            p_ptr[0] /= Z;
            p_ptr[1] /= Z;
            p_ptr += 2;
        }
        if (nClasses % 2 != 0) {
            *p_ptr /= Z;
        }

	// increment log-likelihood
	negLogL -= alpha * log(p[_target[n]]);
	numTerms += 1;

	// increment derivative
        double *df_ptr = df;
        for (int k = 0; k < nClasses - 1; k++) {
            double nu = alpha * p[k];
            const double *f_ptr = &_features[n][0];
            for (int i = nFeaturesDiv4; i != 0; i--) {
                df_ptr[0] += nu * f_ptr[0];
                df_ptr[1] += nu * f_ptr[1];
                df_ptr[2] += nu * f_ptr[2];
                df_ptr[3] += nu * f_ptr[3];
                df_ptr += 4; f_ptr += 4;

            }
            for (int i = 0; i < nFeaturesMod4; i++) {
                (*df_ptr++) += nu * (*f_ptr++);
            }
        }

        if (_target[n] < nClasses - 1) {
            df_ptr = &df[_target[n] * nFeatures];
            const double *f_ptr = &_features[n][0];
            for (int i = nFeaturesDiv4; i != 0; i--) {
                df_ptr[0] -= alpha * f_ptr[0];
                df_ptr[1] -= alpha * f_ptr[1];
                df_ptr[2] -= alpha * f_ptr[2];
                df_ptr[3] -= alpha * f_ptr[3];
                df_ptr += 4; f_ptr += 4;

            }
            for (int i = 0; i < nFeaturesMod4; i++) {
                (*df_ptr++) -= alpha * (*f_ptr++);
            }
	}
    }

    if (numTerms == 0) return 0.0;
    negLogL /= (double)numTerms;
    Eigen::Map<VectorXd>(df, _n) /= (double)numTerms;

    // regularization
    // TODO: make a member function of drwnAdaptiveNode
    switch (_regularizer) {
    case 0: // sum-of-squares
        {
            double weightNorm = 0.0;
            for (unsigned i = 0; i < _n; i++) {
                weightNorm += x[i] * x[i];
                df[i] += _lambda * x[i];
            }

            negLogL += 0.5 * _lambda * weightNorm;
        }
        break;

    case 1: // huber
        {
            double dh;
            for (unsigned i = 0; i < _n; i++) {
                negLogL += _lambda *
                    drwn::huberFunctionAndDerivative(x[i], &dh, 1.0e-3);
                df[i] += _lambda * dh;
            }
        }
        break;

    default:
        DRWN_LOG_ERROR("unsupported regularizer " << _regularizer);
    }

    return negLogL;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Adaptive", drwnMultiClassLogisticNode);

