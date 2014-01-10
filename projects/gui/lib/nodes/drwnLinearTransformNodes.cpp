/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearTransformNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/QR"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnLinearTransformNodes.h"

using namespace std;
using namespace Eigen;

// drwnLinearTransformNode ---------------------------------------------------

drwnLinearTransformNode::drwnLinearTransformNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner)
{
    _nVersion = 100;
    _desc = "Linear/affine transformation";

    // declare propertys
    declareProperty("translation", new drwnVectorProperty(&_translation));
    declareProperty("projection", new drwnMatrixProperty(&_projection));
}

drwnLinearTransformNode::drwnLinearTransformNode(const drwnLinearTransformNode& node) :
    drwnSimpleNode(node), _translation(node._translation), _projection(node._projection)
{
    // declare propertys
    declareProperty("translation", new drwnVectorProperty(&_translation));
    declareProperty("projection", new drwnMatrixProperty(&_projection));
}

drwnLinearTransformNode::~drwnLinearTransformNode()
{
    // do nothing
}

bool drwnLinearTransformNode::forwardFunction(const string& key, const drwnDataRecord *src,
    drwnDataRecord *dst)
{
    if (_projection.cols() == 0)
        return false;

    // TODO: check data dimensions
    dst->data() = MatrixXd::Zero(src->numObservations(), _projection.rows());
    for (int i = 0; i < src->numObservations(); i++) {
        dst->data().row(i) = (src->data().row(i) - _translation.transpose()) *
            _projection.transpose();
    }

    return true;
}

bool drwnLinearTransformNode::backwardGradient(const string& key, drwnDataRecord *src,
    const drwnDataRecord *dst)
{
    if (_projection.cols() == 0)
        return false;

    // TODO: check data dimensions
    src->objective() = dst->objective();
    src->gradient() = dst->gradient() * _projection;

    return true;
}

// drwnRescaleNode ----------------------------------------------------------

drwnRescaleNode::drwnRescaleNode(const char *name, drwnGraph *owner) :
    drwnLinearTransformNode(name, owner), _trainingColour(-1)
{
    _nVersion = 100;
    _desc = "Rescale features (to zero mean and unit variance)";

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
}

drwnRescaleNode::drwnRescaleNode(const drwnRescaleNode& node) :
    drwnLinearTransformNode(node), _trainingColour(node._trainingColour)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
}

drwnRescaleNode::~drwnRescaleNode()
{
    // do nothing
}

void drwnRescaleNode::resetParameters()
{
    // clear parameters
    _translation = VectorXd::Zero(0);
    _projection = MatrixXd::Zero(0,0);
}

void drwnRescaleNode::initializeParameters()
{
    // get input tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    if (tblData == NULL) {
        DRWN_LOG_ERROR("node \"" << getName() << "\" has no data needed for parameter estimation");
        return;
    }

    int dataDim = -1;
    VectorXd dataSum;
    VectorXd dataSum2;
    double dataCount = 0.0;

    // interate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // skip non-matching colour
        if (!getOwner()->getDatabase()->matchColour(*it, _trainingColour)) continue;

        // get data
        const drwnDataRecord *record = tblData->lockRecord(*it);

        // accumulate sufficient statistics
        if (record->numFeatures() != 0) {
            if (dataDim == -1) {
                dataDim = record->numFeatures();
                dataSum = VectorXd::Zero(dataDim);
                dataSum2 = VectorXd::Zero(dataDim);
            }

            DRWN_ASSERT(record->numFeatures() == dataDim);

            // update
            dataSum += record->data().colwise().sum().transpose();
            dataSum2 += record->data().array().square().matrix().colwise().sum().transpose();
            dataCount += (double)record->numObservations();
        }

        // release records
        tblData->unlockRecord(*it);
    }

    // compute projections
    if (dataCount > 0.0) {
        _translation = dataSum / dataCount;
        VectorXd var = (dataSum2 / dataCount -
            _translation.array().square().matrix()).cwiseSqrt();
        for (int i = 0; i < dataDim; i++) {
            if (var[i] < DRWN_DBL_MIN) var[i] = 1.0;
        }
        _projection = (VectorXd::Ones(dataDim).cwiseQuotient(var)).diagonal();
    } else {
        DRWN_LOG_WARNING("not enough data to compute rescaling");
    }

    DRWN_END_PROGRESS;
}

// drwnPCANode --------------------------------------------------------------

drwnPCANode::drwnPCANode(const char *name, drwnGraph *owner) :
    drwnLinearTransformNode(name, owner), _trainingColour(-1), _numOutputDims(1)
{
    _nVersion = 100;
    _desc = "Principal component analysis";

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("outputDim", new drwnIntegerProperty(&_numOutputDims));
}

drwnPCANode::drwnPCANode(const drwnPCANode& node) :
    drwnLinearTransformNode(node), _trainingColour(node._trainingColour),
    _numOutputDims(node._numOutputDims)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("outputDim", new drwnIntegerProperty(&_numOutputDims));
}

drwnPCANode::~drwnPCANode()
{
    // do nothing
}

void drwnPCANode::resetParameters()
{
    // clear parameters
    _translation = VectorXd::Zero(0);
    _projection = MatrixXd::Zero(0,0);
}

void drwnPCANode::initializeParameters()
{
    // get input tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    if (tblData == NULL) {
        DRWN_LOG_ERROR("node \"" << getName() << "\" has no data needed for parameter estimation");
        return;
    }

    int dataDim = -1;
    VectorXd dataSum;
    MatrixXd dataSum2;
    double dataCount = 0.0;

    // interate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // skip non-matching colour
        if (!getOwner()->getDatabase()->matchColour(*it, _trainingColour)) continue;

        // get data
        const drwnDataRecord *record = tblData->lockRecord(*it);

        // accumulate sufficient statistics
        if (record->numFeatures() != 0) {
            if (dataDim == -1) {
                dataDim = record->numFeatures();
                dataSum = VectorXd::Zero(dataDim);
                dataSum2 = MatrixXd::Zero(dataDim, dataDim);
            }

            DRWN_ASSERT(record->numFeatures() == dataDim);

            // update
            dataSum += record->data().colwise().sum().transpose();
            dataSum2 += record->data().transpose() * record->data();
            dataCount += (double)record->numObservations();
        }

        // release records
        tblData->unlockRecord(*it);
    }

    // correct output dimensions if bigger than dataDim
    _numOutputDims = std::min(_numOutputDims, dataDim);

    // compute projections
    if (dataCount > 0.0) {
        _translation = dataSum / dataCount;
        MatrixXd sigma = dataSum2 / dataCount - _translation * _translation.transpose();
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(sigma);

        // TODO: check order of eigenvalues. Is it consistent?
        if (solver.eigenvalues()[0] > solver.eigenvalues()[dataDim - 1]) {
            // check for negative eigenvalues
            if (solver.eigenvalues()[0] <= 0.0) {
                DRWN_LOG_WARNING("all eigenvalues are negative in \"" << getName() << "\"");
            } else {
                for (int i = 1; i < _numOutputDims; i++) {
                    if (solver.eigenvalues()[i] <= 0.0) {
                        _numOutputDims = i;
                        DRWN_LOG_WARNING("reducing number of outputs to " << _numOutputDims << " in \"" << getName() << "\"");
                        break;
                    }
                }

                // construct transformation
                _projection = solver.eigenvectors().topLeftCorner(dataDim, _numOutputDims).transpose();
                for (int i = 0; i < _numOutputDims; i++) {
                    _projection.row(i) /= sqrt(solver.eigenvalues()[i]);
                }
            }
        } else {
            // check for negative eigenvalues
            if (solver.eigenvalues()[dataDim - 1] <= 0.0) {
                DRWN_LOG_WARNING("all eigenvalues are negative in \"" << getName() << "\"");
            } else {
                for (int i = 1; i < _numOutputDims; i++) {
                    if (solver.eigenvalues()[dataDim - i - 1] <= 0.0) {
                        _numOutputDims = i;
                        DRWN_LOG_WARNING("reducing number of outputs to " << _numOutputDims << " in \"" << getName() << "\"");
                        break;
                    }
                }

                // construct transformation
                _projection = solver.eigenvectors().topRightCorner(dataDim, _numOutputDims).transpose();
                for (int i = 0; i < _numOutputDims; i++) {
                    _projection.row(i) /= sqrt(solver.eigenvalues()[dataDim - _numOutputDims + i]);
                }
            }
        }

        DRWN_LOG_VERBOSE("eigenvalues: " << solver.eigenvalues().transpose());
    } else {
        DRWN_LOG_WARNING("not enough data to compute principal components");
    }

    DRWN_END_PROGRESS;
}

// drwnMultiClassLDANode -----------------------------------------------------

drwnMultiClassLDANode::drwnMultiClassLDANode(const char *name, drwnGraph *owner) :
    drwnLinearTransformNode(name, owner), _trainingColour(-1), _lambda(1.0e-3)
{
    _nVersion = 100;
    _desc = "Multi-class linear discriminant analysis";

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "targetIn", "vector of labels"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("regularization", new drwnDoubleRangeProperty(&_lambda, 0.0, DRWN_DBL_MAX));
}

drwnMultiClassLDANode::drwnMultiClassLDANode(const drwnMultiClassLDANode& node) :
    drwnLinearTransformNode(node), _trainingColour(node._trainingColour),
    _lambda(node._lambda)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_trainingColour));
    declareProperty("regularization", new drwnDoubleRangeProperty(&_lambda, 0.0, DRWN_DBL_MAX));
}

drwnMultiClassLDANode::~drwnMultiClassLDANode()
{
    // do nothing
}

void drwnMultiClassLDANode::resetParameters()
{
    // clear parameters
    _translation = VectorXd::Zero(0);
    _projection = MatrixXd::Zero(0,0);
}

void drwnMultiClassLDANode::initializeParameters()
{
    // get input tables
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblTarget = _inputPorts[1]->getTable();
    if ((tblData == NULL) || (tblTarget == NULL)) {
        DRWN_LOG_ERROR("node \"" << getName()
            << "\" has no data or labels needed for parameter estimation");
        return;
    }

    // accumulate class statistics
    int dataDim = -1;
    MatrixXd dataSum2;
    vector<VectorXd *> dataSum;
    vector<double> dataCount;

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

        // TODO: check data is the right format

        // accumulate sufficient statistics
        for (int i = 0; i < recData->numObservations(); i++) {
            // ignore "unknown" class labels
            int classId = (int)recTarget->data()(i);
            if (classId < 0) continue;

            if (dataDim == -1) {
                dataDim = recData->numFeatures();
                dataSum2 = MatrixXd::Zero(dataDim, dataDim);
            }
            DRWN_ASSERT(recData->numFeatures() == dataDim);

            // ensure vectors are the right size
            if ((int)dataSum.size() <= classId) {
                dataSum.reserve(classId + 1);
                dataCount.resize(classId + 1, 0.0);
                while ((int)dataSum.size() <= classId) {
                    dataSum.push_back(new VectorXd(dataDim));
                    *dataSum.back() = VectorXd::Zero(dataDim);
                }
            }

            // update
            *dataSum[classId] += recData->data().row(i);
            dataSum2 += recData->data().row(i).transpose() * recData->data().row(i);
            dataCount[classId] += 1.0;
        }

        // release records
        tblTarget->unlockRecord(*it);
        tblData->unlockRecord(*it);
    }

    // compute projections
    int numClasses = (int)dataSum.size();
    if (numClasses < 2) {
        DRWN_LOG_ERROR("LDA requires more than two classes in \"" << getName() << "\"");
    } else if (numClasses > dataDim) {
        DRWN_LOG_ERROR("LDA requires more features than classes in \"" << getName() << "\"");
    } else {

        // compute class means
        _translation = VectorXd::Zero(dataDim);
        double totalCount = 0.0;
        for (int i = 0; i < numClasses; i++) {
            _translation += *dataSum[i];
            totalCount += dataCount[i];
            if (dataCount[i] > 0.0) {
                *dataSum[i] /= dataCount[i];
            } else {
                DRWN_LOG_WARNING("no examples of class " << i << " in \"" << getName() << "\"");
            }
        }

        DRWN_ASSERT(totalCount > 0.0);

        // global mean and covariance
        _translation /= totalCount;
        dataSum2 = dataSum2 / totalCount - _translation * _translation.transpose();

        // between class covariance
        // TODO: check should not be weighted by class size?
        MatrixXd sigmaB = MatrixXd::Zero(dataDim, dataDim);
        for (int i = 0; i < numClasses; i++) {
            sigmaB += (_translation - *dataSum[i]) *
                (_translation - *dataSum[i]).transpose();
        }
        sigmaB /= (double)numClasses;

        // TODO: regularization

        MatrixXd sigma = dataSum2.inverse() * sigmaB;
        Eigen::EigenSolver<MatrixXd> solver(sigma);

        // construct transformation
        if (solver.eigenvalues()[0].real() > solver.eigenvalues()[dataDim - 1].real()) {
            _projection = solver.eigenvectors().real().topLeftCorner(dataDim, numClasses - 1).transpose();
            for (int i = 0; i < numClasses - 1; i++) {
                _projection.row(i) /= sqrt(solver.eigenvalues()[i].real());
            }
        } else {
            _projection = solver.eigenvectors().real().topRightCorner(dataDim, numClasses - 1).transpose();
            for (int i = 0; i < numClasses - 1; i++) {
                _projection.row(i) /= sqrt(solver.eigenvalues()[dataDim - numClasses + 1 + i].real());
            }
        }

        DRWN_LOG_VERBOSE("eigenvalues: " << solver.eigenvalues().transpose());
    }

    // free memory
    for (unsigned i = 0; i < dataSum.size(); i++) {
        delete dataSum[i];
    }

    DRWN_END_PROGRESS;
}


// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Static", drwnLinearTransformNode);
DRWN_AUTOREGISTERNODE("Adaptive", drwnRescaleNode);
DRWN_AUTOREGISTERNODE("Adaptive", drwnPCANode);
DRWN_AUTOREGISTERNODE("Adaptive", drwnMultiClassLDANode);

