/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDecisionTreeNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

#include "drwnDecisionTreeNode.h"

using namespace std;
using namespace Eigen;

// drwnDecisionTreeNode ------------------------------------------------------

drwnDecisionTreeNode::drwnDecisionTreeNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _trainingColour(-1)
{
    _nVersion = 100;
    _desc = "Multi-class decision tree classifier";

    // declare propertys
    declareProperty("trainingColour", new drwnIntegerProperty(&_trainingColour));
    //declareProperty("subSample", new drwnIntegerProperty(&_subSamplingRate));
    exposeProperties(&_classifier);

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D feature matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "targetIn", "N-by-1 target vector"));
    _inputPorts.push_back(new drwnInputPort(this, "weightsIn", "N-by-1 sample weights (for training)"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-1 output marginals or log-marginals"));
}

drwnDecisionTreeNode::drwnDecisionTreeNode(const drwnDecisionTreeNode& node) :
    drwnNode(node), _trainingColour(node._trainingColour),
    _classifier(node._classifier)
{
    // declare propertys
    declareProperty("trainingColour", new drwnIntegerProperty(&_trainingColour));
    //declareProperty("subSample", new drwnIntegerProperty(&_subSamplingRate));
    exposeProperties(&_classifier);
}

drwnDecisionTreeNode::~drwnDecisionTreeNode()
{
    // do nothing
}

// i/o
bool drwnDecisionTreeNode::save(drwnXMLNode& xml) const
{
    drwnNode::save(xml);
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnDecisionTree", NULL, false);
    return _classifier.save(*node);
}

bool drwnDecisionTreeNode::load(drwnXMLNode& xml)
{
    drwnNode::load(xml);
    drwnXMLNode *node = xml.first_node("drwnDecisionTree");
    if (node != NULL) {
        return _classifier.load(*node);
    }
    return false;
}

void drwnDecisionTreeNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnDecisionTreeNode::updateForwards()
{
    // check that model has been trained
    if (!_classifier.valid()) {
        DRWN_LOG_WARNING("classifier has not been learned");
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

        // TODO: check input dimensions

        // evaluate each row independently
        recordOut->structure() = recordIn->structure();
        recordOut->data() = MatrixXd::Zero(recordIn->numObservations(), _classifier.numClasses());
        for (int i = 0; i < recordIn->numObservations(); i++) {
            // TODO: unneccessary copying
            vector<double> x(_classifier.numFeatures()), y;
            Eigen::Map<VectorXd>(&x[0], _classifier.numFeatures()) = recordIn->data().row(i);
            _classifier.getClassScores(x, y);
            recordOut->data().row(i) = (Eigen::Map<VectorXd>(&y[0], y.size())).transpose();
        }

        // unlock records
        tblOut->unlockRecord(*it);
        tblIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnDecisionTreeNode::resetParameters()
{
    // clear model
    _classifier.initialize(0, 0);
}

void drwnDecisionTreeNode::initializeParameters()
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

    // clear existing decision tree
    resetParameters();

    // accumulate data
    vector<vector<double> > features;
    vector<int> target;
    vector<double> weights;

    // interate over input records
    int numClasses = 0;
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
            //if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
            //    continue;

            // add features
            features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            target.push_back((int)recTarget->data()(i));
            numClasses = std::max(target.back() + 1, numClasses);
            if (tblWeight != NULL) {
                if (recWeight != NULL) {
                    weights.push_back(recWeight->data()(i));
                } else {
                    weights.push_back(1.0);
                }
            }
        }

        // release records
        tblData->unlockRecord(*it);
        tblTarget->unlockRecord(*it);
        if (recWeight != NULL) tblWeight->unlockRecord(*it);
    }

    // learn the tree
    if (features.empty() || (numClasses < 2)) {
        DRWN_LOG_ERROR("insufficient training data for learning");
    } else {
        DRWN_LOG_VERBOSE("training \"" << getName() << "\" with " << features.size()
            << " training examples of length " << features[0].size());

        _classifier.initialize(features[0].size(), numClasses);
        if (weights.empty()) {
            _classifier.train(features, target);
        } else {
            _classifier.train(features, target, weights);
        }
    }

    DRWN_END_PROGRESS;
    DRWN_FCN_TOC;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Adaptive", drwnDecisionTreeNode);

