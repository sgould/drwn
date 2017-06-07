/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfusionMatrixNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnConfusionMatrixNode.h"

using namespace std;
using namespace Eigen;

// drwnConfusionMatrixNode ---------------------------------------------------

drwnConfusionMatrixNode::drwnConfusionMatrixNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _colour(-1), _filename("")
{
    _nVersion = 100;
    _desc = "Computes a confusion matrix";

    _inputPorts.push_back(new drwnInputPort(this, "actual",
            "N-by-K or N-by-1 matrix of classifications"));
    _inputPorts.push_back(new drwnInputPort(this, "predicted",
            "N-by-K or N-by-1 matrix of classifications"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("confusion", new drwnMatrixProperty(&_confusion, true));

    // TODO: add normalization property
}

drwnConfusionMatrixNode::drwnConfusionMatrixNode(const drwnConfusionMatrixNode& node) :
    drwnNode(node), _colour(node._colour), _filename(node._filename),
    _confusion(node._confusion)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("confusion", new drwnMatrixProperty(&_confusion, true));
}

drwnConfusionMatrixNode::~drwnConfusionMatrixNode()
{
    // do nothing
}

// gui
void drwnConfusionMatrixNode::showWindow()
{
    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnGUIBarPlot(this);
        _window->Show();
    }

    updateWindow();
}

void drwnConfusionMatrixNode::updateWindow()
{
    if ((_window == NULL) || (!_window->IsShown())) return;
    ((drwnGUIBarPlot *)_window)->setData(_confusion);
    ((drwnGUIBarPlot *)_window)->setStacked(true);
}

// processing
void drwnConfusionMatrixNode::evaluateForwards()
{
    drwnDataTable *tblActual = _inputPorts[0]->getTable();
    drwnDataTable *tblPredicted = _inputPorts[1]->getTable();
    if ((tblActual == NULL) || (tblPredicted == NULL)) {
        DRWN_LOG_WARNING("node \"" << _name << "\" requires two inputs");
        return;
    }

    _confusion = MatrixXd::Zero(1, 1);

    // interate over records
    vector<string> keys = tblActual->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        if (!getOwner()->getDatabase()->matchColour(*it, _colour)) continue;

        const drwnDataRecord *recActual = tblActual->lockRecord(*it);
        DRWN_ASSERT(recActual != NULL);

        const drwnDataRecord *recPredicted = tblPredicted->lockRecord(*it);
        if (recPredicted == NULL) {
            DRWN_LOG_WARNING("missing predictions for record \"" << *it << "\"");
        } else {

            // check dimensions
            if (recPredicted->numObservations() == 0) {
                DRWN_LOG_WARNING("missing predictions for \"" << *it << "\"");

            } else if (recActual->numObservations() != recPredicted->numObservations()) {
                DRWN_LOG_ERROR("size mismatch between actual (" << recActual->numObservations()
                    << ") and predicted observations (" << recPredicted->numObservations() << ")");

            } else {
                VectorXi actualClass;
                if (recActual->numFeatures() == 1) {
                    actualClass = recActual->data().col(0).cast<int>();
                } else {
                    actualClass = VectorXi::Zero(recActual->numObservations());
                    for (int i = 0; i < recActual->numObservations(); i++) {
                        recActual->data().row(i).maxCoeff(&actualClass[i]);
                    }
                }

                VectorXi predictedClass;
                if (recPredicted->numFeatures() == 1) {
                    predictedClass = recPredicted->data().col(0).cast<int>();
                } else {
                    predictedClass = VectorXi::Zero(recPredicted->numObservations());
                    for (int i = 0; i < recPredicted->numObservations(); i++) {
                        recPredicted->data().row(i).maxCoeff(&predictedClass[i]);
                    }
                }

                int nActual = actualClass.maxCoeff() + 1;
                int nPredicted = predictedClass.maxCoeff() + 1;
                if ((nActual > _confusion.rows()) ||
                    (nPredicted > _confusion.cols())) {
                    // TODO: move to matrix resize function
		  MatrixXd tmp = MatrixXd::Zero(std::max(nActual, (int)_confusion.rows()),
		        std::max(nPredicted, (int)_confusion.cols()));
                    tmp.topLeftCorner(_confusion.rows(), _confusion.cols()) = _confusion;
                    _confusion = tmp;
                }

                for (int i = 0; i < actualClass.rows(); i++) {
                    // skip missing/unknown labels
                    if ((actualClass[i] < 0) || (predictedClass[i] < 0)) {
                        continue;
                    }
                    _confusion(actualClass[i], predictedClass[i]) += 1.0;
                }
            }
        }

        // release records
        tblActual->unlockRecord(*it);
        tblPredicted->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;

    // show overall accuracy
    DRWN_LOG_VERBOSE(getName() << " accuracy is " << _confusion.diagonal().sum() / _confusion.sum());

    // write output
    if (!_filename.empty()) {
        ofstream ofs(_filename.c_str());
        if (ofs.fail()) {
            DRWN_LOG_ERROR("node \"" << _name << "\" could not open file " << _filename);
            return;
        }

        ofs.close();
    }

    updateWindow();
}

void drwnConfusionMatrixNode::resetParameters()
{
    _confusion = MatrixXd::Zero(0, 0);
    updateWindow();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Sink", drwnConfusionMatrixNode);
