/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTRWSInference.cpp
** AUTHOR(S):   Hendra Gunadi <u4971560@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <cstdlib>

#include "drwnBase.h"
#include "drwnIO.h"

#include "drwnTRWSInference.h"

using namespace std;

// drwnTRWSInference -----------------------------------------------------------

int drwnTRWSInference::CONVERGENCE_STEP = 10;
double drwnTRWSInference::EPSILON = 1.0e-6;
double drwnTRWSInference::THETA_CONST = 0.0;

drwnTRWSInference::drwnTRWSInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph), _numNodes(graph.numVariables())
{
    _initialized = false;
}

drwnTRWSInference::~drwnTRWSInference()
{
    clear();
}

void drwnTRWSInference::clear()
{
    int size, j, size_2;
    for (int i = 0; i < _numNodes; i++) {
        _next1[i].clear();
        _prev1[i].clear();
        _next2[i].clear();
        _prev2[i].clear();

        _outFactor1[i].clear();
        _outFactor2[i].clear();
        _inFactor1[i].clear();
        _inFactor2[i].clear();

        if (_flag[i] == 0) delete(_unary[i]);

        size = _outMsg1[i].size();
        for (j = 0; j < size; j++) {
            size_2 = _incMsgOp[i][j].size();
            for (int k = 0; k < size_2; k++) {
                delete(_incMsgOp[i][j][k]);
            }
            delete(_outMsg1[i][j]);
            delete(_tempOutFactor1[i][j]);
        }
        _outMsg1[i].clear();
        _tempOutFactor1[i].clear();

        size = _outMsg2[i].size();
        for (j = 0; j < size; j++) {
            size_2 = _decMsgOp[i][j].size();
            for (int k = 0; k < size_2; k++) {
                delete(_decMsgOp[i][j][k]);
            }
            delete(_outMsg2[i][j]);
            // No need to delete the backward temp factor as it shares the same space with tempOutFactor1
        }
        _outMsg2[i].clear();
        _tempOutFactor2[i].clear();

        delete(_calibrated[i]);

        _margin[i].clear();

        size = _incUnaryOp[i].size();
        for (j = 0; j < size; j++) {
            delete(_incUnaryOp[i][j]);
        }
        size = _decUnaryOp[i].size();
        for (j = 0; j < size; j++) {
            delete(_decUnaryOp[i][j]);
        }
    }

    for (int i = 0; i < _numNodes; i++) {
        _inPrevMsg1[i].clear();
        _inPrevMsg2[i].clear();
        _inMsg1[i].clear();
        _inMsg2[i].clear();
    }

    _next1.clear();
    _prev1.clear();
    _next2.clear();
    _prev2.clear();
    _outFactor1.clear();
    _outFactor2.clear();
    _tempOutFactor1.clear();
    _tempOutFactor2.clear();
    _inFactor1.clear();
    _inFactor2.clear();
    _unary.clear();
    _outMsg1.clear();
    _inPrevMsg1.clear();
    _inMsg1.clear();
    _outMsg2.clear();
    _inPrevMsg2.clear();
    _inMsg2.clear();
    _calibrated.clear();
    _margin.clear();
    
    _incUnaryOp.clear();
    _decUnaryOp.clear();
    _incMsgOp.clear();
    _decMsgOp.clear();

    _initialized = false;
}

pair<double, double> drwnTRWSInference::inference(drwnFullAssignment& mapAssignment)
{
    if (!_initialized) {
        initialize();
    }

    // Temporary variables
    unsigned currentIteration = 0;
    drwnVarUniversePtr universe(_graph.getUniverse());
    int tempSize, tempSize_2;
    vector<double> Ebound;
    bool stop = false;
    double delta;
    drwnTableFactorStorage storage, storage2;

    while((currentIteration < drwnMessagePassingMAPInference::MAX_ITERATIONS) && (!stop)) {
        // One Iteration ---------------
        // Set lower bound to THETA_CONST
        Ebound.push_back(THETA_CONST);

        // For nodes s in V do the following operations in the order of increasing i(s):
        for (int i = 0; i < _numNodes; i++) {
            // Compute calibrated unary theta

            tempSize = (int)(_incUnaryOp[i].size());
            // Sum all the incoming messages
            // Reset the calibrated to unary
            _calibrated[i]->fill(0.0);
            for (int j = 0; j < tempSize; j++) {
                _incUnaryOp[i][j]->execute();   // Calibrated = inMsg1 + unary
            }
            // Add the unary theta
            

            // Normalize unary theta
            delta = (*_calibrated[i])[_calibrated[i]->indexOfMin()];
            _calibrated[i] = &(_calibrated[i]->offset(-1.0 * delta));
            Ebound[currentIteration] += delta;

            // For every outgoing edge, update message
            tempSize = (int) (_outMsg1[i].size());
            int Ns = max((int)_inPrevMsg1[i].size(), (int)_outMsg1[i].size());
            _calibrated[i] = &(_calibrated[i]->scale(1.0 / (double)Ns));
            for (int j = 0; j < tempSize; j++) {
		        _tempOutFactor1[i][j]->fill(0.0);
                tempSize_2 = (int)(_incMsgOp[i][j].size());
                for (int k = 0; k < tempSize_2; k++) {
                    _incMsgOp[i][j][k]->execute();  // OutMsg1 = Minimize(Calibrated + inPrevMsg2 + outFactor1)
                }

                // then normalize the message
                delta = (*_outMsg1[i][j])[_outMsg1[i][j]->indexOfMin()];
                _outMsg1[i][j] = &(_outMsg1[i][j]->offset(-1.0 * delta));
                Ebound[currentIteration] += delta;
            }
        }

        // Reverse the order
        for (int i = _numNodes - 1; i >= 0; i--) {
            // Compute calibrated unary theta

            tempSize = (int)(_decUnaryOp[i].size());
            // Sum all the incoming messages
            // Reset the calibrated unary
            _calibrated[i]->fill(0.0);
            for (int j = 0; j < tempSize; j++) {
                _incUnaryOp[i][j]->execute();   // Calibrated = inMsg2 + unary
            }
            // Add the unary theta

            // Normalize unary theta
            delta = (*_calibrated[i])[_calibrated[i]->indexOfMin()];
            _calibrated[i] = &(_calibrated[i]->offset(-1.0 * delta));
            Ebound[currentIteration] += delta;

            // For every outgoing edge, update message
            tempSize = (int) (_outMsg2[i].size());
            int Ns = max((int)_inPrevMsg2[i].size(), (int)_outMsg2[i].size());
            _calibrated[i] = &(_calibrated[i]->scale(1.0 / (double)Ns));
            for (int j = 0; j < tempSize; j++) {
                _tempOutFactor2[i][j]->fill(0.0);
                tempSize_2 = (int)(_decMsgOp[i][j].size());
                for (int k = 0; k < tempSize_2; k++) {
                    _decMsgOp[i][j][k]->execute();  // OutMsg2 = Minimize(Calibrated + inPrevMsg1 + outFactor2)
                }

                // then normalize the message;
                delta = (*_outMsg2[i][j])[_outMsg2[i][j]->indexOfMin()];
                _outMsg2[i][j] = &(_outMsg2[i][j]->offset(-1.0 * delta));
                Ebound[currentIteration] += delta;
            }
        }

        // Only keep latest CONVERGENCE_STEPs iteration
        if ((int)currentIteration >= CONVERGENCE_STEP) {
            const int size = Ebound.size();
            stop = true;
            // Check for the Ebound for the latest 10 iterations
            for (int i = size - 1; i >= size - CONVERGENCE_STEP; i--) {
                if ((Ebound[i] - Ebound[i - 1]) > EPSILON) {
                    stop = false;
                    break;
                }
            }
            // If they are only changing within the threshold, stopping condition reached
        }
        currentIteration++;
    }

    drwnPartialAssignment tempAssignment;
    vector<const drwnTableFactor*> messages;
    drwnTableFactor factor(universe, &storage);
    factor.addVariable(0);
    factor.fill(0.0);
    messages.push_back(_unary[0]);
    tempSize = (int)(_inPrevMsg2[0].size());
    for (int j = 0; j < tempSize; j++) {
       messages.push_back(_inPrevMsg2[0][j]);
    }
    drwnFactorAdditionOp(&factor, messages).execute();
    factor.assignmentOf(factor.indexOfMin(), tempAssignment);
    for (int i = 1; i < _numNodes; i++) {
	messages.clear();
        drwnTableFactor result1(universe, &storage);
        result1.addVariable(i);
        result1.fill(0.0);
        messages.push_back(_unary[i]);
        // Sum over all the messages coming from the nodes after
        tempSize = (int)(_inPrevMsg2[i].size());
        for (int j = 0; j < tempSize; j++) {
            messages.push_back(_inPrevMsg2[i][j]);
        }
        drwnFactorAdditionOp(&result1, messages).execute();

        // Sum over all the pairwise theta conditioned on previous assignment
        tempSize = (int)(_inFactor1[i].size());
        for (int j = 0; j < tempSize; j++) {
            drwnTableFactor result2(universe, &storage2);
            result2.addVariable(i);
            drwnFactorReduceOp(&result2, _inFactor1[i][j], tempAssignment).execute();
            drwnFactorPlusEqualsOp(&result1, &result2).execute();
        }
        result1.assignmentOf(result1.indexOfMin(), tempAssignment);
    }

    mapAssignment = drwnFullAssignment(tempAssignment);

    return make_pair(_graph.getEnergy(mapAssignment), Ebound.back());
}

void drwnTRWSInference::initialize()
{
    // Temporary variables
    int id1, id2;
    int numFactors = _graph.numFactors();
    drwnTableFactor* factor;
    const drwnTableFactor* temp;
    drwnVarUniversePtr universe(_graph.getUniverse());
    int tempSize;
    _flag.resize(_numNodes);

    // Initialize the space for messages
    _next1.resize(_numNodes);
    _prev1.resize(_numNodes);
    _next2.resize(_numNodes);
    _prev2.resize(_numNodes);
    _outFactor1.resize(_numNodes);
    _outFactor2.resize(_numNodes);
    _tempOutFactor1.resize(_numNodes);
    _tempOutFactor2.resize(_numNodes);
    _inFactor1.resize(_numNodes);
    _inFactor2.resize(_numNodes);
    _unary.resize(_numNodes);
    _outMsg1.resize(_numNodes);
    _inPrevMsg1.resize(_numNodes);
    _inMsg1.resize(_numNodes);
    _outMsg2.resize(_numNodes);
    _inPrevMsg2.resize(_numNodes);
    _inMsg2.resize(_numNodes);
    _calibrated.resize(_numNodes);
    _margin.resize(_numNodes);

    // Read all the factors in the graph
    for (int i = 0; i < numFactors; i++) {
        temp = _graph.getFactor(i);
        tempSize = temp->size();
        // Populate the unary (node)
        if (tempSize == 1) {
            id1 = temp->varId(0);
            _flag[id1] = 1;
            _unary[id1] = temp;

            // Insert factor for calibrated unary table factor
            factor = new drwnTableFactor(universe);
            factor->addVariable(id1);
            _calibrated[id1] = factor;
        }
        // Populate the edge (could be clique of more than 2 variables)
	    else if (tempSize > 1) {
            vector<int> id;
            for (int j = 0; j < tempSize; j++) {
                id.push_back(temp->varId(j));
            }
            int size = id.size();
            for (int j = 1; j < size; j++) {
                for (int k = j - 1; k >= 0; k--) {
                    id1 = id[j];
                    id2 = id[k];

                    if (id1 > id2) {
                        // swap id1 and id2
                        id2 = id1 ^ id2;
                        id1 = id1 ^ id2;
                        id2 = id1 ^ id2;
                    }

                    // Populate the variable ID for the next and previous node. forward next = backward prev
                    _next1[id1].push_back(id2);
                    _prev1[id2].push_back(id1);
                    _prev2[id1].push_back(id2);
                    _next2[id2].push_back(id1);

                    // Create a table for next incoming message = prev outgoing message in increasing order
                    factor = new drwnTableFactor(universe);
                    factor->addVariable(id2);
                    _inPrevMsg1[id2].push_back(factor);
                    _outMsg1[id1].push_back(factor);

                    // Create a table for next incoming message = prev outgoing message in decreased order
                    factor = new drwnTableFactor(universe);
                    factor->addVariable(id1);
                    _inPrevMsg2[id1].push_back(factor);
                    _outMsg2[id2].push_back(factor);

                    // Add to overall incoming message
                    _inMsg1[id1].push_back(_outMsg2[id2].back());
                    _inMsg1[id2].push_back(_inPrevMsg1[id2].back());
                    _inMsg2[id1].push_back(_inPrevMsg2[id1].back());
                    _inMsg2[id2].push_back(_outMsg1[id1].back());

                    // Refer the pairwise theta
                    _inFactor1[id2].push_back(temp);
                    _inFactor2[id1].push_back(temp);
                    _outFactor1[id1].push_back(temp);
                    _outFactor2[id2].push_back(temp);

                    int size = temp->size();
                    factor = new drwnTableFactor(universe);
                    for (int l = 0; l < size; l++) {
                        factor->addVariable(temp->varId(l));
                    }
                    _tempOutFactor1[id1].push_back(factor);
                    _tempOutFactor2[id2].push_back(factor);

                    // Insert other index to be marginalized
                    _margin[id1].insert(id2);
                    _margin[id2].insert(id1);
                }
            }
            id.clear();
        }
    }

    // TODO: Populate the constant factor?

    // Check whether there's variable missing then free the flag to indicate missing factor
    for (int i = 0; i < _numNodes; i++) {
        // Insert 0 table factor if not yet initialized
        if (_flag[i] == 0) {
            // Add unary table factor of 0
            factor = new drwnTableFactor(universe);
            factor->addVariable(i);
            factor->fill(0.0);
            _unary[i] = factor;
            // Add space for calibrated table factor
            factor = new drwnTableFactor(universe);
            factor->addVariable(i);
            factor->fill(0.0);
            _calibrated[i] = factor;
        }
    }

    // Initialize the computation graph
    buildComputationGraph();
    _initialized = true;
}

void drwnTRWSInference::buildComputationGraph() 
{
    _incUnaryOp.resize(_numNodes);
    _decUnaryOp.resize(_numNodes);
    _incMsgOp.resize(_numNodes);
    _decMsgOp.resize(_numNodes);

    int tempSize;
    for (int i = 0; i < _numNodes; i++) {
        // Compute calibrated unary theta

        tempSize = (int)(_inMsg1[i].size());
        // Sum all the incoming messages
        for (int j = 0; j < tempSize; j++) {
            _incUnaryOp[i].push_back(new drwnFactorPlusEqualsOp(_calibrated[i], _inMsg1[i][j]));
        }
        // Add the unary theta
        _incUnaryOp[i].push_back(new drwnFactorPlusEqualsOp(_calibrated[i], _unary[i]));

        // For every outgoing edge, update message
        tempSize = (int) (_outMsg1[i].size());
        _incMsgOp[i].resize(tempSize);
        for (int j = 0; j < tempSize; j++) {
            _incMsgOp[i][j].push_back(new drwnFactorPlusEqualsOp(_tempOutFactor1[i][j], _calibrated[i]));
            _incMsgOp[i][j].push_back(new drwnFactorMinusEqualsOp(_tempOutFactor1[i][j], _inPrevMsg2[i][j]));
            _incMsgOp[i][j].push_back(new drwnFactorPlusEqualsOp(_tempOutFactor1[i][j], _outFactor1[i][j]));
            _incMsgOp[i][j].push_back(new drwnFactorMinimizeOp(_outMsg1[i][j], _tempOutFactor1[i][j], _margin[_next1[i][j]]));
        }
    }

    // Reverse the order
    for (int i = _numNodes - 1; i >= 0; i--) {
        // Compute calibrated unary theta

        tempSize = (int)(_inMsg2[i].size());
        // Sum all the incoming messages
        for (int j = 0; j < tempSize; j++) {
            _decUnaryOp[i].push_back(new drwnFactorPlusEqualsOp(_calibrated[i], _inMsg2[i][j]));
        }
        // Add the unary theta
        _decUnaryOp[i].push_back(new drwnFactorPlusEqualsOp(_calibrated[i], _unary[i]));

        // For every outgoing edge, update message
        tempSize = (int) (_outMsg2[i].size());
        _decMsgOp[i].resize(tempSize);
        for (int j = 0; j < tempSize; j++) {
            _decMsgOp[i][j].push_back(new drwnFactorPlusEqualsOp(_tempOutFactor2[i][j], _calibrated[i]));
            _decMsgOp[i][j].push_back(new drwnFactorMinusEqualsOp(_tempOutFactor2[i][j], _inPrevMsg1[i][j]));
            _decMsgOp[i][j].push_back(new drwnFactorPlusEqualsOp(_tempOutFactor2[i][j], _outFactor2[i][j]));
            _decMsgOp[i][j].push_back(new drwnFactorMinimizeOp(_outMsg2[i][j], _tempOutFactor2[i][j], _margin[_next2[i][j]]));
        }
    }
}

// drwnTRWSInferenceConfig ---------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnTRWSInference
//! \b convergenceStep :: number of iterations included in convergence check (default: 10)\n
//! \b epsilon :: smallest difference in value to be considered as equal (default: 1.0e-6)\n
//! \b thetaConst :: constant parameter (theta) value (default: 0.0)

class drwnTRWSInferenceConfig : public drwnConfigurableModule {
public:
    drwnTRWSInferenceConfig() : drwnConfigurableModule("drwnTRWSInference") { }
    ~drwnTRWSInferenceConfig() { }

    void usage(ostream &os) const {
        os << "    convergenceStep :: number of iterations included in convergence check (default: "
           << drwnTRWSInference::CONVERGENCE_STEP << ")\n";
        os << "      epsilon       :: smallest difference in value to be considered as equal (default: "
           << drwnTRWSInference::EPSILON << ")\n";
        os << "      thetaConst    :: constant parameter (theta) value (default: "
           << drwnTRWSInference::THETA_CONST << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "convergenceStep")) {
            drwnTRWSInference::CONVERGENCE_STEP = std::max(0, atoi(value));
        } else if (!strcmp(name, "epsilon")) {
            drwnTRWSInference::EPSILON = std::max(0.0000000001, atof(value));
        } else if (!strcmp(name, "thetaConst")) {
            drwnTRWSInference::THETA_CONST = atof(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnTRWSInferenceConfig gTRWSInferenceConfig;
