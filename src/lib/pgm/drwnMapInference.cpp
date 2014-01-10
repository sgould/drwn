/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMAPInference.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <vector>
#include <set>
#include <map>
#include <iterator>
#include <iomanip>

#include "drwnBase.h"
#include "drwnPGM.h"

// drwnMAPInference class --------------------------------------------------

drwnMAPInference::drwnMAPInference(const drwnFactorGraph& graph) : _graph(graph)
{
    // do nothing
}

drwnMAPInference::drwnMAPInference(const drwnMAPInference& inf) :
    _graph(inf._graph)
{
    // do nothing
}

drwnMAPInference::~drwnMAPInference()
{
    // do nothing
}

// drwnICMInference class --------------------------------------------------

drwnICMInference::drwnICMInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnICMInference::~drwnICMInference()
{
    // do nothing
}

double drwnICMInference::inference(drwnFullAssignment& mapAssignment)
{
    // initialize MAP assignment
    mapAssignment.resize(_graph.getUniverse()->numVariables(), 0);
    double bestEnergy = _graph.getEnergy(mapAssignment);

    drwnFullAssignment assignment(mapAssignment);

    // iterate until convergence
    bool bConverged = false;
    while (!bConverged) {
        bConverged = true;

        // iterate through all variables
        for (unsigned v = 0; v < mapAssignment.size(); v++) {
            // iterate through all values
            for (int k = 0; k < _graph.getUniverse()->varCardinality(v); k++) {
                // exclude current assignment
                if (mapAssignment[v] == k)
                    continue;

                assignment[v] = k;
                double e = _graph.getEnergy(assignment);

                // update best assignment found so far
                // problems with: -55.8617122200000011389420251362025737762451171875
                // + 6.5225602696727946749888360500335693359375e-16
                if (bestEnergy - e > DRWN_EPSILON) {
                    bestEnergy = e;
                    mapAssignment[v] = k;
                    bConverged = false;
                    //DRWN_LOG_DEBUG("ICM " << bestEnergy << " : " << toString(mapAssignment));
                }
            }

            // revert assignment to best found so far
            assignment[v] = mapAssignment[v];
        }
    }

    return _graph.getEnergy(mapAssignment);
}

// drwnMessagePassingMAPInference class -------------------------------------

unsigned drwnMessagePassingMAPInference::MAX_ITERATIONS = 1000;
double drwnMessagePassingMAPInference::DAMPING_FACTOR = 0.0;

drwnMessagePassingMAPInference::drwnMessagePassingMAPInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnMessagePassingMAPInference::~drwnMessagePassingMAPInference()
{
    // release temporary memory and computation graph
    drwnMessagePassingMAPInference::clear();
}

void drwnMessagePassingMAPInference::clear()
{
    for (unsigned i = 0; i < _computations.size(); i++) {
	delete _computations[i];
    }
    _computations.clear();

    for (unsigned i = 0; i < _intermediateFactors.size(); i++) {
	delete _intermediateFactors[i];
    }
    _intermediateFactors.clear();

    for (unsigned i = 0; i < _forwardMessages.size(); i++) {
        delete _forwardMessages[i];
        delete _backwardMessages[i];
    }
    for (unsigned i = 0; i < _oldForwardMessages.size(); i++) {
        delete _oldForwardMessages[i];
        delete _oldBackwardMessages[i];
    }
    _forwardMessages.clear();
    _backwardMessages.clear();
    _oldForwardMessages.clear();
    _oldBackwardMessages.clear();

    for (unsigned i = 0; i < _sharedStorage.size(); i++) {
        delete _sharedStorage[i];
    }
    _sharedStorage.clear();
}

double drwnMessagePassingMAPInference::inference(drwnFullAssignment& mapAssignment)
{
    // initialize messages and build computation graph
    if (_computations.empty()) {
        initializeMessages();
        buildComputationGraph();
    }

    // iterate over message loop
    DRWN_LOG_VERBOSE("Starting message passing loop...");
    DRWN_LOG_VERBOSE("..." << _graph.numVariables() << " variables; "
        << _graph.numFactors() << " factors; " << _graph.numEdges() << " edges");
    bool bConverged = false;
    int nIteration = 0;

    while (!bConverged) {
	int nConverged = 0;
        bConverged = true;
        nIteration += 1;

	for (int i = 0; i < (int)_computations.size(); i++) {
	    _computations[i]->execute();
	}

#if 0
        // message debugging
        for (int i = 0; i < (int)_forwardMessages.size(); i++) {
            _forwardMessages[i]->dump();
            _backwardMessages[i]->dump();
        }
#endif

        for (int i = 0; i < (int)_forwardMessages.size(); i++) {
            if (_oldForwardMessages[i]->dataCompareAndCopy(*_forwardMessages[i])) {
                nConverged += 1;
            } else {
                bConverged = false;
            }

            if (_oldBackwardMessages[i]->dataCompareAndCopy(*_backwardMessages[i])) {
                nConverged += 1;
            } else {
                bConverged = false;
            }
        }

        DRWN_LOG_VERBOSE("...iteration " << nIteration << " ("
            << nConverged << " of " << (2 * _forwardMessages.size())
            << " messages converged)");

        if ((nIteration >= (int)MAX_ITERATIONS) && !bConverged) {
            DRWN_LOG_WARNING("message passing failed to converge after "
                << nIteration << " iterations (" << nConverged << " of "
                << (2 * _forwardMessages.size()) << " messages converged)");
            break;
        }
    }

    if (bConverged) {
	DRWN_LOG_VERBOSE("...converged in " << nIteration << " iterations");
    }

    // decode solution
    mapAssignment.resize(_graph.numVariables());
    fill(mapAssignment.begin(), mapAssignment.end(), -1);
    decodeBeliefs(mapAssignment);

    return _graph.getEnergy(mapAssignment);
}

void drwnMessagePassingMAPInference::initializeMessages()
{
    // get universe
    drwnVarUniversePtr pUniverse(_graph.getUniverse());

    // set up forward and backward messages
    if (_forwardMessages.empty()) {
	_forwardMessages.reserve(_graph.numEdges());
	_backwardMessages.reserve(_graph.numEdges());
	for (int i = 0; i < _graph.numEdges(); i++) {
            DRWN_LOG_DEBUG("...message between "
                << _graph.getEdge(i).first << " " << toString(_graph[_graph.getEdge(i).first]->getClique())
                << " and "
                << _graph.getEdge(i).second << " " << toString(_graph[_graph.getEdge(i).second]->getClique())
                << " over " << toString(_graph.getSepSet(i)));
	    _forwardMessages.push_back(new drwnTableFactor(pUniverse));
            _forwardMessages.back()->addVariables(_graph.getSepSet(i));
	    _forwardMessages.back()->fill(0.0);
            _backwardMessages.push_back(new drwnTableFactor(*_forwardMessages.back()));
	}
    } else {
        // reset messages to all zeros
	for (unsigned i = 0; i < _forwardMessages.size(); i++) {
            _forwardMessages[i]->fill(0.0);
            _backwardMessages[i]->fill(0.0);
        }
    }

    // storage for old messages (algorithms not needing this should override the
    // initializeMessages method (i.e. this function) and probably the inference method)
    _oldForwardMessages.reserve(_forwardMessages.size());
    _oldBackwardMessages.reserve(_backwardMessages.size());
    for (unsigned i = 0; i < _forwardMessages.size(); i++) {
        _oldForwardMessages.push_back(new drwnTableFactor(*_forwardMessages[i]));
        _oldBackwardMessages.push_back(new drwnTableFactor(*_backwardMessages[i]));
    }
}

void drwnMessagePassingMAPInference::decodeBeliefs(drwnFullAssignment& mapAssignment)
{
    // default: do nothing
}

// drwnMaxProductInference class --------------------------------------------

drwnMaxProdInference::drwnMaxProdInference(const drwnFactorGraph& graph) :
    drwnMessagePassingMAPInference(graph)
{
    // do nothing
}

drwnMaxProdInference::~drwnMaxProdInference()
{
    // do nothing
}

void drwnMaxProdInference::buildComputationGraph()
{
    DRWN_ASSERT(_computations.empty());

    // get universe
    drwnVarUniversePtr pUniverse(_graph.getUniverse());

    // initialize intermediate factors and storage
    _intermediateFactors.reserve(2 * _graph.numEdges());
    _sharedStorage.push_back(new drwnTableFactorStorage());

    // compute reverse edge-to-node indices
    vector<vector<int> > fwdIncidentEdges(_graph.numFactors());
    vector<vector<int> > bckIncidentEdges(_graph.numFactors());
    for (int m = 0; m < _graph.numEdges(); m++) {
        fwdIncidentEdges[_graph.getEdge(m).first].push_back(m);
        bckIncidentEdges[_graph.getEdge(m).second].push_back(m);
    }

    // build computation graph
    for (int m = 0; m < _graph.numEdges(); m++) {
	int fwdIndx = _graph.getEdge(m).first;
	int bckIndx = _graph.getEdge(m).second;
	vector<const drwnTableFactor *> incomingFwdMsgs;
	vector<const drwnTableFactor *> incomingBckMsgs;

	incomingFwdMsgs.push_back(_graph[fwdIndx]);
	incomingBckMsgs.push_back(_graph[bckIndx]);

        for (vector<int>::const_iterator k = fwdIncidentEdges[fwdIndx].begin();
             k != fwdIncidentEdges[fwdIndx].end(); ++k) {
            if (*k != m) incomingFwdMsgs.push_back(_oldBackwardMessages[*k]);
        }
        for (vector<int>::const_iterator k = bckIncidentEdges[fwdIndx].begin();
             k != bckIncidentEdges[fwdIndx].end(); ++k) {
            if (*k != m) incomingFwdMsgs.push_back(_oldForwardMessages[*k]);
        }
        for (vector<int>::const_iterator k = fwdIncidentEdges[bckIndx].begin();
             k != fwdIncidentEdges[bckIndx].end(); ++k) {
            if (*k != m) incomingBckMsgs.push_back(_oldBackwardMessages[*k]);
        }
        for (vector<int>::const_iterator k = bckIncidentEdges[bckIndx].begin();
             k != bckIncidentEdges[bckIndx].end(); ++k) {
            if (*k != m) incomingBckMsgs.push_back(_oldForwardMessages[*k]);
        }

	// forwards
	_intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
	_computations.push_back(new drwnFactorAdditionOp(_intermediateFactors.back(),
                incomingFwdMsgs));
	_computations.push_back(new drwnFactorMinimizeOp(_forwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorLogNormalizeOp(_forwardMessages[m]));
        if (DAMPING_FACTOR > 0.0) {
            _computations.push_back(new drwnFactorWeightedSumOp(_forwardMessages[m],
                _forwardMessages[m], _oldForwardMessages[m], 1.0 - DAMPING_FACTOR, DAMPING_FACTOR));
        }

	// backwards
	_intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
	_computations.push_back(new drwnFactorAdditionOp(_intermediateFactors.back(),
                incomingBckMsgs));
	_computations.push_back(new drwnFactorMinimizeOp(_backwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorLogNormalizeOp(_backwardMessages[m]));
        if (DAMPING_FACTOR > 0.0) {
            _computations.push_back(new drwnFactorWeightedSumOp(_backwardMessages[m],
                _backwardMessages[m], _oldBackwardMessages[m], 1.0 - DAMPING_FACTOR, DAMPING_FACTOR));
        }
    }
}

void drwnMaxProdInference::decodeBeliefs(drwnFullAssignment& mapAssignment)
{
    // compute reverse edge-to-node indices
    vector<vector<int> > fwdIncidentEdges(_graph.numFactors());
    vector<vector<int> > bckIncidentEdges(_graph.numFactors());
    for (int m = 0; m < _graph.numEdges(); m++) {
        fwdIncidentEdges[_graph.getEdge(m).first].push_back(m);
        bckIncidentEdges[_graph.getEdge(m).second].push_back(m);
    }

    // compute final beliefs
    for (int n = 0; n < _graph.numFactors(); n++) {
        // skip non-singleton cliques
        if (_graph[n]->size() != 1) continue;
        int var = _graph[n]->varId(0);

        drwnTableFactor minMarginal(*_graph[n]);
        for (vector<int>::const_iterator m = bckIncidentEdges[n].begin();
             m != bckIncidentEdges[n].end(); ++m) {
            DRWN_ASSERT((_forwardMessages[*m]->size() == 1) &&
                (_forwardMessages[*m]->varId(0) == var));
            for (unsigned i = 0; i < minMarginal.entries(); i++) {
                minMarginal[i] += (*_forwardMessages[*m])[i];
            }
        }

        for (vector<int>::const_iterator m = fwdIncidentEdges[n].begin();
             m != fwdIncidentEdges[n].end(); ++m) {
            DRWN_ASSERT((_backwardMessages[*m]->size() == 1) &&
                (_backwardMessages[*m]->varId(0) == var));
            for (unsigned i = 0; i < minMarginal.entries(); i++) {
                minMarginal[i] += (*_backwardMessages[*m])[i];
            }
        }

        mapAssignment[var] = minMarginal.indexOfMin();
    }

    // compute MAP for any variable not currently assigned
    //! \todo improve this? consisted decoding?
    for (int n = 0; n < _graph.numFactors(); n++) {
        drwnClique c = _graph[n]->getClique();
        drwnClique missingVars;
        for (drwnClique::const_iterator vi = c.begin(); vi != c.end(); vi++) {
            if (mapAssignment[*vi] == -1) {
                missingVars.insert(*vi);
            }
        }

        if (missingVars.empty())
            continue;

        drwnTableFactor minMarginal(*_graph[n]);
        for (vector<int>::const_iterator m = bckIncidentEdges[n].begin();
             m != bckIncidentEdges[n].end(); ++m) {
            drwnFactorPlusEqualsOp(&minMarginal, _forwardMessages[*m]).execute();
        }

        for (vector<int>::const_iterator m = fwdIncidentEdges[n].begin();
             m != fwdIncidentEdges[n].end(); ++m) {
            drwnFactorPlusEqualsOp(&minMarginal, _backwardMessages[*m]).execute();
        }

        drwnPartialAssignment assignment;
        minMarginal.assignmentOf(minMarginal.indexOfMin(), assignment);
        for (drwnPartialAssignment::const_iterator it = assignment.begin();
             it != assignment.end(); it++) {
            if (mapAssignment[it->first] == -1) {
                mapAssignment[it->first] = it->second;
            }
        }
    }
}

// drwnAsyncMaxProductInference class ---------------------------------------

drwnAsyncMaxProdInference::drwnAsyncMaxProdInference(const drwnFactorGraph& graph) :
    drwnMessagePassingMAPInference(graph)
{
    // do nothing
}

drwnAsyncMaxProdInference::~drwnAsyncMaxProdInference()
{
    // do nothing
}

void drwnAsyncMaxProdInference::buildComputationGraph()
{
    DRWN_ASSERT(_computations.empty());

    // get universe
    drwnVarUniversePtr pUniverse(_graph.getUniverse());

    // all intermediate factors share storage
    _intermediateFactors.reserve(2 * _graph.numEdges());
    _sharedStorage.push_back(new drwnTableFactorStorage());

    // compute reverse edge-to-node indices
    vector<vector<int> > fwdIncidentEdges(_graph.numFactors());
    vector<vector<int> > bckIncidentEdges(_graph.numFactors());
    for (int m = 0; m < _graph.numEdges(); m++) {
        fwdIncidentEdges[_graph.getEdge(m).first].push_back(m);
        bckIncidentEdges[_graph.getEdge(m).second].push_back(m);
    }

    // build computation graph
    for (int m = 0; m < _graph.numEdges(); m++) {
	int fwdIndx = _graph.getEdge(m).first;
	vector<const drwnTableFactor *> incomingFwdMsgs;
	incomingFwdMsgs.push_back(_graph[fwdIndx]);

        for (vector<int>::const_iterator k = fwdIncidentEdges[fwdIndx].begin();
             k != fwdIncidentEdges[fwdIndx].end(); ++k) {
            if (*k != m) incomingFwdMsgs.push_back(_backwardMessages[*k]);
        }
        for (vector<int>::const_iterator k = bckIncidentEdges[fwdIndx].begin();
             k != bckIncidentEdges[fwdIndx].end(); ++k) {
            if (*k != m) incomingFwdMsgs.push_back(_forwardMessages[*k]);
        }

	// forwards
	_intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
	_computations.push_back(new drwnFactorAdditionOp(_intermediateFactors.back(),
                incomingFwdMsgs));
	_computations.push_back(new drwnFactorMinimizeOp(_forwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorLogNormalizeOp(_forwardMessages[m]));
        if (DAMPING_FACTOR > 0.0) {
            _computations.push_back(new drwnFactorWeightedSumOp(_forwardMessages[m],
                _forwardMessages[m], _oldForwardMessages[m], 1.0 - DAMPING_FACTOR, DAMPING_FACTOR));
        }
    }

    // build computation graph
    for (int m = _graph.numEdges() - 1; m >= 0; m--) {
	int bckIndx = _graph.getEdge(m).second;
	vector<const drwnTableFactor *> incomingBckMsgs;
	incomingBckMsgs.push_back(_graph[bckIndx]);

        for (vector<int>::const_iterator k = fwdIncidentEdges[bckIndx].begin();
             k != fwdIncidentEdges[bckIndx].end(); ++k) {
            if (*k != m) incomingBckMsgs.push_back(_backwardMessages[*k]);
        }
        for (vector<int>::const_iterator k = bckIncidentEdges[bckIndx].begin();
             k != bckIncidentEdges[bckIndx].end(); ++k) {
            if (*k != m) incomingBckMsgs.push_back(_forwardMessages[*k]);
        }

	// backwards
	_intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
	_computations.push_back(new drwnFactorAdditionOp(_intermediateFactors.back(),
                incomingBckMsgs));
	_computations.push_back(new drwnFactorMinimizeOp(_backwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorLogNormalizeOp(_backwardMessages[m]));
        if (DAMPING_FACTOR > 0.0) {
            _computations.push_back(new drwnFactorWeightedSumOp(_backwardMessages[m],
                _backwardMessages[m], _oldBackwardMessages[m], 1.0 - DAMPING_FACTOR, DAMPING_FACTOR));
        }
    }
}

void drwnAsyncMaxProdInference::decodeBeliefs(drwnFullAssignment& mapAssignment)
{
    // compute reverse edge-to-node indices
    vector<vector<int> > fwdIncidentEdges(_graph.numFactors());
    vector<vector<int> > bckIncidentEdges(_graph.numFactors());
    for (int m = 0; m < _graph.numEdges(); m++) {
        fwdIncidentEdges[_graph.getEdge(m).first].push_back(m);
        bckIncidentEdges[_graph.getEdge(m).second].push_back(m);
    }

    // compute final beliefs
    for (int n = 0; n < _graph.numFactors(); n++) {
        // skip non-singleton cliques
        if (_graph[n]->size() != 1) continue;
        int var = _graph[n]->varId(0);

        drwnTableFactor minMarginal(*_graph[n]);
        for (vector<int>::const_iterator m = bckIncidentEdges[n].begin();
             m != bckIncidentEdges[n].end(); ++m) {
            DRWN_ASSERT((_forwardMessages[*m]->size() == 1) &&
                (_forwardMessages[*m]->varId(0) == var));
            for (unsigned i = 0; i < minMarginal.entries(); i++) {
                minMarginal[i] += (*_forwardMessages[*m])[i];
            }
        }

        for (vector<int>::const_iterator m = fwdIncidentEdges[n].begin();
             m != fwdIncidentEdges[n].end(); ++m) {
            DRWN_ASSERT((_backwardMessages[*m]->size() == 1) &&
                (_backwardMessages[*m]->varId(0) == var));
            for (unsigned i = 0; i < minMarginal.entries(); i++) {
                minMarginal[i] += (*_backwardMessages[*m])[i];
            }
        }

        mapAssignment[var] = minMarginal.indexOfMin();
    }

    // compute MAP for any variable not currently assigned
    //! \todo improve this? consisted decoding?
    for (int n = 0; n < _graph.numFactors(); n++) {
        drwnClique c = _graph[n]->getClique();
        drwnClique missingVars;
        for (drwnClique::const_iterator vi = c.begin(); vi != c.end(); vi++) {
            if (mapAssignment[*vi] == -1) {
                missingVars.insert(*vi);
            }
        }

        if (missingVars.empty())
            continue;

        drwnTableFactor minMarginal(*_graph[n]);
        for (vector<int>::const_iterator m = bckIncidentEdges[n].begin();
             m != bckIncidentEdges[n].end(); ++m) {
            drwnFactorPlusEqualsOp(&minMarginal, _forwardMessages[*m]).execute();
        }

        for (vector<int>::const_iterator m = fwdIncidentEdges[n].begin();
             m != fwdIncidentEdges[n].end(); ++m) {
            drwnFactorPlusEqualsOp(&minMarginal, _backwardMessages[*m]).execute();
        }

        drwnPartialAssignment assignment;
        minMarginal.assignmentOf(minMarginal.indexOfMin(), assignment);
        for (drwnPartialAssignment::const_iterator it = assignment.begin();
             it != assignment.end(); it++) {
            if (mapAssignment[it->first] == -1) {
                mapAssignment[it->first] = it->second;
            }
        }
    }
}

// drwnJunctionTreeInference class ------------------------------------------

drwnJunctionTreeInference::drwnJunctionTreeInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnJunctionTreeInference::~drwnJunctionTreeInference()
{
    // do nothing
}

double drwnJunctionTreeInference::inference(drwnFullAssignment& mapAssignment)
{
    // create new cluster graph (junction tree)
    drwnFactorGraph jt = drwnFactorGraphUtils::createJunctionTree(_graph);

    // run asynchronous max-product inference
    drwnAsyncMaxProdInference maxProduct(jt);
    double e = maxProduct.inference(mapAssignment);

    DRWN_ASSERT(find(mapAssignment.begin(), mapAssignment.end(), -1) == mapAssignment.end());
    DRWN_ASSERT_MSG(fabs(_graph.getEnergy(mapAssignment) - e) < DRWN_EPSILON,
        "E(" << toString(mapAssignment) << ") = " << _graph.getEnergy(mapAssignment) << " != " << e);
    return e;
}

// drwnGEMPLPInference class ------------------------------------------------
// _forwardMessages are \lambda_{c \to s}(x_s)
// _backwardMessages are \lambda_{s}^{-c}(x_s)

drwnGEMPLPInference::drwnGEMPLPInference(const drwnFactorGraph& graph) :
    drwnMessagePassingMAPInference(graph), _maxIterations(MAX_ITERATIONS)
{
    // do nothing
}

drwnGEMPLPInference::~drwnGEMPLPInference()
{
    // do nothing
}

void drwnGEMPLPInference::clear()
{
    drwnMessagePassingMAPInference::clear();

    _cliqueEdges.clear();
    _separatorEdges.clear();
    _separators.clear();
    _edges.clear();
    _lastDualObjective = -DRWN_DBL_MAX;
    _maxIterations = MAX_ITERATIONS;
}

double drwnGEMPLPInference::inference(drwnFullAssignment& mapAssignment)
{
    // initialize messages and build computation graph
    if (_computations.empty()) {
        initializeMessages();
        buildComputationGraph();
    }

    // compute contribution from disconnected factors
    double dualOffset = 0.0;
    for (int c = 0; c < (int)_cliqueEdges.size(); c++) {
        if (_cliqueEdges[c].empty()) {
            const drwnTableFactor *psi = _graph.getFactor(c);
            dualOffset += (*psi)[psi->indexOfMin()];
        }
    }

    // iterate over message loop
    DRWN_LOG_VERBOSE("Starting message passing loop...");
    DRWN_LOG_VERBOSE("..." << _graph.numVariables() << " variables; "
        << _graph.numFactors() << " factors; " << _graph.numEdges() << " edges; "
        << _cliqueEdges.size() << " cliques; " << _separators.size() << " separators");
    bool bConverged = false;
    int nIteration = 0;
#if 0
    mapAssignment.resize(_graph.numVariables(), -1);
    double bestEnergy = DRWN_DBL_MAX;
#else
    mapAssignment.resize(_graph.numVariables(), 0);
    double bestEnergy = _graph.getEnergy(mapAssignment);
#endif

    while (!bConverged) {
        bConverged = true;
        nIteration += 1;

        // pass all messages
	for (int i = 0; i < (int)_computations.size(); i++) {
	    _computations[i]->execute();
	}

        // compute dual
        double dualObjective = 0.0;
        drwnTableFactorStorage storage;
        for (int s = 0; s < (int)_separatorEdges.size(); s++) {
            //! \todo add this to computation graph
            drwnTableFactor phi(_graph.getUniverse(), &storage);
            vector<const drwnTableFactor *> incomingMessages;
            incomingMessages.reserve(_separatorEdges[s].size());
            for (set<int>::const_iterator messageId = _separatorEdges[s].begin();
                 messageId != _separatorEdges[s].end(); ++messageId) {
                incomingMessages.push_back(_forwardMessages[*messageId]);
            }
            drwnFactorAdditionOp(&phi, incomingMessages).execute();
            DRWN_ASSERT(phi.size() == _separators[s].size());
            dualObjective += phi[phi.indexOfMin()];
        }

        // add contribution from disconnected factors
        dualObjective += dualOffset;

        // decode assignment and compute primal
        drwnFullAssignment assignment;
        assignment.resize(_graph.numVariables());
        fill(assignment.begin(), assignment.end(), -1);
        decodeBeliefs(assignment);
        double e = _graph.getEnergy(assignment);
        if (e < bestEnergy) {
            bestEnergy = e;
            mapAssignment = assignment;
        }

        DRWN_LOG_VERBOSE("...iteration " << nIteration
            << "; dual objective " << dualObjective << "; best energy " << bestEnergy);

        // check for convergence of the dual
        bConverged = (dualObjective - _lastDualObjective < DRWN_EPSILON);
        _lastDualObjective = dualObjective;

        if ((nIteration >= (int)_maxIterations) && !bConverged) {
            DRWN_LOG_WARNING("message passing failed to converge after "
                << nIteration << " iterations");
            break;
        }
    }

    if (bConverged) {
	DRWN_LOG_VERBOSE("...converged in " << nIteration << " iterations");
    }

    DRWN_ASSERT(bestEnergy != DRWN_DBL_MAX);
    return bestEnergy;
}

void drwnGEMPLPInference::initializeMessages()
{
    // Get mapping from cliques and separators to edges where an edge is
    // between a clique and a spearator set.
    _separators.clear();
    _edges.clear();
    _cliqueEdges.resize(_graph.numFactors());
    _separatorEdges.clear();
    _lastDualObjective = -DRWN_DBL_MAX;

    // standard GEMPLP
    for (int i = 0; i < _graph.numFactors(); i++) {
        const drwnClique c_i = _graph.getClique(i);
        for (int j = i + 1; j < _graph.numFactors(); j++) {
            const drwnClique c_j = _graph.getClique(j);

            const int k = findSeparatorIndex(c_i, c_j);
            if (k < 0) continue;

            if (find(_edges.begin(), _edges.end(), make_pair(i, k)) == _edges.end()) {
                _cliqueEdges[i].insert(_edges.size());
                _separatorEdges[k].insert(_edges.size());
                _edges.push_back(make_pair(i, k));
            }

            if (find(_edges.begin(), _edges.end(), make_pair(j, k)) == _edges.end()) {
                _cliqueEdges[j].insert(_edges.size());
                _separatorEdges[k].insert(_edges.size());
                _edges.push_back(make_pair(j, k));
            }
        }
    }

    // set up forward and backward messages
    _forwardMessages.reserve(_edges.size());
    _backwardMessages.reserve(_forwardMessages.size());
    for (unsigned i = 0; i < _edges.size(); i++) {
        _forwardMessages.push_back(new drwnTableFactor(_graph.getUniverse()));
        _forwardMessages.back()->addVariables(_separators[_edges[i].second]);
        if (_graph.getFactor(_edges[i].first)->empty()) {
            _forwardMessages.back()->fill(0.0);
        } else {
            // intelligent initialization
            drwnFactorMinimizeOp(_forwardMessages.back(), _graph.getFactor(_edges[i].first)).execute();
            _forwardMessages.back()->scale(1.0 / (double)_cliqueEdges[_edges[i].first].size());
        }
    }

    for (unsigned i = 0; i < _edges.size(); i++) {
        const int cliqueId = _edges[i].first;
        const int separatorId = _edges[i].second;
        _backwardMessages.push_back(new drwnTableFactor(_graph.getUniverse()));
        _backwardMessages.back()->fill(0.0);
        for (set<int>::const_iterator c = _separatorEdges[separatorId].begin();
             c != _separatorEdges[separatorId].end(); ++c) {
            if (_edges[*c].first == cliqueId)
                continue;
            drwnFactorPlusEqualsOp(_backwardMessages.back(), _forwardMessages[*c]).execute();
        }
    }
}

void drwnGEMPLPInference::buildComputationGraph()
{
    DRWN_ASSERT(_computations.empty());

    // intermediate factors for \lambda_{s \to c} and \lambda_{c \to s}
    _intermediateFactors.reserve(2 * _edges.size());
    _sharedStorage.push_back(new drwnTableFactorStorage());
    _sharedStorage.push_back(new drwnTableFactorStorage());

    // since we're implementing coordinate ascent, we need to
    // send all messages orginating from the same node at once
    for (int c = 0; c < _graph.numFactors(); c++) {
        addMessageUpdate(c, _graph.getClique(c), _graph.getFactor(c));
    }
}

void drwnGEMPLPInference::decodeBeliefs(drwnFullAssignment& mapAssignment)
{
    vector<drwnTableFactor *> unary(_graph.numVariables());
    for (int i = 0; i < _graph.numVariables(); i++) {
        unary[i] = new drwnTableFactor(_graph.getUniverse());
        unary[i]->addVariable(i);
    }

    for (int c = 0; c < _graph.numFactors(); c++) {
        if (_graph.getFactor(c)->size() != 1) continue;
        const int v = _graph.getFactor(c)->varId(0);
        drwnFactorPlusEqualsOp(unary[v], _graph.getFactor(c)).execute();
        for (set<int>::const_iterator m = _cliqueEdges[c].begin(); m != _cliqueEdges[c].end(); ++m) {
            drwnFactorPlusEqualsOp(unary[v], _backwardMessages[*m]).execute();
        }
    }

    for (int i = 0; i < _graph.numVariables(); i++) {
        mapAssignment[i] = unary[i]->indexOfMin();
        delete unary[i];
    }
}

int drwnGEMPLPInference::findSeparatorIndex(const drwnClique& cliqueA, const drwnClique& cliqueB)
{
    drwnClique sAB;
    set_intersection(cliqueA.begin(), cliqueA.end(),
        cliqueB.begin(), cliqueB.end(),
        insert_iterator<drwnClique>(sAB, sAB.begin()));

    // return negative index if separator set is empty
    if (sAB.empty()) return -1;

    // find (or add) separator set
    int k = 0;
    for ( ; k < (int)_separators.size(); k++) {
        if (_separators[k] == sAB)
            break;
    }

    if (k == (int)_separators.size()) {
        _separators.push_back(sAB);
        _separatorEdges.push_back(set<int>());
    }

    return k;
}

void drwnGEMPLPInference::addMessageUpdate(int cliqueId, const drwnClique& cliqueVars,
    const drwnTableFactor* psi)
{
    // get universe
    drwnVarUniversePtr pUniverse(_graph.getUniverse());

    // subtract old forward messages from all \lambda_{s \to \hat{c}}
    for (set<int>::const_iterator messageId = _cliqueEdges[cliqueId].begin();
         messageId != _cliqueEdges[cliqueId].end(); ++messageId) {
        const int separatorId = _edges[*messageId].second;
        for (set<int>::const_iterator backwardMessageId = _separatorEdges[separatorId].begin();
             backwardMessageId != _separatorEdges[separatorId].end(); ++backwardMessageId) {
            // skip message from \hat{c} == c
            if (_edges[*backwardMessageId].first == cliqueId)
                continue;
            // \todo *messageId == *backwardMessageId

            DRWN_ASSERT((*messageId < (int)_forwardMessages.size()) &&
                (*backwardMessageId < (int)_backwardMessages.size()));

            _computations.push_back(new drwnFactorMinusEqualsOp(_backwardMessages[*backwardMessageId],
                    _forwardMessages[*messageId]));
        }
    }

    // send all \lambda_{c \to s} messages
    for (set<int>::const_iterator messageId = _cliqueEdges[cliqueId].begin();
         messageId != _cliqueEdges[cliqueId].end(); ++messageId) {
        const int separatorId = _edges[*messageId].second;
        DRWN_ASSERT_MSG(_edges[*messageId].first == cliqueId,
            "[" << *messageId << "] " << _edges[*messageId].first << " == " << cliqueId);

        vector<const drwnTableFactor *> incomingMessages;
        if (psi != NULL) {
            incomingMessages.push_back(psi);
        }
        for (set<int>::const_iterator backwardMessageId = _cliqueEdges[cliqueId].begin();
             backwardMessageId != _cliqueEdges[cliqueId].end(); ++backwardMessageId) {
            if (*backwardMessageId == *messageId) continue;
            incomingMessages.push_back(_backwardMessages[*backwardMessageId]);
        }
        DRWN_ASSERT(!incomingMessages.empty());

        // sum_{hat{s}} \lambda_{\hat{s} \to c} + \theta_c
        _intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
        drwnTableFactor *sumFactor = _intermediateFactors.back();
        sumFactor->addVariables(cliqueVars);
        _computations.push_back(new drwnFactorAdditionOp(sumFactor, incomingMessages));
        // max (sum_{hat{s}} \lambda_{\hat{s} \to c} + \theta_c)
        drwnTableFactor *minFactor = NULL;
        if (cliqueVars.size() > _separators[separatorId].size()) {
            drwnClique residualVars;
            set_difference(cliqueVars.begin(), cliqueVars.end(),
                _separators[separatorId].begin(), _separators[separatorId].end(),
                insert_iterator<drwnClique>(residualVars, residualVars.begin()));
            _intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[1]));
            minFactor = _intermediateFactors.back();
            _computations.push_back(new drwnFactorMinimizeOp(minFactor, sumFactor, residualVars));
        } else {
            minFactor = sumFactor;
        }
        // weighted sum
        double w = 1.0 / (double)_cliqueEdges[cliqueId].size();
        _computations.push_back(new drwnFactorWeightedSumOp(_forwardMessages[*messageId],
                _backwardMessages[*messageId], minFactor, (w - 1.0), w));
    }

    // add new forward messages to all \lambda_{s \to \hat{c}}
    for (set<int>::const_iterator messageId = _cliqueEdges[cliqueId].begin();
         messageId != _cliqueEdges[cliqueId].end(); ++messageId) {
        const int separatorId = _edges[*messageId].second;
        for (set<int>::const_iterator backwardMessageId = _separatorEdges[separatorId].begin();
             backwardMessageId != _separatorEdges[separatorId].end(); ++backwardMessageId) {
            if (*messageId == *backwardMessageId)
                continue;

            DRWN_ASSERT((*messageId < (int)_forwardMessages.size()) &&
                (*backwardMessageId < (int)_backwardMessages.size()));

            _computations.push_back(new drwnFactorPlusEqualsOp(_backwardMessages[*backwardMessageId],
                    _forwardMessages[*messageId]));
        }
    }
}

// drwnSontag08Inference class ----------------------------------------------

unsigned drwnSontag08Inference::WARMSTART_ITERATIONS = 20;
unsigned drwnSontag08Inference::MAX_CLIQUES_TO_ADD = 5;

drwnSontag08Inference::drwnSontag08Inference(const drwnFactorGraph& graph) :
    drwnGEMPLPInference(graph)
{
    // do nothing
}

drwnSontag08Inference::~drwnSontag08Inference()
{
    // do nothing
}

void drwnSontag08Inference::clear()
{
    drwnGEMPLPInference::clear();
    _additionalCliques.clear();
}

double drwnSontag08Inference::inference(drwnFullAssignment& mapAssignment)
{
    // run initial inference
    _maxIterations = MAX_ITERATIONS;
    double energy = drwnGEMPLPInference::inference(mapAssignment);
    double integralityGap = energy - _lastDualObjective;
    if (integralityGap < DRWN_EPSILON) {
        DRWN_LOG_VERBOSE("MAP solution found (integrality gap is " << integralityGap << ")");
        return energy;
    }

    // generate clique candidates
    DRWN_LOG_VERBOSE("generating clique candidates...");
    map<drwnClique, vector<int> > cliqueCandidateSet; // clique-to-factor hyperedges
    findCliqueCandidates(cliqueCandidateSet);

    // iteratively add cliques and re-solve (warm start)
    const drwnVarUniversePtr pUniverse(_graph.getUniverse());
    while (1) {

        // score each clique in cliqueCandidateSet
        multimap<double, drwnClique> scoredCliques;
        drwnTableFactorStorage sumBeliefStorage;
        vector<drwnTableFactor *> beliefs(_graph.numFactors(), (drwnTableFactor *)NULL);
        vector<double> minBeliefs(_graph.numFactors(), DRWN_DBL_MAX);
        for (map<drwnClique, vector<int> >::const_iterator c = cliqueCandidateSet.begin();
             c != cliqueCandidateSet.end(); ++c) {
            double sumMinBeliefs = 0.0;
            drwnTableFactor sumBeliefs(pUniverse, &sumBeliefStorage);
            sumBeliefs.addVariables(c->first);
            sumBeliefs.fill(0.0);
            for (int j = 0; j < (int)c->second.size(); j++) {
                const int e = c->second[j];
                if (beliefs[e] == NULL) {
                    beliefs[e] = new drwnTableFactor(*_graph[e]);
                    for (set<int>::const_iterator backwardMessageId = _cliqueEdges[e].begin();
                         backwardMessageId != _cliqueEdges[e].end(); ++backwardMessageId) {
                        drwnFactorPlusEqualsOp(beliefs[e], _backwardMessages[*backwardMessageId]).execute();
                    }
                    minBeliefs[e] = (*beliefs[e])[beliefs[e]->indexOfMin()];
                }
                sumMinBeliefs += minBeliefs[e];
                drwnFactorPlusEqualsOp(&sumBeliefs, beliefs[e]).execute();
            }

            DRWN_ASSERT(sumBeliefs.size() == c->first.size());
            double score = sumBeliefs[sumBeliefs.indexOfMin()] - sumMinBeliefs;
            if (score == 0.0) continue;
            scoredCliques.insert(pair<double, drwnClique>(-score, c->first));

            DRWN_LOG_DEBUG("score for clique " << toString(c->first)
                << " is " << score << " (" << c->second.size() << " neighbours)");
        }

        for (unsigned i = 0; i < beliefs.size(); i++) {
            if (beliefs[i] != NULL) delete beliefs[i];
        }
        beliefs.clear();

        // add best cliques
        unsigned numCliquesAdded = 0;
        for (multimap<double, drwnClique>::const_iterator it = scoredCliques.begin();
             it != scoredCliques.end(); ++it) {
            if (numCliquesAdded++ >= MAX_CLIQUES_TO_ADD) break;
            DRWN_LOG_VERBOSE(numCliquesAdded << "-th best clique is "
                << toString(it->second) << " (" << - it->first << ")");

            // add clique
            const int cliqueId = (int)_cliqueEdges.size();
            _cliqueEdges.push_back(set<int>());
            map<drwnClique, vector<int> >::iterator c = cliqueCandidateSet.find(it->second);

            for (vector<int>::const_iterator i = c->second.begin(); i != c->second.end(); ++i) {
                const drwnClique c_i = _graph.getClique(*i);
                const int k = findSeparatorIndex(c_i, it->second);
                if (k < 0) continue;

                if (find(_edges.begin(), _edges.end(), make_pair(*i, k)) == _edges.end()) {
                    _cliqueEdges[*i].insert(_edges.size());
                    _separatorEdges[k].insert(_edges.size());
                    _edges.push_back(make_pair(*i, k));
                }

                if (find(_edges.begin(), _edges.end(), make_pair(cliqueId, k)) == _edges.end()) {
                    _cliqueEdges[cliqueId].insert(_edges.size());
                    _separatorEdges[k].insert(_edges.size());
                    _edges.push_back(make_pair(cliqueId, k));
                }
            }

            for (int i = 0; i < (int)_additionalCliques.size(); i++) {
                int k = findSeparatorIndex(_additionalCliques[i], it->second);
                if (k < 0) continue;

                if (find(_edges.begin(), _edges.end(), make_pair(_graph.numFactors() + i, k)) == _edges.end()) {
                    _cliqueEdges[_graph.numFactors() + i].insert(_edges.size());
                    _separatorEdges[k].insert(_edges.size());
                    _edges.push_back(make_pair(_graph.numFactors() + i, k));
                }

                if (find(_edges.begin(), _edges.end(), make_pair(cliqueId, k)) == _edges.end()) {
                    _cliqueEdges[cliqueId].insert(_edges.size());
                    _separatorEdges[k].insert(_edges.size());
                    _edges.push_back(make_pair(cliqueId, k));
                }
            }

            _additionalCliques.push_back(it->second);

            // remove clique from candidate list
            cliqueCandidateSet.erase(c);
        }

        if (numCliquesAdded == 0) {
            DRWN_LOG_ERROR("failed to add additional cliques");

            DRWN_LOG_MESSAGE("running for an additional " << MAX_ITERATIONS << " iterations");
            _maxIterations = MAX_ITERATIONS;
            energy = drwnGEMPLPInference::inference(mapAssignment);
            integralityGap = energy - _lastDualObjective;
            DRWN_LOG_VERBOSE("integrality gap: " << integralityGap);
            if (integralityGap < DRWN_EPSILON) {
                DRWN_LOG_VERBOSE("MAP solution found (integrality gap is " << integralityGap << ")");
            }
            break;
        }

        // reconstruct computation graph
        for (unsigned i = 0; i < _computations.size(); i++) {
            delete _computations[i];
        }
        _computations.clear();

        for (unsigned i = 0; i < _intermediateFactors.size(); i++) {
            delete _intermediateFactors[i];
        }
        _intermediateFactors.clear();

        _forwardMessages.reserve(_edges.size());
        _backwardMessages.reserve(_edges.size());
        for (unsigned i = _forwardMessages.size(); i < _edges.size(); i++) {
            _forwardMessages.push_back(new drwnTableFactor(_graph.getUniverse()));
            _backwardMessages.push_back(new drwnTableFactor(_graph.getUniverse()));
            const int separatorId = _edges[i].second;
            _forwardMessages.back()->addVariables(_separators[separatorId]);
            _forwardMessages.back()->fill(0.0);
            _backwardMessages.back()->addVariables(_separators[separatorId]);
            _backwardMessages.back()->fill(0.0);
        }

        for (unsigned i = 0; i < _edges.size(); i++) {
            const int cliqueId = _edges[i].first;
            const int separatorId = _edges[i].second;
            _backwardMessages[i]->fill(0.0);
            for (set<int>::const_iterator c = _separatorEdges[separatorId].begin();
                 c != _separatorEdges[separatorId].end(); ++c) {
                if (_edges[*c].first == cliqueId)
                    continue;
                drwnFactorPlusEqualsOp(_backwardMessages[i], _forwardMessages[*c]).execute();
            }
        }

        buildComputationGraph();

        // run inference (warm-start)
        _maxIterations = WARMSTART_ITERATIONS;
        energy = drwnGEMPLPInference::inference(mapAssignment);
        integralityGap = energy - _lastDualObjective;
        DRWN_LOG_VERBOSE("integrality gap: " << integralityGap);
        if (integralityGap < DRWN_EPSILON) {
            DRWN_LOG_VERBOSE("MAP solution found (integrality gap is " << integralityGap << ")");
            break;
        }
    }

    return energy;
}

void drwnSontag08Inference::buildComputationGraph()
{
    DRWN_ASSERT(_computations.empty());

    // intermediate factors for \lambda_{s \to c} and \lambda_{c \to s}
    _intermediateFactors.reserve(2 * _edges.size());
    if (_sharedStorage.empty()) {
        _sharedStorage.push_back(new drwnTableFactorStorage());
        _sharedStorage.push_back(new drwnTableFactorStorage());
    }

    // since we're implementing coordinate ascent, we need to
    // send all messages orginating from the same node at once
    for (int c = 0; c < _graph.numFactors(); c++) {
        addMessageUpdate(c, _graph.getClique(c), _graph.getFactor(c));
    }
    for (int c = 0; c < (int)_additionalCliques.size(); c++) {
        addMessageUpdate(c + _graph.numFactors(), _additionalCliques[c]);
    }
}

void drwnSontag08Inference::findCliqueCandidates(map<drwnClique, vector<int> >& cliqueCandidateSet)
{
    const int numFactors = _graph.numFactors();
    const drwnVarUniversePtr pUniverse(_graph.getUniverse());

    for (int e1 = 0; e1 < numFactors; e1++) {
        // skip factors that are not of size 2
        if (_graph[e1]->size() != 2)
            continue;
        // skip factors with singleton dimension
        if ((pUniverse->varCardinality(_graph[e1]->varId(0)) == 1) ||
            (pUniverse->varCardinality(_graph[e1]->varId(1)) == 1)) {
            continue;
        }

        for (int e2 = e1 + 1; e2 < numFactors; e2++) {
            // skip factors that are not of size 2
            if (_graph[e2]->size() != 2)
                continue;
            // skip factors with singleton dimension
            if ((pUniverse->varCardinality(_graph[e2]->varId(0)) == 1) ||
                (pUniverse->varCardinality(_graph[e2]->varId(1)) == 1)) {
                continue;
            }

            drwnClique c;
            c.insert(_graph[e1]->varId(0));
            c.insert(_graph[e1]->varId(1));
            c.insert(_graph[e2]->varId(0));
            c.insert(_graph[e2]->varId(1));
            if (c.size() != 3) continue;
            if (cliqueCandidateSet.find(c) != cliqueCandidateSet.end())
                continue;

            vector<int> edgeIndices;
            edgeIndices.push_back(e1);
            edgeIndices.push_back(e2);
            for (int e = e2 + 1; e < numFactors; e++) {
                if (_graph[e]->size() != 2)
                    continue;
                if (c.find(_graph[e]->varId(0)) == c.end())
                    continue;
                if (c.find(_graph[e]->varId(1)) == c.end())
                    continue;

                edgeIndices.push_back(e);
            }

            cliqueCandidateSet.insert(make_pair(c, edgeIndices));
        }
    }
    DRWN_LOG_VERBOSE("..." << cliqueCandidateSet.size() << " candidates generated");
}

// drwnDualDecompositionInference class -------------------------------------

double drwnDualDecompositionInference::INITIAL_ALPHA = 0.5;
bool drwnDualDecompositionInference::USE_MIN_MARGINALS = false;

drwnDualDecompositionInference::drwnDualDecompositionInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnDualDecompositionInference::~drwnDualDecompositionInference()
{
    // do nothing
}

double drwnDualDecompositionInference::inference(drwnFullAssignment& mapAssignment)
{
    DRWN_LOG_VERBOSE("..." << _graph.numVariables() << " variables; "
        << _graph.numFactors() << " factors; " << _graph.numEdges() << " edges");

#if 1
    // initial assignment (using ICM)
    double bestEnergy = drwnICMInference(_graph).inference(mapAssignment);
#else
    // initial assignment
    mapAssignment.resize(_graph.numVariables(), 0);
    double bestEnergy = _graph.getEnergy(mapAssignment);
#endif
    double bestDualEnergy = -DRWN_DBL_MAX;

    // working graph for updating unary potentials
    drwnFactorGraph reparameterizedGraph(_graph);

    const drwnVarUniversePtr pUniverse(_graph.getUniverse());
    drwnTableFactorStorage storage;

    // variable to factor mappings
    vector<set<int> > varSlaveMapping;
    varSlaveMapping.resize(_graph.numVariables());
    for (int i = 0; i < _graph.numFactors(); i++) {
        drwnClique c = _graph.getClique(i);
        for (drwnClique::const_iterator q = c.begin(); q != c.end(); ++q) {
            varSlaveMapping[*q].insert(i);
        }
    }

    // assignments
    vector<drwnFullAssignment> slaveAssignments(_graph.numFactors(),
        drwnFullAssignment(_graph.numVariables(), -1));
    vector<vector<drwnTableFactor> > minMarginals;
    if (USE_MIN_MARGINALS) {
        minMarginals.resize(_graph.numFactors(),
            vector<drwnTableFactor>(_graph.numVariables(), drwnTableFactor(pUniverse)));
    }

    // gradients
    vector<vector<double> > delta(_graph.numVariables(), vector<double>());
    for (int q = 0; q < _graph.numVariables(); q++) {
        delta[q].resize(pUniverse->varCardinality(q));
    }
    vector<bool> bVarConverged(_graph.numVariables());

    // repeat until convergence
    bool bConverged = false;
    unsigned nIterations = 0;
    double dualityGap = 1.0;
    while (!bConverged && (nIterations < drwnMessagePassingMAPInference::MAX_ITERATIONS)) {

        // optimize each slave
        double dualEnergy = 0.0;
        for (int i = 0; i < _graph.numFactors(); i++) {
            const drwnTableFactor *phi = reparameterizedGraph[i];
            const int indx = phi->indexOfMin();
            dualEnergy += (*phi)[indx];
            phi->assignmentOf(indx, slaveAssignments[i]);
        }
        if (dualEnergy > bestDualEnergy) {
            bestDualEnergy = dualEnergy;
            dualityGap = bestEnergy - bestDualEnergy;
        }
        if (dualityGap <= DRWN_EPSILON)
            break;

        // update prices (and check for convergence)
        bConverged = true;
        for (int q = 0; q < _graph.numVariables(); q++) {
            fill(delta[q].begin(), delta[q].end(), 0.0);
            for (set<int>::const_iterator it = varSlaveMapping[q].begin();
                 it != varSlaveMapping[q].end(); ++it) {
                if (USE_MIN_MARGINALS) {
                    // compute min-marginals for subgradient
                    const drwnTableFactor *phi = reparameterizedGraph[*it];
                    drwnClique clq(phi->getClique());
                    clq.erase(q);
                    drwnFactorMinimizeOp(&minMarginals[*it][q], phi, clq).execute();
                    minMarginals[*it][q].scale(-1.0);
                    drwnFactorExpAndNormalizeOp(&minMarginals[*it][q]).execute();
                    DRWN_ASSERT(minMarginals[*it][q].entries() == delta[q].size());

                    //minMarginals[*it][q].fill(0.0);
                    //minMarginals[*it][q][slaveAssignments[*it][q]] = 1.0;

                    for (int v = 0; v < (int)delta[q].size(); v++) {
                        delta[q][v] += minMarginals[*it][q][v];
                    }
                } else {
                    // use argmin for subgradient
                    delta[q][slaveAssignments[*it][q]] += 1.0;
                }
            }

            bVarConverged[q] = true;
            for (int v = 0; v < pUniverse->varCardinality(q); v++) {
                bVarConverged[q] = bVarConverged[q] &&
                    ((delta[q][v] == 0.0) || (delta[q][v] == (double)varSlaveMapping[q].size()));
                delta[q][v] /= (double)varSlaveMapping[q].size();
            }

            if (bVarConverged[q]) {
                DRWN_LOG_DEBUG("...all slaves agree on variable " << q);
            } else {
                bConverged = false;
            }
        }

        // update alpha as dualityGap / dg^2
        //! \todo change for min-marginal subgradient
        double dg2 = 0.0;
        for (int q = 0; q < _graph.numVariables(); q++) {
            dg2 += (double)varSlaveMapping[q].size();
        }

        double alpha = INITIAL_ALPHA * dualityGap / dg2;
        //double alpha = INITIAL_ALPHA / sqrt((double)nIterations + 1.0);

        // some assignments may be impossible
        if (!isfinite(dualityGap)) {
            alpha = INITIAL_ALPHA / sqrt((double)nIterations + 1.0);
        }

        // take gradient step
        drwnFullAssignment assignment(_graph.numVariables(), -1);
        for (int q = 0; q < _graph.numVariables(); q++) {
            assignment[q] = drwn::argmax<double>(delta[q]); // most voted for assignment
            if (bVarConverged[q]) continue;

            // variable has not converged, so update prices
            drwnTableFactor dU(pUniverse, &storage);
            dU.addVariable(q);
            Eigen::Map<VectorXd>(&dU[0], dU.entries()) = -alpha *
                Eigen::Map<VectorXd>(&delta[q][0], delta[q].size());

            if (USE_MIN_MARGINALS) {
                for (set<int>::const_iterator it = varSlaveMapping[q].begin();
                     it != varSlaveMapping[q].end(); ++it) {
                    drwnFactorPlusEqualsOp(reparameterizedGraph[*it], &dU).execute();
                    drwnFactorPlusEqualsOp(reparameterizedGraph[*it],
                        &minMarginals[*it][q], alpha).execute();
                }
            } else {
                for (set<int>::const_iterator it = varSlaveMapping[q].begin();
                     it != varSlaveMapping[q].end(); ++it) {
                    dU[slaveAssignments[*it][q]] += alpha;
                    drwnFactorPlusEqualsOp(reparameterizedGraph[*it], &dU).execute();
                    dU[slaveAssignments[*it][q]] -= alpha;
                }
            }
        }

        // next iteration
        double primalEnergy = _graph.getEnergy(assignment);
        if (primalEnergy < bestEnergy) {
            bestEnergy = primalEnergy;
            mapAssignment = assignment;
        }
        dualityGap = bestEnergy - bestDualEnergy;
        DRWN_LOG_VERBOSE("...iteration " << nIterations << " dual: " << dualEnergy
            << "; integrality gap: " << dualityGap);
        nIterations += 1;
    }

    if (nIterations >= drwnMessagePassingMAPInference::MAX_ITERATIONS) {
        DRWN_LOG_WARNING("dual decomposition failed to converge after " << nIterations << " iterations");
    } else {
        DRWN_LOG_VERBOSE("dual decomposition converged after " << nIterations << " iterations");
    }

    return bestEnergy;
}

// drwnMAPInferenceConfig ---------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnMAPInference
//! \b maxIterations :: maximum number message passing iterations (default: 1000)\n
//! \b damping :: damping factor for message updates (default: 0.0)\n
//! \b warmStartIterations :: maximum number of iterations after adding constraints, Sontag et al., UAI 2008 (default: 20)\n
//! \b initialAlpha :: initial step-size for dual-decomposition (default: 0.5)

class drwnMAPInferenceConfig : public drwnConfigurableModule {
public:
    drwnMAPInferenceConfig() : drwnConfigurableModule("drwnMAPInference") { }
    ~drwnMAPInferenceConfig() { }

    void usage(ostream &os) const {
        os << "      maxIterations :: maximum number of message passing iterations (default: "
           << drwnMessagePassingMAPInference::MAX_ITERATIONS << ")\n";
        os << "            damping :: damping factor for message updates (default: "
           << drwnMessagePassingMAPInference::DAMPING_FACTOR << ")\n";
        os << "warmStartIterations :: maximum number of iterations after adding constraints (default: "
           << drwnSontag08Inference::WARMSTART_ITERATIONS << ")\n";
        os << "       initialAlpha :: initial step-size for dual-decomposition (default: "
           << drwnDualDecompositionInference::INITIAL_ALPHA << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxIterations")) {
            drwnMessagePassingMAPInference::MAX_ITERATIONS = std::max(0, atoi(value));
        } else if (!strcmp(name, "damping")) {
            drwnMessagePassingMAPInference::DAMPING_FACTOR = std::min(std::max(0.0, atof(value)), 1.0);
        } else if (!strcmp(name, "warmStartIterations")) {
            drwnSontag08Inference::WARMSTART_ITERATIONS = std::max(0, atoi(value));
        } else if (!strcmp(name, "initialAlpha")) {
            drwnDualDecompositionInference::INITIAL_ALPHA = std::max(0.0, atof(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnMAPInferenceConfig gMAPInferenceConfig;
