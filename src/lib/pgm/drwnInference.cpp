/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInference.cpp
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

// drwnInference class -----------------------------------------------------

drwnInference::drwnInference(const drwnFactorGraph& graph) : _graph(graph)
{
    // do nothing
}

drwnInference::drwnInference(const drwnInference& inf) :
    _graph(inf._graph)
{
    // do nothing
}

drwnInference::~drwnInference()
{
    // do nothing
}

drwnFactorGraph drwnInference::varMarginals() const
{
    drwnFactorGraph marginals(_graph.getUniverse());
    for (int i = 0; i < _graph.getUniverse()->numVariables(); i++) {
        drwnTableFactor *psi = new drwnTableFactor(_graph.getUniverse());
        psi->addVariable(i);
        marginal(*psi);
        marginals.addFactor(psi);
    }
    return marginals;
}

// drwnMessagePassingInference class ----------------------------------------

unsigned drwnMessagePassingInference::MAX_ITERATIONS = 1000;

drwnMessagePassingInference::drwnMessagePassingInference(const drwnFactorGraph& graph) :
    drwnInference(graph)
{
    // do nothing
}

drwnMessagePassingInference::~drwnMessagePassingInference()
{
    // release temporary memory and computation graph
    drwnMessagePassingInference::clear();
}

void drwnMessagePassingInference::clear()
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

bool drwnMessagePassingInference::inference()
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

        // execute computation graph
	for (int i = 0; i < (int)_computations.size(); i++) {
	    _computations[i]->execute();
	}

        // check for convergence and copy messages
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

    return bConverged;
}

void drwnMessagePassingInference::initializeMessages()
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
	    _forwardMessages.back()->fill(1.0);
            _backwardMessages.push_back(new drwnTableFactor(*_forwardMessages.back()));
	}
    } else {
        // reset messages to all ones
	for (unsigned i = 0; i < _forwardMessages.size(); i++) {
            _forwardMessages[i]->fill(1.0);
            _backwardMessages[i]->fill(1.0);
        }
    }

    // storage for old messages
    // \todo: some algorithms may not use this
    _oldForwardMessages.reserve(_forwardMessages.size());
    _oldBackwardMessages.reserve(_backwardMessages.size());
    for (unsigned i = 0; i < _forwardMessages.size(); i++) {
        _oldForwardMessages.push_back(new drwnTableFactor(*_forwardMessages[i]));
        _oldBackwardMessages.push_back(new drwnTableFactor(*_backwardMessages[i]));
    }
}

// drwnSumProductInference class --------------------------------------------

drwnSumProdInference::drwnSumProdInference(const drwnFactorGraph& graph) :
    drwnMessagePassingInference(graph)
{
    // do nothing
}

drwnSumProdInference::~drwnSumProdInference()
{
    // do nothing
}

void drwnSumProdInference::marginal(drwnTableFactor& belief) const
{
    // initialize belief
    int factorIndx = _graph.findFactor(belief.getClique());
    DRWN_ASSERT(factorIndx >= 0);
    belief = *_graph[factorIndx];

    // compute final beliefs
    for (int m = 0; m < _graph.numEdges(); m++) {
        if (_graph.getEdge(m).first == factorIndx) {
            drwnFactorTimesEqualsOp(&belief, _backwardMessages[m]).execute();
            belief.normalize();
        }
        if (_graph.getEdge(m).second == factorIndx) {
            drwnFactorTimesEqualsOp(&belief, _forwardMessages[m]).execute();
            belief.normalize();
        }
    }

    DRWN_ASSERT(belief.entries() == _graph[factorIndx]->entries());
}

void drwnSumProdInference::buildComputationGraph()
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
	_computations.push_back(new drwnFactorProductOp(_intermediateFactors.back(),
				    incomingFwdMsgs));
	_computations.push_back(new drwnFactorMarginalizeOp(_forwardMessages[m],
				    _intermediateFactors.back()));
	_computations.push_back(new drwnFactorNormalizeOp(_forwardMessages[m]));

	// backwards
	_intermediateFactors.push_back(new drwnTableFactor(pUniverse, _sharedStorage[0]));
	_computations.push_back(new drwnFactorProductOp(_intermediateFactors.back(),
				    incomingBckMsgs));
	_computations.push_back(new drwnFactorMarginalizeOp(_backwardMessages[m],
				    _intermediateFactors.back()));
	_computations.push_back(new drwnFactorNormalizeOp(_backwardMessages[m]));
    }
}

// drwnAsyncSumProductInference class ---------------------------------------

drwnAsyncSumProdInference::drwnAsyncSumProdInference(const drwnFactorGraph& graph) :
    drwnSumProdInference(graph)
{
    // do nothing
}

drwnAsyncSumProdInference::~drwnAsyncSumProdInference()
{
    // do nothing
}

void drwnAsyncSumProdInference::buildComputationGraph()
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
	_computations.push_back(new drwnFactorProductOp(_intermediateFactors.back(),
                incomingFwdMsgs));
	_computations.push_back(new drwnFactorMarginalizeOp(_forwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorNormalizeOp(_forwardMessages[m]));
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
	_computations.push_back(new drwnFactorProductOp(_intermediateFactors.back(),
                incomingBckMsgs));
	_computations.push_back(new drwnFactorMarginalizeOp(_backwardMessages[m],
                _intermediateFactors.back()));
	_computations.push_back(new drwnFactorNormalizeOp(_backwardMessages[m]));
    }
}
