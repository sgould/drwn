/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFactorGraph.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <list>
#include <algorithm>
#include <iterator>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnPGM.h"

using namespace std;
using namespace Eigen;

// drwnFactorGraph ---------------------------------------------------------

drwnFactorGraph::drwnFactorGraph() : _pUniverse(NULL)
{
    // do nothing
}

drwnFactorGraph::drwnFactorGraph(const drwnVarUniversePtr& u) : _pUniverse(u)
{
    // do nothing
}

drwnFactorGraph::drwnFactorGraph(const drwnFactorGraph& g) :
    _pUniverse(g._pUniverse), _edges(g._edges)
{
    _factors.resize(g._factors.size(), NULL);
    for (unsigned i = 0; i < g._factors.size(); i++) {
        if (g._factors[i] != NULL) {
            _factors[i] = g._factors[i]->clone();
        }
    }
}

drwnFactorGraph::~drwnFactorGraph()
{
    for (vector<drwnTableFactor *>::const_iterator it = _factors.begin(); it != _factors.end(); ++it) {
        delete *it;
    }
    _factors.clear();
}

void drwnFactorGraph::addFactor(drwnTableFactor *psi)
{
    DRWN_ASSERT(psi != NULL);

    // check that universe is set and that they match
    if (_pUniverse == NULL) _pUniverse = psi->getUniverse();
    DRWN_ASSERT(_pUniverse == psi->getUniverse());

    // add the factor to the graph
    _factors.push_back(psi);
}

void drwnFactorGraph::copyFactor(const drwnTableFactor *psi)
{
    DRWN_ASSERT(psi != NULL);
    addFactor(psi->clone());
}

void drwnFactorGraph::deleteFactor(int indx)
{
    DRWN_ASSERT((indx >= 0) && (indx < (int)_factors.size()));
    delete _factors[indx];

    // preserve order of remaining factors (otherwise just swap back() and indx)
    for (unsigned i = indx; i < _factors.size() - 1; i++) {
        _factors[i] = _factors[i + 1];
    }
    _factors.pop_back();

    // udpate edges
    vector<_drwnFactorEdge> newEdges;
    newEdges.reserve(_edges.size());
    for (vector<_drwnFactorEdge>::const_iterator e = _edges.begin(); e != _edges.end(); e++) {
        if ((e->first == indx) || (e->second == indx))
            continue;

        newEdges.push_back(_drwnFactorEdge(e->first > indx ? e->first - 1 : e->first,
                e->second > indx ? e->second - 1 : e->second, e->third));
    }
    std::swap(_edges, newEdges);
}

int drwnFactorGraph::findFactor(const drwnClique& clique, bool bAllowSuperset) const
{
    if (!bAllowSuperset) {
        for (int indx = 0; indx < (int)_factors.size(); indx++) {
            if (_factors[indx]->getClique() == clique)
                return indx;
        }
    } else {
        for (int indx = 0; indx < (int)_factors.size(); indx++) {
            const drwnClique c(_factors[indx]->getClique());
            if (c.size() < clique.size())
                continue;

            drwnClique s;
            set_intersection(clique.begin(), clique.end(), c.begin(), c.end(),
                insert_iterator<drwnClique>(s, s.begin()));
            if (s.size() == clique.size())
                return indx;
        }
    }

    return -1;
}

drwnClique drwnFactorGraph::getClique(int indx) const
{
    return _factors[indx]->getClique();
}

drwnClique drwnFactorGraph::getSepSet(const drwnFactor& psiA, const drwnFactor& psiB)
{
    drwnClique sepset;
    drwnClique cA = psiA.getClique();
    drwnClique cB = psiB.getClique();
    set_intersection(cA.begin(), cA.end(), cB.begin(), cB.end(),
        insert_iterator<drwnClique>(sepset, sepset.begin()));

    return sepset;
}

// energy
double drwnFactorGraph::getEnergy(const drwnFullAssignment& x) const
{
    double e = 0.0;

    for (vector<drwnTableFactor *>::const_iterator it = _factors.begin(); it != _factors.end(); ++it) {
        e += (*it)->value(x);
    }

    return e;
}

// graph connectivity
bool drwnFactorGraph::addEdge(const drwnEdge& e)
{
    DRWN_ASSERT(((unsigned)e.first < _factors.size()) && ((unsigned)e.second < _factors.size()));
    DRWN_ASSERT(e.first != e.second);

    // check if edge already exists
    vector<_drwnFactorEdge>::const_iterator it = _edges.begin();
    while (it != _edges.end()) {
        if (((it->first == e.first) && (it->second == e.second)) ||
            ((it->second == e.first) && (it->first == e.second))) {
            break;
        }
        ++it;
    }

    if (it != _edges.end()) {
        DRWN_LOG_WARNING("edge " << toString(e) << " already exists in the graph");
        return false;
    }

    // add the edge
    _edges.push_back(_drwnFactorEdge(e.first, e.second, drwnClique()));
    set_intersection(_factors[e.first]->getOrderedVars().begin(), _factors[e.first]->getOrderedVars().end(),
        _factors[e.second]->getOrderedVars().begin(), _factors[e.second]->getOrderedVars().end(),
        insert_iterator<drwnClique>(_edges.back().third, _edges.back().third.begin()));

    return true;
}

bool drwnFactorGraph::connectGraph(const set<drwnEdge>& edges)
{
    _edges.clear();
    _edges.reserve(edges.size());
    for (set<drwnEdge>::const_iterator it = edges.begin(); it != edges.end(); it++) {
        _edges.push_back(_drwnFactorEdge(it->first, it->second, drwnClique()));

        set_intersection(_factors[it->first]->getOrderedVars().begin(), _factors[it->first]->getOrderedVars().end(),
            _factors[it->second]->getOrderedVars().begin(), _factors[it->second]->getOrderedVars().end(),
            insert_iterator<drwnClique>(_edges.back().third, _edges.back().third.begin()));
    }

    return true;
}

bool drwnFactorGraph::connectGraph()
{
    DRWN_FCN_TIC;

    // clear existing edges
    _edges.clear();

    // find factors containing each variable
    vector<vector<int> > factorIndex(numVariables());
    for (unsigned i = 0; i < _factors.size(); i++) {
        for (unsigned j = 0; j < _factors[i]->size(); j++) {
            factorIndex[_factors[i]->varId(j)].push_back(i);
        }
    }

#if 0
    // debugging
    for (int i = 0; i < numVariables(); i++) {
        DRWN_LOG_DEBUG("variable " << i << " is in factors " << toString(factorIndex[i]));
    }
#endif

    // find max-spanning-tree for each variable
    vector<drwnEdge> candidateEdges;
    vector<double> weights;
    for (int n = 0; n < numVariables(); n++) {
        // check for missing variables
        if (factorIndex[n].empty()) continue;

	// create weighted graph
	candidateEdges.clear();
	weights.clear();
	for (unsigned i = 0; i < factorIndex[n].size() - 1; i++) {
	    for (unsigned j = i + 1; j < factorIndex[n].size(); j++) {
		candidateEdges.push_back(drwnEdge(factorIndex[n][i], factorIndex[n][j]));
		drwnClique s;
		set_intersection(_factors[factorIndex[n][i]]->getOrderedVars().begin(),
		    _factors[factorIndex[n][i]]->getOrderedVars().end(),
		    _factors[factorIndex[n][j]]->getOrderedVars().begin(),
		    _factors[factorIndex[n][j]]->getOrderedVars().end(),
		    insert_iterator<drwnClique>(s, s.begin()));
		weights.push_back(-(double)s.size());
	    }
	}

        // check for unary variables
        if (candidateEdges.empty()) continue;

	// find spanning tree
	vector<drwnEdge> spanningTree = minSpanningTree(numFactors(),
            candidateEdges, weights);

	// add edges and separators to cluster graph
	for (vector<drwnEdge>::const_iterator it = spanningTree.begin();
	     it != spanningTree.end(); ++it) {
            DRWN_ASSERT(it->first != it->second);
	    int existingIndx = 0;
	    while (existingIndx < (int)_edges.size()) {
		if ((_edges[existingIndx].first == it->first) &&
		    (_edges[existingIndx].second == it->second)) {
		    break;
		}
		existingIndx++;
	    }

	    if (existingIndx < (int)_edges.size()) {
		_edges[existingIndx].third.insert(n);
	    } else {
		_edges.push_back(_drwnFactorEdge(it->first, it->second, drwnClique()));
		_edges.back().third.insert(n);
	    }
	}
    }

    DRWN_FCN_TOC;
    return true;
}

// Connect graph using the bethe-approximation to the energy
// functional. All messages pass through marginals.
bool drwnFactorGraph::connectBetheApprox()
{
    _edges.clear();

    // find singleton connecting nodes
    vector<int> singletonNodes(numVariables(), -1);
    for (int i = 0; i < (int)_factors.size(); i++) {
	if (_factors[i]->size() != 1)
	    continue;
	if (singletonNodes[_factors[i]->varId(0)] == -1) {
	    singletonNodes[_factors[i]->varId(0)] = i;
	}
    }

    // require singleton nodes
    for (int i = 0; i < numVariables(); i++) {
	if (singletonNodes[i] == -1) {
	    DRWN_LOG_FATAL("missing singleton node for variable " << i);
	    return false;
	}
    }

    // connect clique to singleton nodes
    for (int i = 0; i < (int)_factors.size(); i++) {
	if ((_factors[i]->size() == 1) &&
            (singletonNodes[_factors[i]->varId(0)] == i))
	    continue;

        for (unsigned j = 0; j < _factors[i]->size(); j++) {
	    DRWN_ASSERT(singletonNodes[_factors[i]->varId(j)] != -1);
	    _edges.push_back(_drwnFactorEdge(i, singletonNodes[_factors[i]->varId(j)],
                    drwnClique()));
            _edges.back().third.insert(_factors[i]->varId(j));
	}
    }

    return true;
}

// i/o
bool drwnFactorGraph::save(drwnXMLNode& xml) const
{
    // write universe
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnVarUniverse", NULL, false);
    _pUniverse->save(*node);

    // write factors
    for (unsigned i = 0; i < _factors.size(); i++) {
        node = drwnAddXMLChildNode(xml, "factor", NULL, false);
        drwnAddXMLAttribute(*node, "type", _factors[i]->type(), false);
        _factors[i]->save(*node);
    }

    // write edges
    std::stringstream s;
    for (unsigned e = 0; e < _edges.size(); e++) {
        if (e != 0) s << "\n";
        s << _edges[e].first << " " << _edges[e].second;
    }
    drwnAddXMLChildNode(xml, "edges", s.str().c_str(), false);

    return true;
}

bool drwnFactorGraph::load(drwnXMLNode& xml)
{
    // clear existing factors and edges
    for (vector<drwnTableFactor *>::const_iterator it = _factors.begin(); it != _factors.end(); ++it) {
        delete *it;
    }
    _factors.clear();
    _edges.clear();

    // load universe
    drwnXMLNode *node = xml.first_node("drwnVarUniverse");
    if (_pUniverse == NULL) {
        DRWN_LOG_DEBUG("creating new variable universe...");
        //_pUniverse = drwnVarUniversePtr(new drwnVarUniverse());
        drwnVarUniversePtr tmp(new drwnVarUniverse());
        _pUniverse = tmp;
    }
    DRWN_LOG_DEBUG("loading variable universe...");
    _pUniverse->load(*node);

    // load factors
    const int numFactors = drwnCountXMLChildren(xml, "factor");
    _factors.reserve(numFactors);
    DRWN_LOG_DEBUG("loading " << numFactors << " factors...");
    for (node = xml.first_node("factor"); node != NULL; node = node->next_sibling("factor")) {
        // \todo check factor type (use factor factory)
        _factors.push_back(new drwnTableFactor(_pUniverse));
        _factors.back()->load(*node);
    }

    // load edges
    node = xml.first_node("edges");
    if (node != NULL) {
        vector<int> edgeList;
        drwn::parseString<int>(string(drwnGetXMLText(*node)), edgeList);
        DRWN_ASSERT(edgeList.size() % 2 == 0);
        _edges.reserve(edgeList.size() / 2);
        for (unsigned i = 0; i < edgeList.size(); i += 2) {
            _edges.push_back(_drwnFactorEdge(edgeList[i], edgeList[i + 1], drwnClique()));
        }

        // compute sep-sets
        computeSeparatorSets();
    }

    return true;
}

void drwnFactorGraph::computeSeparatorSets()
{
    for (unsigned i = 0; i < _edges.size(); i++) {
        drwnClique a(_factors[_edges[i].first]->getClique());
        drwnClique b(_factors[_edges[i].second]->getClique());
	set_intersection(a.begin(), a.end(), b.begin(), b.end(),
	    insert_iterator<drwnClique>(_edges[i].third, _edges[i].third.begin()));
    }
}

// drwnFactorGraph utilities -----------------------------------------------

void drwnFactorGraphUtils::writeDottyOutput(const char *filename, const drwnFactorGraph &graph)
{
    DRWN_ASSERT(filename != NULL);
    ofstream ofs(filename);
    DRWN_ASSERT(!ofs.fail());

    ofs << "graph factorGraph {\n";

    for (int i = 0; i < graph.numFactors(); i++) {
        ofs << "  \"" << toString(graph[i]->getClique()) << "\"\n";
    }

    for (int i = 0; i < graph.numEdges(); i++) {
        drwnEdge e = graph.getEdge(i);
        ofs << "  \"" << toString(graph[e.first]->getClique()) << "\" -- \""
            << toString(graph[e.second]->getClique()) << "\";\n";
    }

    ofs << "};\n";
}

vector<set<int> > drwnFactorGraphUtils::variableAdjacencyList(const drwnFactorGraph& graph)
{
    vector<set<int> > adjList(graph.numVariables());

    for (int i = 0; i < graph.numFactors(); i++) {
        drwnClique ci = graph[i]->getClique();
        for (drwnClique::const_iterator it = ci.begin(); it != ci.end(); it++) {
            for (drwnClique::const_iterator jt = ci.begin(); jt != it; jt++) {
                adjList[*it].insert(*jt);
                adjList[*jt].insert(*it);
            }
        }
    }

    return adjList;
}

drwnFactorGraph drwnFactorGraphUtils::createJunctionTree(const drwnFactorGraph& graph)
{
    // find edges between variables
    set<drwnEdge> edgeSet;
    for (int i = 0; i < graph.numFactors(); i++) {
        drwnClique ci = graph[i]->getClique();
        for (drwnClique::const_iterator it = ci.begin(); it != ci.end(); it++) {
            for (drwnClique::const_iterator jt = ci.begin(); jt != ci.end(); jt++) {
                if (*it < *jt) {
                    edgeSet.insert(drwnEdge(*it, *jt));
                }
            }
        }
    }

    vector<drwnEdge> edges(edgeSet.begin(), edgeSet.end());

    // triangulate graph
    triangulateGraph(graph.numVariables(), edges, DRWN_MINFILLIN);

    // find maximal cliques
    vector<int> nodeOrder;
    bool success = maxCardinalitySearch(graph.numVariables(), edges, nodeOrder);
    if (!success) {
        for (vector<drwnEdge>::const_iterator it = edges.begin(); it != edges.end(); it++) {
            DRWN_LOG_ERROR(it->first << " -- " << it->second);
        }
        DRWN_LOG_FATAL("triangulation failed with " << toString(nodeOrder));
    }
    vector<drwnClique> maxCliques = variableEliminationCliques(edges, nodeOrder);

#if 1
    int treeWidth = 1;
    double maxFactorEntries = 0.0;
    for (vector<drwnClique>::const_iterator c = maxCliques.begin(); c != maxCliques.end(); c++) {
        treeWidth = std::max(treeWidth, (int)c->size() - 1);
        maxFactorEntries = std::max(maxFactorEntries, graph.getUniverse()->logStateSpace(*c));

    }
    DRWN_LOG_VERBOSE("junction tree has tree-width of " << treeWidth);
    DRWN_LOG_VERBOSE("maximum factor entries is " << exp(maxFactorEntries));
#endif

    // create new cluster graph (junction tree)
    drwnFactorGraph jt(graph.getUniverse());

    // add high-order cliques
    for (vector<drwnClique>::const_iterator c = maxCliques.begin(); c != maxCliques.end(); c++) {
        DRWN_LOG_DEBUG("...adding clique " << toString(*c));
        drwnTableFactor *psi = new drwnTableFactor(graph.getUniverse());
        psi->addVariables(*c);
        jt.addFactor(psi);
    }

    for (int i = 0; i < graph.numFactors(); i++) {
        drwnClique c = graph.getClique(i);
        int indx = findSuperset(maxCliques, c);
        DRWN_ASSERT_MSG(indx != -1, toString(c));
        drwnFactorPlusEqualsOp(jt[indx], graph[i]).execute();
    }

    // add edges
    jt.connectGraph();

    return jt;
}

double drwnFactorGraphUtils::removeUniformFactors(drwnFactorGraph& graph)
{
    //! \todo reduce factors where variable cardinality is one

    double e = 0.0;

    // remove uniform factors
    for (int i = graph.numFactors() - 1; i >= 0; i--) {
        if (graph[i]->size() == 1) continue; // don't remove singleton factors
        const drwnTableFactor *f = graph[i];
        int indx = f->indexOfMin();
        if (f->indexOfMax() == indx) {
            e += (*f)[indx];
            DRWN_LOG_DEBUG("deleteing factor over " << toString(f->getClique()));
            graph.deleteFactor(i);
        }
    }

#if 1
    // check for disconnected unary factors
    vector<bool> connected(graph.numVariables(), false);
    for (int i = 0; i < graph.numFactors(); i++) {
        const drwnClique c = graph.getClique(i);
        if (c.size() == 1) continue;
        for (drwnClique::const_iterator it = c.begin(); it != c.end(); ++it) {
            connected[*it] = true;
        }
    }

    for (int i = 0; i < graph.numVariables(); i++) {
        if (!connected[i]) {
            DRWN_LOG_WARNING("variable " << i << " is disconnected");
        }
    }
#endif

    return e;
}

void drwnFactorGraphUtils::absorbSmallFactors(drwnFactorGraph& graph, bool bIncludeUnary)
{
    // disconnect graph
    graph.connectGraph(set<drwnEdge>());

    // look for factors to remove
    for (int i = graph.numFactors() - 1; i >= 0; i--) {
        const drwnClique clique(graph.getClique(i));
        if (!bIncludeUnary && (clique.size() == 1))
            continue;

        list<int> factors;
        for (int j = 0; j < graph.numFactors(); j++) {
            const drwnClique c(graph.getClique(j));
            if ((i == j) || (c.size() < clique.size()))
                continue;

            drwnClique s;
            set_intersection(clique.begin(), clique.end(), c.begin(), c.end(),
                insert_iterator<drwnClique>(s, s.begin()));
            if (s.size() == clique.size()) {
                factors.push_back(j);
            }
        }

        if (factors.empty()) continue;

        // distribute factor across higher-order factors
        graph[i]->scale(1.0 / (double)factors.size());
        for (list<int>::const_iterator it = factors.begin(); it != factors.end(); it++) {
            DRWN_LOG_DEBUG("absorbing factor over " << toString(clique)
                << " into factor over " << toString(graph[*it]->getClique()));
            drwnFactorPlusEqualsOp(graph[*it], graph[i]).execute();
        }

        // delete factor
        graph.deleteFactor(i);
    }
}

void drwnFactorGraphUtils::mergeDuplicateFactors(drwnFactorGraph& graph)
{
    // find duplicate cliques
    map<drwnClique, list<int> > clique2factors;
    for (int i = 0; i < graph.numFactors(); i++) {
        drwnClique c(graph.getClique(i));
        map<drwnClique, list<int> >::iterator it = clique2factors.find(c);
        if (it == clique2factors.end()) {
            it = clique2factors.insert(it, make_pair(c, list<int>()));
        }
        it->second.push_back(i);
    }

    // merge duplicate cliques
    set<int, greater<int> > pendingDeletion;
    for (map<drwnClique, list<int> >::const_iterator it = clique2factors.begin();
         it != clique2factors.end(); ++it) {
        if (it->second.size() == 1) continue;

        list<int>::const_iterator jt = it->second.begin();
        drwnTableFactor *phi = graph[*jt++];
        while (jt != it->second.end()) {
            drwnFactorPlusEqualsOp(phi, graph[*jt]);
            pendingDeletion.insert(*jt);
            ++jt;
        }
    }

    // delete duplicate cliques (from back of list first)
    for (set<int, greater<int> >::const_iterator it = pendingDeletion.begin();
         it != pendingDeletion.end(); ++it) {
        graph.deleteFactor(*it);
    }
}
