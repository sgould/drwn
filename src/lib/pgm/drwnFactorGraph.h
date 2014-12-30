/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFactorGraph.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <list>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnGraphUtils.h"
#include "drwnTableFactor.h"

using namespace std;
using namespace Eigen;

// drwnFactorGraph ---------------------------------------------------------
//! Container and utility functions for factor graphs.
//!
//! \todo template on a particular factor type (could be all table factors,
//! for example, or generic drwnFactors)

class drwnFactorGraph : public drwnStdObjIface {
 protected:
    drwnVarUniversePtr _pUniverse;  //!< all variables in the universe
    vector<drwnTableFactor *> _factors;  //!< list of factors in the model

    typedef drwnTriplet<int, int, drwnClique> _drwnFactorEdge;
    vector<_drwnFactorEdge> _edges; //!< list of edges and sep-sets

 public:
    //! construct an empty factor graph
    drwnFactorGraph();
    //! construct a factor graph based on variables in universe \p u
    drwnFactorGraph(const drwnVarUniversePtr& u);
    //! copy constructor
    drwnFactorGraph(const drwnFactorGraph& g);
    virtual ~drwnFactorGraph();

    // access functions
    virtual const char *type() const { return "drwnFactorGraph"; }
    virtual drwnFactorGraph *clone() const { return new drwnFactorGraph(*this); }

    //! return the universe of variables for this factor graph
    const drwnVarUniversePtr& getUniverse() const { return _pUniverse; }
    //! returns the number of variables in the universe
    int numVariables() const { return (_pUniverse == NULL) ? 0 : _pUniverse->numVariables(); }
    //! returns the number of factors in the graph
    int numFactors() const { return (int)_factors.size(); }
    //! returns the number of (hyper-)edges in the graph
    int numEdges() const { return (int)_edges.size(); }

    //! add a factor to the graph and takes ownership
    void addFactor(drwnTableFactor *psi);
    //! add a factor to the graph by copying
    void copyFactor(const drwnTableFactor *psi);
    //! returns the factor at index \b indx
    const drwnTableFactor* getFactor(int indx) const { return _factors[indx]; }
    //! delete a factor from the graph
    void deleteFactor(int indx);
    //! returns the index of a factor defined over the clique (or a superset of it)
    int findFactor(const drwnClique& clique, bool bAllowSuperset = false) const;
    //! returns the clique for the factor at index \b indx
    drwnClique getClique(int indx) const;
    //! returns the indices for the factors adjacent to edge \b eindx
    drwnEdge getEdge(int eindx) const { return drwnEdge(_edges[eindx].first, _edges[eindx].second); }
    //! returns the set of separator variables for edge \b eindx
    drwnClique getSepSet(int eindx) const { return _edges[eindx].third; }
    //! returns the set of separator variables between factors \b psiA and \b psiB
    static drwnClique getSepSet(const drwnFactor& psiA, const drwnFactor& psiB);

    //! compute energy for a given assignment
    double getEnergy(const drwnFullAssignment& x) const;

    // graph connectivity
    //! add an edge to the graph (and compute separator set)
    bool addEdge(const drwnEdge& e);
    //! connect graph with given set of edges
    bool connectGraph(const set<drwnEdge>& edges);
    //! connect graph using max-spanning tree for each variable
    virtual bool connectGraph();
    //! connect graph using the bethe-approximation
    bool connectBetheApprox();

    // check running intersection property
    /*
    bool checkRunIntProp();
    */

    // i/o
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    // operators
    drwnTableFactor* operator[](unsigned indx) { return _factors[indx]; }
    const drwnTableFactor* operator[](unsigned indx) const { return _factors[indx]; }

    //bool operator==(const drwnFactorGraph& g) const;


 protected:
    //! computes the sep-sets for each edge in _edges
    void computeSeparatorSets();
};

// drwnFactorGraph utilities -----------------------------------------------
//! Utility functions for factor graphs.

namespace drwnFactorGraphUtils {
    //! Generate output for dotty (Graphviz)
    void writeDottyOutput(const char *filename, const drwnFactorGraph &graph);

    //! Returns the neighbours (i.e., other variables appearing in the same
    //! clique) of each variable
    vector<set<int> > variableAdjacencyList(const drwnFactorGraph& graph);

    //! Create a junction tree from a factor graph
    drwnFactorGraph createJunctionTree(const drwnFactorGraph& graph);

    //! Removes uniform factors from the graph (returns energy of factors removed)
    double removeUniformFactors(drwnFactorGraph& graph);

    //! Absorbs smaller (log-space) factors into larger ones (operates inline). The
    //! returned graph is disconnected. If \p bIncludeUnary is \p false then unary
    //! potentials are left untouched.
    void absorbSmallFactors(drwnFactorGraph& graph, bool bIncludeUnary = true);

    //! Merges (log-space) factors over identical cliques. Operates inline.
    void mergeDuplicateFactors(drwnFactorGraph& graph);
};
