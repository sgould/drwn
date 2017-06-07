/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMaxFlow.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cassert>
#include <vector>
#include <map>
#include <deque>

using namespace std;

class drwnMaxFlow;

// drwnMaxFlow --------------------------------------------------------------
//! Interface for maxflow/min-cut algorithms (for minimizing submodular
//! quadratic pseudo-Boolean functions)
//!
//! Residual capacities are updated in-place. See \ref drwnMaxFlowDoc. Negative
//! edge weights are allowed between terminals (source or target) and added to
//! the constant. Supports dynamic graph-cuts (see Kohli and Torr, PAMI 2007).

class drwnMaxFlow {
 protected:
    //! tree states
    static const unsigned char FREE = 0x00;
    static const unsigned char SOURCE = 0x01;
    static const unsigned char TARGET = 0x02;

    //! references target node j and edge weights (i,j) and (j,i)
    typedef map<int, pair<unsigned, unsigned> > _drwnCapacitatedEdge;

    vector<double> _sourceEdges; //!< edges leaving the source
    vector<double> _targetEdges; //!< edges entering the target
    vector<double> _edgeWeights; //!< internal edge weights
    vector<_drwnCapacitatedEdge> _nodes; //!< nodes and their outgoing internal edges
    double _flowValue;           //!< current flow value (includes constant)

    vector<unsigned char> _cut;  //!< identifies which side of the cut a node falls

 public:
    //! construct a maxflow/mincut problem with estimated maxNodes
    drwnMaxFlow(unsigned maxNodes = 0) : _flowValue(0.0) {
        _sourceEdges.reserve(maxNodes);
        _targetEdges.reserve(maxNodes);
        _edgeWeights.reserve(2 * maxNodes);
        _nodes.reserve(maxNodes);
    }

    //! destructor
    virtual ~drwnMaxFlow() {
        // do nothing
    }

    //! get number of nodes in the graph
    size_t numNodes() const { return _nodes.size(); }

    //! reset all edge capacities to zero (but don't free the graph)
    virtual void reset();
    //! clear the graph and internal datastructures
    virtual void clear();

    //! add nodes to the graph (returns the id of the first node added)
    inline int addNodes(unsigned n = 1) {
        int nodeId = (int)_nodes.size();
        _nodes.resize(_nodes.size() + n);
        _sourceEdges.resize(_nodes.size(), 0.0);
        _targetEdges.resize(_nodes.size(), 0.0);
        return nodeId;
    }

    //! add constant flow to graph
    void addConstant(double c) {
        _flowValue += c;
    }

    //! add edge from s to nodeId
    inline void addSourceEdge(int u, double cap) {
        DRWN_ASSERT((u >= 0) && (u < (int)_nodes.size()));
        if (cap < 0.0) { _flowValue += cap; _targetEdges[u] -= cap; }
        else _sourceEdges[u] += cap;
    }

    //! add edge from nodeId to t
    inline void addTargetEdge(int u, double cap) {
        DRWN_ASSERT((u >= 0) && (u < (int)_nodes.size()));
        if (cap < 0.0) { _flowValue += cap; _sourceEdges[u] -= cap; }
        else _targetEdges[u] += cap;
    }

    //! add edge from u to v and edge from v to u
    //! (requires cap_uv + cap_vu >= 0)
    inline void addEdge(int u, int v, double cap_uv, double cap_vu = 0.0) {
        DRWN_ASSERT((u >= 0) && (u < (int)_nodes.size()));
        DRWN_ASSERT((v >= 0) && (v < (int)_nodes.size()));
        DRWN_ASSERT(u != v);

        _drwnCapacitatedEdge::const_iterator it = _nodes[u].find(v);
        if (it == _nodes[u].end()) {
            DRWN_ASSERT(cap_uv + cap_vu >= 0.0);
            if (cap_uv < 0.0) {
                _nodes[u].insert(make_pair(v, make_pair(_edgeWeights.size(), _edgeWeights.size() + 1)));
                _nodes[v].insert(make_pair(u, make_pair(_edgeWeights.size() + 1, _edgeWeights.size())));
                _edgeWeights.push_back(0.0);
                _edgeWeights.push_back(cap_uv + cap_vu);
                _sourceEdges[u] -= cap_uv;
                _targetEdges[v] -= cap_uv;
                _flowValue += cap_uv;
            } else if (cap_vu < 0.0) {
                _nodes[u].insert(make_pair(v, make_pair(_edgeWeights.size(), _edgeWeights.size() + 1)));
                _nodes[v].insert(make_pair(u, make_pair(_edgeWeights.size() + 1, _edgeWeights.size())));
                _edgeWeights.push_back(cap_uv + cap_vu);
                _edgeWeights.push_back(0.0);
                _sourceEdges[v] -= cap_vu;
                _targetEdges[u] -= cap_vu;
                _flowValue += cap_vu;
            } else {
                _nodes[u].insert(make_pair(v, make_pair(_edgeWeights.size(), _edgeWeights.size() + 1)));
                _nodes[v].insert(make_pair(u, make_pair(_edgeWeights.size() + 1, _edgeWeights.size())));
                _edgeWeights.push_back(cap_uv);
                _edgeWeights.push_back(cap_vu);
            }
        } else {
            const double w_u = _edgeWeights[it->second.first] += cap_uv;
            const double w_v = _edgeWeights[it->second.second] += cap_vu;
            DRWN_ASSERT(w_u + w_v >= 0.0);
            if (w_u < 0.0) {
                _edgeWeights[it->second.first] = 0.0;
                _edgeWeights[it->second.second] += w_u;
                _sourceEdges[u] -= w_u;
                _targetEdges[v] -= w_u;
                _flowValue += w_u;
            } else if (w_v < 0.0) {
                _edgeWeights[it->second.first] += w_v;
                _edgeWeights[it->second.second] = 0.0;
                _sourceEdges[v] -= w_v;
                _targetEdges[u] -= w_v;
                _flowValue += w_v;
            }
        }
    }

    //! solve the max-flow problem and return the flow
    virtual double solve() = 0;

    //! return true if \p u is in the s-set after calling \ref solve.
    //! (note that sometimes a node can be in either S or T)
    bool inSetS(int u) const { return (_cut[u] == SOURCE); }
    //! return true if \p u is in the t-set after calling \ref solve
    //! (note that sometimes a node can be in either S or T)
    bool inSetT(int u) const { return (_cut[u] == TARGET); }

    //! returns the residual capacity for an edge (use -1 for terminal so
    //! that (-1,-1) represents the current flow)
    double operator()(int u, int v) const;

 protected:
    //! pre-augment s-u-t and s-u-v-t paths
    void preAugmentPaths();
};

// drwnEdmondsKarpMaxFlow ---------------------------------------------------
//! Implementation of Edmonds-Karp maxflow/min-cut algorithm

class drwnEdmondsKarpMaxFlow : public drwnMaxFlow {
 public:
    drwnEdmondsKarpMaxFlow(unsigned maxNodes = 0) : drwnMaxFlow(maxNodes) {
        // do nothing
    }
    virtual ~drwnEdmondsKarpMaxFlow() {
        // do nothing
    }

    virtual double solve();

 protected:
    void findCuts();
};

// drwnBKMaxFlow ------------------------------------------------------------
//! Implementation of Boykov and Kolmogorov's maxflow algorithm
//!
//! See Boykov and Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow
//! Algorithms for Energy Minimization in Vision", PAMI 2004 for a description
//! of the algorithm.
//!

class drwnBKMaxFlow : public drwnMaxFlow {
 protected:
    static const int TERMINAL = -1;  //! _parents flag for terminal state

    //! search tree and active list
    vector<int> _parents;
    vector<pair<double *, double *> > _weightptrs;
    drwnIndexQueue _activeList;

 public:
    drwnBKMaxFlow(unsigned maxNodes = 0) : drwnMaxFlow(maxNodes) {
        // do nothing
    }
    virtual ~drwnBKMaxFlow() {
        // do nothing
    }

    virtual void reset();
    virtual void clear();
    virtual double solve();

 protected:
    //! initialize trees from source and target
    void initializeTrees();
    //! expand trees until a path is found (or no path (-1, -1))
    pair<int, int> expandTrees();
    //! augment the path found by expandTrees; return orphaned subtrees
    void augmentBKPath(const pair<int, int>& path, deque<int>& orphans);
    //! adopt orphaned subtrees
    void adoptOrphans(deque<int>& orphans);
    //! return true if u is an ancestor of v
    bool isAncestor(int u, int v) const;
};
