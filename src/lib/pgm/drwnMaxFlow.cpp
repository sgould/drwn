/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMaxFlow.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>
#include <algorithm>

#include "drwnBase.h"
#include "drwnMaxFlow.h"

using namespace std;

// drwnMaxFlow and drwnBKMaxFlow statics ------------------------------------

#ifdef __APPLE__
const unsigned char drwnMaxFlow::FREE;
const unsigned char drwnMaxFlow::SOURCE;
const unsigned char drwnMaxFlow::TARGET;
const int drwnBKMaxFlow::TERMINAL;
#endif

// drwnMaxFlow --------------------------------------------------------------

void drwnMaxFlow::reset()
{
    _flowValue = 0.0;
    fill(_sourceEdges.begin(), _sourceEdges.end(), 0.0);
    fill(_targetEdges.begin(), _targetEdges.end(), 0.0);
    fill(_edgeWeights.begin(), _edgeWeights.end(), 0.0);
    fill(_cut.begin(), _cut.end(), FREE);
}

void drwnMaxFlow::clear()
{
    _flowValue = 0.0;
    _sourceEdges.clear();
    _targetEdges.clear();
    _edgeWeights.clear();
    _nodes.clear();
    _cut.clear();
}

double drwnMaxFlow::operator()(int u, int v) const
{
    if ((u < 0) && (v < 0)) return _flowValue;
    if (u < 0) { return _sourceEdges[v]; }
    if (v < 0) { return _targetEdges[u]; }
    _drwnCapacitatedEdge::const_iterator it = _nodes[u].find(v);
    if (it == _nodes[u].end()) return 0.0;
    return _edgeWeights[it->second.first];
}

void drwnMaxFlow::preAugmentPaths()
{
    for (int u = 0; u < (int)_nodes.size(); u++) {
        // augment s-u-t paths
        if ((_sourceEdges[u] > 0.0) && (_targetEdges[u] > 0.0)) {
            const double cap = std::min(_sourceEdges[u], _targetEdges[u]);
            _flowValue += cap;
            _sourceEdges[u] -= cap;
            _targetEdges[u] -= cap;
        }

        if (_sourceEdges[u] == 0.0) continue;

        // augment s-u-v-t paths
        for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
            const int v = it->first;
            if ((_edgeWeights[it->second.first] == 0.0) || (_targetEdges[v] == 0.0)) continue;
            const double w = std::min(_edgeWeights[it->second.first], std::min(_sourceEdges[u], _targetEdges[v]));
            _sourceEdges[u] -= w;
            _targetEdges[v] -= w;
            _edgeWeights[it->second.first] -= w;
            _edgeWeights[it->second.second] += w;
            _flowValue += w;

            if (_sourceEdges[u] == 0.0) break;
        }
    }
}

// drwnEdmondsKarpMaxFlow ---------------------------------------------------

double drwnEdmondsKarpMaxFlow::solve()
{
    DRWN_FCN_TIC;

    // find max-flow
    const int UNSEEN_NODE = -2;
    const int SOURCE_NODE = -1;
    vector<int> frontier(_nodes.size());
    vector<int> backtrack(_nodes.size());

    // pre-augment s->u->t and s->u->v->t paths
    preAugmentPaths();

    // augment remaining paths
    while (true) {
        // find augmenting path (BFS)
        int frontierHead = 0;
        int frontierTail = 0;
        fill(backtrack.begin(), backtrack.end(), UNSEEN_NODE);

        // add initial nodes
        for (int u = 0; u < (int)_nodes.size(); u++) {
            if (_sourceEdges[u] > 0.0) {
                frontier[frontierTail++] = u;
                backtrack[u] = SOURCE_NODE;
            }
        }

        // find path
        while ((frontierHead != frontierTail) && (_targetEdges[frontier[frontierHead]] == 0.0)) {
            int u = frontier[frontierHead++];

            for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
                if ((_edgeWeights[it->second.first] > 0.0) && (backtrack[it->first] == UNSEEN_NODE)) {
                    frontier[frontierTail++] = it->first;
                    backtrack[it->first] = u;
                }
            }
        }

        if (frontierHead == frontierTail) break;

        // update residuals
        int u = frontier[frontierHead];
        double c = _targetEdges[u];
        while (backtrack[u] != SOURCE_NODE) {
            c = std::min(c, _edgeWeights[_nodes[backtrack[u]][u].first]);
            u = backtrack[u];
        }
        c = std::min(c, _sourceEdges[u]);

        //DRWN_ASSERT(c > 0.0);
        u = frontier[frontierHead];
        _targetEdges[u] -= c;
        while (backtrack[u] != SOURCE_NODE) {
            _drwnCapacitatedEdge::const_iterator it = _nodes[backtrack[u]].find(u);
            _edgeWeights[it->second.first] -= c;
            _edgeWeights[it->second.second] += c;
            u = backtrack[u];
        }
        _sourceEdges[u] -= c;

        // update flow
        _flowValue = _flowValue + c;
    }

    // set node state (cut)
    findCuts();

    DRWN_FCN_TOC;
    return _flowValue;
}

void drwnEdmondsKarpMaxFlow::findCuts()
{
    // initialize cut
    _cut.resize(_nodes.size());
    fill(_cut.begin(), _cut.end(), FREE);

    // BFS from source
    vector<int> frontier(_nodes.size());
    int frontierHead = 0;
    int frontierTail = 0;
    for (int u = 0; u < (int)_nodes.size(); u++) {
        if (_sourceEdges[u] > 0.0) {
            frontier[frontierTail++] = u;
            _cut[u] = SOURCE;
        }
    }

    while (frontierHead != frontierTail) {
        int u = frontier[frontierHead++];

        for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
            if ((_edgeWeights[it->second.first] > 0.0) && (_cut[it->first] == FREE)) {
                frontier[frontierTail++] = it->first;
                _cut[it->first] = SOURCE;
            }
        }
    }

    // BFS from target
    frontierHead = frontierTail = 0;
    for (int u = 0; u < (int)_nodes.size(); u++) {
        if (_targetEdges[u] > 0.0) {
            frontier[frontierTail++] = u;
            _cut[u] = TARGET;
        }
    }

    while (frontierHead != frontierTail) {
        int u = frontier[frontierHead++];

        for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
            if ((_edgeWeights[it->second.second] > 0.0) && (_cut[it->first] == FREE)) {
                frontier[frontierTail++] = it->first;
                _cut[it->first] = TARGET;
            }
        }
    }
}

// drwnBKMaxFlow ------------------------------------------------------------
//
// s -> u_i -> u_j -> ... -> v_j -> v_i -> t
// parent[u_i] = terminal, tree[u_i] = s
// parent[u_j] = u_i, tree[u_j] = s | active
// parent[v_i] = terminal, tree[v_i] = t
// parent[v_j] = v_i, tree[v_j] = t | active
// activeSet = {u_j, v_j}
//
// u on an augmenting path is an orphan if
// 1) tree[u] = s and (parent[u], u) = 0, or
// 2) tree[u] = t and (u, parent[u]) = 0
//
// \todo speed up check for activeSet by adding flag to _cut
//

void drwnBKMaxFlow::reset()
{
    drwnMaxFlow::reset();
    _parents.clear();
    _weightptrs.clear();
    _activeList.resize(_nodes.size());
}

void drwnBKMaxFlow::clear()
{
    drwnMaxFlow::clear();
    _parents.clear();
    _weightptrs.clear();
    _activeList.resize(_nodes.size());
}

double drwnBKMaxFlow::solve()
{
    DRWN_FCN_TIC;

    // initialize search tree and active set
    _cut.resize(_nodes.size());
    fill(_cut.begin(), _cut.end(), FREE);
    _parents.resize(_nodes.size());
    _weightptrs.resize(_nodes.size());
    _activeList.resize(_nodes.size());

    // pre-augment paths
    preAugmentPaths();

    // initialize search trees
    initializeTrees();

    deque<int> orphans;
    while (!_activeList.empty()) {
        //DRWN_LOG_DEBUG("current flow: " << _flowValue);
        const pair<int, int> path = expandTrees();
        augmentBKPath(path, orphans);
        if (!orphans.empty()) {
            adoptOrphans(orphans);
        }
    }

    DRWN_FCN_TOC;
    return _flowValue;
}

void drwnBKMaxFlow::initializeTrees()
{
    // initialize search tree
    for (int u = 0; u < (int)_nodes.size(); u++) {
        if (_sourceEdges[u] > 0.0) {
            _cut[u] = SOURCE;
            _parents[u] = TERMINAL;
            _activeList.push_back(u);
        } else if (_targetEdges[u] > 0.0) {
            _cut[u] = TARGET;
            _parents[u] = TERMINAL;
            _activeList.push_back(u);
        }
    }
}

pair<int, int> drwnBKMaxFlow::expandTrees()
{
    // expand trees looking for augmenting paths
    while (!_activeList.empty()) {
        const int u = _activeList.front();

        if (_cut[u] == SOURCE) {
            for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
                if (_edgeWeights[it->second.first] > 0.0) {
                    if (_cut[it->first] == FREE) {
                        _cut[it->first] = SOURCE;
                        _parents[it->first] = u;
                        _weightptrs[it->first] = make_pair(&_edgeWeights[it->second.first], &_edgeWeights[it->second.second]);
                        _activeList.push_back(it->first);
                    } else if (_cut[it->first] == TARGET) {
                        // found augmenting path
                        return make_pair(u, it->first);
                    }
                }
            }
        } else {
            for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
                if (_cut[it->first] == TARGET) continue;
                if (_edgeWeights[it->second.second] > 0.0) {
                    if (_cut[it->first] == FREE) {
                        _cut[it->first] = TARGET;
                        _parents[it->first] = u;
                        _weightptrs[it->first] = make_pair(&_edgeWeights[it->second.second], &_edgeWeights[it->second.first]);
                        _activeList.push_back(it->first);
                    } else if (_cut[it->first] == SOURCE) {
                        // found augmenting path
                        return make_pair(it->first, u);
                    }
                }
            }
        }

        // remove node from active set
        _activeList.pop_front();
    }

    return make_pair(TERMINAL, TERMINAL);
}

void drwnBKMaxFlow::augmentBKPath(const pair<int, int>& path, deque<int>& orphans)
{
    if ((path.first == TERMINAL) && (path.second == TERMINAL))
        return;

    // find path capacity

    // backtrack
    _drwnCapacitatedEdge::const_iterator e = _nodes[path.first].find(path.second);
    double c = _edgeWeights[e->second.first];

    int u = path.first;
    while (_parents[u] != TERMINAL) {
        c = std::min(c, *_weightptrs[u].first);
        u = _parents[u];
        //DRWN_ASSERT(_cut[u] == SOURCE);
    }
    c = std::min(c, _sourceEdges[u]);

    // forward track
    u = path.second;
    while (_parents[u] != TERMINAL) {
        c = std::min(c, *_weightptrs[u].first);
        u = _parents[u];
        //DRWN_ASSERT(_cut[u] == TARGET);
    }
    c = std::min(c, _targetEdges[u]);

    // augment path
    _flowValue += c;
    //DRWN_LOG_DEBUG("path capacity: " << c);

    // backtrack
    u = path.first;
    while (_parents[u] != TERMINAL) {
        *_weightptrs[u].first -= c;
        *_weightptrs[u].second += c;
        if (*_weightptrs[u].first == 0.0) {
            orphans.push_back(u);
        }
        u = _parents[u];
    }
    _sourceEdges[u] -= c;
    if (_sourceEdges[u] == 0.0) {
        orphans.push_back(u);
    }

    // link
    _edgeWeights[e->second.first] -= c;
    _edgeWeights[e->second.second] += c;

    // forward track
    u = path.second;
    while (_parents[u] != TERMINAL) {
        *_weightptrs[u].first -= c;
        *_weightptrs[u].second += c;
        if (*_weightptrs[u].first == 0.0) {
            orphans.push_back(u);
        }
        u = _parents[u];
    }
    _targetEdges[u] -= c;
    if (_targetEdges[u] == 0.0) {
        orphans.push_back(u);
    }
}

void drwnBKMaxFlow::adoptOrphans(deque<int>& orphans)
{
#if 0
    // re-initialize
    orphans.clear();
    fill(_cut.begin(), _cut.end(), FREE);
    _activeList.clear();
    initializeTrees();
#else
    // clear parents for all orphans to prevent orphan from being added multiple times
    for (deque<int>::iterator it = orphans.begin(); it != orphans.end(); ++it) {
        _parents[*it] = TERMINAL;
    }

    // find new parent for orphaned subtree or free it
    while (!orphans.empty()) {
        const int u = orphans.back();
        const char treeLabel = _cut[u];
        orphans.pop_back();

        // can occur if same node is inserted into orphans multiple times
        // (e.g., when the edge weight goes to zero between descendants)
        //if (treeLabel == FREE) continue;
        //DRWN_ASSERT(treeLabel != FREE);

        // look for new parent
        bool bFreeOrphan = true;
#if 1
        for (_drwnCapacitatedEdge::const_iterator jt = _nodes[u].begin(); jt != _nodes[u].end(); ++jt) {
            // skip if different trees
            if (_cut[jt->first] != treeLabel) continue;

            // check edge capacity
            if (_edgeWeights[(_cut[jt->first] == TARGET) ? jt->second.first : jt->second.second] == 0.0)
                continue;

            // check that u is not an ancestor of jt->first
#if 1
            int v = jt->first;
            while ((v != u) && (v != TERMINAL)) {
                v = _parents[v];
            }
            if (v != TERMINAL) continue;
#else
            if (isAncestor(u, jt->first)) continue;
#endif
            // add as parent
            if (_cut[jt->first] == TARGET) {
                _parents[u] = jt->first;
                _weightptrs[u] = make_pair(&_edgeWeights[jt->second.first], &_edgeWeights[jt->second.second]);
            } else {
                _parents[u] = jt->first;
                _weightptrs[u] = make_pair(&_edgeWeights[jt->second.second], &_edgeWeights[jt->second.first]);
            }

            bFreeOrphan = false;
            break;
        }
#endif

        // free the orphan subtree and remove it from the active set
        if (bFreeOrphan) {
            for (_drwnCapacitatedEdge::const_iterator jt = _nodes[u].begin(); jt != _nodes[u].end(); ++jt) {
                if ((_cut[jt->first] == treeLabel) && (_parents[jt->first] == u)) {
                    orphans.push_back(jt->first);
                    _activeList.push_back(jt->first);
              //} else if (_cut[jt->first] != FREE) {
                  //_activeList.push_back(jt->first);
                }
            }

            // mark inactive and free
            _activeList.erase(u);
            _cut[u] = FREE;
        }
    }
#endif
}

bool drwnBKMaxFlow::isAncestor(int u, int v) const
{
    while (v != u) {
        if (v == TERMINAL) return false;
        v = _parents[v];
    }
    return true;
}
