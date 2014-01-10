/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
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

// drwnMaxFlow statics ------------------------------------------------------

#ifdef __APPLE__
const unsigned char drwnMaxFlow::FREE;
const unsigned char drwnMaxFlow::SOURCE;
const unsigned char drwnMaxFlow::TARGET;
#endif

// drwnMaxFlow --------------------------------------------------------------

void drwnMaxFlow::reset()
{
    _flowValue = 0.0;
    fill(_sourceEdges.begin(), _sourceEdges.end(), 0.0);
    fill(_targetEdges.begin(), _targetEdges.end(), 0.0);
    fill(_edgeWeights.begin(), _edgeWeights.end(), 0.0);
    fill(_cut.begin(), _cut.end(), FREE);
    _history.clear();
}

void drwnMaxFlow::clear()
{
    _flowValue = 0.0;
    _sourceEdges.clear();
    _targetEdges.clear();
    _edgeWeights.clear();
    _nodes.clear();
    _cut.clear();
    _history.clear();
}

void drwnMaxFlow::augmentPath(const drwnAugmentingPath& path)
{
    if (path.path.empty()) return;

    // find path capacity
    double c = (*this)(-1, path.path.front());
    list<int>::const_iterator jt = path.path.begin();
    while (c > 0.0) {
        list<int>::const_iterator it = jt++;
        if (jt == path.path.end()) break;
        c = std::min(c, (*this)(*it, *jt));
    }
    c = std::min(c, (*this)(path.path.back(), -1));

    // augment the path
    if (c > 0.0) {
        pair<int, int> e(-1, -1);
        _sourceEdges[path.path.front()] -= c;
        list<int>::const_iterator jt = path.path.begin();
        while (true) {
            list<int>::const_iterator it = jt++;
            if (jt == path.path.end()) break;

            _drwnCapacitatedEdge::const_iterator ei = _nodes[*it].find(*jt);
            if (_edgeWeights[ei->second.first] == c) {
                _edgeWeights[ei->second.first] = 0.0;
                e = make_pair(*it, *jt);
            } else {
                _edgeWeights[ei->second.first] -= c;
            }
            _edgeWeights[ei->second.second] += c;
        }
        _targetEdges[path.path.back()] -= c;

        // update history with augmenting path
        if (_maintainHistory) {
            if (_sourceEdges[path.path.front()] == 0.0) {
                e = make_pair(-1, path.path.front());
            } else if (_targetEdges[path.path.back()] == 0.0) {
                e = make_pair(path.path.back(), -1);
            }
            _history.push_back(path);
        }
    }
}

void drwnMaxFlow::augmentPaths(const list<drwnAugmentingPath>& paths)
{
    for (list<drwnAugmentingPath>::const_iterator it = paths.begin(); it != paths.end(); ++it) {
        this->augmentPath(*it);
    }
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

            // keep track of augmenting paths
            if (_maintainHistory) {
                _history.push_back(drwnAugmentingPath(u, this));
            }
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

            // keep track of augmenting paths
            if (_maintainHistory) {
                _history.push_back(drwnAugmentingPath(u, v, this));
            }

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

        // keep track of augmenting paths
        if (_maintainHistory) {
            drwnAugmentingPath p;
            int u = frontier[frontierHead];
            p.edge = make_pair(u, -1);
            p.path.push_front(u);
            while (backtrack[u] != SOURCE_NODE) {
                if ((*this)(backtrack[u], u) < (*this)(p.edge.first, p.edge.second)) {
                    p.edge = make_pair(backtrack[u], u);
                }
                u = backtrack[u];
                p.path.push_front(u);
            }
            if ((*this)(-1, u) < (*this)(p.edge.first, p.edge.second)) {
                p.edge = make_pair(-1, u);
            }

            _history.push_back(p);
        }

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
    clearActive();
}

void drwnBKMaxFlow::clear()
{
    drwnMaxFlow::clear();
    _parents.clear();
    clearActive();
}

double drwnBKMaxFlow::solve()
{
    DRWN_FCN_TIC;

    // initialize search tree and active set
    _cut.resize(_nodes.size());
    fill(_cut.begin(), _cut.end(), FREE);
    _parents.resize(_nodes.size());

    clearActive();

    // pre-augment paths
    preAugmentPaths();

    // initialize search trees
    initializeTrees();

    deque<int> orphans;
    while (!isActiveSetEmpty()) {
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
            _parents[u].first = TERMINAL;
            markActive(u);
        } else if (_targetEdges[u] > 0.0) {
            _cut[u] = TARGET;
            _parents[u].first = TERMINAL;
            markActive(u);
        }
    }
}

pair<int, int> drwnBKMaxFlow::expandTrees()
{
    // expand trees looking for augmenting paths
    while (!isActiveSetEmpty()) {
        const int u = _activeHead;

        if (_cut[u] == SOURCE) {
            for (_drwnCapacitatedEdge::const_iterator it = _nodes[u].begin(); it != _nodes[u].end(); it++) {
                if (_edgeWeights[it->second.first] > 0.0) {
                    if (_cut[it->first] == FREE) {
                        _cut[it->first] = SOURCE;
                        _parents[it->first] = make_pair(u, make_pair(&_edgeWeights[it->second.first], &_edgeWeights[it->second.second]));
                        markActive(it->first);
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
                        _parents[it->first] = make_pair(u, make_pair(&_edgeWeights[it->second.second], &_edgeWeights[it->second.first]));
                        markActive(it->first);
                    } else if (_cut[it->first] == SOURCE) {
                        // found augmenting path
                        return make_pair(it->first, u);
                    }
                }
            }
        }

        // remove node from active set
        markInActive(u);
    }

    return make_pair(TERMINAL, TERMINAL);
}

void drwnBKMaxFlow::augmentBKPath(const pair<int, int>& path, deque<int>& orphans)
{
    if ((path.first == TERMINAL) && (path.second == TERMINAL))
        return;

    // keep track of augmenting paths
    if (_maintainHistory) {
        drwnAugmentingPath p;
        p.edge = path;
        int u = path.first;
        p.path.push_front(u);
        while (_parents[u].first != TERMINAL) {
            if ((*this)(_parents[u].first, u) < (*this)(p.edge.first, p.edge.second)) {
                p.edge = make_pair(_parents[u].first, u);
            }
            u = _parents[u].first;
            p.path.push_front(u);
        }
        u = path.second;
        p.path.push_back(u);
        while (_parents[u].first != TERMINAL) {
            if ((*this)(u, _parents[u].first) < (*this)(p.edge.first, p.edge.second)) {
                p.edge = make_pair(u, _parents[u].first);
            }
            u = _parents[u].first;
            p.path.push_back(u);
        }
        _history.push_back(p);
    }

    // find path capacity

    // backtrack
    _drwnCapacitatedEdge::const_iterator e = _nodes[path.first].find(path.second);
    double c = _edgeWeights[e->second.first];

    int u = path.first;
    while (_parents[u].first != TERMINAL) {
        c = std::min(c, *_parents[u].second.first);
        u = _parents[u].first;
        //DRWN_ASSERT(_cut[u] == SOURCE);
    }
    c = std::min(c, _sourceEdges[u]);

    // forward track
    u = path.second;
    while (_parents[u].first != TERMINAL) {
        c = std::min(c, *_parents[u].second.first);
        u = _parents[u].first;
        //DRWN_ASSERT(_cut[u] == TARGET);
    }
    c = std::min(c, _targetEdges[u]);

    // augment path
    _flowValue += c;
    //DRWN_LOG_DEBUG("path capacity: " << c);

    // backtrack
    u = path.first;
    while (_parents[u].first != TERMINAL) {
        *_parents[u].second.first -= c;
        *_parents[u].second.second += c;
        if (*_parents[u].second.first == 0.0) {
            orphans.push_front(u);
        }
        u = _parents[u].first;
    }
    _sourceEdges[u] -= c;
    if (_sourceEdges[u] == 0.0) {
        orphans.push_front(u);
    }

    // link
    _edgeWeights[e->second.first] -= c;
    _edgeWeights[e->second.second] += c;

    // forward track
    u = path.second;
    while (_parents[u].first != TERMINAL) {
        *_parents[u].second.first -= c;
        *_parents[u].second.second += c;
        if (*_parents[u].second.first == 0.0) {
            orphans.push_front(u);
        }
        u = _parents[u].first;
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
    clearActive();
    initializeTrees();
#else
    // find new parent for orphaned subtree or free it
    while (!orphans.empty()) {
        const int u = orphans.front();
        const char treeLabel = _cut[u];
        orphans.pop_front();

        // can occur if same node is inserted into orphans multiple times
        if (treeLabel == FREE) continue;
        //DRWN_ASSERT(treeLabel != FREE);

        // look for new parent
        bool bFreeOrphan = true;
#if 1
        for (_drwnCapacitatedEdge::const_iterator jt = _nodes[u].begin(); jt != _nodes[u].end(); ++jt) {
            // skip if different trees
            if (_cut[jt->first] != treeLabel) continue;

            // check edge capacity
            if (((treeLabel == TARGET) && (_edgeWeights[jt->second.first] <= 0.0)) ||
                ((treeLabel == SOURCE) && (_edgeWeights[jt->second.second] <= 0.0)))
                continue;

            // check that u is not an ancestor of jt->first
            int v = jt->first;
            while ((v != u) && (v != TERMINAL)) {
                v = _parents[v].first;
            }
            if (v != TERMINAL) continue;

            // add as parent
            if (treeLabel == SOURCE) {
                _parents[u] = make_pair(jt->first, make_pair(&_edgeWeights[jt->second.second], &_edgeWeights[jt->second.first]));
            } else {
                _parents[u] = make_pair(jt->first, make_pair(&_edgeWeights[jt->second.first], &_edgeWeights[jt->second.second]));
            }
            bFreeOrphan = false;
            break;
        }
#endif

        // free the orphan subtree and remove it from the active set
        if (bFreeOrphan) {
            for (_drwnCapacitatedEdge::const_iterator jt = _nodes[u].begin(); jt != _nodes[u].end(); ++jt) {
                if ((_cut[jt->first] == treeLabel) && (_parents[jt->first].first == u)) {
                    orphans.push_front(jt->first);
                    markActive(jt->first);
                } else if (_cut[jt->first] != FREE) {
                    markActive(jt->first);
                }
            }

            // mark inactive and free
            markInActive(u);
            _cut[u] = FREE;
        }
    }
#endif
}
