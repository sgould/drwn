/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphThreadedMoves.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

//! \todo multi-thread randproj moves

#pragma once

#include "drwnNNGraph.h"
#include "drwnNNGraphMoves.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphThreadedMoves --------------------------------------------------
//! Templated drwnNNGraph search moves (multi-threaded).

namespace drwnNNGraphThreadedMoves {
    //! randomly initialize edges (matched) for all active images (keeps existing
    //! matches unless an improvement is found)
    template<class DistanceMetric>
    void initialize(drwnNNGraph& graph, const DistanceMetric& M);

    //! randomly initialize edges (matched) for all active images (keeps existing
    //! matches unless an improvement is found) using default metric
    inline void initialize(drwnNNGraph& graph) { initialize(graph, drwnNNGraphDefaultMetric()); }

    //! rescore all matches based on current features (which may have changed
    //! since graph was created)
    template<class DistanceMetric>
    double rescore(drwnNNGraph& graph, const DistanceMetric& M);

    //! rescore all matches based on current features (which may have changed
    //! since graph was created) using default metric
    inline double rescore(drwnNNGraph& graph) { return rescore(graph, drwnNNGraphDefaultMetric()); }

    //! perform one update iteration (including enrichment and exhaustive)
    //! on all active images
    template<class DistanceMetric>
    void update(drwnNNGraph& graph, const DistanceMetric& M);

    //! perform one update iteration (including enrichment and exhaustive)
    //! on all active images using default metric
    inline void update(drwnNNGraph& graph) { update(graph, drwnNNGraphDefaultMetric()); }

    //! perform single update iteration of propagate, local and search moves
    //! on a specific image
    template<class DistanceMetric>
    void updateImage(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M);

    //! random projection moves that project all superpixels onto a random
    //! direction, sort and compare adjacent superpixels if from different
    //! images (returns true if a better match was found)
    template<class DistanceMetric>
    bool randproj(drwnNNGraph& graph, const DistanceMetric& M);

    //! enrichment: inverse (from target to source) and forward (from
    //! source to target's target) (returns true if a better match was found)
    template<class DistanceMetric>
    bool enrichment(drwnNNGraph& graph, const DistanceMetric& M);

    //! threading initialize functor
    template<class DistanceMetric>
    class drwnNNGraphThreadedInitializeJob : public drwnThreadJob {
    protected:
        const DistanceMetric& _M;        //!< distance metric to use during moves
        set<unsigned> _imgIndxes;        //!< indexes of images for this job
        drwnNNGraph& _graph;             //!< graph for updating (includes features)

    public:
        drwnNNGraphThreadedInitializeJob(const set<unsigned>& imgIndxes, drwnNNGraph& g,
            const DistanceMetric& M) : _M(M), _imgIndxes(imgIndxes), _graph(g) { /* do nothing */ }
        ~drwnNNGraphThreadedInitializeJob() { /* do nothing */ }

        void operator()() {
            for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
                drwnNNGraphMoves::initialize(_graph, *it, _M);
            }
        }
    };

    //! threading rescore functor
    template<class DistanceMetric>
    class drwnNNGraphThreadedRescoreJob : public drwnThreadJob {
    protected:
        const DistanceMetric& _M;        //!< distance metric to use during moves
        set<unsigned> _imgIndxes;        //!< indexes of images for this job
        drwnNNGraph& _graph;             //!< graph for updating (includes features)

    public:
        drwnNNGraphThreadedRescoreJob(const set<unsigned>& imgIndxes, drwnNNGraph& g,
            const DistanceMetric& M) : _M(M), _imgIndxes(imgIndxes), _graph(g) { /* do nothing */ }
        ~drwnNNGraphThreadedRescoreJob() { /* do nothing */ }

        void operator()() {
            for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
                drwnNNGraphMoves::rescore(_graph, *it, _M);
            }
        }
    };

    //! threading update functor
    template<class DistanceMetric>
    class drwnNNGraphThreadedUpdateJob : public drwnThreadJob {
    protected:
        const DistanceMetric& _M;        //!< distance metric to use during moves
        set<unsigned> _imgIndxes;        //!< indexes of images for this job
        drwnNNGraph& _graph;             //!< graph for updating (includes features)

    public:
        drwnNNGraphThreadedUpdateJob(const set<unsigned>& imgIndxes, drwnNNGraph& g,
            const DistanceMetric& M) : _M(M), _imgIndxes(imgIndxes), _graph(g) { /* do nothing */ }
        ~drwnNNGraphThreadedUpdateJob() { /* do nothing */ }

        void operator()() {
            for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
                drwnNNGraphThreadedMoves::updateImage(_graph, *it, _M);
            }
        }
    };

    //! threading exhaustive functor
    template<class DistanceMetric>
    class drwnNNGraphThreadedExhaustiveJob : public drwnThreadJob {
    protected:
        const DistanceMetric& _M;        //!< distance metric to use during moves
        set<drwnNNGraphNodeIndex> _nodeIndxes; //!< indexes of nodes for this job
        drwnNNGraph& _graph;             //!< graph for updating (includes features)

    public:
        drwnNNGraphThreadedExhaustiveJob(const set<drwnNNGraphNodeIndex>& nodeIndxes, drwnNNGraph& g,
            const DistanceMetric& M) : _M(M),  _nodeIndxes(nodeIndxes), _graph(g) { /* do nothing */ }
        ~drwnNNGraphThreadedExhaustiveJob() { /* do nothing */ }

        void operator()() {
            for (set<drwnNNGraphNodeIndex>::const_iterator it = _nodeIndxes.begin(); it != _nodeIndxes.end(); ++it) {
                drwnNNGraphMoves::exhaustive(_graph, *it, _M);
            }
        }
    };
};

// drwnNNGraphThreadedMoves implementation -----------------------------------

template<class DistanceMetric>
void drwnNNGraphThreadedMoves::initialize(drwnNNGraph& graph, const DistanceMetric& M)
{
    // prepare thread data
    const unsigned nJobs = std::min((unsigned)graph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        if (!graph[imgIndx].bSourceMatchable) {
            DRWN_LOG_DEBUG("...skipping initialization of " << graph[imgIndx].name());
            continue;
        }

        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphThreadedInitializeJob<DistanceMetric> *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphThreadedInitializeJob<DistanceMetric>(imgIndxes[i], graph, M);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();
}

template<class DistanceMetric>
double drwnNNGraphThreadedMoves::rescore(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;

    // prepare thread data
    const unsigned nJobs = std::min((unsigned)graph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphThreadedRescoreJob<DistanceMetric> *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphThreadedRescoreJob<DistanceMetric>(imgIndxes[i], graph, M);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();

    DRWN_FCN_TOC;
    return graph.energy().first;
}

template<class DistanceMetric>
void drwnNNGraphThreadedMoves::update(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;

    // try improve a bad match (from a random (active) image)
    if (drwnNNGraph::DO_EXHAUSTIVE > 0) {
        // sum scores of nodes from active images
        double totalWeight = 0.0;
        for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
            if (!graph[imgIndx].bSourceMatchable) continue;
            for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
                const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                if (e.empty()) continue;

                totalWeight += (double)e.front().weight;
            }
        }

        // randomly sample n nodes
        set<drwnNNGraphNodeIndex> sampledNodes;
        for (int i = 0; i < drwnNNGraph::DO_EXHAUSTIVE; i++) {

            // sample the weight
            double weight = totalWeight * drand48();

            // find the node
            for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
                if (!graph[imgIndx].bSourceMatchable) continue;
                for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
                    const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                    if (e.empty()) continue;

                    // skip if already sampled this node
                    if (sampledNodes.find(drwnNNGraphNodeIndex(imgIndx, segId)) != sampledNodes.end())
                        continue;

                    weight -= (double)e.front().weight;
                    if (weight <= 0.0) {
                        // add node to set of samples and remove from totalWeight
                        sampledNodes.insert(drwnNNGraphNodeIndex(imgIndx, segId));
                        totalWeight -= (double)e.front().weight;
                        break;
                    }
                }
                if (weight <= 0.0) break;
            }
        }

        // start threads for exhuastive search on nodes
        if (sampledNodes.size() == 1) {
            drwnNNGraphMoves::exhaustive(graph, *sampledNodes.begin(), M);
        } else {
            drwnThreadPool threadPool;
            threadPool.start();
            vector<drwnNNGraphThreadedExhaustiveJob<DistanceMetric> *> jobs;
            for (set<drwnNNGraphNodeIndex>::const_iterator it = sampledNodes.begin(); it != sampledNodes.end(); ++it) {
                set<drwnNNGraphNodeIndex> nodeIndexes;
                nodeIndexes.insert(*it);
                jobs.push_back(new drwnNNGraphThreadedExhaustiveJob<DistanceMetric>(nodeIndexes, graph, M));
                threadPool.addJob(jobs.back());
            }
            threadPool.finish();

            for (unsigned i = 0; i < jobs.size(); i++) {
                delete jobs[i];
            }
            jobs.clear();
        }
    }

    // random projection
    if (drwnNNGraph::DO_RANDPROJ > 0) {
        randproj(graph, M);
    }

    // propagation moves
    if (drwnNNGraph::DO_PROPAGATE || drwnNNGraph::DO_LOCAL || drwnNNGraph::DO_SEARCH) {
        // prepare thread data
        const unsigned nJobs = std::min((unsigned)graph.numImages(),
            std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
        vector<set<unsigned> > imgIndxes(nJobs);
        for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
            if (!graph[imgIndx].bSourceMatchable) {
                DRWN_LOG_DEBUG("...skipping update of " << graph[imgIndx].name());
                continue;
            }

            imgIndxes[imgIndx % nJobs].insert(imgIndx);
        }

        // start threads
        drwnThreadPool threadPool(nJobs);
        threadPool.start();
        vector<drwnNNGraphThreadedUpdateJob<DistanceMetric> *> jobs(nJobs);
        for (unsigned i = 0; i < nJobs; i++) {
            jobs[i] = new drwnNNGraphThreadedUpdateJob<DistanceMetric>(imgIndxes[i], graph, M);
            threadPool.addJob(jobs[i]);
        }
        threadPool.finish();

        for (unsigned i = 0; i < jobs.size(); i++) {
            delete jobs[i];
        }
        jobs.clear();
    }

    // enrichment (forward and inverse)
    if (drwnNNGraph::DO_ENRICHMENT) {
        enrichment(graph, M);
    }

    DRWN_FCN_TOC;
}

template<class DistanceMetric>
void drwnNNGraphThreadedMoves::updateImage(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M)
{
    // perform update on each node
    drwnNNGraphNodeIndex u(imgIndx, 0);
    for (u.segId = 0; u.segId < graph[u.imgIndx].numNodes(); u.segId++) {
        if (drwnNNGraph::DO_LOCAL) {
            drwnNNGraphMoves::local(graph, u, M);
        }
        if (drwnNNGraph::DO_PROPAGATE) {
            drwnNNGraphMoves::propagate(graph, u, M);
        }
        if (drwnNNGraph::DO_SEARCH) {
            drwnNNGraphMoves::search(graph, u, M);
        }
    }
}

template<class DistanceMetric>
bool drwnNNGraphThreadedMoves::randproj(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;
    bool bChanged = false;

    // pick a random direction
    VectorXf mu(graph[0][0].features.size());
    for (int i = 0; i < mu.rows(); i++) {
        mu[i] = float(drand48() - 0.5);
    }
    mu /= mu.norm();

    // project all superpixels
    vector<pair<float, drwnNNGraphNodeIndex> > projections;
    projections.reserve(graph.numNodes());

    drwnNNGraphNodeIndex u;
    for (u.imgIndx = 0; u.imgIndx < graph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < graph[u.imgIndx].numNodes(); u.segId++) {
            const float x = mu.dot(graph[u].features);
            projections.push_back(make_pair(x, u));
        }
    }

    // sort
    sort(projections.begin(), projections.end());

    // score
    unsigned dbCountChanged = 0;
    unsigned inext = 0;
    for (unsigned i = 0; i < projections.size() - 1; i++) {
        const drwnNNGraphNodeIndex &u = projections[i].second;

        // don't update inactive images
        if (!graph[u.imgIndx].bSourceMatchable || graph[u].edges.empty()) {
            if (inext <= i) inext += 1;
            continue;
        }

        // skip if images are in the same equivalence class
        while (inext != projections.size()) {
            if (graph[projections[inext].second.imgIndx].bTargetMatchable &&
                !graph.inSameEqvClass(u.imgIndx, projections[inext].second.imgIndx))
                break;
            inext += 1;
        }

        if (inext == projections.size()) break;

        float uworst = sqrt(graph[u].edges.back().weight);
        for (unsigned j = inext; j < std::min(projections.size(), (size_t)(inext + drwnNNGraph::DO_RANDPROJ)); j++) {
            // cauchy-schwarz check
            if ((projections[j].first - projections[i].first) >= uworst) {
                break;
            }

            // equivalence class check
            if (!graph[projections[j].second.imgIndx].bTargetMatchable) continue;
            if (graph.inSameEqvClass(u.imgIndx, projections[j].second.imgIndx))
                continue;

            // evaluate edge candidate
            const drwnNNGraphNodeIndex &v = projections[j].second;

            if (M.isFinite(graph[u], graph[v])) {
                const float w = M.score(graph[u], graph[v]);
                if (graph[u].insert(drwnNNGraphEdge(v, w))) {
                    uworst = sqrt(graph[u].edges.back().weight);
                    dbCountChanged += 1;
                    bChanged = true;
                }
            }
        }
    }

    DRWN_LOG_DEBUG("randproj() improved " << dbCountChanged << " matches");
    DRWN_FCN_TOC;
    return bChanged;
}

template<class DistanceMetric>
bool drwnNNGraphThreadedMoves::enrichment(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;
    bool bChanged = false;

    // inverse enrichment
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        // skip image if not target matchable
        if (!graph[imgIndx].bTargetMatchable) continue;

        for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
            const drwnNNGraphEdgeList& el = graph[imgIndx][segId].edges;
            for (drwnNNGraphEdgeList::const_iterator kt = el.begin(); kt != el.end(); ++kt) {

                // skip if not active
                if (!graph[kt->targetNode.imgIndx].bSourceMatchable)
                    continue;

                // evaluate reverse edge
                const drwnNNGraphEdge e(drwnNNGraphNodeIndex(imgIndx, segId), kt->weight);
                if (graph[kt->targetNode].insert(e)) {
                    bChanged = true;
                }
            }
        }
    }

    // forward enrichment
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        // only do forward enrichment on active images
        if (!graph[imgIndx].bSourceMatchable) continue;

        for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
            // needed to prevent invalid iterators when updating e
            const drwnNNGraphEdgeList el(graph[imgIndx][segId].edges);

            int nCompared = (int)drwnNNGraph::K; // prevent quadratic growth in K
            for (drwnNNGraphEdgeList::const_iterator it = el.begin(); it != el.end(); ++it) {
                const drwnNNGraphEdgeList& r = graph[it->targetNode].edges;

                for (drwnNNGraphEdgeList::const_iterator kt = r.begin(); kt != r.end(); ++kt) {
                    // check that we're not matching back to ourself (or any other
                    // image in the same equivalence class) and is matchable
                    if (!graph[kt->targetNode.imgIndx].bTargetMatchable) continue;
                    if (graph.inSameEqvClass(kt->targetNode.imgIndx, imgIndx)) continue;

                    // check that we have processed this pair previously
                    if ((it->status == DRWN_NNG_PROCESSED_TWICE) &&
                        (kt->status == DRWN_NNG_PROCESSED_TWICE)) continue;

                    // check that edge is legal
                    if (!M.isFinite(graph[imgIndx][segId], graph[kt->targetNode]))
                        continue;

                    const float w = M.score(graph[imgIndx][segId], graph[kt->targetNode]);
                    if (graph[imgIndx][segId].insert(drwnNNGraphEdge(kt->targetNode, w))) {
                        bChanged = true;
                    }

                    nCompared -= 1;
                    if (nCompared < 0) break;
                }

                if (nCompared < 0) break;
            }
        }
    }

    DRWN_FCN_TOC;
    return bChanged;
}
