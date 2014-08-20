/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphMoves.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include "drwnNNGraph.h"

using namespace std;
using namespace Eigen;

// drwnNNGraph node distance metrics -----------------------------------------
//! Implements the scoring functions needed by the search moves. Required
//! member functions are isMatchable(), isFinite() and score().

class drwnNNGraphDefaultMetric {
 public:
    bool isMatchable(const drwnNNGraphNode& a) const { return true; }
    bool isFinite(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const { return true; }
    float score(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.features - b.features).squaredNorm();
        //return a.squaredNorm + b.squaredNorm - 2.0 * a.features.dot(b.features);
    }
};

class drwnNNGraphLabelsEqualMetric {
 public:
    bool isMatchable(const drwnNNGraphNode& a) const { return true; }
    bool isFinite(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.label == b.label) || (a.label == -1) || (b.label == -1);
    }
    float score(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.features - b.features).squaredNorm();
    }
};

class drwnNNGraphLabelsNotEqualMetric {
 public:
    bool isMatchable(const drwnNNGraphNode& a) const { return true; }
    bool isFinite(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.label != b.label);
    }
    float score(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.features - b.features).squaredNorm();
    }
};

class drwnNNGraphLabelsEqualMetricNoUnknown {
 public:
    bool isMatchable(const drwnNNGraphNode& a) const { return (a.label != -1); }
    bool isFinite(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.label == b.label) && (a.label != -1);
    }
    float score(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.features - b.features).squaredNorm();
    }
};

class drwnNNGraphLabelsNotEqualMetricNoUnknown {
 public:
    bool isMatchable(const drwnNNGraphNode& a) const { return (a.label != -1); }
    bool isFinite(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.label != b.label) && (a.label != -1) && (b.label != -1);
    }
    float score(const drwnNNGraphNode& a, const drwnNNGraphNode& b) const {
        return (a.features - b.features).squaredNorm();
    }
};

// drwnNNGraphMoves ----------------------------------------------------------
//! Templated drwnNNGraph search moves.

namespace drwnNNGraphMoves {
    //! randomly initialize edges (matched) for all active images (keeps existing
    //! matches unless an improvement is found)
    template<class DistanceMetric>
    void initialize(drwnNNGraph& graph, const DistanceMetric& M);

    //! randomly initialize edges (matched) for all active images (keeps existing
    //! matches unless an improvement is found) using default metric
    inline void initialize(drwnNNGraph& graph) { initialize(graph, drwnNNGraphDefaultMetric()); }

    //! randomly initialize edges (matched) for a given image (keeps existing
    //! matches unless an improvement is found)
    template<class DistanceMetric>
    void initialize(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M);

    //! rescore all matches based on current features (which may have changed
    //! since graph was created)
    template<class DistanceMetric>
    double rescore(drwnNNGraph& graph, const DistanceMetric& M);

    //! rescore all matches based on current features (which may have changed
    //! since graph was created) using default metric
    inline double rescore(drwnNNGraph& graph) { return rescore(graph, drwnNNGraphDefaultMetric()); }

    //! rescore all matches for a given image based on current features (which
    //! may have changed since graph was created)
    template<class DistanceMetric>
    double rescore(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M);

    //! update initial edges using FLANN (with possible self-matches in both source and
    //! destination matchable)
    template<class DistanceMetric>
    void flann(drwnNNGraph& graph, const DistanceMetric& M);

    //! perform one update iteration (including enrichment and exhaustive)
    //! on all active images
    template<class DistanceMetric>
    void update(drwnNNGraph& graph, const DistanceMetric& M);

    //! perform one update iteration (including enrichment and exhaustive)
    //! on all active images using default metric
    inline void update(drwnNNGraph& graph) { update(graph, drwnNNGraphDefaultMetric()); }

    //! propagate good matches from supeprixel \p u to its neighbours
    //! (returns true if a better match was found)
    template<class DistanceMetric>
    bool propagate(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M);

    //! random search across all images for superpixel \p u
    //! (returns true if a better match was found)
    template<class DistanceMetric>
    bool search(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M);

    //! local neighbourhood search around current match for superpixel \p u
    //! (returns true if a better match was found)
    template<class DistanceMetric>
    bool local(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M);

    //! random projection moves that project all superpixels onto a random
    //! direction, sort and compare adjacent superpixels if from different
    //! images (returns true if a better match was found)
    template<class DistanceMetric>
    bool randproj(drwnNNGraph& graph, const DistanceMetric& M);

    //! enrichment: inverse (from target to source) and forward (from
    //! source to target's target) (returns true if a better match was found)
    template<class DistanceMetric>
    bool enrichment(drwnNNGraph& graph, const DistanceMetric& M);

    //! exhaustive search for best match across entire graph --- use sparingly
    //! (returns true if a better match was found)
    template<class DistanceMetric>
    bool exhaustive(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M);
};

// drwnNNGraphMoves implementation -------------------------------------------

template<class DistanceMetric>
void drwnNNGraphMoves::initialize(drwnNNGraph& graph, const DistanceMetric& M)
{
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        if (!graph[imgIndx].bSourceMatchable) {
            DRWN_LOG_DEBUG("...skipping initialization of " << graph[imgIndx].name());
            continue;
        }

        initialize(graph, imgIndx, M);
    }
}

template<class DistanceMetric>
void drwnNNGraphMoves::initialize(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M)
{
    DRWN_FCN_TIC;
    DRWN_LOG_DEBUG("...initializing " << graph[imgIndx].name());

    // random number generator state
#if !(defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__))
    unsigned short int xsubi[3] = {0, 0, 0};
    xsubi[0] = random();
#endif

    // initialize image indexes for fast subsampling
    vector<unsigned> indexes;
    indexes.reserve(graph.numImages() - 1);
    for (unsigned i = 0; i < graph.numImages(); i++) {
        if (!graph[i].bTargetMatchable) continue;
        if (graph.inSameEqvClass(i, imgIndx)) continue;
        indexes.push_back(i);
    }

    bool bLessMatches = false;
    const unsigned kMatches = std::min((unsigned)indexes.size(), drwnNNGraph::K);

    // randomly match all superpixels in the image
    for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
        // check if node is matchable
        if (!M.isMatchable(graph[imgIndx][segId])) continue;

        // check for existing edges (i.e., image is being re-initialized)
        const bool bHasExistingMatches = !graph[imgIndx][segId].edges.empty();

#if 0
        drwn::shuffle(indexes);
#elif defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
        drwn::shuffle(indexes);
#else
        {
            // thread safe shuffle
            unsigned int seed = (unsigned int)erand48(xsubi);
            const size_t n = indexes.size();
            for (size_t i = 0; i < n - 1; i++) {
                size_t j = rand_r(&seed) % (n - i);
                std::swap(indexes[i], indexes[i + j]);
            }
        }
#endif
        unsigned matchesAdded = 0;
        for (unsigned k = 0; k < indexes.size(); k++) {

            // determine valid nodes to match against
            vector<unsigned> segIndexes;
            segIndexes.reserve(graph[indexes[k]].numNodes());
            for (unsigned i = 0; i < graph[indexes[k]].numNodes(); i++) {
                if (M.isFinite(graph[imgIndx][segId], graph[indexes[k]][i])) {
                    segIndexes.push_back(i);
                }
            }
            if (segIndexes.empty()) continue;

            drwnNNGraphEdge e;
            e.targetNode.imgIndx = indexes[k];
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
            e.targetNode.segId = (uint16_t)segIndexes[drand48() * segIndexes.size()];
#else
            e.targetNode.segId = (uint16_t)segIndexes[erand48(xsubi) * segIndexes.size()];
#endif
            e.weight = M.score(graph[imgIndx][segId], graph[e.targetNode]);
            graph[imgIndx][segId].edges.push_back(e);

            // check if we've added the required number of matches
            if (++matchesAdded >= kMatches)
                break;
        }

        // check if not enough matches were found
        bLessMatches = bLessMatches || (graph[imgIndx][segId].edges.size() < kMatches);

        // remove excess matches if existing
        if (bHasExistingMatches) {
            // remove duplicate target images
            graph[imgIndx][segId].edges.sort(drwnNNGraphSortByImage());
            drwnNNGraphEdgeList::iterator kt = graph[imgIndx][segId].edges.begin();
            drwnNNGraphEdgeList::iterator jt = kt++;
            while (kt != graph[imgIndx][segId].edges.end()) {
                if (kt->targetNode.imgIndx == jt->targetNode.imgIndx) {
                    kt = graph[imgIndx][segId].edges.erase(kt);
                } else {
                    jt = kt++;
                }
            }
        }

        // sort matches from best to worst
        graph[imgIndx][segId].edges.sort();

        // resize to correct number of matches per node
        if (bHasExistingMatches) {
            graph[imgIndx][segId].edges.resize(kMatches);
        }
    }

    // issue warning for too few matches
    if (bLessMatches) {
        DRWN_LOG_WARNING("some nodes in " << graph[imgIndx].name()
            << " have out-degree less than " << drwnNNGraph::K);
    }

    DRWN_FCN_TOC;
}

template<class DistanceMetric>
double drwnNNGraphMoves::rescore(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;
    double energy = 0.0;

    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        energy += rescore(graph, imgIndx, M);
    }

    DRWN_FCN_TOC;
    return energy;
}

template<class DistanceMetric>
double drwnNNGraphMoves::rescore(drwnNNGraph& graph, unsigned imgIndx, const DistanceMetric& M)
{
    double energy = 0.0;

    for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {
        drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
        for (drwnNNGraphEdgeList::iterator kt = e.begin(); kt != e.end(); ++kt) {
            if (M.isFinite(graph[imgIndx][segId], graph[kt->targetNode])) {
                kt->weight = M.score(graph[imgIndx][segId], graph[kt->targetNode]);
                energy += kt->weight;
            } else {
                //! \todo delete the node
                DRWN_TODO;
            }
        }
        e.sort();
    }

    return energy;
}

template<class DistanceMetric>
void drwnNNGraphMoves::flann(drwnNNGraph& graph, const DistanceMetric& M)
{
    DRWN_FCN_TIC;

    // extract features to match
    const size_t numFeatures = graph[0][0].features.size();
    vector<drwnNNGraphNodeIndex> sampleIndexes;
    for (unsigned i = 0; i < graph.numImages(); i++) {
        if (!graph[i].bTargetMatchable) continue;

        sampleIndexes.reserve(sampleIndexes.size() + graph[i].numNodes());
        for (unsigned j = 0; j < graph[i].numNodes(); j++) {
            sampleIndexes.push_back(drwnNNGraphNodeIndex(i, j));
        }
    }

    cv::Mat features(sampleIndexes.size(), numFeatures, CV_32FC1);
    for (unsigned i = 0; i < sampleIndexes.size(); i++) {
        for (unsigned d = 0; d < numFeatures; d++) {
            features.at<float>(i, d) = graph[sampleIndexes[i]].features[d];
        }
    }

    // build kd-tree
    cv::flann::KDTreeIndexParams indexParams;
    cv::flann::Index kdtree(features, indexParams);

    // run queries
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        if (!graph[imgIndx].bSourceMatchable) continue;

        const unsigned numQueries = graph[imgIndx].numNodes();
        cv::Mat queries(numQueries, numFeatures, CV_32FC1);
        for (unsigned i = 0; i < numQueries; i++) {
            for (unsigned d = 0; d < numFeatures; d++) {
                queries.at<float>(i, d) = graph[imgIndx][i].features[d];
            }
        }

        cv::Mat indexes(numQueries, drwnNNGraph::K, CV_32SC1);
        cv::Mat dists(numQueries, drwnNNGraph::K, CV_32FC1);
        kdtree.knnSearch(queries, indexes, dists, drwnNNGraph::K, cv::flann::SearchParams(64));

        // extract nearest neighbours
        for (unsigned i = 0; i < numQueries; i++) {
            const drwnNNGraphNodeIndex u(imgIndx, i);
            for (unsigned k = 0; k < drwnNNGraph::K; k++) {
                const drwnNNGraphNodeIndex v(sampleIndexes[indexes.at<int>(i, k)]);
                if (graph.inSameEqvClass(u.imgIndx, v.imgIndx)) continue;
                if (!M.isMatchable(graph[v])) continue;
                if (M.isFinite(graph[u], graph[v])) {
                    const float w = M.score(graph[u], graph[v]);
                    graph[u].insert(drwnNNGraphEdge(v, w));
                }
            }
        }
    }

    DRWN_FCN_TOC;
}

template<class DistanceMetric>
void drwnNNGraphMoves::update(drwnNNGraph& graph, const DistanceMetric& M)
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

        // run exhaustive search on nodes
        for (set<drwnNNGraphNodeIndex>::const_iterator it = sampledNodes.begin(); it != sampledNodes.end(); ++it) {
            exhaustive(graph, *it, M);
        }
    }

    // random projection
    if (drwnNNGraph::DO_RANDPROJ > 0) {
        randproj(graph, M);
    }

    // propagation moves
    if (drwnNNGraph::DO_PROPAGATE || drwnNNGraph::DO_LOCAL || drwnNNGraph::DO_SEARCH) {
        drwnNNGraphNodeIndex u;
        for (u.imgIndx = 0; u.imgIndx < graph.numImages(); u.imgIndx++) {
            // check if image is in the active set
            if (!graph[u.imgIndx].bSourceMatchable) {
                DRWN_LOG_DEBUG("...skipping " << graph[u.imgIndx].name());
                continue;
            }

            // perform update on each node
            for (u.segId = 0; u.segId < graph[u.imgIndx].numNodes(); u.segId++) {
                if (drwnNNGraph::DO_LOCAL) {
                    local(graph, u, M);
                }
                if (drwnNNGraph::DO_PROPAGATE) {
                    propagate(graph, u, M);
                }
                if (drwnNNGraph::DO_SEARCH) {
                    search(graph, u, M);
                }
            }
        }
    }

    // enrichment (forward and inverse)
    if (drwnNNGraph::DO_ENRICHMENT) {
        enrichment(graph, M);
    }

    DRWN_FCN_TOC;
}

template<class DistanceMetric>
bool drwnNNGraphMoves::propagate(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M)
{
    drwnNNGraphEdgeList& el = graph[u].edges;
    if (el.empty()) return false;

    bool bChanged = false;
    for (drwnNNGraphEdgeList::iterator kt = el.begin(); kt != el.end(); ++kt) {

        // check if match has already been processed in a previous iteration
        if (kt->status == DRWN_NNG_DIRTY) {
            kt->status = DRWN_NNG_PROCESSED_ONCE;
        } else if (kt->status == DRWN_NNG_PROCESSED_ONCE) {
            kt->status = DRWN_NNG_PROCESSED_TWICE;
            continue;
        } else continue;

        // iterate around neighbours of (u,v)
        const drwnNNGraphNodeIndex& v(kt->targetNode);

        for (set<uint16_t>::const_iterator ut = graph[u].spatialNeighbours.begin();
             ut != graph[u].spatialNeighbours.end(); ++ut) {

            // check if this node has edges associated with it
            if (graph[u.imgIndx][*ut].edges.empty()) continue;

            // look at all neighbours of current edge
            for (set<uint16_t>::const_iterator vt = graph[v].spatialNeighbours.begin();
                 vt != graph[v].spatialNeighbours.end(); ++vt) {

                // check that edge is legal
                if (!M.isFinite(graph[u.imgIndx][*ut], graph[v.imgIndx][*vt]))
                    continue;

                // score the edge
                const float w = M.score(graph[u.imgIndx][*ut], graph[v.imgIndx][*vt]);
                const drwnNNGraphEdge e(drwnNNGraphNodeIndex(v.imgIndx, *vt), w);
                if (graph[u.imgIndx][*ut].insert(e)) {
                    bChanged = true;
                }
            }
        }
    }

    return bChanged;
}

template<class DistanceMetric>
bool drwnNNGraphMoves::search(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M)
{
    // construct list of valid target images
    vector<unsigned> validIndexes;
    validIndexes.reserve(graph.numImages() - 1);
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        if (!graph[imgIndx].bTargetMatchable) continue;
        if (!graph.inSameEqvClass(imgIndx, u.imgIndx)) {
            validIndexes.push_back(imgIndx);
        }
    }

    if (validIndexes.empty()) return false;

    // randomly sample a valid segment from a valid image
    drwnNNGraphEdge e;
    e.targetNode.imgIndx = validIndexes[(unsigned)(drand48() * validIndexes.size())];

    // determine valid nodes to match against
    vector<unsigned> segIndexes;
    segIndexes.reserve(graph[e.targetNode.imgIndx].numNodes());
    for (unsigned i = 0; i < graph[e.targetNode.imgIndx].numNodes(); i++) {
        if (M.isFinite(graph[u], graph[e.targetNode.imgIndx][i])) {
            segIndexes.push_back(i);
        }
    }
    if (segIndexes.empty()) return false;

    e.targetNode.segId = (uint16_t)segIndexes[drand48() * segIndexes.size()];
    e.weight = M.score(graph[u], graph[e.targetNode]);
    const bool bChanged = graph[u].insert(e);

    return bChanged;
}

template<class DistanceMetric>
bool drwnNNGraphMoves::local(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M)
{
    drwnNNGraphEdgeList& el = graph[u].edges;

    bool bChanged = false;
    for (drwnNNGraphEdgeList::iterator kt = el.begin(); kt != el.end(); ++kt) {
        if (kt->status != DRWN_NNG_DIRTY) continue;

        const drwnNNGraphNodeIndex v(kt->targetNode);
        for (set<uint16_t>::const_iterator it = graph[v].spatialNeighbours.begin();
             it != graph[v].spatialNeighbours.end(); ++it) {

            if (!M.isFinite(graph[u], graph[v.imgIndx][*it]))
                continue;

            const float w = M.score(graph[u], graph[v.imgIndx][*it]);
            if (w < kt->weight) {
                kt->weight = w;
                kt->targetNode.segId = *it;
                kt->status = DRWN_NNG_DIRTY;
                bChanged = true;
            }
        }
    }

    // sort edges by weight
    if (bChanged) {
        el.sort(drwnNNGraphSortByScore());
    }

    return bChanged;
}

template<class DistanceMetric>
bool drwnNNGraphMoves::randproj(drwnNNGraph& graph, const DistanceMetric& M)
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

        // skip if images are in the same equivalence class or not matchable
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
bool drwnNNGraphMoves::enrichment(drwnNNGraph& graph, const DistanceMetric& M)
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

template<class DistanceMetric>
bool drwnNNGraphMoves::exhaustive(drwnNNGraph& graph, const drwnNNGraphNodeIndex& u, const DistanceMetric& M)
{
    DRWN_FCN_TIC;
    drwnNNGraphEdgeList& el = graph[u].edges;

    bool bChanged = false;
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        if (!graph[imgIndx].bTargetMatchable) continue;
        if (graph.inSameEqvClass(imgIndx, u.imgIndx)) continue;

        drwnNNGraphEdge bestEdge;
        if (!el.empty()) bestEdge = el.back();

        for (unsigned segId = 0; segId < graph[imgIndx].numNodes(); segId++) {

            const drwnNNGraphNodeIndex v(imgIndx, segId);
            if (!M.isFinite(graph[u], graph[v]))
                continue;

            const float w = M.score(graph[u], graph[v]);
            if (w < bestEdge.weight) {
                bestEdge = drwnNNGraphEdge(v, w);
            }
        }

        if (graph[u].insert(bestEdge)) {
            bChanged = true;
        }
    }

    DRWN_FCN_TOC;
    return bChanged;
}
