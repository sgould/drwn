/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphLearn.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

#include "drwnNNGraph.h"
#include "drwnNNGraphLearn.h"
#include "drwnNNGraphMoves.h"
#include "drwnNNGraphThreadedMoves.h"

using namespace std;
using namespace Eigen;

#define USE_THREADING
//#undef USE_THREADING

#define FAST_AND_FAT
//#undef FAST_AND_FAT

//! \todo features are replicated for positive and negative nn graphs which is
//! a memory waste. Try refactor with features external.

// graph update thread --------------------------------------------------------

template<class DistanceMetric>
class drwnNNGraphMoveUpdateThread : public drwnThreadJob {
protected:
    DistanceMetric _M;
    drwnNNGraph *_graph;

public:
    pair<double, double> energy;

public:
    drwnNNGraphMoveUpdateThread(drwnNNGraph *graph) :
        _graph(graph), energy(0.0, 0.0) { /* do nothing */ }
    drwnNNGraphMoveUpdateThread(drwnNNGraph *graph, const DistanceMetric& M) :
        _M(M), _graph(graph), energy(0.0, 0.0) { /* do nothing */ }
    ~drwnNNGraphMoveUpdateThread() { /* do nothing */ }

    void operator()() {
        //drwnNNGraphMoves::update(*_graph, _M);
        drwnNNGraphThreadedMoves::update(*_graph, _M);
        energy = _graph->energy();
    }
};

// data matrix thread ---------------------------------------------------------

class drwnNNGraphDataMatrixThread : public drwnThreadJob {
protected:
    const drwnNNGraphLearner *_learner;
    MatrixXd *_X;
    set<unsigned> _imgIndxes;

public:
    drwnNNGraphDataMatrixThread(const drwnNNGraphLearner *learner, MatrixXd *X, unsigned imgIndx) :
        _learner(learner), _X(X) { _imgIndxes.insert(imgIndx); }
    drwnNNGraphDataMatrixThread(const drwnNNGraphLearner *learner, MatrixXd *X, const set<unsigned>& imgIndxes) :
        _learner(learner), _X(X), _imgIndxes(imgIndxes) { /* do nothing */ }
    ~drwnNNGraphDataMatrixThread() { /* do nothing */ }

    void operator()() {

        const drwnNNGraph& graph = _learner->getSrcGraph();
        const drwnNNGraph& posGraph = _learner->getPosGraph();

        for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
            MatrixXf localX = MatrixXf::Zero(_X->rows(), _X->cols());
            for (unsigned segId = 0; segId < posGraph[*it].numNodes(); segId++) {
                const drwnNNGraphNodeIndex u(*it, segId);
                const drwnNNGraphEdgeList& e = posGraph[u].edges;
                for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                    const drwnNNGraphNodeIndex v(kt->targetNode);
                    const VectorXf delta(graph[u].features - graph[v].features);
                    //localX += delta * delta.transpose();
                    localX.selfadjointView<Eigen::Upper>().rankUpdate(delta);
                }
            }

            lock();
            //(*_X) += localX.cast<double>();
            (*_X) += localX.cast<double>().selfadjointView<Eigen::Upper>();
            unlock();
        }
    }
};

// full subgradient thread ---------------------------------------------------

class drwnNNGraphSubGradientThread : public drwnThreadJob {
protected:
    const drwnNNGraphLearner *_learner;
    MatrixXd *_G;

    // indexed by u = (imgIndx, segIndx); contains {(v, w)}
    drwnNNGraphNodeAnnotation<drwnNNGraphLearnViolatedConstraints> *_lastUpdated;
    set<unsigned> _imgIndxes;

public:
    drwnNNGraphSubGradientThread(const drwnNNGraphLearner *learner, MatrixXd *G,
        drwnNNGraphNodeAnnotation<drwnNNGraphLearnViolatedConstraints> *lastUpdated, unsigned imgIndx) :
        _learner(learner), _G(G), _lastUpdated(lastUpdated) { _imgIndxes.insert(imgIndx); }
    drwnNNGraphSubGradientThread(const drwnNNGraphLearner *learner, MatrixXd *G,
        drwnNNGraphNodeAnnotation<drwnNNGraphLearnViolatedConstraints> *lastUpdated, const set<unsigned>& imgIndxes) :
        _learner(learner), _G(G), _lastUpdated(lastUpdated), _imgIndxes(imgIndxes) { /* do nothing */ }
    ~drwnNNGraphSubGradientThread() { /* do nothing */ }

    void operator()() {
        MatrixXf localG = MatrixXf::Zero(_G->rows(), _G->cols());

        const drwnNNGraph& graph = _learner->getSrcGraph();
        const vector<double>& labelWeights = _learner->getLabelWeights();

        for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
            for (unsigned segId = 0; segId < graph[*it].numNodes(); segId++) {
                const drwnNNGraphNodeIndex u(*it, segId);

                const float alpha = labelWeights.empty() ? 1.0f : (float)labelWeights[graph[u].label];

                const drwnNNGraphEdgeList& ev = _learner->getPosGraph()[u].edges;
                const drwnNNGraphEdgeList& ew = _learner->getNegGraph()[u].edges;

                for (drwnNNGraphEdgeList::const_iterator vt = ev.begin(); vt != ev.end(); ++vt) {
                    for (drwnNNGraphEdgeList::const_iterator wt = ew.begin(); wt != ew.end(); ++wt) {
                        const double xi_uvw = vt->weight - wt->weight + 1.0;

                        drwnNNGraphLearnViolatedConstraints::iterator it =
                            (*_lastUpdated)[u].find(make_pair(vt->targetNode, wt->targetNode));

                        // check if subgradient needs updating
                        if ((xi_uvw <= 0.0) && (it == (*_lastUpdated)[u].end())) continue;
                        if ((xi_uvw > 0.0) && (it != (*_lastUpdated)[u].end())) continue;

                        const VectorXf delta_v(graph[u].features - graph[vt->targetNode].features);
                        const VectorXf delta_w(graph[u].features - graph[wt->targetNode].features);

                        // update subgradient
                        if (xi_uvw <= 0.0) {
                            localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_v, -alpha);
                            localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_w, alpha);
                            (*_lastUpdated)[u].erase(it);
                        } else {
                            localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_v, alpha);
                            localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_w, -alpha);
                            (*_lastUpdated)[u].insert(make_pair(vt->targetNode, wt->targetNode));
                        }
                    }
                }
            }
        }

        lock();
        (*_G) += localG.cast<double>().selfadjointView<Eigen::Upper>();
        unlock();
    }
};

// sparse subgradient thread -------------------------------------------------

class drwnNNGraphSparseSubGradientThread : public drwnThreadJob {
protected:
    const drwnNNGraphLearner *_learner;
    MatrixXd *_G;

    // indexed by u = (imgIndx, segIndx); contains (valid, v, w)
    drwnNNGraphNodeAnnotation<drwnNNGraphLearnSparseViolatedConstraint> *_lastUpdated;

    set<unsigned> _imgIndxes;

public:
    drwnNNGraphSparseSubGradientThread(const drwnNNGraphLearner *learner, MatrixXd *G,
        drwnNNGraphNodeAnnotation<drwnNNGraphLearnSparseViolatedConstraint> *lastUpdated, unsigned imgIndx) :
        _learner(learner), _G(G), _lastUpdated(lastUpdated) { _imgIndxes.insert(imgIndx); }
    drwnNNGraphSparseSubGradientThread(const drwnNNGraphLearner *learner, MatrixXd *G,
        drwnNNGraphNodeAnnotation<drwnNNGraphLearnSparseViolatedConstraint> *lastUpdated, const set<unsigned>& imgIndxes) :
        _learner(learner), _G(G), _lastUpdated(lastUpdated), _imgIndxes(imgIndxes) { /* do nothing */ }
    ~drwnNNGraphSparseSubGradientThread() { /* do nothing */ }

    void operator()() {
        MatrixXf localG = MatrixXf::Zero(_G->rows(), _G->cols());

        const drwnNNGraph& graph = _learner->getSrcGraph();
        const drwnNNGraph& posGraph = _learner->getPosGraph();
        const drwnNNGraph& negGraph = _learner->getNegGraph();
        const vector<double>& labelWeights = _learner->getLabelWeights();

        for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
            for (unsigned segId = 0; segId < graph[*it].numNodes(); segId++) {
                const drwnNNGraphNodeIndex u(*it, segId);
                if (posGraph[u].edges.empty() || negGraph[u].edges.empty()) continue;

                const double xi_u = posGraph[u].edges.back().weight -
                    negGraph[u].edges.front().weight + 1.0;

                // node satisfies constraint
                if (xi_u <= 0.0) {
                    if ((*_lastUpdated)[u].first) {
                        const float alpha = labelWeights.empty() ? 1.0f : (float)labelWeights[graph[u].label];

                        const drwnNNGraphNodeIndex& v_prev((*_lastUpdated)[u].second);
                        const drwnNNGraphNodeIndex& w_prev((*_lastUpdated)[u].third);

                        const VectorXf delta_v_prev(graph[u].features - graph[v_prev].features);
                        const VectorXf delta_w_prev(graph[u].features - graph[w_prev].features);

                        localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_v_prev, -alpha);
                        localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_w_prev, alpha);
                    }
                    (*_lastUpdated)[u].first = false;
                    continue;
                }

                // node does not satisfy constraint
                const drwnNNGraphNodeIndex& v(posGraph[u].edges.back().targetNode);
                const drwnNNGraphNodeIndex& w(negGraph[u].edges.front().targetNode);
                const drwnNNGraphNodeIndex& v_prev((*_lastUpdated)[u].second);
                const drwnNNGraphNodeIndex& w_prev((*_lastUpdated)[u].third);

                if ((*_lastUpdated)[u].first && (v_prev == v) && (w_prev == w)) continue;

                const float alpha = labelWeights.empty() ? 1.0f : (float)labelWeights[graph[u].label];

                // subtract previous violated constraint
                if ((*_lastUpdated)[u].first) {
                    const VectorXf delta_v_prev(graph[u].features - graph[v_prev].features);
                    const VectorXf delta_w_prev(graph[u].features - graph[w_prev].features);
                    localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_v_prev, -alpha);
                    localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_w_prev, alpha);
                }

                // add current violated constraint
                const VectorXf delta_v(graph[u].features - graph[v].features);
                const VectorXf delta_w(graph[u].features - graph[w].features);

                localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_v, alpha);
                localG.selfadjointView<Eigen::Upper>().rankUpdate(delta_w, -alpha);

                // update cache flags
                (*_lastUpdated)[u].first = true;
                (*_lastUpdated)[u].second = v;
                (*_lastUpdated)[u].third = w;
            }
        }

        lock();
        (*_G) += localG.cast<double>().selfadjointView<Eigen::Upper>();
        unlock();
    }
};

// graph project features thread ----------------------------------------------

class drwnNNGraphProjectFeaturesThread : public drwnThreadJob {
protected:
    set<unsigned> _imgIndxes;
    const drwnNNGraph& _srcGraph;
    drwnNNGraph& _posGraph;
    drwnNNGraph& _negGraph;
    const MatrixXf &_Lt;

#ifdef FAST_AND_FAT
    static map<unsigned, MatrixXf> _imgFeatureData;
#endif

public:
    drwnNNGraphProjectFeaturesThread(unsigned imgIndx, const drwnNNGraph& srcGraph,
        drwnNNGraph& posGraph, drwnNNGraph& negGraph, const MatrixXf& Lt) : _srcGraph(srcGraph),
        _posGraph(posGraph), _negGraph(negGraph), _Lt(Lt) { _imgIndxes.insert(imgIndx); }
    drwnNNGraphProjectFeaturesThread(const set<unsigned>& imgIndxes, const drwnNNGraph& srcGraph,
        drwnNNGraph& posGraph, drwnNNGraph& negGraph, const MatrixXf& Lt) : _imgIndxes(imgIndxes),
        _srcGraph(srcGraph), _posGraph(posGraph), _negGraph(negGraph), _Lt(Lt) { /* do nothing */ }
    ~drwnNNGraphProjectFeaturesThread() { /* do nothing */ }

    void operator()() {
#ifdef FAST_AND_FAT
        //! Performing feature transformation on a block results
        //! is about a factor of two speedup. This suggests moving
        //! feature storage from nodes to images. However, the
        //! change will make introduction of different node
        //! types (with different sized feature vectors) difficult.
        //! It also means that the metric functors would need to
        //! be passed a copy of the graph in addition to the nodes.
        for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
            lock();
            map<unsigned, MatrixXf>::const_iterator ft = _imgFeatureData.find(*it);
            if (ft == _imgFeatureData.end()) {
                MatrixXf X(_Lt.cols(), _srcGraph[*it].numNodes());
                for (unsigned segId = 0; segId < _srcGraph[*it].numNodes(); segId++) {
                    X.col(segId) = _srcGraph[*it][segId].features;
                }
                ft = _imgFeatureData.insert(_imgFeatureData.end(), make_pair(*it, X));
            }
            unlock();

            const MatrixXf X = _Lt.triangularView<Eigen::Upper>() * ft->second;
            for (unsigned segId = 0; segId < _srcGraph[*it].numNodes(); segId++) {
                _negGraph[*it][segId].features = _posGraph[*it][segId].features = X.col(segId);
            }
        }
#else
        const TriangularView<MatrixXf, Eigen::Upper> Lt(_Lt);
        for (set<unsigned>::const_iterator it = _imgIndxes.begin(); it != _imgIndxes.end(); ++it) {
            for (unsigned segId = 0; segId < _srcGraph[*it].numNodes(); segId++) {
                _negGraph[*it][segId].features = _posGraph[*it][segId].features = Lt * _srcGraph[*it][segId].features;
            }
        }
#endif
    }
};

#ifdef FAST_AND_FAT
map<unsigned, MatrixXf> drwnNNGraphProjectFeaturesThread::_imgFeatureData;
#endif

// drwnNNGraphLearner --------------------------------------------------------

double drwnNNGraphLearner::ALPHA_ZERO = 5.0e-6;
unsigned drwnNNGraphLearner::METRIC_ITERATIONS = 500;
unsigned drwnNNGraphLearner::SEARCH_ITERATIONS = 20;

drwnNNGraphLearner::drwnNNGraphLearner(const drwnNNGraph& graph, double lambda) :
    _graph(graph), _lambda(lambda), _posGraph(graph), _negGraph(graph), _dim(0)
{
    DRWN_ASSERT(graph.numImages() > 0);
    _dim = graph[0][0].features.rows();
    DRWN_LOG_VERBOSE("initializing metric learner with " << graph.numImages()
        << " images and " << _dim << "-dimensional features");
}

drwnNNGraphLearner::~drwnNNGraphLearner()
{
    // do nothing
}

double drwnNNGraphLearner::learn(unsigned maxCycles)
{
    // re-initialize the graph edges
#if 0
    drwnNNGraphThreadedMoves::initialize(_posGraph, drwnNNGraphLabelsEqualMetric());
    drwnNNGraphThreadedMoves::initialize(_negGraph, drwnNNGraphLabelsNotEqualMetric());
#else
    drwnNNGraphThreadedMoves::initialize(_posGraph, drwnNNGraphLabelsEqualMetricNoUnknown());
    drwnNNGraphThreadedMoves::initialize(_negGraph, drwnNNGraphLabelsNotEqualMetricNoUnknown());
    DRWN_LOG_VERBOSE("...graphs have (+) " << _posGraph.numEdges() << " (-) " << _negGraph.numEdges() << " edges");
#endif

    // iterate for maxCycles
    for (unsigned nCycle = 0; nCycle < maxCycles; nCycle++) {

        // nearest neighbour search
        nearestNeighbourUpdate(nCycle, SEARCH_ITERATIONS);

        // metric optimization
        double obj = computeObjective();
        MatrixXd L_best = getTransform();
        double obj_best = obj;

        MatrixXd G_avg;
        startMetricCycle();
        for (unsigned nIterations = 0; nIterations < METRIC_ITERATIONS; nIterations++) {

            // display progress
            DRWN_LOG_MESSAGE("...learning iteration " << nCycle << "." << nIterations
                << "; objective " << obj);

            // compute gradient
            const MatrixXd G = computeSubGradient();

            // filtered subgradient
            if (nIterations == 0) {
                G_avg = G;
            } else {
                const double beta = 0.5;
                G_avg = beta * G + (1.0 - beta) * G_avg;
            }

            // take subgradient step
            const double alpha = ALPHA_ZERO / (nCycle + sqrt(nIterations + 1.0));
            subGradientStep(G_avg, alpha);

            // update features
            updateGraphFeatures();

            // compute objective and update state
            obj = computeObjective();
            if (obj < obj_best) {
                obj_best = obj;
                L_best = getTransform();
                ALPHA_ZERO *= 1.05;
            } else if (obj > (1.0 + 1.0e-3) * obj_best) {
                ALPHA_ZERO *= 0.5;
                setTransform(L_best);
                obj = obj_best;
            }
        }

        // keep best metric found so far
        if (obj_best < obj) {
            setTransform(L_best);
        }
        endMetricCycle();
    }

    // final objective
    const double obj = computeObjective();
    DRWN_LOG_MESSAGE("final learning objective " << obj);
    return obj;
}

double drwnNNGraphLearner::computeObjective() const
{
    DRWN_FCN_TIC;
    // \lambda \sum_u \alpha(y_u) \sum_{v \in {\cal N}_u^{+}} d_M(u,v)
#if 0
    double obj = 0.0;
    drwnNNGraphNodeIndex u(0, 0);
    for (u.imgIndx = 0; u.imgIndx < _posGraph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < _posGraph[u.imgIndx].numNodes(); u.segId++) {
            const drwnNNGraphEdgeList& e = _posGraph[u].edges;
            for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                obj += kt->weight;
            }
        }
    }
    obj *= _lambda;
#else
    const MatrixXd L = getTransform();
    const MatrixXd M = L.transpose() * L;
    double obj = _lambda * M.cwiseProduct(_X).sum();
#endif

    // \sum_u \alpha(y_u) [\max_{v,w} d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
    // (decomposes and constant cost given the sorted order of edges)
    obj += computeLossFunction();

    DRWN_FCN_TOC;
    return obj;
}

double drwnNNGraphLearner::computeLossFunction() const
{
    double loss = 0.0;

    // \sum_u \alpha(y_u) \sum_{v,w} [d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
    drwnNNGraphNodeIndex u(0, 0);
    for (u.imgIndx = 0; u.imgIndx < _posGraph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < _posGraph[u.imgIndx].numNodes(); u.segId++) {

            const double alpha = _labelWeights.empty() ? 1.0 : _labelWeights[_graph[u].label];
            const drwnNNGraphEdgeList& ev = _posGraph[u].edges;
            const drwnNNGraphEdgeList& ew = _negGraph[u].edges;

            for (drwnNNGraphEdgeList::const_iterator vt = ev.begin(); vt != ev.end(); ++vt) {
                for (drwnNNGraphEdgeList::const_iterator wt = ew.begin(); wt != ew.end(); ++wt) {
                    const double xi_uvw = alpha * (vt->weight - wt->weight + 1.0);
                    loss += std::max(xi_uvw, 0.0);
                }
            }
        }
    }

    return loss;
}

void drwnNNGraphLearner::updateGraphFeatures()
{
    DRWN_FCN_TIC;
    const MatrixXf L = this->getTransform().cast<float>();

#ifndef USE_THREADING
    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        for (unsigned segId = 0; segId < _posGraph[imgIndx].numNodes(); segId++) {
            _posGraph[imgIndx][segId].features = L * _graph[imgIndx][segId].features;
        }
    }

    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        for (unsigned segId = 0; segId < _posGraph[imgIndx].numNodes(); segId++) {
            _negGraph[imgIndx][segId].features = _posGraph[imgIndx][segId].features;
        }
    }
#else
    // prepare thread data
    const unsigned nJobs = std::min((unsigned)_posGraph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphProjectFeaturesThread *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphProjectFeaturesThread(imgIndxes[i], _graph, _posGraph, _negGraph, L);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();
#endif

    drwnNNGraphThreadedMoves::rescore(_posGraph);
    drwnNNGraphThreadedMoves::rescore(_negGraph);
    DRWN_FCN_TOC;
}

void drwnNNGraphLearner::nearestNeighbourUpdate(unsigned nCycle, unsigned maxIterations)
{
    DRWN_FCN_TIC;

#ifndef USE_THREADING
    // nearest neighbour search
    for (unsigned nIterations = 0; nIterations < maxIterations; nIterations++) {

        // perform an update
        drwnNNGraphMoves::update(_posGraph, drwnNNGraphLabelsEqualMetric());
        drwnNNGraphMoves::update(_negGraph, drwnNNGraphLabelsNotEqualMetric());

        // check energy
        pair<double, double> e_pos = _posGraph.energy();
        pair<double, double> e_neg = _negGraph.energy();
        DRWN_LOG_MESSAGE("...search iteration " << nCycle << "." << nIterations
            << "; energy (+) " << e_pos.first << " (-) " << e_neg.first);
    }
#else
    drwnNNGraphMoveUpdateThread<drwnNNGraphLabelsEqualMetric> posThread(&_posGraph, drwnNNGraphLabelsEqualMetric());
    drwnNNGraphMoveUpdateThread<drwnNNGraphLabelsNotEqualMetric> negThread(&_negGraph, drwnNNGraphLabelsNotEqualMetric());

    // threaded nearest neighbour search
    {
        drwnThreadPool threadPool(2);
        for (unsigned nIterations = 0; nIterations < maxIterations; nIterations++) {

            // perform an update
            threadPool.start();
            threadPool.addJob(&posThread);
            threadPool.addJob(&negThread);
            threadPool.finish();

            // report energy
            DRWN_LOG_MESSAGE("...search iteration " << nCycle << "." << nIterations
                << "; energy (+) " << posThread.energy.first << " (-) " << negThread.energy.first);
        }
    }
#endif

    // cache data matrix
    DRWN_LOG_VERBOSE("...caching data matrix of size " << _dim << "-by-" << _dim);
    _X = MatrixXd::Zero(_dim, _dim);
#ifndef USE_THREADING
    drwnNNGraphNodeIndex u(0, 0);
    for (u.imgIndx = 0; u.imgIndx < _posGraph.numImages(); u.imgIndx++) {
        MatrixXf localX = MatrixXf::Zero(_dim, _dim);
        for (u.segId = 0; u.segId < _posGraph[u.imgIndx].numNodes(); u.segId++) {
            const drwnNNGraphEdgeList& e = _posGraph[u].edges;
            for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                const drwnNNGraphNodeIndex v(kt->targetNode);
                const VectorXf delta(_graph[u].features - _graph[v].features);
                localX += delta * delta.transpose();
            }
        }
        _X += localX.cast<double>();
    }
#else
    {
        // prepare thread data
        const unsigned nJobs = std::min((unsigned)_posGraph.numImages(),
            std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
        vector<set<unsigned> > imgIndxes(nJobs);
        for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
            imgIndxes[imgIndx % nJobs].insert(imgIndx);
        }

        // start threads
        drwnThreadPool threadPool(nJobs);
        threadPool.start();
        vector<drwnNNGraphDataMatrixThread *> jobs(nJobs);
        for (unsigned i = 0; i < nJobs; i++) {
            jobs[i] = new drwnNNGraphDataMatrixThread(this, &_X, imgIndxes[i]);
            threadPool.addJob(jobs[i]);
        }
        threadPool.finish();

        for (unsigned i = 0; i < jobs.size(); i++) {
            delete jobs[i];
        }
        jobs.clear();
    }
#endif

    DRWN_FCN_TOC;
}

void drwnNNGraphLearner::startMetricCycle()
{
    // do nothing
}

void drwnNNGraphLearner::endMetricCycle()
{
    // do nothing
}

MatrixXd drwnNNGraphLearner::initializeTransform() const
{
    // accumulate second-order statistics
    drwnSuffStats stats(_dim);
    vector<double> x(_dim);
    drwnNNGraphNodeIndex u(0, 0);
    for (u.imgIndx = 0; u.imgIndx < _graph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < _graph[u.imgIndx].numNodes(); u.segId++) {
            Eigen::Map<VectorXd>(&x[0], x.size()) = _graph[u].features.cast<double>();
            stats.accumulate(x);
        }
    }

    // create diagonal Mahalanobis transform matrix
    MatrixXd L = MatrixXd::Identity(_dim, _dim);
    for (unsigned i = 0; i < _dim; i++) {
        const double mu = stats.sum(i) / (stats.count() + DRWN_DBL_MIN);
        const double sigma = stats.sum2(i, i) / (stats.count() + DRWN_DBL_MIN) - mu * mu;
        if (sigma < DRWN_DBL_MIN) {
            L(i, i) = 0.0;
        } else {
            L(i, i) = 1.0 / sqrt(sigma);
        }
    }

    return (1.0 / (double)_dim) * L;
}

// drwnNNGraphLLearner -------------------------------------------------------

drwnNNGraphMLearner::drwnNNGraphMLearner(const drwnNNGraph& graph, double lambda) :
    drwnNNGraphLearner(graph, lambda)
{
    _updateCache.initialize(graph, drwnNNGraphLearnViolatedConstraints());
    _G = MatrixXd::Zero(_dim, _dim);
    setTransform(initializeTransform());
}

drwnNNGraphMLearner::~drwnNNGraphMLearner()
{
    // do nothing
}

void drwnNNGraphMLearner::setTransform(const MatrixXd& L)
{
    DRWN_ASSERT((L.rows() == (int)_dim) && (L.cols() == (int)_dim));
    _M = L.transpose() * L;
    updateGraphFeatures();
}

MatrixXd drwnNNGraphMLearner::getTransform() const
{
    return _M.llt().matrixU();
}

MatrixXd drwnNNGraphMLearner::computeSubGradient()
{
    DRWN_FCN_TIC;

    // gradient of \lambda \sum_u \alpha(y_u) \sum_{v \in {\cal N}_u^{+}} d_M(u,v)
    MatrixXd G = _lambda * _X;

    // subgradient of \sum_u \alpha(y_u) \sum_{v,w} [d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
    // prepare thread data
    const unsigned nJobs = std::min((unsigned)_posGraph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphSubGradientThread *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphSubGradientThread(this, &_G, &_updateCache, imgIndxes[i]);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();

    DRWN_FCN_TOC;
    return G + _G;
}

void drwnNNGraphMLearner::subGradientStep(const MatrixXd& G, double alpha)
{
    DRWN_FCN_TIC;

    // gradient step
    _M -= alpha * G;

    // project onto psd
    SelfAdjointEigenSolver<MatrixXd> es;
    es.compute(_M);

    const VectorXd d = es.eigenvalues().real();
    if ((d.array() < 0.0).any()) {
        const MatrixXd V = es.eigenvectors();
        _M = V * d.cwiseMax(VectorXd::Constant(d.rows(), DRWN_EPSILON)).asDiagonal() * V.inverse();
    }

    DRWN_FCN_TOC;
}

void drwnNNGraphMLearner::startMetricCycle()
{
    _updateCache.reset(drwnNNGraphLearnViolatedConstraints());
    _G = MatrixXd::Zero(_dim, _dim);
}

// drwnNNGraphLLearner -------------------------------------------------------

drwnNNGraphLLearner::drwnNNGraphLLearner(const drwnNNGraph& graph, double lambda) :
    drwnNNGraphLearner(graph, lambda)
{
    _updateCache.initialize(graph, drwnNNGraphLearnViolatedConstraints());
    _G = MatrixXd::Zero(_dim, _dim);
    setTransform(initializeTransform());
}

drwnNNGraphLLearner::~drwnNNGraphLLearner()
{
    // do nothing
}

void drwnNNGraphLLearner::setTransform(const MatrixXd& Lt)
{
    DRWN_ASSERT((Lt.rows() == (int)_dim) && (Lt.cols() == (int)_dim));
    _Lt = Lt.triangularView<Eigen::Upper>();
    updateGraphFeatures();
}

MatrixXd drwnNNGraphLLearner::getTransform() const
{
    return _Lt;
}

MatrixXd drwnNNGraphLLearner::computeSubGradient()
{
    DRWN_FCN_TIC;
    MatrixXd G = MatrixXd::Zero(_Lt.rows(), _Lt.cols());

    // gradient of \lambda \sum_u \alpha(y_u) \sum_{v \in {\cal N}_u^{+}} d_M(u,v)
    G = _lambda * _X;

    // subgradient of \sum_u \alpha(y_u) \sum_{v,w} [d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
    // prepare thread data
    const unsigned nJobs = std::min((unsigned)_posGraph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphSubGradientThread *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphSubGradientThread(this, &_G, &_updateCache, imgIndxes[i]);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();

    G += _G;

    DRWN_FCN_TOC;
    return (2.0 * _Lt * G).triangularView<Eigen::Upper>();
}

void drwnNNGraphLLearner::subGradientStep(const MatrixXd& G, double alpha)
{
    _Lt -= alpha * G;
}

void drwnNNGraphLLearner::startMetricCycle()
{
    _updateCache.reset(drwnNNGraphLearnViolatedConstraints());
    _G = MatrixXd::Zero(_dim, _dim);
}

// drwnNNGraphSparseLearner --------------------------------------------------

drwnNNGraphSparseLearner::drwnNNGraphSparseLearner(const drwnNNGraph& graph, double lambda) :
    drwnNNGraphLearner(graph, lambda)
{
    // do nothing
}

drwnNNGraphSparseLearner::~drwnNNGraphSparseLearner()
{
    // do nothing
}

double drwnNNGraphSparseLearner::computeLossFunction() const
{
    double loss = 0.0;

    // \sum_u \alpha(y_u) [\max_{v,w} d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
    // (decomposes and constant cost given the sorted order of edges)
    drwnNNGraphNodeIndex u(0, 0);
    for (u.imgIndx = 0; u.imgIndx < _posGraph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < _posGraph[u.imgIndx].numNodes(); u.segId++) {
            if (_posGraph[u].edges.empty() || _negGraph[u].edges.empty())
                continue;

            const double alpha = _labelWeights.empty() ? 1.0 : _labelWeights[_graph[u].label];
            const double xi_u = alpha * (_posGraph[u].edges.back().weight -
                _negGraph[u].edges.front().weight + 1.0);
            loss += std::max(xi_u, 0.0);
        }
    }

    return loss;
}

// drwnNNGraphLSparseLearner -------------------------------------------------

drwnNNGraphLSparseLearner::drwnNNGraphLSparseLearner(const drwnNNGraph& graph, double lambda) :
    drwnNNGraphSparseLearner(graph, lambda)
{
    _updateCache.initialize(graph, drwnNNGraphLearnSparseViolatedConstraint(false, drwnNNGraphNodeIndex(), drwnNNGraphNodeIndex()));
    _G = MatrixXd::Zero(_dim, _dim);
    setTransform(initializeTransform());
}

drwnNNGraphLSparseLearner::~drwnNNGraphLSparseLearner()
{
    // do nothing
}

void drwnNNGraphLSparseLearner::setTransform(const MatrixXd& Lt)
{
    DRWN_ASSERT((Lt.rows() == (int)_dim) && (Lt.cols() == (int)_dim));
    _Lt = Lt;
    updateGraphFeatures();
}

MatrixXd drwnNNGraphLSparseLearner::getTransform() const
{
    return _Lt;
}

MatrixXd drwnNNGraphLSparseLearner::computeSubGradient()
{
    DRWN_FCN_TIC;
    MatrixXd G = MatrixXd::Zero(_Lt.rows(), _Lt.cols());

    // gradient of \lambda \sum_u \alpha(y_u) \sum_{v \in {\cal N}_u^{+}} d_M(u,v)
    drwnNNGraphNodeIndex u(0, 0);
#if 0
    for (u.imgIndx = 0; u.imgIndx < _posGraph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < _posGraph[u.imgIndx].numNodes(); u.segId++) {
            const drwnNNGraphEdgeList& e = _posGraph[u].edges;
            for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                const drwnNNGraphNodeIndex v(kt->targetNode);
                const VectorXd delta((_graph[u].features - _graph[v].features).cast<double>());
                G += delta * delta.transpose();
            }
        }
    }
    G *= _lambda;
#else
    G = _lambda * _X;
#endif

    // subgradient of \sum_u \alpha(y_u) [\max_{v,w} d_M(u,v) - d_M(u,w) + 1]_{\geq 0}
#ifndef USE_THREADING
    DRWN_TODO;
#else
    // prepare thread data
    const unsigned nJobs = std::min((unsigned)_posGraph.numImages(),
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    vector<set<unsigned> > imgIndxes(nJobs);
    for (unsigned imgIndx = 0; imgIndx < _posGraph.numImages(); imgIndx++) {
        imgIndxes[imgIndx % nJobs].insert(imgIndx);
    }

    // start threads
    drwnThreadPool threadPool(nJobs);
    threadPool.start();
    vector<drwnNNGraphSparseSubGradientThread *> jobs(nJobs);
    for (unsigned i = 0; i < nJobs; i++) {
        jobs[i] = new drwnNNGraphSparseSubGradientThread(this, &_G, &_updateCache, imgIndxes[i]);
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();
#endif

    G += _G;

    DRWN_FCN_TOC;
    //return 2.0 * _Lt * G;
    return (2.0 * _Lt * G).triangularView<Eigen::Upper>();
}

void drwnNNGraphLSparseLearner::subGradientStep(const MatrixXd& G, double alpha)
{
    _Lt -= alpha * G;
}

void drwnNNGraphLSparseLearner::startMetricCycle()
{
    _updateCache.reset(drwnNNGraphLearnSparseViolatedConstraint(false, drwnNNGraphNodeIndex(), drwnNNGraphNodeIndex()));
    _G = MatrixXd::Zero(_dim, _dim);
}

// drwnNNGraphLearnerConfig -------------------------------------------------
//! \addtogroup drwnConfigSettings
//! \section drwnNNGraphLearner
//! \b alpha0          :: initial step size\n
//! \b metricIters     :: metric learning iterations\n
//! \b searchIters     :: search iterations during learning\n

class drwnNNGraphLearnerConfig : public drwnConfigurableModule {
public:
    drwnNNGraphLearnerConfig() : drwnConfigurableModule("drwnNNGraphLearner") { }
    ~drwnNNGraphLearnerConfig() { }

    void usage(ostream &os) const {
        os << "      alpha0           :: initial step size (default: "
           << drwnNNGraphLearner::ALPHA_ZERO << ")\n";
        os << "      metricIters      :: metric learning iterations (default: "
           << drwnNNGraphLearner::METRIC_ITERATIONS << ")\n";
        os << "      searchIters      :: search iterations during learning (default: "
           << drwnNNGraphLearner::SEARCH_ITERATIONS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "alpha0")) {
            drwnNNGraphLearner::ALPHA_ZERO = std::max(atof(value), DRWN_DBL_MIN);
        } else if (!strcmp(name, "metricIters")) {
            drwnNNGraphLearner::METRIC_ITERATIONS = atoi(value);
        } else if (!strcmp(name, "searchIters")) {
            drwnNNGraphLearner::SEARCH_ITERATIONS = atoi(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnNNGraphLearnerConfig gNNGraphLearnerConfig;
