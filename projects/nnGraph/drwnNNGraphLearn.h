/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphLearn.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include "drwnNNGraph.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphLearner --------------------------------------------------------
//! Learn the distance metric base class with full set of constraints (i.e.,
//! loss function over all targets and imposters).

class drwnNNGraphLearner {
 public:
    static double ALPHA_ZERO;
    static unsigned METRIC_ITERATIONS;
    static unsigned SEARCH_ITERATIONS;

 protected:
    const drwnNNGraph& _graph;
    double _lambda;
    vector<double> _labelWeights;

    drwnNNGraph _posGraph;
    drwnNNGraph _negGraph;

    unsigned _dim;              //!< feature vector dimensions
    MatrixXd _X;                //!< cached (x_u - x_v) (x_u - x_v)^T

 public:
    drwnNNGraphLearner(const drwnNNGraph& graph, double lambda);
    virtual ~drwnNNGraphLearner();

    virtual void setTransform(const MatrixXd& Lt) = 0;
    virtual MatrixXd getTransform() const = 0;
    virtual double learn(unsigned maxCycles);

    const drwnNNGraph& getSrcGraph() const { return _graph; }
    const drwnNNGraph& getPosGraph() const { return _posGraph; }
    const drwnNNGraph& getNegGraph() const { return _negGraph; }

    void clearLabelWeights() { _labelWeights.clear(); }
    void setLabelWeights(const vector<double>& w) {
        DRWN_LOG_VERBOSE("...setting label weights to " << toString(w));
        _labelWeights = w;
    };
    const vector<double>& getLabelWeights() const { return _labelWeights; }

 protected:
    virtual double computeObjective() const;
    virtual double computeLossFunction() const;
    virtual MatrixXd computeSubGradient() = 0;
    virtual void subGradientStep(const MatrixXd& G, double alpha) = 0;

    virtual void startMetricCycle();
    virtual void endMetricCycle();

    void updateGraphFeatures();
    void nearestNeighbourUpdate(unsigned nCycle, unsigned maxIterations);

    //! initialize transform as feature whitener (diagonal Mahalanobis)
    MatrixXd initializeTransform() const;
};

// drwnNNGraphMLearner -------------------------------------------------------
//! Learn the distance metric M = LL^T as M.

typedef set<pair<drwnNNGraphNodeIndex, drwnNNGraphNodeIndex> > drwnNNGraphLearnViolatedConstraints;

class drwnNNGraphMLearner : public drwnNNGraphLearner {
 protected:
    MatrixXd _M;

    MatrixXd _G;
    drwnNNGraphNodeAnnotation<drwnNNGraphLearnViolatedConstraints> _updateCache;

 public:
    drwnNNGraphMLearner(const drwnNNGraph& graph, double lambda);
    ~drwnNNGraphMLearner();

    void setTransform(const MatrixXd& Lt);
    MatrixXd getTransform() const;

 protected:
    MatrixXd computeSubGradient();
    void subGradientStep(const MatrixXd& G, double alpha);

    void startMetricCycle();
};

// drwnNNGraphLLearner -------------------------------------------------------
//! Learn the distance metric M = LL^T as L^T.

class drwnNNGraphLLearner : public drwnNNGraphLearner {
 protected:
    MatrixXd _Lt;

    MatrixXd _G;
    drwnNNGraphNodeAnnotation<drwnNNGraphLearnViolatedConstraints> _updateCache;

 public:
    drwnNNGraphLLearner(const drwnNNGraph& graph, double lambda);
    ~drwnNNGraphLLearner();

    void setTransform(const MatrixXd& Lt);
    MatrixXd getTransform() const;

 protected:
    MatrixXd computeSubGradient();
    void subGradientStep(const MatrixXd& G, double alpha);

    void startMetricCycle();
};

// drwnNNGraphSparseLearner --------------------------------------------------
//! Learn the distance metric base class with sparse set of constraints (i.e.,
//! loss function over further target and nearest imposter only).

class drwnNNGraphSparseLearner : public drwnNNGraphLearner {
 public:
    drwnNNGraphSparseLearner(const drwnNNGraph& graph, double lambda);
    ~drwnNNGraphSparseLearner();

 protected:
    virtual double computeLossFunction() const;
};

// drwnNNGraphLFastLearner ---------------------------------------------------
//! Learn the distance metric M = LL^T as L^T.

typedef drwnTriplet<bool, drwnNNGraphNodeIndex, drwnNNGraphNodeIndex>
    drwnNNGraphLearnSparseViolatedConstraint;

class drwnNNGraphLSparseLearner : public drwnNNGraphSparseLearner {
 protected:
    MatrixXd _Lt;

    MatrixXd _G;
    drwnNNGraphNodeAnnotation<drwnNNGraphLearnSparseViolatedConstraint> _updateCache;

 public:
    drwnNNGraphLSparseLearner(const drwnNNGraph& graph, double lambda);
    ~drwnNNGraphLSparseLearner();

    void setTransform(const MatrixXd& Lt);
    MatrixXd getTransform() const;

 protected:
    MatrixXd computeSubGradient();
    void subGradientStep(const MatrixXd& G, double alpha);

    void startMetricCycle();
};
