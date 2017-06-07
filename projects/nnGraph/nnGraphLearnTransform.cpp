/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphLearnTransform.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// eigen matrix library headers
#include "Eigen/Core"
#include "Eigen/Eigenvalues"

// opencv library headers
#include "cv.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

#include "drwnNNGraph.h"
#include "drwnNNGraphLearn.h"

using namespace std;
using namespace Eigen;

// utility functions ---------------------------------------------------------

drwnSuffStats computeSufficientStatistics(const drwnNNGraph& graph)
{
    drwnNNGraphNodeIndex u(0, 0);
    const int dim = graph[u].features.rows();
    drwnSuffStats stats(dim, DRWN_PSS_FULL);
    vector<double> x(dim);
    for (u.imgIndx = 0; u.imgIndx < graph.numImages(); u.imgIndx++) {
        for (u.segId = 0; u.segId < graph[u.imgIndx].numNodes(); u.segId++) {
            Eigen::Map<VectorXd>(&x[0], x.size()) = graph[u].features.cast<double>();
            stats.accumulate(x);
        }
    }

    return stats;
}

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphLearnTransform [OPTIONS] <graphFile>\n";
    cerr << "OPTIONS:\n"
         << "  -t <type>         :: transform type (PCA, Whitener, Mahalanobis, LMNNSparse, LMNN (default), LMNN-M)\n"
         << "  -m <iterations>   :: maximum learning iterations (default: 1)\n"
         << "  -i <filename>     :: initialize from previously trained tranform\n"
         << "  -o <filename>     :: output transform\n"
         << "  -labelWeighting   :: weight nodes by inverse of class occurrence\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *xformType = "LMNN";
    int maxIterations = 1;
    const char *inXformFile = NULL;
    const char *outXformFile = NULL;
    bool bLabelWeighting = false;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-t", xformType)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-i", inXformFile)
        DRWN_CMDLINE_STR_OPTION("-o", outXformFile)
        DRWN_CMDLINE_BOOL_OPTION("-labelWeighting", bLabelWeighting)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

    // initialize dataset
    const char *inGraphFile = DRWN_CMDLINE_ARGV[0];
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << inGraphFile << "...");

    drwnNNGraph graph;
    graph.read(inGraphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("..." << graph.numNodesWithLabel(-1) << " labeled as unknown");
    DRWN_LOG_VERBOSE("..." << graph[0][0].features.size() << "-dimensional features");

    // compute label weighting
    vector<double> labelWeights;
    if (bLabelWeighting) {
        drwnNNGraphNodeIndex u(0, 0);
        for (u.imgIndx = 0; u.imgIndx < graph.numImages(); u.imgIndx++) {
            for (u.segId = 0; u.segId < graph[u.imgIndx].numNodes(); u.segId++) {
                if (graph[u].label < 0) continue;
                if (labelWeights.size() <= (size_t)graph[u].label) {
                    labelWeights.resize(graph[u].label + 1, 0.0);
                }
                labelWeights[graph[u].label] += 1.0;
            }
        }

        double maxWeight = 0.0;
        for (unsigned i = 0; i < labelWeights.size(); i++) {
            labelWeights[i] = 1.0 / (labelWeights[i] + 1.0);
            maxWeight = std::max(maxWeight, labelWeights[i]);
        }

        for (unsigned i = 0; i < labelWeights.size(); i++) {
            labelWeights[i] /= maxWeight;
        }
    }

    // learning
    DRWN_LOG_MESSAGE("Learning feature transform " << xformType << "...");
    drwnFeatureTransform *xform = NULL;

    if ((strcasecmp(xformType, "LMNN") == 0) ||
        (strcasecmp(xformType, "LMNN-M") == 0) ||
        (strcasecmp(xformType, "LMNNSparse") == 0)) {

        drwnNNGraphLearner *learner = NULL;
        if (strcasecmp(xformType, "LMNN") == 0) {
            learner = new drwnNNGraphLLearner(graph, 1.0e-6);
        } else if (strcasecmp(xformType, "LMNN-M") == 0) {
            learner = new drwnNNGraphMLearner(graph, 1.0e-6);
        } else {
            learner = new drwnNNGraphLSparseLearner(graph, 1.0e-6);
        }
        DRWN_ASSERT(learner != NULL);

        // set weights for each label
        if (bLabelWeighting) {
            learner->setLabelWeights(labelWeights);
        }

        // load from initialization file
        if (inXformFile != NULL) {
            DRWN_LOG_VERBOSE("...loading initial transform from " << inXformFile);
            drwnLinearTransform initXform;
            initXform.read(inXformFile);
            learner->setTransform(initXform.projection());
        }

        // learn the transform
        learner->learn(maxIterations);

        // save the transform
        const MatrixXd L = learner->getTransform();
        xform = new drwnLinearTransform(VectorXd::Zero(L.rows()), L);

        delete learner;

    } else if (strcasecmp(xformType, "PCA") == 0) {
        const drwnSuffStats stats = computeSufficientStatistics(graph);
        xform = new drwnPCA(stats, 0.999);
        DRWN_LOG_VERBOSE("...maps from " << ((drwnPCA *)xform)->numInputs() << " dimensions to "
            << ((drwnPCA *)xform)->numOutputs() << " dimensions");

    } else if (strcasecmp(xformType, "Whitener") == 0) {
        const drwnSuffStats stats = computeSufficientStatistics(graph);
        xform = new drwnFeatureWhitener(stats);

    } else if (strcasecmp(xformType, "Mahalanobis") == 0) {
        const drwnSuffStats stats = computeSufficientStatistics(graph);
        const VectorXd mu = stats.firstMoments() / stats.count();
        const MatrixXd Sigma = stats.secondMoments() / stats.count() -
            mu * mu.transpose() + 1.0e-9 * MatrixXd::Identity(stats.size(), stats.size());
        const MatrixXd L(Sigma.llt().matrixL());
        xform = new drwnLinearTransform(mu, L.inverse());
    }
    DRWN_ASSERT_MSG(xform != NULL, "unknown transform " << xformType);

    // save the transform
    if (outXformFile) {
        xform->write(outXformFile);
    }

    // clean up
    delete xform;
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
