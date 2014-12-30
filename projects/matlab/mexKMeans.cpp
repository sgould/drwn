/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexKMeans.cpp
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

// matlab headers
#include "mex.h"
#include "matrix.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: centroids = mexKMeans(k, features, [weights, [options]]);\n");
    mexPrintf("  features :: N-by-D matrix of (transposed) feature vectors\n");
    mexPrintf("  weights  :: N-vector of non-negative weights (or empty)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("  maxiters :: maximum number of iterations (default: infinity)\n");
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs == 0) || (nlhs > 1)) {
        usage();
        return;
    }

    if ((nrhs < 2) || (nrhs > 4)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    DRWN_ASSERT_MSG((nrhs < 3) || (mxIsEmpty(prhs[2])) ||
        (mxGetM(prhs[2]) == mxGetM(prhs[1])), "mismatch between features and weights");

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    options[string("maxiters")] = string("-1");
    if (nrhs == 4) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    const int maxIterations = atoi(options[string("maxiters")].c_str());
    if (maxIterations >= 0) drwnKMeans::MAX_ITERATIONS = maxIterations; 

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse k, features and weights
    const int K = mxGetScalar(prhs[0]);
    DRWN_ASSERT_MSG(K > 0, "k must be positive");

    vector<vector<double> > features;
    drwnMatlabUtils::mxArrayToVector(prhs[1], features);

    vector<double> weights;
    if ((nrhs > 2) && (!mxIsEmpty(prhs[2]))) {
        weights.resize(features.size(), 0.0);
        const double *p = mxGetPr(prhs[2]);
        for (unsigned i = 0; i < weights.size(); i++) {
            weights[i] = p[i];
        }
    }

    // run k-means
    drwnKMeans clusters(K);
    if (weights.empty()) {
        clusters.train(features);
    } else {
        clusters.train(features, weights);
    }

    // return centroids
    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleMatrix(K, clusters.numFeatures(), mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (int col = 0; col < clusters.numFeatures(); col++) {
            for (int row = 0; row < K; row++) {
                *p++ = clusters.getCentroids()(row, col);
            }
        }
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
