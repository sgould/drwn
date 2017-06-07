/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexMaxFlow.cpp
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
#include "drwnPGM.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: [value, cut] = mexMaxFlow(edgeList, [options]);\n");
    mexPrintf("  edgeList :: n-by-3 array of weighted directed edges\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if (nrhs == 0) {
        usage();
        return;
    }

    if ((nrhs < 1) && (nrhs > 2)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 2) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse edges to construct max flow graph
    DRWN_LOG_DEBUG("Building max-flow graph...");
    drwnBKMaxFlow maxFlowGraph;
    const int m = mxGetNumberOfElements(prhs[0]);
    if (m % 3 != 0) {
        DRWN_LOG_ERROR("edgeList must be an n-by-3 array");
        return;
    }
    const int headOffset = 0;
    const int tailOffset = m / 3;
    const int weightOffset = 2 * m / 3;
    for (int i = 0; i < m / 3; i++) {
        const int u = (int)mxGetPr(prhs[0])[i + headOffset];
        const int v = (int)mxGetPr(prhs[0])[i + tailOffset];
        const double w = mxGetPr(prhs[0])[i + weightOffset];
        DRWN_LOG_DEBUG("...adding edge (" << u << ", " << v << ") with weight " << w);
        if (std::max(u, v) >= maxFlowGraph.numNodes()) {
            maxFlowGraph.addNodes(std::max(u, v) - maxFlowGraph.numNodes() + 1);
        }

        if ((u < 0) && (v < 0)) {
            maxFlowGraph.addConstant(w);
        } else if (u < 0) {
            maxFlowGraph.addSourceEdge(v, w);
        } else if (v < 0) {
            maxFlowGraph.addTargetEdge(u, w);
        } else {
            maxFlowGraph.addEdge(u, v, w);
        }
    }

    // run max-flow/min-cut
    DRWN_LOG_DEBUG("Finding minimum cut...");
    const double value = maxFlowGraph.solve();
    DRWN_LOG_DEBUG("...value is " << value);

    // extract value
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
        mxGetPr(plhs[0])[0] = value;
    }

    // extract cut
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(maxFlowGraph.numNodes(), 1, mxREAL);
        double *p = mxGetPr(plhs[1]);
        for (unsigned i = 0; i < maxFlowGraph.numNodes(); i++) {
            p[i] = maxFlowGraph.inSetS(i) ? 0.0 : 1.0;
        }
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
