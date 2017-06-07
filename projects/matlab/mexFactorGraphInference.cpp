/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexFactorGraphInference.cpp
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
    mexPrintf("USAGE: assignment = mexFactorGraphInference(universe, factors, [edges, [options]]);\n");
    mexPrintf("  universe :: n-vector of variable cardinalities\n");
    mexPrintf("  factors  :: structure array of factors (.vars and .data)\n");
    mexPrintf("  edges    :: m-by-2 array of edges between factors\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("  method  :: inference method (default: drwnAsyncMaxProdInference)\n");
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

    if ((nrhs < 2) && (nrhs > 4)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    options[string("method")] = string("drwnAsyncMaxProdInference");
    if (nrhs == 4) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse universe
    drwnVarUniversePtr universe(new drwnVarUniverse());
    for (int i = 0; i < mxGetNumberOfElements(prhs[0]); i++) {
        universe->addVariable((int)mxGetPr(prhs[0])[i]);
    }

    // parse factor graph
    drwnFactorGraph graph(universe);
    for (int i = 0; i < mxGetNumberOfElements(prhs[1]); i++) {
        graph.addFactor(drwnMatlabUtils::parseFactor(universe, prhs[1], i));
    }
    
    // parse edges
    if ((nrhs > 2) && (!mxIsEmpty(prhs[2]))) {
        const int m = mxGetNumberOfElements(prhs[2]);
        if (m % 2 != 0) {
            DRWN_LOG_ERROR("edges must be m-by-2 array");
            return;
        }
        set<drwnEdge> edges;
        for (int i = 0; i < m / 2; i++) {
            edges.insert(drwnEdge(mxGetPr(prhs[2])[i], mxGetPr(prhs[2])[i + m / 2]));            
        }
        graph.connectGraph(edges);
    } else {
        graph.connectGraph();
    }

    if (drwnLogger::getLogLevel() >= DRWN_LL_DEBUG) {
        graph.dump();
    }

    // run MAP inference
    DRWN_LOG_DEBUG("using inference method " << options[string("method")]);
    double e = drwnFactorGraphUtils::removeUniformFactors(graph);
    drwnMAPInference *inf = drwnMAPInferenceFactory::get().create(options[string("method")].c_str(), graph);
    if (inf == NULL) {
        DRWN_LOG_ERROR("unknown inference method " << options[string("method")]);
        DRWN_LOG_ERROR("options are: " << toString(drwnMAPInferenceFactory::get().getRegisteredClasses()));
        return;
    }

    drwnFullAssignment assignment;
    e += inf->inference(assignment).first;
    DRWN_LOG_MESSAGE("map assignment has energy " << e);
    DRWN_LOG_VERBOSE("map assignment is " << toString(assignment));
    DRWN_ASSERT(universe->numVariables() == assignment.size());
    delete inf;

    // extract MAP assignment
    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleMatrix(assignment.size(), 1, mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (unsigned i = 0; i < assignment.size(); i++) {
            p[i] = (double)assignment[i];
        }
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
