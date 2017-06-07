/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexGetLinearTransform.cpp
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
    mexPrintf("USAGE: [translation, projection] = mexGetLinearTransform(xmlFile, [options]);\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs != 1) && (nrhs != 2)) {
        usage();
        return;
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 2) {
        drwnMatlabUtils::parseOptions(prhs[1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // load feature transform
    char *xmlFilename = mxArrayToString(prhs[0]);
    DRWN_LOG_VERBOSE("Loading transform from " << xmlFilename << "...");

    drwnLinearTransform xform;
    xform.read(xmlFilename);

    // return translation and projection
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleMatrix(xform.numInputs(), 1, mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (int row = 0; row < xform.numInputs(); row++) {
            *p++ = xform.translation()(row);
        }
    }

    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(xform.numOutputs(), xform.numInputs(), mxREAL);
        double *p = mxGetPr(plhs[1]);
        for (int col = 0; col < xform.numInputs(); col++) {
            for (int row = 0; row < xform.numOutputs(); row++) {
                *p++ = xform.projection()(row, col);
            }
        }
    }

    // clean up and print profile information
    mxFree(xmlFilename);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
