/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexSetLinearTransform.cpp
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
    mexPrintf("USAGE: mexSetLinearTransform(xmlFile, translation, projection, [options]);\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs != 3) && (nrhs != 4)) {
        usage();
        return;
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 4) {
        drwnMatlabUtils::parseOptions(prhs[3], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // set parameters
    VectorXd translation;
    MatrixXd projection;
    drwnMatlabUtils::mxArrayToEigen(prhs[1], translation);
    drwnMatlabUtils::mxArrayToEigen(prhs[2], projection);

    drwnLinearTransform xform;
    xform.set(translation, projection);

    // save feature transform
    char *xmlFilename = mxArrayToString(prhs[0]);
    DRWN_LOG_VERBOSE("Saving transform to " << xmlFilename << "...");
    xform.write(xmlFilename);

    // clean up and print profile information
    mxFree(xmlFilename);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
