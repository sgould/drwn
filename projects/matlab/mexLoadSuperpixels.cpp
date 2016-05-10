/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexLoadSuperpixels.cpp
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
#include "drwnVision.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

// main -----------------------------------------------------------------------

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\nDESCRIPITON:\n");
    mexPrintf("  Loads superpixels from a previously saved drwnSuperpixelContainer object.\n");
    mexPrintf("USAGE: sp = mexLoadSuperpixels(filename, [options]);\n");
    mexPrintf("  filename :: filename containing the superpixels\n");
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

    char *filename = mxArrayToString(prhs[0]);
    DRWN_LOG_VERBOSE("Loading superpixels from " << filename << "...");

    // load superpixels
    drwnSuperpixelContainer container;
    if (drwnFileExists(filename)) {
        ifstream ifs(filename, ios::binary);
        container.read(ifs);
        ifs.close();
    } else {
        DRWN_LOG_WARNING("file " << filename << " does not exist");
    }

    // return superpixels (as 1-based)
    if ((nlhs == 1) && (!container.empty())) {
        mwSize dims[3];
        dims[0] = container.height();
        dims[1] = container.width();
        dims[2] = container.channels();

        plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double *px = mxGetPr(plhs[0]);

        for (int c = 0; c < container.channels(); c++) {
            for (int x = 0; x < container.width(); x++) {
                for (int y = 0; y < container.height(); y++) {
                    *px++ = (double)container[c].at<int>(y, x) + 1;
                }
            }
        }
    }

    // clean up and print profile information
    mxFree(filename);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
