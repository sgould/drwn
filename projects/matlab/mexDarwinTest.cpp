/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexDarwinTest.cpp
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
    mexPrintf("USAGE: mexDarwinTest(data, [options]);\n");
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

    if ((nrhs != 1) && (nrhs != 2)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    options[string("verbose")] = string("0");
    options[string("profile")] = string("0");
    if (nrhs == 2) {
        drwnMatlabUtils::parseOptions(prhs[1], options);
    }

    if (atoi(options[string("verbose")].c_str()) != 0) {
        drwnLogger::setLogLevel(DRWN_LL_VERBOSE);
    }
    drwnCodeProfiler::enabled = (atoi(options[string("profile")].c_str()) != 0);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // show options
    for (map<string, string>::const_iterator it = options.begin(); it != options.end(); ++it) {
        DRWN_LOG_VERBOSE(it->first << " = " << it->second);
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
