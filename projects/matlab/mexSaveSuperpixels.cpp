/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexSaveSuperpixels.cpp
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

// matrix parser --------------------------------------------------------------

template<typename T>
void drwnParseMexSuperpixels(const mxArray *array, drwnSuperpixelContainer& container)
{
    const int nRows = mxGetDimensions(array)[0]; // number of rows
    const int nCols = mxGetDimensions(array)[1]; // number of columns
    const int nChannels = mxGetDimensions(array)[2]; // number of channels
    DRWN_LOG_DEBUG("parsing a " << nRows << "-by-" << nCols << "-by-" << nChannels << " matrix...");

    T *px = (T *)mxGetPr(array);
    for (int c = 0; c < nChannels; c++) {
        cv::Mat m(nRows, nCols, CV_32SC1);
        for (int x = 0; x < nCols; x++) {
            for (int y = 0; y < nRows; y++) {
                m.at<int>(y, x) = (int)(*px++) - 1;
            }
        }
        container.addSuperpixels(m);
    }
}

// main -----------------------------------------------------------------------

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\nDESCRIPITON:\n");
    mexPrintf("  Saves superpixel maps to a drwnSuperpixelContainer object on disk.\n");
    mexPrintf("USAGE: mexSaveSuperpixels(filename, sp, [options]);\n");
    mexPrintf("  filename :: filename to write superpixels to\n");
    mexPrintf("  sp       :: H-by-W-by-C superpixel maps (1-based)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs != 2) && (nrhs != 3)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 3) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // add superpixles to container
    drwnSuperpixelContainer container;
    DRWN_ASSERT_MSG(mxGetNumberOfDimensions(prhs[1]) == 3, "expecting multi-channel superpixel matrix");

    switch (mxGetClassID(prhs[1])) {
    case mxDOUBLE_CLASS:
        drwnParseMexSuperpixels<double>(prhs[1], container);
        break;
    case mxUINT8_CLASS:
        drwnParseMexSuperpixels<uint8_T>(prhs[1], container);
        break;
    case mxINT32_CLASS:
        drwnParseMexSuperpixels<int32_T>(prhs[1], container);
        break;
    default:
        DRWN_LOG_FATAL("unexpected superpixel matrix type, try 'sp = double(sp);'");
    }

    // save the superpixel container
    char *filename = mxArrayToString(prhs[0]);
    DRWN_LOG_VERBOSE("Saving superpixels to " << filename << "...");

    ofstream ofs(filename, ios::out | ios::binary);
    container.write(ofs);
    ofs.close();

    // clean up and print profile information
    mxFree(filename);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}
