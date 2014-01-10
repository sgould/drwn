/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexImageCRF.cpp
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

// openCV headers
#include "cxcore.h"

// matlab headers
#include "mex.h"
#include "matrix.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

// function prototypes --------------------------------------------------------

cv::Mat parseImage(const mxArray *m);

// main -----------------------------------------------------------------------

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: x = mexImageCRF(image, unary, lambda_P, [regions, lambda_H], [options]);\n");
    mexPrintf("  image    :: H-by-W-by-3 image\n");
    mexPrintf("  unary    :: H-by-W-by-L unary potentials\n");
    mexPrintf("  lambda_P :: contrast-sensitive pairwise smoothness weight (>= 0)\n");
    mexPrintf("  regions  :: H-by-W-by-K region map for robust potts potentials\n");
    mexPrintf("  lambda_H :: robust potts weight (>= 0)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs != 3) && (nrhs != 4) && (nrhs != 5) && (nrhs != 6)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if ((nrhs == 4) || (nrhs == 6)) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse image and create a multi-class seg instance object
    cv::Mat image = parseImage(prhs[0]);
    DRWN_ASSERT(image.data != NULL);
    const int H = image.rows;
    const int W = image.cols;

    drwnSegImageInstance instance(image);

    // parse unary potentials
    vector<MatrixXd> unary;
    drwnMatlabUtils::mxArrayToEigen(prhs[1], unary);
    const int L = (int)unary.size();
    DRWN_ASSERT_MSG(L > 1, "invalid number of labels");
    DRWN_ASSERT_MSG((unary[0].rows() == H) && (unary[0].cols() == W),
        "unary potentials must match image size " << H << "-by-" << W);

    instance.unaries.resize(H * W, vector<double>(L));
    for (int l = 0; l < L; l++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                instance.unaries[instance.pixel2Indx(x, y)][l] = unary[l](y, x);
            }
        }
    }

    // parse pairwise contrast weight
    const double lambda_p = mxGetScalar(prhs[2]);
    DRWN_ASSERT_MSG(lambda_p >= 0.0, "lambda_P must be non-negative");

    // parse robust potts parameters
    if (nrhs >= 5) {
        vector<MatrixXd> regions;
        drwnMatlabUtils::mxArrayToEigen(prhs[3], regions);
        for (int k = 0; k < regions.size(); k++) {
            cv::Mat seg(H, W, CV_32SC1);
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    seg.at<int>(y, x) = regions[k](y, x);
                }
            }
            instance.superpixels.addSuperpixels(seg);
        }
    }

    const double lambda_h = (nrhs < 5) ? 0.0 : mxGetScalar(prhs[4]);
    DRWN_ASSERT_MSG(lambda_h >= 0.0, "lambda_H must be non-negative");

    // create inference object
    drwnRobustPottsCRFInference inf;
    inf.alphaExpansion(&instance, lambda_p, lambda_h);

    // return solution
    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleMatrix(H, W, mxREAL);
        double *px = mxGetPr(plhs[0]);
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                *px++ = (double)instance.pixelLabels(y, x) + 1;
            }
        }
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

// private functions -------------------------------------------------------

cv::Mat parseImage(const mxArray *m)
{
    DRWN_ASSERT((m != NULL) && (mxIsNumeric(m)));

    mwSize dims = mxGetNumberOfDimensions(m);
    DRWN_ASSERT_MSG((dims == 2) || ((dims == 3) && (mxGetDimensions(m)[2] == 3)),
        "image must be H-by-W or H-by-W-by-3");
    const int H = mxGetDimensions(m)[0]; // height
    const int W = mxGetDimensions(m)[1]; // width
    cv::Mat img = (dims == 2 ? cv::Mat(H, W, CV_8UC1) : cv::Mat(H, W, CV_8UC3));

    const double *p = mxGetPr(m);
    for (int c = 0; c < img.channels(); c++) {
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                img.at<unsigned char>(y, img.channels() * x + c) = (unsigned char)(255 * (*p++));
            }
        }
    }

    return img;
}
