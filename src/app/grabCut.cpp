/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    grabCut.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implementation of the grabCut algorithm by Rother et al., 2004.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// opencv library headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// main ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./grabCut [OPTIONS] <image> (<mask>)\n";
    cerr << "OPTIONS:\n"
         << "  -k <num>          :: number of mixture components (default: 5)\n"
         << "  -m <samples>      :: max samples to use when learning colour models\n"
         << "  -o <dir>          :: set output directory for segmentation masks and images\n"
         << "  -s <scale>        :: rescale input\n"
         << "  -w <weight>       :: use pairwise weight provided (otherwise tries many)\n"
         << "  -x                :: visualize\n"
         << "  -scm <file>       :: save final colour models to <file>\n"
         << "  -lcm <file>       :: load initial colour models from <file>\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // default parameters
    const char *outDir = NULL;
    double scale = 1.0;
    double weight = -1.0;
    bool bVisualize = false;
    const char *finalColourModelFile = NULL;
    const char *initialColourModelFile = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-k", drwnGrabCutInstance::numMixtures)
        DRWN_CMDLINE_INT_OPTION("-m", drwnGrabCutInstance::maxSamples)
        DRWN_CMDLINE_STR_OPTION("-o", outDir)
        DRWN_CMDLINE_REAL_OPTION("-s", scale)
        DRWN_CMDLINE_REAL_OPTION("-w", weight)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
        DRWN_CMDLINE_STR_OPTION("-scm", finalColourModelFile)
        DRWN_CMDLINE_STR_OPTION("-lcm", initialColourModelFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if ((DRWN_CMDLINE_ARGC != 1) && (DRWN_CMDLINE_ARGC != 2)) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // load image
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    const char *maskFilename = DRWN_CMDLINE_ARGC == 1 ? NULL : DRWN_CMDLINE_ARGV[1];
    cv::Mat img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_COLOR);

    // load mask
    cv::Mat mask(img.rows, img.cols, CV_8UC1);
    if (maskFilename == NULL) {
        cv::Rect bb = drwnInputBoundingBox(string("annotate"), img);
        DRWN_ASSERT((bb.width > 0) && (bb.height > 0));
        mask.setTo(cv::Scalar(drwnGrabCutInstance::MASK_BG));
        mask(bb) = cvScalar(drwnGrabCutInstance::MASK_C_FG); 
    } else {
        mask = cv::imread(string(maskFilename), CV_LOAD_IMAGE_GRAYSCALE);
    }

    // rescale image and mask
    if (scale != 1.0) {
        drwnResizeInPlace(img, (int)(scale * img.rows), (int)(scale * img.cols), CV_INTER_LINEAR);
        drwnResizeInPlace(mask, img.rows, img.cols, CV_INTER_NN);
    }

    // show image and mask
    if (bVisualize) {
        drwnShowDebuggingImage(img, string("image"), false);
        drwnShowDebuggingImage(mask, string("mask"), false);
    }
    drwnGrabCutInstance::bVisualize = bVisualize;

    // run grabCut with different weights
    const double minWeight = (weight < 0.0) ? 1.0 : weight;
    const double maxWeight = (weight < 0.0) ? 256.0 : weight;
    drwnGrabCutInstance model;
    model.name = drwn::strBaseName(imgFilename);
    for (double w = minWeight; w <= maxWeight; ) {
        // initialize model
        model.initialize(img, mask, initialColourModelFile);
        model.setBaseModelWeights(1.0, 0.0, w);
        cv::Mat seg = model.inference();

        // save segmentation mask
        if (outDir != NULL) {
            string wStr = drwn::strReplaceSubstr(toString(0.01 * (int)(w * 100)), ".", "_");
            string filename = string(outDir) + drwn::strBaseName(string(imgFilename)) +
                string("_mask_") + wStr + string(".png");
            DRWN_LOG_VERBOSE("writing segmentation to " << filename << "...");
            cv::imwrite(filename, seg);

            filename = string(outDir) + drwn::strBaseName(string(imgFilename)) +
                string("_img_") + wStr + string(".png");
            DRWN_LOG_VERBOSE("writing segmented image to " << filename << "...");
            cv::Mat m(img.clone());
            cv::compare(seg, cv::Scalar(0), seg, CV_CMP_EQ);
            m.setTo(cv::Scalar(255, 0, 0), seg);
            cv::imwrite(filename, m);
        }

        // update weight
        if (w == 0.0) w = 1.0; else w *= 2.0;
    }

    // save final colour model file
    if (finalColourModelFile != NULL) {
        DRWN_LOG_VERBOSE("writing colour models to " << finalColourModelFile << "...");
        model.saveColourModels(finalColourModelFile);
    }

    if (bVisualize && (weight >= 0.0)) {
        cvWaitKey(-1);
    }

    // show profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

