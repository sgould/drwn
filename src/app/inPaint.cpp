/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    inPaint.cpp
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

// opencv library headers
#include "cv.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./inPaint [OPTIONS] <img> <mask>\n";
    cerr << "OPTIONS:\n"
         << "  -dilate <n>       :: dilate mask by <n> pixels before inpainting\n"
         << "  -infilled         :: allow copying from already infilled region\n"
         << "  -p <n>            :: patch radius (2 <n> + 1)-by-(2 <n> + 1) (default: 3)\n"
         << "  -o <filename>     :: output image name\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    unsigned patchRadius = 3;
    int dilateAmount = 0;
    const char *outImage = NULL;
    bool bAllowCopyFromInfilled = false;
    bool bVisualize = false;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-dilate", dilateAmount)
        DRWN_CMDLINE_BOOL_OPTION("-infilled", bAllowCopyFromInfilled)
        DRWN_CMDLINE_INT_OPTION("-p", patchRadius)
        DRWN_CMDLINE_STR_OPTION("-o", outImage)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    DRWN_ASSERT(patchRadius > 0);
    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    const char *maskFilename = DRWN_CMDLINE_ARGV[1];

    // load image and mask
    DRWN_LOG_VERBOSE("Processing " << imgFilename << "...");
    cv::Mat image = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(image.data != NULL, imgFilename);
    cv::Mat mask = cv::imread(maskFilename, CV_LOAD_IMAGE_GRAYSCALE);
    DRWN_ASSERT_MSG(mask.data != NULL, maskFilename);
    DRWN_ASSERT(mask.size() == image.size());

    // dilate mask
    if (dilateAmount > 0) {
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(2 * dilateAmount + 1, 2 * dilateAmount + 1), cv::Point(dilateAmount, dilateAmount));
        cv::dilate(mask, mask, element);
    }

    // show masked image
    if (bVisualize) {
        cv::Mat canvas = image.clone();
        drwnShadeRegion(canvas, mask, CV_RGB(0, 255, 0), 1.0, DRWN_FILL_DIAG, 1);
        drwnDrawRegionBoundaries(canvas, mask, CV_RGB(0, 255, 0), 2);
        drwnShowDebuggingImage(canvas, "image", false);
    }

    // initialize inpainter and perform inpainting
    drwnImageInPainter inpainter(patchRadius, bAllowCopyFromInfilled);
    inpainter.bVisualize = bVisualize;
    cv::Mat infilledImage = inpainter.fill(image, mask);

    // pause if visualizing
    if (bVisualize) cv::waitKey(-1);

    // write output image
    if (outImage != NULL) {
        cv::imwrite(outImage, infilledImage);
    }

    // clean up
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
