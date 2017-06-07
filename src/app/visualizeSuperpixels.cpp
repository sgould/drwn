/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    visualizeSuperpixels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for visualizing superpixels stored in a superpixel container
**  object.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// opencv library headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./visualizeSuperpixels [OPTIONS] <superpixelContainer>\n";
    cerr << "OPTIONS:\n"
         << "  -colourById       :: colour superpixels by identifier\n"
         << "  -i <image>        :: image filename for overlaying superpixels\n"
         << "  -o <image>        :: output image filename\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH = 1024;
    drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT = 1024;

    const char *inImage = NULL;
    const char *outImage = NULL;
    bool bColourById = false;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_BOOL_OPTION("-colourById", bColourById)
        DRWN_CMDLINE_STR_OPTION("-i", inImage)
        DRWN_CMDLINE_STR_OPTION("-o", outImage)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    const char *containerFilename = DRWN_CMDLINE_ARGV[0];
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read superpixels
    drwnSuperpixelContainer container;
    ifstream ifs(containerFilename, ios::binary);
    DRWN_ASSERT_MSG(!ifs.fail(), "error opening file " << containerFilename);
    container.read(ifs);
    ifs.close();

    // read image if available
    cv::Mat img;
    if (inImage != NULL) {
        img = cv::imread(inImage, CV_LOAD_IMAGE_COLOR);
        DRWN_ASSERT_MSG((img.rows == container.height()) && (img.cols == container.width()),
            "mismatch between image size and superpixel container size, " << toString(img)
            << " vs. " << container.height() << "-by-" << container.width());
    } else {
        img = cv::Mat(container.height(), container.width(), CV_8UC3, cv::Scalar::all(255));
    }

    const cv::Mat canvas = container.visualize(img, bColourById);
    if (bVisualize) {
        drwnShowDebuggingImage(canvas, string("Superpixels"), true);
    }

    // save
    if (outImage != NULL) {
        cv::imwrite(string(outImage), canvas);
    }

    // clean up
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
