/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    visualizeImageDataset.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for visualizing image sets.
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
    cerr << "USAGE: ./visualizeImageDataset [OPTIONS] <imgList>\n";
    cerr << "OPTIONS:\n"
         << "  -i <dir>          :: input directory (default: .)\n"
         << "  -o <filename>     :: output filename\n"
         << "  -s <scale>        :: combined image scale factor (default: 1.0)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    //! \todo add options for sizing, file extensions, etc.
    const char *inDir = ".";
    const char *outFilename = NULL;
    double scaleFactor = 1.0;
    int baseWidth = -1;
    int baseHeight = -1;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", inDir)
        DRWN_CMDLINE_STR_OPTION("-o", outFilename)
        DRWN_CMDLINE_REAL_OPTION("-s", scaleFactor)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    const char *imgList = DRWN_CMDLINE_ARGV[0];

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read image list
    DRWN_LOG_MESSAGE("Reading image list from " << imgList << "...");
    vector<string> baseNames = drwnReadFile(imgList);
    DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");

    // load images
    vector<cv::Mat> views;
    for (vector<string>::const_iterator it = baseNames.begin(); it != baseNames.end(); ++it) {
        string filename = string(inDir) + string("/") + (*it) + string(".jpg");
        cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        DRWN_ASSERT_MSG(img.data != NULL, filename);
        if ((baseWidth < 0) || (baseHeight < 0)) {
            baseWidth = img.cols;
            baseHeight = img.rows;
        } else {
            drwnResizeInPlace(img, cv::Size(baseWidth, baseHeight));
        }

        views.push_back(img);
    }

    cv::Mat canvas = drwnCombineImages(views);
    drwnResizeInPlace(canvas, cv::Size((int)(scaleFactor * canvas.cols),
            (int)(scaleFactor * canvas.rows)));

    if (bVisualize) {
        drwnShowDebuggingImage(canvas, string(imgList), true);
    }

    // save
    if (outFilename != NULL) {
        cv::imwrite(string(outFilename), canvas);
    }

    // clean up
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
