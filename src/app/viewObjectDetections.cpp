/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    viewObjectDetections.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Visualization of detected objects.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

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
    cerr << "USAGE: ./viewObjectDetections [OPTIONS] <image> <objects>\n";
    cerr << "OPTIONS:\n"
         << "  -nms <area>       :: apply non-maximal overlap suppression\n"
         << "  -o <filename>     :: output image filename\n"
         << "  -s scale          :: output scale (default: 0.5)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // default parameters
    double nmsAreaOverlap = 0.0;
    const char *outDir = NULL;
    double scale = 0.5;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_REAL_OPTION("-nms", nmsAreaOverlap)
        DRWN_CMDLINE_STR_OPTION("-o", outDir)
        DRWN_CMDLINE_REAL_OPTION("-s", scale)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // load image and objects
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    cv::Mat img = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, "error loading " << imgFilename);
    DRWN_LOG_VERBOSE("..." << toString(img) << " image loaded");

    const char *objFilename = DRWN_CMDLINE_ARGV[1];
    drwnObjectList objects;
    objects.read(objFilename);
    DRWN_LOG_VERBOSE("..." << objects.size() << " objects read");

    // apply non-maximal overlap suppression
    if (nmsAreaOverlap > 0.0) {
        int n = objects.nonMaximalSuppression(nmsAreaOverlap);
        DRWN_LOG_VERBOSE("..." << n << " objects removed by non-maximal suppression");
    } else {
        objects.sort();
    }

    // draw bounding boxes
    vector<cv::Mat> views;
    for (drwnObjectList::const_iterator it = objects.begin(); it != objects.end(); ++it) {
        views.push_back(img.clone());
        drwnDrawBoundingBox(views.back(), it->extent, CV_RGB(255, 0, 0));
        cv::putText(views.back(), it->name.c_str(), cv::Point(it->extent.x, it->extent.y - 4),
            CV_FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2, CV_AA);
    }
    cv::Mat canvas = drwnCombineImages(views);
    drwnResizeInPlace(canvas, cv::Size((int)(scale * canvas.cols), (int)(scale * canvas.rows)));

    if (bVisualize) {
        drwnShowDebuggingImage(canvas, strBaseName(imgFilename), true);
    }

    // save
    if (outDir != NULL) {
        string outFilename = string(outDir) + string("/") + strBaseName(imgFilename) + string("_objects.png");
        cv::imwrite(outFilename, canvas);
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

