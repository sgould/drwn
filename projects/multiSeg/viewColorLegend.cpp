/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    viewColorLegend.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for visualizing segmentation colors and category names.
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
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

#define WINDOW_NAME "viewColorLegend"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./viewColorLegend [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << "  -o <filename>     :: save legend to disk (default: none)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *outImage = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", outImage)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // generate legend
    vector<cv::Mat> categories;
    const set<int> keys = gMultiSegRegionDefs.keys();
    for (set<int>::const_iterator it = keys.begin(); it != keys.end(); ++it) {
        unsigned cId = gMultiSegRegionDefs.color(*it);
        cv::Scalar colour(drwnMultiSegRegionDefinitions::blue(cId),
            drwnMultiSegRegionDefinitions::green(cId),
            drwnMultiSegRegionDefinitions::red(cId));
        cv::Mat box(96, 128, CV_8UC3, colour);
        cv::rectangle(box, cv::Point(0, 0), cv::Point(box.cols - 1, box.rows - 1),
            CV_RGB(255, 255, 255), 2, CV_AA);

        const string name = gMultiSegRegionDefs.name(*it);
        int baseline;
        cv::Size textExtent = cv::getTextSize(name, CV_FONT_HERSHEY_SIMPLEX, 0.71, 2, &baseline);
        cv::putText(box, name, cv::Point((box.cols - textExtent.width) / 2,
                (box.rows - textExtent.height)/2), CV_FONT_HERSHEY_SIMPLEX, 0.71, CV_RGB(255, 255, 255), 3);
        cv::putText(box, name, cv::Point((box.cols - textExtent.width) / 2,
                (box.rows - textExtent.height)/2), CV_FONT_HERSHEY_SIMPLEX, 0.71, CV_RGB(0, 0, 0), 2);

        categories.push_back(box);
    }

    // generate visualization
    cv::Mat canvas;
    if (bVisualize || (outImage != NULL)) {
        canvas = drwnCombineImages(categories, 2);
    }

    // save output image
    if (outImage != NULL) {
        cv::imwrite(string(outImage), canvas);
    }

    // show image
    if (bVisualize) {
        cv::namedWindow(string(WINDOW_NAME));
        cv::imshow(string(WINDOW_NAME), canvas);
        cv::waitKey(-1);
    }

    // clean up and print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
