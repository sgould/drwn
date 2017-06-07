/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    selfPatchMatchDemo.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./patchMatchDemo [OPTIONS] <img>\n";
    cerr << "OPTIONS:\n"
         << "  -m <iterations>   :: maximum number of iterations\n"
         << "  -o <filebase>     :: output nearest neighbour field\n"
         << "  -p <radius>       :: patch radius (default: 4 for 9x9 patch)\n"
         << "  -x                :: visualize matches\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int maxIterations = 10;
    unsigned patchRadius = 4;
    const char *outFilebase = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-o", outFilebase)
        DRWN_CMDLINE_INT_OPTION("-p", patchRadius)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read images
    const string imgName = string(DRWN_CMDLINE_ARGV[0]);
    cv::Mat img = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, imgName);

    // run patch match
    cv::Mat nnf;
    cv::Mat costs = drwnSelfPatchMatch(img, cv::Size(patchRadius, patchRadius), nnf, 0.0, maxIterations);
    DRWN_LOG_VERBOSE("...final self matching energy " << cv::sum(costs)[0]);

    if (bVisualize) {
        vector<cv::Mat> views;
        cv::split(nnf, views);
        views[0] = drwnCreateHeatMap(views[0], DRWN_COLORMAP_RAINBOW);
        views[1] = drwnCreateHeatMap(views[1], DRWN_COLORMAP_RAINBOW);
        
        views.push_back(costs.clone());
        drwnScaleToRange(views.back(), 0.0, 1.0);
        views.back() = drwnCreateHeatMap(views.back(), DRWN_COLORMAP_RAINBOW);

        views.push_back(drwnNNFRepaint(img, nnf));
        drwnShowDebuggingImage(views, "selfPatchMatchDemo", false);
    }

    // run patch match on flipped image
    cv::Mat nnfFlipped;
    cv::Mat imgFlipped;
    cv::flip(img, imgFlipped, 1);
    cv::Mat costsFlipped = drwnBasicPatchMatch(img, imgFlipped, cv::Size(patchRadius, patchRadius),
        nnfFlipped, maxIterations);
    DRWN_LOG_VERBOSE("...final flipped matching energy " << cv::sum(costs)[0]);

    if (bVisualize) {
        vector<cv::Mat> views;
        cv::split(nnfFlipped, views);
        views[0] = drwnCreateHeatMap(views[0], DRWN_COLORMAP_RAINBOW);
        views[1] = drwnCreateHeatMap(views[1], DRWN_COLORMAP_RAINBOW);
        
        views.push_back(costsFlipped.clone());
        drwnScaleToRange(views.back(), 0.0, 1.0);
        views.back() = drwnCreateHeatMap(views.back(), DRWN_COLORMAP_RAINBOW);

        views.push_back(drwnNNFRepaint(imgFlipped, nnfFlipped));
        drwnShowDebuggingImage(views, "selfPatchMatchDemo flipped", true);
    }

    // save nearest neighbour field
    if (outFilebase != NULL) {
        DRWN_LOG_VERBOSE("...writing nearest neighbour field to "
            << outFilebase << ".x.png and " << outFilebase << ".y.png");
        vector<cv::Mat> directions;
        cv::split(nnf, directions);
        cv::imwrite(string(outFilebase) + string(".x.png"), directions[0]);
        cv::imwrite(string(outFilebase) + string(".y.png"), directions[1]);
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
