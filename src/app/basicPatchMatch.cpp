/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    basicPatchMatch.cpp
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
    cerr << "USAGE: ./basicPatchMatch [OPTIONS] <imgA> <imgB> (<maskA> (<maskB>))\n";
    cerr << "OPTIONS:\n"
         << "  -negate           :: negate masks\n"
         << "  -m <iterations>   :: maximum number of iterations\n"
         << "  -o <filebase>     :: output nearest neighbour field\n"
         << "  -p <size>         :: patch size (<size>-by-<size>)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    bool bNegateMask = false;
    unsigned maxIterations = 2;
    unsigned patchSize = 8;
    const char *outFilebase = NULL;
    bool bVisualize = false;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_BOOL_OPTION("-negate", bNegateMask)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-o", outFilebase)
        DRWN_CMDLINE_INT_OPTION("-p", patchSize)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if ((DRWN_CMDLINE_ARGC < 2) || (DRWN_CMDLINE_ARGC > 4)) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    const char *imgNameA = DRWN_CMDLINE_ARGV[0];
    const char *imgNameB = DRWN_CMDLINE_ARGV[1];
    const char *maskNameA = (DRWN_CMDLINE_ARGC < 3) ? NULL : DRWN_CMDLINE_ARGV[2];
    const char *maskNameB = (DRWN_CMDLINE_ARGC < 4) ? NULL : DRWN_CMDLINE_ARGV[3];

    // load images
    cv::Mat imgA = cv::imread(imgNameA, CV_LOAD_IMAGE_COLOR);
    cv::Mat imgB = cv::imread(imgNameB, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(imgA.data != NULL, imgNameA);
    DRWN_ASSERT_MSG(imgB.data != NULL, imgNameB);

    cv::Mat maskA, maskB;
    if (maskNameA != NULL) {
        maskA = cv::imread(maskNameA, CV_LOAD_IMAGE_GRAYSCALE);
        DRWN_ASSERT_MSG(maskA.data != NULL, maskNameA);
        if (bNegateMask) maskA = (maskA == 0x00);
    }
    if (maskNameB != NULL) {
        maskB = cv::imread(maskNameB, CV_LOAD_IMAGE_GRAYSCALE);
        DRWN_ASSERT_MSG(maskB.data != NULL, maskNameB);
        if (bNegateMask) maskB = (maskB == 0x00);
    }

    // run patch match
    drwnMaskedPatchMatch pm(imgA, imgB, maskA, maskB, cv::Size(patchSize, patchSize));
    pm.search(maxIterations);
    DRWN_LOG_VERBOSE("...final matching energy " << pm.energy());

    // save nearest neighbour field
    if (outFilebase != NULL) {
        DRWN_LOG_VERBOSE("...writing nearest neighbour field to " 
            << outFilebase << ".x.png and " << outFilebase << ".y.png");
        vector<cv::Mat> directions;
        cv::split(pm.nnf(), directions);
        cv::imwrite(string(outFilebase) + string(".x.png"), directions[0]);
        cv::imwrite(string(outFilebase) + string(".y.png"), directions[1]);
    }

    // visualize
    if (bVisualize) {
        drwnShowDebuggingImage(pm.visualize(), "NNF", true);
    }

    // clean up
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
