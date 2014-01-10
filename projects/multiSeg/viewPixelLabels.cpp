/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    viewPixelLabels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for visualizing pixel labels.
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

#define WINDOW_NAME "viewPixelLabels"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./viewPixelLabels [OPTIONS] (<imageList>|<baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -inLabels <ext>   :: extension for label input (default: .txt)\n"
         << "  -outImages <ext>  :: extension for image output (default: none)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *inLabelExt = ".txt";
    const char *outImageExt = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-inLabels", inLabelExt)
        DRWN_CMDLINE_STR_OPTION("-outImages", outImageExt)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read list of evaluation images
    const char *imageList = DRWN_CMDLINE_ARGV[0];

    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }

    // iterate over images
    for (int i = 0; i < (int)baseNames.size(); i++) {
        // load image and labels
        const string imgFilename = gMultiSegConfig.filename("imgDir", baseNames[i], "imgExt");
        drwnSegImageInstance instance(imgFilename.c_str());

        const string lblFilename = gMultiSegConfig.filebase("outputDir", baseNames[i]) +
            string(inLabelExt);
        drwnReadMatrix(instance.pixelLabels, lblFilename.c_str());
        
        // generate visualization
        cv::Mat canvas;
        if (bVisualize || (outImageExt != NULL)) {
            canvas = drwnMultiSegVis::visualizeInstance(instance);
        }

        // show image
        if (bVisualize) {
            cv::namedWindow(string(WINDOW_NAME));
            cv::imshow(string(WINDOW_NAME), canvas);
            cv::waitKey(100);
        }

        // save output image
        if (outImageExt != NULL) {
            const string filename = gMultiSegConfig.filebase("outputDir", baseNames[i]) + 
                string(outImageExt);
            cv::imwrite(filename, canvas);
        }
    }

    // wait for keypress if only image
    if (bVisualize && (baseNames.size() == 1)) {
        cv::waitKey(-1);
    }

    // clean up and print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
