/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    convertPixelLabels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for converting (colour) annotated images to label files.
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
#include "drwnML.h"
#include "drwnVision.h"

#define WINDOW_NAME "convertPixelLabels"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./convertPixelLabels [OPTIONS] (<imageList>|<baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -i <ext>          :: extension for image input (default: .bmp)\n"
         << "  -o <ext>          :: extension for label output (default: .txt)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *inExt = ".bmp";
    const char *outExt = ".txt";
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", inExt)
        DRWN_CMDLINE_STR_OPTION("-o", outExt)
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

    // check for input directory
    if (!drwnDirExists(gMultiSegConfig.filebase("lblDir", "").c_str())) {
        DRWN_LOG_FATAL("input/output labels directory " << gMultiSegConfig.filebase("lblDir", "")
            << " does not exist");
    }

    // construct colour-to-key table
    map<unsigned int, int> table;

    set<int> keys(gMultiSegRegionDefs.keys());
    for (set<int>::const_iterator ik = keys.begin(); ik != keys.end(); ++ik) {
        table[gMultiSegRegionDefs.color(*ik)] = *ik;
    }

    for (map<unsigned int, int>::const_iterator it = table.begin(); it != table.end(); ++it) {
        DRWN_LOG_DEBUG("colour (" << (int)gMultiSegRegionDefs.red(it->first) << ", "
            << (int)gMultiSegRegionDefs.green(it->first) << ", "
            << (int)gMultiSegRegionDefs.blue(it->first)
            << ") corresponds to label " << it->second);
    }

    // iterate over images
    for (int i = 0; i < (int)baseNames.size(); i++) {

        // load annotated images
        const string inFilename = gMultiSegConfig.filebase("lblDir", baseNames[i]) +
            string(inExt);
        cv::Mat img = cv::imread(inFilename, CV_LOAD_IMAGE_COLOR);
        DRWN_ASSERT_MSG(img.data != NULL, "could not load image " << inFilename);

        // show image
        if (bVisualize) {
            cv::namedWindow(WINDOW_NAME);
            cv::imshow(WINDOW_NAME, img);
            cv::waitKey(100);
        }

        // write label file
        const string outFilename = gMultiSegConfig.filebase("lblDir", baseNames[i]) +
            string(outExt);
        ofstream ofs(outFilename.c_str());

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (x != 0) ofs << " ";

                unsigned char red = img.at<unsigned char>(y, 3 * x + 2);
                unsigned char green = img.at<unsigned char>(y, 3 * x + 1);
                unsigned char blue = img.at<unsigned char>(y, 3 * x + 0);

                unsigned int c = (red << 16) | (green << 8) | blue;
                map<unsigned int, int>::const_iterator it = table.find(c);
                ofs << ((it == table.end()) ? -1 : it->second);
            }
            ofs << "\n";
        }

        ofs.close();
    }

    // wait for keypress if only image
    if (bVisualize && (baseNames.size() == 1)) {
        cv::waitKey(-1);
    }

    // clean up and print profile information
    cv::destroyAllWindows();
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
