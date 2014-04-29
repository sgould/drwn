/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    generateSuperpixels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
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
#include "drwnVision.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./generateSuperpixels [OPTIONS] <img>\n";
    cerr << "OPTIONS:\n"
         << "  -m <method>       :: segmentation method (SUPERPIXEL (default), SLIC or KMEANS)\n"
         << "  -g <grid>         :: set grid size for superpixel/SLIC segmentation (default: 10)\n"
         << "  -k <clusters>     :: number of clusters for k-means segmentation (default: 100)\n"
         << "  -o <filename>     :: save segmentation to file <filename> (use .txt extension)\n"
         << "                    :: for text; else drwnSuperpixelContainer format is used)\n"
         << "  -a <filename>     :: append segmentation to file <filename> (only supports\n"
         << "                    :: drwnSuperpixelContainer format)\n"
         << "  -medianBlur       :: apply a median blurring filter before segmenting\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *method = "SUPERPIXEL";
    unsigned gridSize = 10;
    unsigned numClusters = 100;
    const char *outFile = NULL;
    const char *appendFile = NULL;
    bool bMedianBlur = false;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-m", method)
        DRWN_CMDLINE_INT_OPTION("-g", gridSize)
        DRWN_CMDLINE_INT_OPTION("-k", numClusters)
        DRWN_CMDLINE_STR_OPTION("-o", outFile)
        DRWN_CMDLINE_STR_OPTION("-a", appendFile)
        DRWN_CMDLINE_BOOL_OPTION("-medianBlur", bMedianBlur)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    DRWN_ASSERT_MSG((appendFile == NULL) || (outFile == NULL),
        "options -a and -o are mutually exclusive");
    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // load image
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    cv::Mat img = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, imgFilename);

    if (bMedianBlur) {
        cv::medianBlur(img, img, 3);
    }

    // run superpixel algorithm
    cv::Mat seg;
    if (!strcasecmp(method, "SUPERPIXEL")) {
        seg = drwnFastSuperpixels(img, gridSize);
    } else if (!strcasecmp(method, "SLIC")) {
        cv::Mat imgLab;
        cv::cvtColor(img, imgLab, CV_BGR2Lab);
        seg = drwnSLICSuperpixels(imgLab, gridSize * gridSize);
    } else if (!strcasecmp(method, "KMEANS")) {
        seg = drwnKMeansSegments(img, numClusters);
    } else {
        DRWN_LOG_FATAL("unrecognized method " << method);
    }
    DRWN_LOG_VERBOSE(cv::norm(seg, cv::NORM_INF) << " superpixels generated");

    // visualize
    if (bVisualize) {
        cv::Mat canvas = img.clone();
        drwnAverageRegions(canvas, seg);
        drwnDrawRegionBoundaries(canvas, seg, CV_RGB(0, 0, 0));
        //cv::Mat canvas = drwnCreateHeatMap(seg);
        drwnShowDebuggingImage(canvas, string("segmentation"), true);
    }

    // save segmentation
    if (appendFile != NULL) {
        drwnSuperpixelContainer container;

        ifstream ifs(appendFile, ios::binary);
        DRWN_ASSERT_MSG(!ifs.fail(), "file " << appendFile << " does not exist");
        container.read(ifs);
        ifs.close();

        container.addSuperpixels(seg);

        ofstream ofs (appendFile, ios::binary);
        container.write(ofs);
        ofs.close();

    } else if (outFile != NULL) {
        const string ext = drwn::strExtension(string(outFile));

        if (ext.compare("txt") == 0) {
            // write as human readable text file
            ofstream ofs(outFile);
            for (int y = 0; y < seg.rows; y++) {
                for (int x = 0; x < seg.cols; x++) {
                    if (x != 0) ofs << " ";
                    ofs << seg.at<int>(y, x);
                }
                ofs << "\n";
            }
            ofs.close();
        } else {
            // write as drwnSuperpixelContainer object
            drwnSuperpixelContainer container;
            container.addSuperpixels(seg);
            ofstream ofs (outFile, ios::binary);
            container.write(ofs);
            ofs.close();
        }
    }

    // clean up
    cv::destroyAllWindows();
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
