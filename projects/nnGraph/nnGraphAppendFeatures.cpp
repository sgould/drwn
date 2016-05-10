/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphAppendFeatures.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for appending external superpixel features.
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
#include "drwnML.h"
#include "drwnVision.h"

#include "drwnNNGraph.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphAppendFeatures [OPTIONS] <graph> (<ext>)+\n";
    cerr << "OPTIONS:\n"
         << "  -d <directory>    :: directory containing feature files (default: cached)\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *featureDir = "cached";

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-d", featureDir)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC < 2) {
        usage();
        return -1;
    }

    const char *graphFile = DRWN_CMDLINE_ARGV[0];

    vector<string> featureExts;
    for (int i = 1; i < DRWN_CMDLINE_ARGC; i++) {
        featureExts.push_back(DRWN_CMDLINE_ARGV[i]);
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

    // read graph
    drwnNNGraph graph;
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << graphFile << "...");
    graph.read(graphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");
    DRWN_LOG_VERBOSE("...with " << graph[0][0].features.size() << "-dimensional features");

    // load feature files and append to graph
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        const drwnNNGraphImageData image(graph[imgIndx].name());

        for (unsigned fIndx = 0; fIndx < featureExts.size(); fIndx++) {
            const string featureFilename = string(featureDir) + DRWN_DIRSEP + graph[imgIndx].name() + featureExts[fIndx];
            DRWN_ASSERT_MSG(drwnFileExists(featureFilename.c_str()), featureFilename);

            // determine if the features are in image or binary format
            string ext = drwn::strExtension(featureFilename);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // read feature file
            cv::Mat features;
            if (ext.compare("png") == 0) {
                features = cv::imread(featureFilename, CV_LOAD_IMAGE_GRAYSCALE);                
            } else {
                cv::Mat features = cv::Mat(image.height(), image.width(), CV_8UC1);
                ifstream ifs(featureFilename.c_str(), ios::binary | ios::in);
                for (unsigned y = 0; y < image.height(); y++) {
                    char *p = (char *)features.ptr<unsigned char>(y);
                    ifs.read(p, image.width() * sizeof(char));
                }
                ifs.close();
            }

            // append features
            graph[imgIndx].appendNodeFeatures(image, features);
        }
    }

    // write graph
    DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << graphFile << "...");
    DRWN_LOG_VERBOSE("...with " << graph[0][0].features.size() << "-dimensional features");
    graph.write(graphFile);

    // clean up
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
