/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphOptimize.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for building/updating a nearest neighbour graph
**              over superpixels.
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
#include "drwnNNGraphMoves.h"
#include "drwnNNGraphThreadedMoves.h"
#include "drwnNNGraphVis.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphOptimize [OPTIONS] <imgList>\n";
    cerr << "OPTIONS:\n"
         << "  -i <filename>     :: input graph filename (for initialization)\n"
         << "  -m <iterations>   :: maximum iterations\n"
         << "  -o <filename>     :: output graph filename (can be same as input)\n"
         << "  -eqv <imgList>    :: load equivalence classes (can be repeated)\n"
         << "  -not <imgList>    :: list of images that are not target matchable\n"
         << "  -enforceLabels    :: only match nodes with the same label (like eqv at label level)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT = 1024;
    drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH = 1024;

    const char *inGraphFile = NULL;
    const char *outGraphFile = NULL;
    vector<const char *> eqvFiles;
    const char *notMatchable = NULL;
    int maxIterations = 100;
    bool bEnforceLabels = false;
    bool bVisualize = false;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", inGraphFile)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
        DRWN_CMDLINE_VEC_OPTION("-eqv", eqvFiles)
        DRWN_CMDLINE_STR_OPTION("-not", notMatchable)
        DRWN_CMDLINE_BOOL_OPTION("-enforceLabels", bEnforceLabels)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);
    const time_t startTime = std::time(NULL);

    const char *imgList = DRWN_CMDLINE_ARGV[0];
    vector<string> baseNames;
    DRWN_LOG_MESSAGE("Reading image list from " << imgList << "...");
    baseNames = drwnReadFile(imgList);
    DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");

    // initialize dataset
    drwnNNGraph graph;

    if (inGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << inGraphFile << "...");
        graph.read(inGraphFile);
        DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");

        // disable images (and re-enable if in image list)
        for (unsigned i = 0; i < graph.numImages(); i++) {
            graph[i].bSourceMatchable = false;
        }
    }

    // append images not in dataset / only search on images in this list
    DRWN_LOG_MESSAGE("Adding images to drwnNNGraph...");
    vector<drwnNNGraphImageData> imageData;
    imageData.reserve(baseNames.size());
    for (unsigned i = 0; i < baseNames.size(); i++) {
        if (bVisualize) {
            imageData.push_back(drwnNNGraphImageData(baseNames[i]));
        }
        const int indx = graph.findImage(baseNames[i]);
        if (indx < 0) {
            if (bVisualize) {
                graph.appendImage(imageData.back());
            } else {
                graph.appendImage(drwnNNGraphImage(drwnNNGraphImageData(baseNames[i])));
            }
        } else {
            graph[indx].bSourceMatchable = true;
        }
    }
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");

    // set not target matchable images
    if (notMatchable != NULL) {
        vector<string> notBaseNames = drwnReadFile(notMatchable);
        DRWN_LOG_VERBOSE("Setting " << notBaseNames.size() << " images not matchable...");
        for (vector<string>::const_iterator it = notBaseNames.begin(); it != notBaseNames.end(); ++it) {
            const int imgIndx = graph.findImage(*it);
            if (imgIndx < 0) {
                DRWN_LOG_WARNING("could not find " << *it << " in graph for not matchable image");
            } else {
                graph[imgIndx].bTargetMatchable = false;
            }
        }
    }

    // load and set equivalence classes
    for (int eqvClass = 0; eqvClass < (int)eqvFiles.size(); eqvClass++) {
        vector<string> eqvBaseNames = drwnReadFile(eqvFiles[eqvClass]);
        DRWN_LOG_VERBOSE("Setting equivalence class " << eqvClass << " for " << eqvBaseNames.size() << " images...");
        for (vector<string>::const_iterator it = eqvBaseNames.begin(); it != eqvBaseNames.end(); ++it) {
            const int imgIndx = graph.findImage(*it);
            if (imgIndx < 0) {
                DRWN_LOG_WARNING("could not find " << *it << " in graph for equivalence class " << eqvClass);
            } else {
                graph[imgIndx].eqvClass = eqvClass;
            }
        }
    }

    // initialize the graph
    if (bEnforceLabels) {
        DRWN_LOG_VERBOSE("...only matching nodes with the SAME label");
        drwnNNGraphThreadedMoves::initialize(graph, drwnNNGraphLabelsEqualMetric());
    } else {
        drwnNNGraphThreadedMoves::initialize(graph);
#if 0
        drwnNNGraphMoves::flann(graph, drwnNNGraphDefaultMetric());
#endif
    }
    DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");
    DRWN_LOG_VERBOSE("...with " << graph[0][0].features.size() << "-dimensional features");

    pair<double, double> lastEnergy = graph.energy();
    DRWN_LOG_MESSAGE("...iteration 0 (" << (int)difftime(std::time(NULL), startTime)
        << "); energy " << lastEnergy.first << ", best " << lastEnergy.second);

    // iterate moves
    int nIterations = 0;
    while (nIterations < maxIterations) {
        nIterations += 1;

        // perform an update
        if (bEnforceLabels) {
            drwnNNGraphThreadedMoves::update(graph, drwnNNGraphLabelsEqualMetric());
        } else {
            drwnNNGraphThreadedMoves::update(graph);
        }

        // visualizations
        if (bVisualize) {
            drwnShowDebuggingImage(drwnNNGraphVis::visualizeRetarget(imageData, graph),
                "match retarget", false);
            drwnShowDebuggingImage(drwnNNGraphVis::visualizeImageIndex(imageData, graph),
                "match source", false);
            drwnShowDebuggingImage(drwnNNGraphVis::visualizeMatchQuality(imageData, graph),
                "match quality", false);
        }

        // check energy
        pair<double, double> e = graph.energy();
        DRWN_LOG_MESSAGE("...iteration " << nIterations << " ("
            << (int)difftime(std::time(NULL), startTime)
            << "); energy " << e.first << ", best " << e.second);

        if (e.first == lastEnergy.first) break;
        lastEnergy = e;
    }

    DRWN_LOG_MESSAGE("nnGraphOptimize converged after " << (nIterations + 1) << " iterations");

    // write graph
    if (outGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << outGraphFile << "...");
        graph.write(outGraphFile);
    }

    // wait for key press
    if (bVisualize) cv::waitKey(-1);

    // clean up
    if (bVisualize) cv::destroyAllWindows();
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
