/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphRelabelNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for relabeling superpixel nodes.
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

// helper functions ----------------------------------------------------------

void updateNodeLabels(drwnNNGraphImage& image, const drwnNNGraphImageData& data)
{
    DRWN_ASSERT((int)image.numNodes() == data.segments().size());

    const int nLabels = data.labels().maxCoeff() + 1;
    if (nLabels <= 0) {
        DRWN_LOG_WARNING("image " << data.name() << " has no labels");
        for (unsigned segId = 0; segId < data.numSegments(); segId++) {
            image[segId].label = -1;
        }
        return;
    }

    vector<VectorXi> labelCounts(data.numSegments(), VectorXi::Zero(nLabels));
    vector<int> unknownCounts(data.numSegments(), 0);
    for (int c = 0; c < data.segments().channels(); c++) {
        for (unsigned y = 0; y < data.height(); y++) {
            for (unsigned x = 0; x < data.width(); x++) {
                const int segId = data.segments()[c].at<int>(y, x);
                if (segId < 0) continue;
                const int lblId = data.labels()(y, x);
                if (lblId < 0) {
                    unknownCounts[segId] += 1;
                } else {
                    labelCounts[segId][lblId] += 1;
                }
            }
        }
    }

    int lbl;
    for (unsigned segId = 0; segId < data.numSegments(); segId++) {
        const int maxCount = labelCounts[segId].maxCoeff(&lbl);
        if (maxCount >= unknownCounts[segId]) {
            image[segId].label = lbl;
        } else {
            image[segId].label = -1;
        }
    }
}

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphRelabelNodes [OPTIONS] <graph>\n";
    cerr << "OPTIONS:\n"
         << "  -o <filename>     :: output graph filename (default: same as input)\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *inGraphFile = NULL;
    const char *outGraphFile = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC < 1) {
        usage();
        return -1;
    }

    inGraphFile = DRWN_CMDLINE_ARGV[0];
    if (outGraphFile == NULL) outGraphFile = inGraphFile;

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

    // read graph
    drwnNNGraph graph;
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << inGraphFile << "...");
    graph.read(inGraphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");

    // relabel nodes
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        const drwnNNGraphImageData imageData(graph[imgIndx].name());
        updateNodeLabels(graph[imgIndx], imageData);
    }

    // write graph
    DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << outGraphFile << "...");
    graph.write(outGraphFile);

    // clean up
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
