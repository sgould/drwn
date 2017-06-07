/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphApplyTransform.cpp
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
#include "drwnML.h"
#include "drwnVision.h"

#include "drwnNNGraph.h"
#include "drwnNNGraphMoves.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphApplyTransform [OPTIONS] <xformFile> <graphFile>\n";
    cerr << "OPTIONS:\n"
         << "  -o <graphFile>    :: output graph filename (can be same as input)\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *outGraphFile = NULL;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    const char *inXformFile = DRWN_CMDLINE_ARGV[0];
    const char *inGraphFile = DRWN_CMDLINE_ARGV[1];

    // load the transform
    drwnFeatureTransform *featureTransform =
        drwnFeatureTransformFactory::get().createFromFile(inXformFile);
    DRWN_ASSERT_MSG(featureTransform != NULL, inXformFile);

    // load the graph and transform features
    drwnNNGraph graph;
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << inGraphFile << "...");
    graph.read(inGraphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");

    DRWN_LOG_MESSAGE("Transforming features...");
    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        graph[imgIndx].transformNodeFeatures(*featureTransform);
    }

    drwnNNGraphMoves::rescore(graph);

    // write graph
    if (outGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << outGraphFile << "...");
        graph.write(outGraphFile);
    }

    // clean up
    delete featureTransform;
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
