/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphMerge.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Merges two or more graphs. Repeated images are ignored. An
**              optional feature transform can be applied to the graphs.
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
    cerr << "USAGE: ./nnGraphMerge [OPTIONS] (<graphFile>)+\n";
    cerr << "OPTIONS:\n"
         << "  -o <graphFile>    :: output graph filename (can be same as one of the inputs)\n"
         << "  -t <xformFile>    :: apply transformation to all but first <graphFile>\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *outGraphFile = NULL;
    const char *xformFile = NULL;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
        DRWN_CMDLINE_STR_OPTION("-t", xformFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC < 2) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");

    // load the feature transform if provided
    drwnFeatureTransform *featureTransform = NULL;
    if (xformFile != NULL) {
        featureTransform = drwnFeatureTransformFactory::get().createFromFile(xformFile);
        DRWN_ASSERT_MSG(featureTransform != NULL, xformFile);
    }

    // load the first graph
    drwnNNGraph graph;
    DRWN_LOG_MESSAGE("Appending drwnNNGraph from " << DRWN_CMDLINE_ARGV[0] << "...");
    graph.read(DRWN_CMDLINE_ARGV[0]);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");
    DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");

    // load subsequent graphs and append
    for (int fileIndx = 1; fileIndx < DRWN_CMDLINE_ARGC; fileIndx++) {
        const char *inGraphFile = DRWN_CMDLINE_ARGV[fileIndx];

        drwnNNGraph additionalNodes;
        DRWN_LOG_MESSAGE("Appending drwnNNGraph from " << inGraphFile << "...");
        additionalNodes.read(inGraphFile);

        // add images (without edges)
        vector<int> imgIndxMapping(additionalNodes.numImages(), -1);
        for (unsigned imgIndx = 0; imgIndx < additionalNodes.numImages(); imgIndx++) {
            int newIndx = graph.findImage(additionalNodes[imgIndx].name());
            if (newIndx < 0) {
                // add image if not already in the graph
                newIndx = graph.appendImage(additionalNodes[imgIndx]);

                // clear edges
                graph[newIndx].clearEdges();

                // apply feature transform
                if (featureTransform != NULL) {
                    graph[newIndx].transformNodeFeatures(*featureTransform);
                }
            }
            imgIndxMapping[imgIndx] = newIndx;
        }

        // add edges (renumbered appropriately)
        for (unsigned imgIndx = 0; imgIndx < additionalNodes.numImages(); imgIndx++) {
            const int newIndx = imgIndxMapping[imgIndx];
            DRWN_ASSERT(graph[newIndx].numNodes() == additionalNodes[imgIndx].numNodes());
            for (unsigned segId = 0; segId < additionalNodes[imgIndx].numNodes(); segId++) {
                const bool bHasExistingMatches = !graph[newIndx][segId].edges.empty();
                const drwnNNGraphEdgeList& el = additionalNodes[imgIndx][segId].edges;
                for (drwnNNGraphEdgeList::const_iterator kt = el.begin(); kt != el.end(); ++kt) {
                    drwnNNGraphEdge e(*kt);
                    e.targetNode.imgIndx = imgIndxMapping[e.targetNode.imgIndx];
                    if (bHasExistingMatches) {
                        graph[newIndx][segId].insert(e);
                    } else {
                        graph[newIndx][segId].edges.push_back(e);
                    }
                }
            }
        }

        DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
        DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");
        DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");
    }

    // write graph
    if (outGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << outGraphFile << "...");
        graph.write(outGraphFile);
    }

    // clean up
    if (featureTransform != NULL)
        delete featureTransform;
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
