/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphFLANN.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for building a nearest neighbour graph over
**              superpixels using FLANN.
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
#include "drwnNNGraphVis.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphFLANN [OPTIONS] <imgList>\n";
    cerr << "OPTIONS:\n"
         << "  -i <filename>     :: input graph filename (for initialization)\n"
         << "  -o <filename>     :: output graph filename (can be same as input)\n"
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
    bool bVisualize = false;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", inGraphFile)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

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
    DRWN_LOG_VERBOSE("...with " << graph[0][0].features.size() << "-dimensional features");

    // construct data for matching
    const size_t numFeatures = graph[0][0].features.size();
    vector<drwnNNGraphNodeIndex> sampleIndexes;
    vector<drwnNNGraphNodeIndex> queryIndexes;
    int numSamples = 0;
    int numQueries = 0;
    for (unsigned i = 0; i < graph.numImages(); i++) {
        if (graph[i].bSourceMatchable) {
            queryIndexes.reserve(queryIndexes.size() + graph[i].numNodes());
            for (unsigned j = 0; j < graph[i].numNodes(); j++) {
                queryIndexes.push_back(drwnNNGraphNodeIndex(i, j));
            }
            numQueries += graph[i].numNodes();
        } else {
            sampleIndexes.reserve(sampleIndexes.size() + graph[i].numNodes());
            for (unsigned j = 0; j < graph[i].numNodes(); j++) {
                sampleIndexes.push_back(drwnNNGraphNodeIndex(i, j));
            }
            numSamples += graph[i].numNodes();
        }
    }

    DRWN_LOG_VERBOSE("..." << numSamples << " samples and " << numQueries << " queries");

    cv::Mat features(numSamples, numFeatures, CV_32FC1);
    for (unsigned i = 0; i < sampleIndexes.size(); i++) {
        for (unsigned d = 0; d < numFeatures; d++) {
            features.at<float>(i, d) = graph[sampleIndexes[i]].features[d];
        }
    }

    cv::Mat queries(numQueries, numFeatures, CV_32FC1);
    for (unsigned i = 0; i < queryIndexes.size(); i++) {
        for (unsigned d = 0; d < numFeatures; d++) {
            queries.at<float>(i, d) = graph[queryIndexes[i]].features[d];
        }
    }

    // build kd-tree
    DRWN_LOG_VERBOSE("building kd-tree on " << numSamples << " samples...");
    cv::flann::KDTreeIndexParams indexParams;
    //cv::flann::LinearIndexParams indexParams;
    cv::flann::Index kdtree(features, indexParams);

    // query the kd-tree
    DRWN_LOG_VERBOSE("querying kd-tree on " << numQueries << " queries...");
    cv::Mat indexes(numQueries, drwnNNGraph::K, CV_32SC1); 
    cv::Mat dists(numQueries, drwnNNGraph::K, CV_32FC1);
    kdtree.knnSearch(queries, indexes, dists, drwnNNGraph::K, cv::flann::SearchParams(64));

    // extract nearest neighbours
    for (unsigned i = 0; i < queryIndexes.size(); i++) {
        for (unsigned k = 0; k < drwnNNGraph::K; k++) {        
            drwnNNGraphEdge e(sampleIndexes[indexes.at<int>(i, k)], dists.at<float>(i, k));
            graph[queryIndexes[i]].edges.push_back(e);
        }
    }

    DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");

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
