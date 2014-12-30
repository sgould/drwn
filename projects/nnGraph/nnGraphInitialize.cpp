/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphInitialize.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for initializing / augmenting a drwnNNGraph.
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

// threading -----------------------------------------------------------------

class drwnNNGraphAppendImageJob : public drwnThreadJob {
protected:
    drwnNNGraph *_graph;
    string _baseName;

public:
    drwnNNGraphAppendImageJob(drwnNNGraph *graph, const string& baseName) :
        _graph(graph), _baseName(baseName) { /* do nothing */ }
    ~drwnNNGraphAppendImageJob() { /* do nothing */ }

    void operator()() {
        const drwnNNGraphImageData data(_baseName);
        const drwnNNGraphImage img(data);
        lock();
        _graph->appendImage(img);
        unlock();
    }
};

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphInitialize [OPTIONS] <imgList>\n";
    cerr << "OPTIONS:\n"
         << "  -i <filename>     :: input graph filename (for appending)\n"
         << "  -o <filename>     :: output graph filename (can be same as input)\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *inGraphFile = NULL;
    const char *outGraphFile = NULL;

    // process commandline arguments
    srand48((unsigned)time(NULL));
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", inGraphFile)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
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
    }

    // append images not in dataset
    DRWN_LOG_MESSAGE("Adding images to drwnNNGraph...");
#if 0
    for (unsigned i = 0; i < baseNames.size(); i++) {
        const int indx = graph.findImage(baseNames[i]);
        if (indx < 0) {
            graph.appendImage(drwnNNGraphImage(drwnNNGraphImageData(baseNames[i])));
        }
    }
#else
    drwnThreadPool threadPool;
    vector<drwnNNGraphAppendImageJob *> jobs;
    for (unsigned i = 0; i < baseNames.size(); i++) {
        const int indx = graph.findImage(baseNames[i]);
        if (indx < 0) {
            jobs.push_back(new drwnNNGraphAppendImageJob(&graph, baseNames[i]));
        }
    }

    threadPool.start();
    for (unsigned i = 0; i < jobs.size(); i++) {
        threadPool.addJob(jobs[i]);
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }
    jobs.clear();
#endif

    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");

    // write graph
    if (outGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Writing drwnNNGraph to " << outGraphFile << "...");
        graph.write(outGraphFile);
    }

    // clean up
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
