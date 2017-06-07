/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinGraph.cpp
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

// darwin library headers
#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnNodes.h"

using namespace std;
using namespace Eigen;

// prototypes ----------------------------------------------------------------

void printCache();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinGraph [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << "  -nodeRegistry     :: show node registry\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    /*
    if (argc == 1) {
        usage();
        return 0;
    }
    */

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("-nodeRegistry")
        vector<string> groupNames = drwnNodeFactory::get().getGroups();
        for (vector<string>::const_iterator ig = groupNames.begin(); ig != groupNames.end(); ig++) {
            cout << *ig << "\n";
        }
        DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());
    
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // run tests
    DRWN_LOG_VERBOSE("creating graph...");
    drwnGraph graph("test graph");
    drwnDatabase *db = drwnDbManager::get().openDatabase("testdb");
    graph.setDatabase(db);

    printCache();

    graph.addNode(new drwnTextFileSourceNode("nodeA"));
    graph.addNode(new drwnTextFileSinkNode("nodeB"));
    graph.addNode(new drwnTextFileSourceNode("nodeC"));

    graph.addNode(new drwnRandomSourceNode("srcNodeA"));
    graph.addNode(new drwnStdOutSinkNode("snkNodeA"));

    //graph.write(cout);

    // connect the graph
    graph.connectNodes(graph.getNode("srcNodeA"), "dataOut",
        graph.getNode("snkNodeA"), "dataIn");

    // evaluate the graph
    graph.getNode("srcNodeA")->evaluateForwards();
    printCache();
    graph.getNode("snkNodeA")->evaluateForwards();
    printCache();

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print(cerr);
    return 0;
}

// private functions ---------------------------------------------------------

void printCache()
{
    drwnDataCache& cache = drwnDataCache::get();
    cout << "cache: " << cache.getSize() << " of " 
         << cache.getSizeLimit() << " entries (" 
         << setprecision(3) << (100.0 * cache.getSize() / cache.getSizeLimit()) << "%)\n"
         << "       " << cache.getMemoryUsed() << " of " 
         << cache.getMemoryLimit() << " bytes ("
         << setprecision(3) << (100.0 * cache.getMemoryUsed() / cache.getMemoryLimit()) << "%)\n";
    cout << "\n";
}
