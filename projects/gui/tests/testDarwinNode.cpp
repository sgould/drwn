/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinNode.cpp
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

void testPCANode();
void testBoostingNode();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinNode [OPTIONS] (<command> (<node>))*\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "COMMANDS:\n"
         << "  list              :: list all registered nodes\n"
         << "  properties <node> :: show default properties for node <node>\n"
         << "  test <node>       :: regression test for node <node>\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    drwnNode *node;

    // TODO: refactor

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("list")
            cout << "--- NODE REGISTRY ---\n";
            vector<string> groups = drwnNodeFactory::get().getGroups();
            for (vector<string>::const_iterator it = groups.begin(); it != groups.end(); it++) {
                cout << " * " << *it << "\n";
                vector<string> nodes = drwnNodeFactory::get().getNodes(it->c_str());
                for (vector<string>::const_iterator jt = nodes.begin(); jt != nodes.end(); jt++) {
                    cout << "   - " << *jt << "\n";
                }
            }
            cout << "--- ---" << endl;
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("properties", p)
            cout << "--- NODE: " << p[0] << " ---\n";
            node = drwnNodeFactory::get().create(p[0]);
            node->printProperties(cout);
            delete node;
            cout << "--- ---" << endl;
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("test", p)
            drwnCodeProfiler::tic(drwnCodeProfiler::getHandle(p[0]));

            const char *testNodeStr = p[0];
            if (!strcasecmp(testNodeStr, "PCANode")) {
                testPCANode();
            } else if (!strcasecmp(testNodeStr, "BoostedClassifierNode")) {
                testBoostingNode();
            } else {
                DRWN_LOG_FATAL("unknown node \"" << testNodeStr << "\" for testing\n");
                usage();
                return -1;
            }

            drwnCodeProfiler::toc(drwnCodeProfiler::getHandle(p[0]));
        DRWN_CMDLINE_OPTION_END(1)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print(cerr);
    return 0;
}

// regression tests ----------------------------------------------------------

const double IRISDATA[150][4] = {
    {5.1, 3.5, 1.4, 0.2},
    {4.9, 3.0, 1.4, 0.2},
    {4.7, 3.2, 1.3, 0.2},
    {4.6, 3.1, 1.5, 0.2},
    {5.0, 3.6, 1.4, 0.2},
    {5.4, 3.9, 1.7, 0.4},
    {4.6, 3.4, 1.4, 0.3},
    {5.0, 3.4, 1.5, 0.2},
    {4.4, 2.9, 1.4, 0.2},
    {4.9, 3.1, 1.5, 0.1},
    {5.4, 3.7, 1.5, 0.2},
    {4.8, 3.4, 1.6, 0.2},
    {4.8, 3.0, 1.4, 0.1},
    {4.3, 3.0, 1.1, 0.1},
    {5.8, 4.0, 1.2, 0.2},
    {5.7, 4.4, 1.5, 0.4},
    {5.4, 3.9, 1.3, 0.4},
    {5.1, 3.5, 1.4, 0.3},
    {5.7, 3.8, 1.7, 0.3},
    {5.1, 3.8, 1.5, 0.3},
    {5.4, 3.4, 1.7, 0.2},
    {5.1, 3.7, 1.5, 0.4},
    {4.6, 3.6, 1.0, 0.2},
    {5.1, 3.3, 1.7, 0.5},
    {4.8, 3.4, 1.9, 0.2},
    {5.0, 3.0, 1.6, 0.2},
    {5.0, 3.4, 1.6, 0.4},
    {5.2, 3.5, 1.5, 0.2},
    {5.2, 3.4, 1.4, 0.2},
    {4.7, 3.2, 1.6, 0.2},
    {4.8, 3.1, 1.6, 0.2},
    {5.4, 3.4, 1.5, 0.4},
    {5.2, 4.1, 1.5, 0.1},
    {5.5, 4.2, 1.4, 0.2},
    {4.9, 3.1, 1.5, 0.1},
    {5.0, 3.2, 1.2, 0.2},
    {5.5, 3.5, 1.3, 0.2},
    {4.9, 3.1, 1.5, 0.1},
    {4.4, 3.0, 1.3, 0.2},
    {5.1, 3.4, 1.5, 0.2},
    {5.0, 3.5, 1.3, 0.3},
    {4.5, 2.3, 1.3, 0.3},
    {4.4, 3.2, 1.3, 0.2},
    {5.0, 3.5, 1.6, 0.6},
    {5.1, 3.8, 1.9, 0.4},
    {4.8, 3.0, 1.4, 0.3},
    {5.1, 3.8, 1.6, 0.2},
    {4.6, 3.2, 1.4, 0.2},
    {5.3, 3.7, 1.5, 0.2},
    {5.0, 3.3, 1.4, 0.2},
    {7.0, 3.2, 4.7, 1.4},
    {6.4, 3.2, 4.5, 1.5},
    {6.9, 3.1, 4.9, 1.5},
    {5.5, 2.3, 4.0, 1.3},
    {6.5, 2.8, 4.6, 1.5},
    {5.7, 2.8, 4.5, 1.3},
    {6.3, 3.3, 4.7, 1.6},
    {4.9, 2.4, 3.3, 1.0},
    {6.6, 2.9, 4.6, 1.3},
    {5.2, 2.7, 3.9, 1.4},
    {5.0, 2.0, 3.5, 1.0},
    {5.9, 3.0, 4.2, 1.5},
    {6.0, 2.2, 4.0, 1.0},
    {6.1, 2.9, 4.7, 1.4},
    {5.6, 2.9, 3.6, 1.3},
    {6.7, 3.1, 4.4, 1.4},
    {5.6, 3.0, 4.5, 1.5},
    {5.8, 2.7, 4.1, 1.0},
    {6.2, 2.2, 4.5, 1.5},
    {5.6, 2.5, 3.9, 1.1},
    {5.9, 3.2, 4.8, 1.8},
    {6.1, 2.8, 4.0, 1.3},
    {6.3, 2.5, 4.9, 1.5},
    {6.1, 2.8, 4.7, 1.2},
    {6.4, 2.9, 4.3, 1.3},
    {6.6, 3.0, 4.4, 1.4},
    {6.8, 2.8, 4.8, 1.4},
    {6.7, 3.0, 5.0, 1.7},
    {6.0, 2.9, 4.5, 1.5},
    {5.7, 2.6, 3.5, 1.0},
    {5.5, 2.4, 3.8, 1.1},
    {5.5, 2.4, 3.7, 1.0},
    {5.8, 2.7, 3.9, 1.2},
    {6.0, 2.7, 5.1, 1.6},
    {5.4, 3.0, 4.5, 1.5},
    {6.0, 3.4, 4.5, 1.6},
    {6.7, 3.1, 4.7, 1.5},
    {6.3, 2.3, 4.4, 1.3},
    {5.6, 3.0, 4.1, 1.3},
    {5.5, 2.5, 4.0, 1.3},
    {5.5, 2.6, 4.4, 1.2},
    {6.1, 3.0, 4.6, 1.4},
    {5.8, 2.6, 4.0, 1.2},
    {5.0, 2.3, 3.3, 1.0},
    {5.6, 2.7, 4.2, 1.3},
    {5.7, 3.0, 4.2, 1.2},
    {5.7, 2.9, 4.2, 1.3},
    {6.2, 2.9, 4.3, 1.3},
    {5.1, 2.5, 3.0, 1.1},
    {5.7, 2.8, 4.1, 1.3},
    {6.3, 3.3, 6.0, 2.5},
    {5.8, 2.7, 5.1, 1.9},
    {7.1, 3.0, 5.9, 2.1},
    {6.3, 2.9, 5.6, 1.8},
    {6.5, 3.0, 5.8, 2.2},
    {7.6, 3.0, 6.6, 2.1},
    {4.9, 2.5, 4.5, 1.7},
    {7.3, 2.9, 6.3, 1.8},
    {6.7, 2.5, 5.8, 1.8},
    {7.2, 3.6, 6.1, 2.5},
    {6.5, 3.2, 5.1, 2.0},
    {6.4, 2.7, 5.3, 1.9},
    {6.8, 3.0, 5.5, 2.1},
    {5.7, 2.5, 5.0, 2.0},
    {5.8, 2.8, 5.1, 2.4},
    {6.4, 3.2, 5.3, 2.3},
    {6.5, 3.0, 5.5, 1.8},
    {7.7, 3.8, 6.7, 2.2},
    {7.7, 2.6, 6.9, 2.3},
    {6.0, 2.2, 5.0, 1.5},
    {6.9, 3.2, 5.7, 2.3},
    {5.6, 2.8, 4.9, 2.0},
    {7.7, 2.8, 6.7, 2.0},
    {6.3, 2.7, 4.9, 1.8},
    {6.7, 3.3, 5.7, 2.1},
    {7.2, 3.2, 6.0, 1.8},
    {6.2, 2.8, 4.8, 1.8},
    {6.1, 3.0, 4.9, 1.8},
    {6.4, 2.8, 5.6, 2.1},
    {7.2, 3.0, 5.8, 1.6},
    {7.4, 2.8, 6.1, 1.9},
    {7.9, 3.8, 6.4, 2.0},
    {6.4, 2.8, 5.6, 2.2},
    {6.3, 2.8, 5.1, 1.5},
    {6.1, 2.6, 5.6, 1.4},
    {7.7, 3.0, 6.1, 2.3},
    {6.3, 3.4, 5.6, 2.4},
    {6.4, 3.1, 5.5, 1.8},
    {6.0, 3.0, 4.8, 1.8},
    {6.9, 3.1, 5.4, 2.1},
    {6.7, 3.1, 5.6, 2.4},
    {6.9, 3.1, 5.1, 2.3},
    {5.8, 2.7, 5.1, 1.9},
    {6.8, 3.2, 5.9, 2.3},
    {6.7, 3.3, 5.7, 2.5},
    {6.7, 3.0, 5.2, 2.3},
    {6.3, 2.5, 5.0, 1.9},
    {6.5, 3.0, 5.2, 2.0},
    {6.2, 3.4, 5.4, 2.3},
    {5.9, 3.0, 5.1, 1.8}
};

int IRISLABLES[150] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};

void testPCANode()
{
    drwnNode *node;
    drwnGraph *graph = new drwnGraph("Test Network");
    graph->addNode(node = new drwnRandomSourceNode("Data"));
    node->setProperty(node->findProperty("numRecords"), 1);
    node->setProperty(node->findProperty("numFeatures"), 10);
    node->setProperty(node->findProperty("minObservations"), 1000);
    node->setProperty(node->findProperty("maxObservations"), 1000);
    graph->addNode(node = new drwnStdOutSinkNode("Output"));

    graph->addNode(node = new drwnPCANode("Test Node"));
    node->setProperty(node->findProperty("outputDim"), 3);

    node->getInputPort("dataIn")->connect(graph->getNode(0)->getOutputPort(0));
    node->getOutputPort("dataOut")->connect(graph->getNode(1)->getInputPort(0));

    graph->initializeParameters();

    cout << "Translation Vector:\n"
         << node->getVectorProperty(node->findProperty("translation")).transpose() << "\n"
         << "Projection Matrix\n"
         << node->getMatrixProperty(node->findProperty("projection"))
         << endl;
}

void testBoostingNode()
{
    // TODO
    DRWN_ASSERT(false);
}
