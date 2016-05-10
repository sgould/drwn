/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    demoClassifier.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Demonstration commandline classifier using the Darwin framework.
**
*****************************************************************************/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnNodes.h"

using namespace std;

// main ----------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./demoClassifier [OPTIONS] <data file> <labels file>\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // process commandline propertys
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        // TODO
    DRWN_END_CMDLINE_PROCESSING(usage());
    
    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));
    const char *DATAFILE = DRWN_CMDLINE_ARGV[0];
    const char *LABELFILE = DRWN_CMDLINE_ARGV[1];

    // build graph
    drwnGraph *graph = new drwnGraph("Demo Classifier");

    drwnNode *node;
    graph->addNode(node = new drwnTextFileSourceNode("Data"));
    node->setProperty(node->findProperty("filename"), DATAFILE);
    graph->addNode(node = new drwnTextFileSourceNode("Labels"));
    node->setProperty(node->findProperty("filename"), LABELFILE);

    graph->addNode(node = new drwnMultiClassLogisticNode("Classifier"));
    node->getInputPort("dataIn")->connect(graph->getNode(0)->getOutputPort(0));
    node->getInputPort("targetIn")->connect(graph->getNode(1)->getOutputPort(0));

    graph->addNode(node = new drwnStdOutSinkNode("Output"));
    node->getInputPort("dataIn")->connect(graph->getNode(2)->getOutputPort(0));

    //graph->setDatabase(drwnDbManager::get().openDatabase("tmpdb"));
    graph->setDatabase(drwnDbManager::get().openMemoryDatabase());

    // load in data
    graph->getNode(0)->initializeForwards();
    graph->getNode(0)->evaluateForwards();
    graph->getNode(0)->finalizeForwards();

    graph->getNode(1)->initializeForwards();
    graph->getNode(1)->evaluateForwards();
    graph->getNode(1)->finalizeForwards();

    graph->getNode(2)->initializeParameters();
    graph->getNode(2)->initializeForwards();
    graph->getNode(2)->evaluateForwards();
    graph->getNode(2)->finalizeForwards();

    graph->getNode(3)->initializeForwards();
    graph->getNode(3)->evaluateForwards();
    graph->getNode(3)->finalizeForwards();
    
    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print(cerr);
    return 0;
}


