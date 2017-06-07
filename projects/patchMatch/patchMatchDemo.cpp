/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    patchMatchDemo.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./patchMatchDemo [OPTIONS] <imgA> <imgB>\n";
    cerr << "OPTIONS:\n"
         << "  -m <iterations>   :: maximum iterations (default: 100)\n"
         << "  -o <filestem>     :: save PatchMatchGraph to <filestem>\n"
         << "  -x                :: visualize matches\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *outputStem = NULL;
    int maxIterations = 100;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-o", outputStem)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read images
    const string nameA = string(DRWN_CMDLINE_ARGV[0]);
    const string nameB = string(DRWN_CMDLINE_ARGV[1]);

    cv::Mat imgA = cv::imread(nameA, CV_LOAD_IMAGE_COLOR);
    cv::Mat imgB = cv::imread(nameB, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(imgA.data != NULL, nameA);
    DRWN_ASSERT_MSG(imgB.data != NULL, nameB);

    // construct patch match graph
    drwnPatchMatchImagePyramid::MAX_LEVELS = 1;
    drwnPatchMatchImagePyramid::MAX_SIZE = 320;
#if 1
    drwnPatchMatchGraph::K = 1;
#else
    drwnPatchMatchImageRecord::ALLOW_MULTIPLE = true;
    drwnPatchMatchGraph::K = 2;
#endif
    drwnPatchMatchGraph graph;
    graph.appendImage(nameA);
    graph.appendImage(nameB);

    // initialize the patchMatchGraph
    drwnPatchMatchGraphLearner learner(graph);
    learner.initialize();

    pair<double, double> lastEnergy = graph.energy();
    DRWN_LOG_MESSAGE("0. energy = " << lastEnergy.first);

    // learn the patchMatchGraph
    int nIterations = 0;
    drwnPatchMatchGraphRepaint repaint(graph);
    while (nIterations < maxIterations) {

        // perform update
        learner.update();

        // show image, best match quality, worst match quality, repainted image
        if (bVisualize) {
            vector<cv::Mat> views;
            views.push_back(imgA);
            views.push_back(drwnPatchMatchVis::visualizeMatchQuality(graph, 0));
            if (drwnPatchMatchGraph::K > 1) {
                views.push_back(drwnPatchMatchVis::visualizeMatchQuality(graph, 0, 0.0, drwnPatchMatchGraph::K - 1));
            }
            views.push_back(repaint.retarget(0));
            views.push_back(imgB);
            views.push_back(drwnPatchMatchVis::visualizeMatchQuality(graph, 1));
            if (drwnPatchMatchGraph::K > 1) {
                views.push_back(drwnPatchMatchVis::visualizeMatchQuality(graph, 1, 0.0, drwnPatchMatchGraph::K - 1));
            }
            views.push_back(repaint.retarget(1));
            drwnShowDebuggingImage(views, string("patchMatchDemo"), false, 2);
        }

        // check energy
        pair<double, double> e = graph.energy();
        DRWN_LOG_MESSAGE((nIterations + 1) << ". energy = " << e.first);
        if (e.first == lastEnergy.first) break;
        lastEnergy = e;

        nIterations += 1;
    }

    // show convergence message and wait for key press if visualizing
    DRWN_LOG_MESSAGE("patchMatch converged after " << (nIterations + 1)
        << " iterations with energy " << lastEnergy.first);
    if (bVisualize) cvWaitKey(-1);

    // save patchMatchGraph
    if (outputStem != NULL) {
        DRWN_LOG_DEBUG("writing patchMatchGraph to " << outputStem);
        graph.write(outputStem);
#if 1
        // debug
        drwnPatchMatchGraph graph2;
        graph2.read(outputStem);
        DRWN_LOG_VERBOSE("saved graph has energy " << graph.energy().first);
        DRWN_LOG_VERBOSE("loaded graph has energy " << graph2.energy().first);
#endif
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
