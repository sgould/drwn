/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    buildPatchMatchGraph.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Implements the PatchMatchGraph algorithm on sets of images.
**  See Gould and Zhang (ECCV 2012) and Barnes et al. (ECCV 2010).
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
#include "cxcore.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./buildPatchMatchGraph [OPTIONS] <imgList>\n";
    cerr << "OPTIONS:\n"
         << "  -d <directory>    :: image directory (default: .)\n"
         << "  -e <extension>    :: image filename extension (default: .jpg)\n"
         << "  -i <filename>     :: input graph filename (for initialization)\n"
         << "  -o <filename>     :: output graph filename (can be same as input)\n"
         << "  -m <iterations>   :: maximum iterations\n"
         << "  -eqv <filename>   :: load equivalence classes (can be repeated)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *imgDirectory = ".";
    const char *imgExtension = ".jpg";
    const char *inGraphFile = NULL;
    const char *outGraphFile = NULL;
    vector<const char *> eqvFiles;
    int maxIterations = 10;
    bool bVisualize = false;
    bool bPause = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-d", imgDirectory)
        DRWN_CMDLINE_STR_OPTION("-e", imgExtension)
        DRWN_CMDLINE_STR_OPTION("-i", inGraphFile)
        DRWN_CMDLINE_STR_OPTION("-o", outGraphFile)
        DRWN_CMDLINE_VEC_OPTION("-eqv", eqvFiles)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_BOOL_OPTION("-pause", bPause)
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
    drwnPatchMatchGraph graph;
    graph.imageDirectory = string(imgDirectory);
    graph.imageExtension = string(imgExtension);

    if (inGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Loading PatchMatchGraph from " << inGraphFile << "...");
        graph.read(inGraphFile);
        DRWN_LOG_MESSAGE("...graph has " << graph.size() << " images");

        // disable images (and re-enable if in image list)
        for (unsigned i = 0; i < graph.size(); i++) {
            graph[i].bActive = false;
        }

        // append images not in dataset / only search on images in list
        for (unsigned i = 0; i < baseNames.size(); i++) {
            const int indx = graph.findImage(baseNames[i]);
            if (indx < 0) {
                graph.appendImage(baseNames[i]);
            } else {
                graph[indx].bActive = true;
            }
        }

    } else {
        DRWN_LOG_MESSAGE("Adding images to PatchMatchGraph...");
        graph.appendImages(baseNames);
    }

    // load and set equivalence classes
    for (int eqvClass = 0; eqvClass < (int)eqvFiles.size(); eqvClass++) {
        vector<string> eqvBaseNames = drwnReadFile(eqvFiles[eqvClass]);
        for (vector<string>::const_iterator it = eqvBaseNames.begin(); it != eqvBaseNames.end(); ++it) {
            const int imgIndx = graph.findImage(*it);
            if (imgIndx < 0) {
                DRWN_LOG_WARNING("could not find " << *it << " in graph for equivalence class " << eqvClass);
            } else {
                graph[imgIndx].eqvClass = eqvClass;
            }
        }
    }

    // check that the dataset contains enough images
    if (graph.size() <= drwnPatchMatchGraph::K) {
        drwnPatchMatchGraph::K = graph.size() - 1;
        DRWN_LOG_WARNING("...reducing drwnPatchMatchGraph::K to " << drwnPatchMatchGraph::K);
    }

    // show settings
    DRWN_LOG_MESSAGE("Initializing PatchMatchGraph...");
    DRWN_LOG_VERBOSE("  Patch Size: " << graph.patchWidth() << "-by-" << graph.patchHeight());
    DRWN_LOG_VERBOSE("  Max. Image Size: " << drwnPatchMatchImagePyramid::MAX_SIZE);
    DRWN_LOG_VERBOSE("  Max. Pyramid Levels: " << drwnPatchMatchImagePyramid::MAX_LEVELS);
    DRWN_LOG_VERBOSE("  Matches per pixel: " << drwnPatchMatchGraph::K << " @ " <<
        (drwnPatchMatchGraphLearner::TOP_VAR_PATCHES * 100.0) << "%");
    DRWN_LOG_VERBOSE("  Search Decay Rate: " << drwnPatchMatchGraphLearner::SEARCH_DECAY_RATE);
    DRWN_LOG_VERBOSE("  Forward Enrichment: " << drwnPatchMatchGraphLearner::FORWARD_ENRICHMENT_K);
    DRWN_LOG_VERBOSE("  Inverse Enrichment: " << (drwnPatchMatchGraphLearner::DO_INVERSE_ENRICHMENT ? "yes" : "no"));
    DRWN_LOG_VERBOSE("  Allow Horizontal Flips: "
        << ((drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00 ? "yes" : "no"));
    DRWN_LOG_VERBOSE("  Allow Vertical Flips: "
        << ((drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_VFLIP) != 0x00 ? "yes" : "no"));
    DRWN_LOG_VERBOSE("  Local Search: " << (drwnPatchMatchGraphLearner::DO_LOCAL_SEARCH ? "yes" : "no"));
    DRWN_LOG_VERBOSE("  Random Exhuastive Search: " << (drwnPatchMatchGraphLearner::DO_EXHAUSTIVE ? "yes" : "no"));

    // initialize graph
    drwnPatchMatchGraphLearner learner(graph);
    learner.initialize();
    pair<double, double> lastEnergy = graph.energy();
    DRWN_LOG_VERBOSE("  Matchable Pixels: " << drwnPatchMatchUtils::countMatchablePixels(graph));
    DRWN_LOG_MESSAGE("0. (" << millisecondsToString(1000 * difftime(std::time(NULL), startTime))
        << ") energy = " << lastEnergy.first << " top = " << lastEnergy.second);

#if 1
    if (bVisualize) {
        // show quality
        drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchQuality(graph), string("quality"), false);
        // show transform
        drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchTransforms(graph), string("transforms"), false);
        // show image targets
        drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchTargets(graph), string("targets"), bPause);
    }
#endif

    int nIterations = 0;
    while (nIterations < maxIterations) {

        // perform update
        learner.update();

#if 1
        //! \todo move into monitor function?
        if (bVisualize) {
            // show quality
            drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchQuality(graph), string("quality"), false);
            // show transform
            drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchTransforms(graph), string("transforms"), false);
            // show image targets
            drwnShowDebuggingImage(drwnPatchMatchVis::visualizeMatchTargets(graph), string("targets"), bPause);
        }
#endif

        // check energy
        pair<double, double> e = graph.energy();
        DRWN_LOG_MESSAGE((nIterations + 1) << ". ("
            << millisecondsToString(1000 * difftime(std::time(NULL), startTime))
            << ") energy = " << e.first << " top = " << e.second);
        if (e.first == lastEnergy.first) break;
        lastEnergy = e;

        nIterations += 1;
    }

    DRWN_LOG_MESSAGE("patchMatch converged after " << (nIterations + 1) << " iterations");

    // write graph
    if (outGraphFile != NULL) {
        DRWN_LOG_MESSAGE("Writing PatchMatchGraph to " << outGraphFile << "...");
        graph.write(outGraphFile);
    }

    // clean up and print profile information
    if ((bVisualize) && (graph.size() == 2)) {
        cvWaitKey(-1);
    }

    cvDestroyAllWindows();
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
