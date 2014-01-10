/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    patchMatchLabelTransfer.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Performs label transfer from image in a PatchMatchGraph to a
**  test image. This is a variant of the method described in Gould and Zhang
**  (ECCV 2012).
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>

// eigen matrix library headers
#include "Eigen/Core"

// opencv library headers
#include "cv.h"
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
    cerr << "USAGE: ./patchMatchLabelTransfer [OPTIONS] <graph> (<imgList> | <baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -outLabels <ext>  :: extension for label output (default: none)\n"
         << "  -outImages <ext>  :: extension for image output (default: none)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *outLabelExt = NULL;
    const char *outImageExt = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-outLabels", outLabelExt)
        DRWN_CMDLINE_STR_OPTION("-outImages", outImageExt)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

    const char *graphFile = DRWN_CMDLINE_ARGV[0];
    const char *evalList = DRWN_CMDLINE_ARGV[1];

    vector<string> baseNames;
    if (drwnFileExists(evalList)) {
        DRWN_LOG_MESSAGE("Reading evaluation list from " << evalList << "...");
        baseNames = drwnReadFile(evalList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << evalList << "...");
        baseNames.push_back(string(evalList));
    }

    // load graph
    DRWN_LOG_MESSAGE("Loading PatchMatchGraph from " << graphFile << "...");
    drwnPatchMatchGraph graph;
    graph.read(graphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.size() << " images");

    // load labels
    map<string, MatrixXi> labels;
    for (unsigned i = 0; i < graph.size(); i++) {
        const string lblFilename = gMultiSegConfig.filename("lblDir", graph[i].name(), "lblExt");
        DRWN_LOG_DEBUG("loading groundtruth labels from " << lblFilename << "...");
        if (!drwnFileExists(lblFilename.c_str())) {
            DRWN_LOG_WARNING("groundtruth does not exist for " << graph[i].name());
        } else {
            MatrixXi L = MatrixXi::Zero(graph[i].height(), graph[i].width());
            drwnReadMatrix(L, lblFilename.c_str());
            labels[graph[i].name()] = L;
        }
    }

    // do label transfer
    const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
    for (unsigned i = 0; i < baseNames.size(); i++) {
        // find index of test image
        const int imgIndx = graph.findImage(baseNames[i]);
        if (imgIndx < 0) {
            DRWN_LOG_ERROR("could not find image " << baseNames[i] << " in the graph");
            continue;
        }
        DRWN_LOG_VERBOSE("processing " << graph[imgIndx].width() << "-by-"
            << graph[imgIndx].height() << " image " << baseNames[i] << "...");

        // create marginals for labels
        vector<MatrixXd> marginals(nLabels,
            MatrixXd::Zero(graph[imgIndx].height(), graph[imgIndx].width()));

        // iterate over matches
        for (unsigned imgScale = 0; imgScale < graph[imgIndx].levels(); imgScale++) {
            for (int y = 0; y < graph[imgIndx][imgScale].height() - (int)graph.patchHeight() + 1; y++) {
                for (int x = 0; x < graph[imgIndx][imgScale].width() - (int)graph.patchWidth() + 1; x++) {
                    const cv::Rect tgtRect = drwnTransformROI(cv::Rect(x, y, graph.patchWidth(), graph.patchHeight()),
                        cv::Size(graph[imgIndx][imgScale].width(), graph[imgIndx][imgScale].height()),
                        cv::Size(graph[imgIndx].width(), graph[imgIndx].height()));

                    const drwnPatchMatchEdgeList& e =
                        graph.edges(drwnPatchMatchNode(imgIndx, imgScale, x, y));

                    double rank = 1.0;
                    for (drwnPatchMatchEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                        const drwnPatchMatchNode &v = kt->targetNode;
                        const cv::Rect srcRect = drwnTransformROI(cv::Rect(v.xPosition, v.yPosition, graph.patchWidth(), graph.patchHeight()),
                            cv::Size(graph[v.imgIndx][v.imgScale].width(), graph[v.imgIndx][v.imgScale].height()),
                            cv::Size(graph[v.imgIndx].width(), graph[v.imgIndx].height()));

                        map<string, MatrixXi>::const_iterator lt = labels.find(graph[v.imgIndx].name());
                        DRWN_ASSERT_MSG(lt != labels.end(), graph[v.imgIndx].name());

                        for (int yy = 0; yy < tgtRect.height; yy++) {
                            for (int xx = 0; xx < tgtRect.width; xx++) {
                                const int lbl = lt->second(srcRect.y + yy * srcRect.height / tgtRect.height,
                                    srcRect.x + xx * srcRect.width / tgtRect.width);
                                if (lbl < 0) continue;
                                marginals[lbl](tgtRect.y + yy, tgtRect.x + xx) += 1.0 / rank;
                            }
                        }

                        rank += 1.0;
                    }
                }
            }
        }

        // find most likely labeling for instance
        const string imgFilename = gMultiSegConfig.filename("imgDir", baseNames[i], "imgExt");
        drwnSegImageInstance instance(imgFilename.c_str(), baseNames[i].c_str());
        instance.pixelLabels = MatrixXi::Constant(graph[imgIndx].height(), graph[imgIndx].width(), -1);
        for (int y = 0; y < instance.pixelLabels.rows(); y++) {
            for (int x = 0; x < instance.pixelLabels.cols(); x++) {
                double p = 0.0;
                for (int lbl = 0; lbl < nLabels; lbl++) {
                    if (marginals[lbl](y, x) > p) {
                        p = marginals[lbl](y, x);
                        instance.pixelLabels(y, x) = lbl;
                    }
                }
            }
        }

        // save labels
        if (outLabelExt != NULL) {
            string filename = gMultiSegConfig.filebase("outputDir", baseNames[i]) + string(outLabelExt);
            ofstream ofs(filename.c_str());
            ofs << instance.pixelLabels << "\n";
            ofs.close();
        }

        // visualize and save images
        if (bVisualize || (outImageExt != NULL)) {
            cv::Mat canvas = drwnMultiSegVis::visualizeInstance(instance);
            if (bVisualize) {
                drwnShowDebuggingImage(canvas, string("predicted labels"), false);
            }
            if (outImageExt != NULL) {
                string filename = gMultiSegConfig.filebase("outputDir", baseNames[i]) + string(outImageExt);
                cv::imwrite(filename.c_str(), canvas);
            }
        }
    }

    // clean up and print profile information
    if ((bVisualize) && (baseNames.size() == 1)) {
        cv::waitKey(-1);
    }

    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
