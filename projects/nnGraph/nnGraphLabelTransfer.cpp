/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphLabelTransfer.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION: Application for transfering labels across an drwnNNGraph.
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

//! \todo no need to load graph features

// label distribution ----------------------------------------------------------

typedef drwnNNGraphNodeAnnotation<VectorXd> drwnNNGraphLabelDistributions;

// load data thread ------------------------------------------------------------

class loadDataJob : public drwnThreadJob {
protected:
    drwnNNGraphLabelDistributions& _labelDistributions;
    const drwnNNGraph& _graph;
    unsigned _imgIndx;

public:
    loadDataJob(drwnNNGraphLabelDistributions& labelDistributions, const drwnNNGraph& graph, unsigned imgIndx) :
        _labelDistributions(labelDistributions), _graph(graph), _imgIndx(imgIndx) { /* do nothing */ }
    ~loadDataJob() { /* do nothing */ }

    void operator()() {
        const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
        drwnNNGraphImageData data(_graph[_imgIndx].name());
        DRWN_ASSERT(data.numSegments() == _graph[_imgIndx].numNodes());
        lock();
        _labelDistributions[_imgIndx] = data.getSegmentLabelMarginals(nLabels);
        unlock();
    }
};

// inference thread ------------------------------------------------------------

class labelTransferJob : public drwnThreadJob {
public:
    const drwnNNGraph *graph;
    const drwnNNGraphLabelDistributions *labelDistributions;

    double pairwiseSmoothness;

    const char *outLabelExt;
    const char *outImageExt;
    bool bVisualize;

    string baseName;

public:
    labelTransferJob() : graph(NULL), labelDistributions(NULL), pairwiseSmoothness(0.0),
        outLabelExt(NULL), outImageExt(NULL), bVisualize(false) { /* do nothing */ }
    ~labelTransferJob() { /* do nothing */ }

    void operator()() {
        DRWN_ASSERT((graph != NULL) && (labelDistributions != NULL));
        const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
        const int imgIndx = graph->findImage(baseName);
        DRWN_ASSERT_MSG(imgIndx >= 0, baseName);

        // load image data
        drwnNNGraphImageData data(baseName);

        lock();
        DRWN_LOG_STATUS("processing " << data.width() << "-by-"
            << data.height() << " image " << baseName << "...");
        unlock();
        DRWN_ASSERT_MSG(data.numSegments() == (*graph)[imgIndx].numNodes(),
            data.numSegments() << " != " << (*graph)[imgIndx].numNodes());

        // find most likely labeling for instance
        drwnSegImageInstance instance(data.image(), baseName.c_str());
        instance.unaries.resize(instance.size(), vector<double>(nLabels, 0.0));

        for (int y = 0; y < instance.pixelLabels.rows(); y++) {
            for (int x = 0; x < instance.pixelLabels.cols(); x++) {
                VectorXd marginals = VectorXd::Constant(nLabels, DRWN_EPSILON);

                for (int c = 0; c < data.segments().channels(); c++) {
                    const int segId = data.segments()[c].at<int>(y, x);
                    if (segId < 0) continue;

                    const drwnNNGraphEdgeList& e = (*graph)[imgIndx][segId].edges;
                    if (e.empty()) continue;

                    // accumulate label votes
                    double rank = 1.0;
                    for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {

                        // check that labels exist for the matching image
                        if ((*labelDistributions)[kt->targetNode.imgIndx].empty())
                            continue;

                        const double w = 1.0 / rank;
                        marginals += w * (*labelDistributions)[kt->targetNode];
                        rank += 1.0;
                    }
                }

                Eigen::Map<VectorXd>(&instance.unaries[instance.pixel2Indx(x, y)][0], nLabels) =
                    -1.0 * marginals.array().log();
            }
        }

        // infer labels
        drwnPixelSegCRFInference inf;
        inf.alphaExpansion(&instance, pairwiseSmoothness);

        // save labels
        if (outLabelExt != NULL) {
            string filename = gMultiSegConfig.filebase("outputDir", baseName) + string(outLabelExt);
            ofstream ofs(filename.c_str());
            ofs << instance.pixelLabels << "\n";
            ofs.close();
        }

        // visualize and save images
        if (bVisualize || (outImageExt != NULL)) {
            cv::Mat canvas = drwnMultiSegVis::visualizeInstance(instance);
            if (bVisualize) {
                lock();
                string wndName = string("predicted labels (thread: ") + toString(threadId()) + string(")");
                drwnShowDebuggingImage(canvas, wndName, false);
                unlock();
            }
            if (outImageExt != NULL) {
                string filename = gMultiSegConfig.filebase("outputDir", baseName) + string(outImageExt);
                cv::imwrite(filename.c_str(), canvas);
            }
        }
    }
};

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphLabelTransfer [OPTIONS] <graph> (<imgList> | <baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -pairwise <p>     :: strength of pairwise smoothness prior (default: 0.0)\n"
         << "  -outLabels <ext>  :: extension for label output (default: none)\n"
         << "  -outImages <ext>  :: extension for image output (default: none)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    double pairwiseSmoothness = 0.0;
    const char *outLabelExt = NULL;
    const char *outImageExt = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_REAL_OPTION("-pairwise", pairwiseSmoothness)
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
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << graphFile << "...");
    drwnNNGraph graph;
    graph.read(graphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");
    DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");

    const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
    DRWN_LOG_VERBOSE("...transferring " << nLabels << " labels");

    // load image data and superpixel labels
    DRWN_LOG_MESSAGE("Loading image data...");
    drwnNNGraphLabelDistributions labelDistributions(graph, VectorXd());
    drwnThreadPool threadPool;
    vector<loadDataJob *> loadJobs;
    threadPool.start();
    for (unsigned i = 0; i < graph.numImages(); i++) {
        loadJobs.push_back(new loadDataJob(labelDistributions, graph, i));
        threadPool.addJob(loadJobs.back());
    }
    threadPool.finish();

    for (unsigned i = 0; i < loadJobs.size(); i++) {
        delete loadJobs[i];
    }

    // do label transfer
    threadPool.start();
    vector<labelTransferJob *> jobs;
    for (unsigned i = 0; i < baseNames.size(); i++) {
        // find index of test image
        const int imgIndx = graph.findImage(baseNames[i]);
        if (imgIndx < 0) {
            DRWN_LOG_ERROR("could not find image " << baseNames[i] << " in the graph");
            continue;
        }

        jobs.push_back(new labelTransferJob());
        jobs.back()->graph = &graph;
        jobs.back()->labelDistributions = &labelDistributions;
        jobs.back()->pairwiseSmoothness = pairwiseSmoothness;
        jobs.back()->outLabelExt = outLabelExt;
        jobs.back()->outImageExt = outImageExt;
        jobs.back()->bVisualize = bVisualize;
        jobs.back()->baseName = baseNames[i];

        threadPool.addJob(jobs.back());
    }
    threadPool.finish();

    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }

    // clean up and print profile information
    if ((bVisualize) && (baseNames.size() == 1)) {
        cv::waitKey(-1);
    }

    // clean up
    if (bVisualize) cv::destroyAllWindows();
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
