/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    nnGraphOnlineLabelTransfer.cpp
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
#include "drwnNNGraphMoves.h"
#include "drwnNNGraphThreadedMoves.h"
#include "drwnNNGraphVis.h"

using namespace std;
using namespace Eigen;

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

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./nnGraphOnlineLabelTransfer [OPTIONS] <graph> <imgFile> <segFile>\n";
    cerr << "OPTIONS:\n"
         << "  -m <iterations>   :: maximum iterations\n"
         << "  -t <xformFile>    :: apply transformation to features from <imgFile>\n"
         << "  -pairwise <p>     :: strength of pairwise smoothness prior (default: 0.0)\n"
         << "  -outLabel <file>  :: output label filename (default: none)\n"
         << "  -outImage <file>  :: output image filename (default: none)\n"
         << "  -labelCache <file>:: filename for storing label cache (default: none)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int maxIterations = 100;
    const char *xformFile = NULL;
    double pairwiseSmoothness = 0.0;
    const char *outLabel = NULL;
    const char *outImage = NULL;
    const char *labelCache = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-m", maxIterations)
        DRWN_CMDLINE_STR_OPTION("-t", xformFile)
        DRWN_CMDLINE_REAL_OPTION("-pairwise", pairwiseSmoothness)
        DRWN_CMDLINE_STR_OPTION("-outLabel", outLabel)
        DRWN_CMDLINE_STR_OPTION("-outImage", outImage)
        DRWN_CMDLINE_STR_OPTION("-labelCache", labelCache)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 3) {
        usage();
        return -1;
    }

    const int hMain = drwnCodeProfiler::getHandle("main");
    drwnCodeProfiler::tic(hMain);

    const char *graphFile = DRWN_CMDLINE_ARGV[0];
    const char *imageFile = DRWN_CMDLINE_ARGV[1];
    const char *segFile = DRWN_CMDLINE_ARGV[2];

    // load graph
    DRWN_LOG_MESSAGE("Loading drwnNNGraph from " << graphFile << "...");
    drwnNNGraph graph;
    graph.read(graphFile);
    DRWN_LOG_MESSAGE("...graph has " << graph.numImages() << " images");
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");
    DRWN_LOG_VERBOSE("...graph has " << graph.numEdges() << " edges");

    const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
    DRWN_LOG_VERBOSE("...transferring " << nLabels << " labels");

    // load superpixel labels
    drwnNNGraphLabelDistributions labelDistributions(graph, VectorXd());
    if ((labelCache != NULL) && drwnFileExists(labelCache)) {
        DRWN_LOG_MESSAGE("Loading label data from cache...");
        ifstream ifs(labelCache, ios::binary);
        DRWN_ASSERT_MSG(!ifs.fail(), labelCache);
        for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
            uint32_t n;
            ifs.read((char *)&n, sizeof(uint32_t));
            labelDistributions[imgIndx].resize(n);
            vector<double> x(nLabels);
            for (unsigned segId = 0; segId < n; segId++) {
                ifs.read((char *)&x[0], nLabels * sizeof(double));
                labelDistributions[imgIndx][segId] = Eigen::Map<VectorXd>(&x[0], nLabels);
            }
            DRWN_ASSERT_MSG(!ifs.fail(), labelCache);
        }
    } else {
        // load image data and construct superpixel labels
        DRWN_LOG_MESSAGE("Loading label data from images...");
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

        // save labels to cache
        if (labelCache != NULL) {
            ofstream ofs(labelCache, ios::binary);
            DRWN_ASSERT_MSG(!ofs.fail(), labelCache);
            for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
                uint32_t n = (uint32_t)labelDistributions[imgIndx].size();
                ofs.write((char *)&n, sizeof(uint32_t));
                for (unsigned segId = 0; segId < labelDistributions[imgIndx].size(); segId++) {
                    ofs.write((char *)labelDistributions[imgIndx][segId].data(), nLabels * sizeof(double));
                }
            }
        }
    }

    // set up model
    drwnPixelSegModel model;
    model.learnPixelContrastWeight(pairwiseSmoothness);

    // load image and segments
    cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, imageFile);
    drwnSuperpixelContainer segments;
    ifstream ifs(segFile, ios::binary);
    DRWN_ASSERT_MSG(!ifs.fail(), segFile);
    segments.read(ifs);
    ifs.close();

    drwnNNGraphImageData data(img, segments);

    // add image to graph and set all other images to disabled
    for (unsigned i = 0; i < graph.numImages(); i++) {
        graph[i].bSourceMatchable = false;
    }
    int imgIndx = graph.appendImage(data);
    graph[imgIndx].bTargetMatchable = false;

    // apply feature transform
    if (xformFile != NULL) {
        drwnFeatureTransform *featureTransform =
            drwnFeatureTransformFactory::get().createFromFile(xformFile);
        DRWN_ASSERT(featureTransform != NULL);
        graph[imgIndx].transformNodeFeatures(*featureTransform);
        delete featureTransform;
    }

    // run some search moves
    drwnNNGraphThreadedMoves::initialize(graph);
    pair<double, double> lastEnergy = graph.energy();
    DRWN_LOG_MESSAGE("...iteration 0; energy " << lastEnergy.first << ", best " << lastEnergy.second);

    // iterate moves
    int nIterations = 0;
    while (nIterations < maxIterations) {
        nIterations += 1;

        // perform an update
        drwnNNGraphThreadedMoves::update(graph);

        // check energy
        pair<double, double> e = graph.energy();
        DRWN_LOG_MESSAGE("...iteration " << nIterations << "; energy " << e.first << ", best " << e.second);

        if (e.first == lastEnergy.first) break;
        lastEnergy = e;
    }

    // do label transfer
    drwnSegImageInstance instance(img, "");
    instance.unaries.resize(instance.size(), vector<double>(nLabels, 0.0));

    for (int y = 0; y < instance.pixelLabels.rows(); y++) {
        for (int x = 0; x < instance.pixelLabels.cols(); x++) {
            VectorXd marginals = VectorXd::Constant(nLabels, DRWN_EPSILON);

            for (int c = 0; c < data.segments().channels(); c++) {
                const int segId = data.segments()[c].at<int>(y, x);
                if (segId < 0) continue;

                const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                if (e.empty()) continue;

                // accumulate label votes
                double rank = 1.0;
                for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {

                    // check that labels exist for the matching image
                    if (labelDistributions[kt->targetNode.imgIndx].empty())
                        continue;

                    const double w = 1.0 / rank;
                    marginals += w * labelDistributions[kt->targetNode];
                    rank += 1.0;
                }
            }

            Eigen::Map<VectorXd>(&instance.unaries[instance.pixel2Indx(x, y)][0], nLabels) =
                -1.0 * marginals.array().log();
        }
    }

    // infer labels
    model.inferPixelLabels(&instance);

    // save labels
    if (outLabel != NULL) {
        ofstream ofs(outLabel);
        ofs << instance.pixelLabels << "\n";
        ofs.close();
    }

    // visualize and save images
    if (bVisualize || (outImage != NULL)) {
        cv::Mat canvas = drwnMultiSegVis::visualizeInstance(instance);
        if (outImage != NULL) {
            cv::imwrite(outImage, canvas);
        }
        if (bVisualize) {
            drwnShowDebuggingImage(canvas, "nnGraphOnlineLabelTransfer", true);
        }
    }

    // clean up and print profile information
    if (bVisualize) cv::destroyAllWindows();
    drwnCodeProfiler::toc(hMain);
    drwnCodeProfiler::print();
    return 0;
}
