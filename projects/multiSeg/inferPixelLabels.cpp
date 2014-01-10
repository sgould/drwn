/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    inferPixelLabels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for inferring pixel labels using a learned multi-class image
**  labeling CRF.
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
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

#define WINDOW_NAME "inferPixelLabels"

using namespace std;
using namespace Eigen;

// threading ---------------------------------------------------------------

class InferenceThread : public drwnThreadJob {
public:
    static const char *outLabelExt;        // extension for output labels
    static const char *outImageExt;        // extension for output images
    static const char *outUnaryExt;        // extension for output unary potentials
    static bool bVisualize;                // visualization flag

public:
    const drwnPixelSegModel *model;        // model
    const string *baseName;                // instance

public:
    InferenceThread() : model(NULL), baseName(NULL) { /* do nothing */ }
    ~InferenceThread() { /* do nothing */ }

    void operator()() {
        DRWN_ASSERT((model != NULL) && (baseName != NULL));
        if (drwnThreadPool::MAX_THREADS == 0) {
            DRWN_LOG_STATUS("..." << *baseName);
        } else {
            DRWN_LOG_STATUS("..." << *baseName << " (tid: " << threadId() << ")");
        }

        // load image
        string imgFilename = gMultiSegConfig.filename("imgDir", *baseName, "imgExt");
        drwnSegImageInstance instance(imgFilename.c_str());

        // load superpixels
        if (model->getRobustPottsWeight() != 0.0) {
            const string segFilename = gMultiSegConfig.filename("segDir", *baseName, "segExt");
            if (!drwnFileExists(segFilename.c_str())) {
                lock();
                DRWN_LOG_WARNING("superpixel file \"" << segFilename << "\" does not exist");
                unlock();
            } else {
                ifstream ifs(segFilename.c_str(), ios::binary);
                instance.superpixels.read(ifs);
                ifs.close();
            }
        }

        // infer labels
        model->inferPixelLabels(&instance);

        // visualize
        if (bVisualize) {
            lock();
            string wndName = string(WINDOW_NAME) + toString(threadId());
            cvNamedWindow(wndName.c_str());
            cv::Mat canvas = drwnMultiSegVis::visualizeInstance(instance);
            cv::imshow(wndName, canvas);
            cv::waitKey(100);
            unlock();
        }

        // save labels
        if (outLabelExt != NULL) {
            string filename = gMultiSegConfig.filebase("outputDir", *baseName) + string(outLabelExt);
            ofstream ofs(filename.c_str());
            ofs << instance.pixelLabels << "\n";
            ofs.close();
        }

        // save images
        if (outImageExt != NULL) {
            string filename = gMultiSegConfig.filebase("outputDir", *baseName) + string(outImageExt);
            cv::Mat canvas = drwnMultiSegVis::visualizeInstance(instance);
            cv::imwrite(filename, canvas);
        }

        // save unary potentials
        if (outUnaryExt != NULL) {
            string filename = gMultiSegConfig.filebase("outputDir", *baseName) + string(outUnaryExt);
            ofstream ofs(filename.c_str());
            DRWN_ASSERT_MSG(!ofs.fail(), filename);
            for (int i = 0; i < instance.size(); i++) {
                ofs << toString(instance.unaries[i]) << "\n";
            }
            ofs.close();
        }
    }
};

const char *InferenceThread::outLabelExt = NULL;
const char *InferenceThread::outImageExt = NULL;
const char *InferenceThread::outUnaryExt = NULL;
bool InferenceThread::bVisualize = false;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./inferPixelLabels [OPTIONS] (<evalList>|<baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -outLabels <ext>  :: extension for label output (default: none)\n"
         << "  -outImages <ext>  :: extension for image output (default: none)\n"
         << "  -outUnary <ext>   :: extension for dumping unary potentials (default: none)\n"
         << "  -pairwise <w>     :: override learned pairwise weight\n"
         << "  -robustpotts <w>  :: override learned robust potts weight\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    double pairwiseWeight = -1.0;
    double robustPottsWeight = -1.0;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-outLabels", InferenceThread::outLabelExt)
        DRWN_CMDLINE_STR_OPTION("-outImages", InferenceThread::outImageExt)
        DRWN_CMDLINE_STR_OPTION("-outUnary", InferenceThread::outUnaryExt)
        DRWN_CMDLINE_REAL_OPTION("-pairwise", pairwiseWeight)
        DRWN_CMDLINE_REAL_OPTION("-robustpotts", robustPottsWeight)
        DRWN_CMDLINE_BOOL_OPTION("-x", InferenceThread::bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read list of evaluation images
    const char *evalList = DRWN_CMDLINE_ARGV[0];

    vector<string> baseNames;
    if (drwnFileExists(evalList)) {
        DRWN_LOG_MESSAGE("Reading evaluation list from " << evalList << "...");
        baseNames = drwnReadFile(evalList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << evalList << "...");
        baseNames.push_back(string(evalList));
    }

    // load model
    drwnPixelSegModel* model = new drwnPixelSegModel();

    string modelFilename = gMultiSegConfig.filebase("modelsDir", "pixelSegModel.xml");
    DRWN_LOG_MESSAGE("Reading model from " << modelFilename);
    model->read(modelFilename.c_str());
    if (pairwiseWeight >= 0.0) {
        DRWN_LOG_MESSAGE("...overriding pairwise weight with " << pairwiseWeight);
        model->learnPixelContrastWeight(pairwiseWeight);
    }
    if (robustPottsWeight >= 0.0) {
        DRWN_LOG_MESSAGE("...overriding robust potts weight with " << robustPottsWeight);
        model->learnRobustPottsWeight(robustPottsWeight);
    }

    // process images (threaded)
    DRWN_LOG_MESSAGE("Processing " << baseNames.size() << " images...");

    drwnThreadPool threadPool;
    vector<InferenceThread> threadArgs;
    threadArgs.resize(baseNames.size());
    threadPool.start();
    for (int i = 0; i < (int)baseNames.size(); i++) {
        threadArgs[i].model = model;
        threadArgs[i].baseName = &baseNames[i];

        threadPool.addJob(&threadArgs[i]);
    }
    threadPool.finish();

    // wait for keypress if only image
    if (InferenceThread::bVisualize && (baseNames.size() == 1)) {
        cvWaitKey(-1);
    }

    // clean up and print profile information
    delete model;
    cvDestroyAllWindows();
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
