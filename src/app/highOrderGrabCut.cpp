/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    highOrderGrabCut.cpp
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
    cerr << "USAGE: ./highOrderGrabCut [OPTIONS] <img> (<mask>)\n";
    cerr << "OPTIONS:\n"
         << "  -k <k>            :: number of clusters for k-means segmentation (default: 10)\n"
         << "  -c <components>   :: number of mixture components (default: 5)\n"
         << "  -m <samples>      :: max samples to use when learning colour models\n"
         << "  -t <type>         :: colour model type (\"GMM\" (default) or \"Histogram\")\n"
         << "  -outMask <name>   :: filename for saving mask (default: none)\n"
         << "  -outLabels <name> :: filename for label output (default: none)\n"
         << "  -outImage <name>  :: filename for image output (default: none)\n"
         << "  -pairwise <w>     :: override learned pairwise weight\n"
         << "  -highorder <w>    :: weight for high order term (default: 0.0)\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}


// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH = 1024;
    drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT = 1024;

    int numClusters = 10;
    const char *colourModelType = "GMM";
    const char *outLabelName = NULL;
    const char *outMaskName = NULL;
    const char *outImageName = NULL;
    double pairwiseWeight = 0.0;
    double highOrderWeight = 0.0;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-k", numClusters)
        DRWN_CMDLINE_INT_OPTION("-c", drwnGrabCutInstanceGMM::numMixtures)
        DRWN_CMDLINE_INT_OPTION("-m", drwnGrabCutInstanceGMM::maxSamples)
        DRWN_CMDLINE_STR_OPTION("-t", colourModelType)
        DRWN_CMDLINE_STR_OPTION("-outMask", outMaskName)
        DRWN_CMDLINE_STR_OPTION("-outLabels", outLabelName)
        DRWN_CMDLINE_STR_OPTION("-outImage", outImageName)
        DRWN_CMDLINE_REAL_OPTION("-pairwise", pairwiseWeight)
        DRWN_CMDLINE_REAL_OPTION("-highorder", highOrderWeight)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if ((DRWN_CMDLINE_ARGC != 1) && (DRWN_CMDLINE_ARGC != 2)) {
        usage();
        return -1;
    }

    DRWN_ASSERT(numClusters > 1);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // initialize for foreground/background segmentation
    gMultiSegRegionDefs.initializeForDataset(DRWN_DS_FGBG);

    // load image
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    const char *maskFilename = DRWN_CMDLINE_ARGC == 1 ? NULL : DRWN_CMDLINE_ARGV[1];
    cv::Mat img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_COLOR);

    // load mask
    cv::Mat mask(img.rows, img.cols, CV_8UC1);
    if (maskFilename == NULL) {
        cv::Rect bb = drwnInputBoundingBox(string("annotate"), img);
        DRWN_ASSERT((bb.width > 0) && (bb.height > 0));
        mask.setTo(cv::Scalar(drwnGrabCutInstance::MASK_BG));
        mask(bb) = cvScalar(drwnGrabCutInstance::MASK_C_FG); 
    } else {
        mask = cv::imread(string(maskFilename), CV_LOAD_IMAGE_GRAYSCALE);
    }

    if (outMaskName != NULL) {
        cv::imwrite(string(outMaskName), mask);
    }

    // show image and mask
    if (bVisualize) {
        drwnShowDebuggingImage(img, string("image"), false);
        drwnShowDebuggingImage(mask, string("mask"), false);
    }

    // learn foreground and background colour models
    drwnGrabCutInstance *model = NULL;
    if (!strcmp(colourModelType, "GMM")) {
        model = new drwnGrabCutInstanceGMM();
    } else if (!strcmp(colourModelType, "Histogram")) {
        model = new drwnGrabCutInstanceHistogram();
    }
    DRWN_ASSERT_MSG(model != NULL, "unknown model type \"" << colourModelType << "\"");

    model->name = drwn::strBaseName(imgFilename);
    model->initialize(img, mask);

    // create instance
    drwnSegImageInstance instance(img, model->name.c_str());
    instance.unaries.resize(instance.size(), vector<double>(2, 0.0));
    const cv::Mat unary = model->unaryPotentials();
    const cv::Mat bg = model->knownBackground();
    const cv::Mat fg = model->knownForeground();
    for (int y = 0; y < instance.height(); y++) {
        const float *p = unary.ptr<const float>(y);
        for (int x = 0; x < instance.width(); x++) {
            if (bg.at<unsigned char>(y, x) != 0x00) {
                instance.unaries[instance.pixel2Indx(x, y)][0] = DRWN_DBL_MAX;
            } else if (fg.at<unsigned char>(y, x) != 0x00) {
                instance.unaries[instance.pixel2Indx(x, y)][1] = DRWN_DBL_MAX;
            } else {
                instance.unaries[instance.pixel2Indx(x, y)][1] = (double)p[x];
            }
        }
    }

    // generate soft superpixels
    cv::Mat filteredImg(img.rows, img.cols, CV_8UC3);
    cv::GaussianBlur(img, filteredImg, cv::Size(3, 3), 0);
    instance.superpixels.addSuperpixels(drwnKMeansSegments(filteredImg, numClusters));

    // visualize soft segmentation
    if (bVisualize) {
        drwnShowDebuggingImage(drwnCreateHeatMap(instance.superpixels[0]), "kMeansSegmentation", false);
    }

    // infer labels
    drwnRobustPottsCRFInference inf(0.25);
    inf.alphaExpansion(&instance, pairwiseWeight, highOrderWeight);

    // visualize and save image
    cv::Mat canvas;
    if (bVisualize || (outImageName != NULL)) {
        canvas = drwnMultiSegVis::visualizeInstance(instance);
        if (bVisualize) {
            drwnShowDebuggingImage(canvas, "HighOrderGrabCutResults", false);
        }
        if (outImageName != NULL) {
            cv::imwrite(string(outImageName), canvas);
        }
    }

    // save labels
    if (outLabelName != NULL) {
        ofstream ofs(outLabelName);
        ofs << instance.pixelLabels << "\n";
        ofs.close();
    }

    // clean up and print profile information
    if (bVisualize) cv::waitKey(-1);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
