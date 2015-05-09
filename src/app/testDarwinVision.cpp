/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinVision.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Testbench for drwnVision library.
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

// opencv headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// prototypes ----------------------------------------------------------------

int handleOpenCVError(int status, const char* func_name,
    const char* err_msg, const char* file_name,
    int line, void* userdata )
{
    DRWN_LOG_FATAL("OpenCV error: " << err_msg << " in "
        << file_name << " at line " << line);
    return 0;
}

void testImagePadding(const char *filename);
void testImageRotation(const char *filename);
void testTemplateMatcher(const char *filename);
void testHeatMap();
void testTextonFeatures(const char *filename);
void testHOGFeatures(const char *filename);
void testDenseHOGFeatures(const char *filename);
void timeHOGFeatures(const char *filename, int nIterations);
void testPartsInference();
void testFilterBankResponse(const char *filename);
void testSuperpixelContainer(const char *imgFile, const char *segFile);
void testSuperpixelContainerBasic();
void testSuperpixelContainerIO(const char *imgFile);
void testImageCache(const char *directory);
void testImagePyramidCache(const char *directory);
void testImageInPainting(const char *imgFile);
void testColourHistogram(const char *imgFile);
void testMaskedPatchMatch();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinVision [OPTIONS] (<test>)*\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "TESTS:\n"
         << "  padding <img>     :: test drwnPadImage\n"
         << "  rotate <img>      :: test drwnRotateImage\n"
         << "  templates <img>   :: test drwnTemplateMatcher\n"
         << "  heatmap           :: test drwnHeatMap\n"
         << "  texton <img>      :: test texton features\n"
         << "  hog <img>         :: test HOG features\n"
         << "  dhog <img>        :: test dense HOG features\n"
         << "  timehog <img> <n> :: time HOG features for <n> iterations\n"
         << "  partsinf          :: parts inference\n"
         << "  filters <img>     :: test drwnFilterBankResponse\n"
         << "  spc <img> <seg>   :: test drwnSuperpixelContainer\n"
         << "  spcbasic          :: test drwnSuperpixelContainer\n"
         << "  spcio <img>       :: test drwnSuperpixelContainer I/O\n"
         << "  drwnImageCache <d> :: test drwnImageCache on directory <d>\n"
         << "  drwnImagePyramidCache <d> :: test drwnImagePryamidCache on directory <d>\n"
         << "  drwnInPaint <img> :: test drwnInPaint on image <img>\n"
         << "  drwnColourHistogram <img> :: test drwnColourHistogram class on image <img>\n"
         << "  drwnMaskedPatchMatch :: test drwnMaskedPatchMatch\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    cvRedirectError(handleOpenCVError);

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_OPTION_BEGIN("padding", p)
            testImagePadding(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("rotate", p)
            testImageRotation(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("templates", p)
            testTemplateMatcher(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_FLAG_BEGIN("heatmap")
            testHeatMap();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("texton", p)
            testTextonFeatures(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("hog", p)
            testHOGFeatures(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("dhog", p)
            testDenseHOGFeatures(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("timehog", p)
            timeHOGFeatures(p[0], atoi(p[1]));
        DRWN_CMDLINE_OPTION_END(2)
        DRWN_CMDLINE_FLAG_BEGIN("partsinf")
            testPartsInference();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("filters", p)
            testFilterBankResponse(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("spc", p)
            testSuperpixelContainer(p[0], p[1]);
        DRWN_CMDLINE_OPTION_END(2)
        DRWN_CMDLINE_FLAG_BEGIN("spcbasic")
            testSuperpixelContainerBasic();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("spcio", p)
            testSuperpixelContainerIO(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("drwnImageCache", p)
            testImageCache(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("drwnImagePyramidCache", p)
            testImagePyramidCache(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("drwnInPaint", p)
            testImageInPainting(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("drwnColourHistogram", p)
            testColourHistogram(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_FLAG_BEGIN("drwnMaskedPatchMatch")
            testMaskedPatchMatch();
        DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// regression tests ----------------------------------------------------------

void testImagePadding(const char *imgFilename)
{
    // load image
    cv::Mat img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, imgFilename);

    // page version
    cv::Rect page((int)(-img.cols / 4), (int)(-img.rows / 2),
        2 * img.cols, (int)(1.75 * img.rows));
    cv::Mat paddedImg = drwnPadImage(img, page);
    cv::rectangle(paddedImg, cv::Point(-page.x, -page.y),
        cv::Point(-page.x + img.cols, -page.y + img.rows), cv::Scalar(0, 0, 255), 2);

    drwnShowDebuggingImage(paddedImg, string("testImagePadding"), true);

    // margin version
    const int margin = std::min(img.cols / 4, img.rows / 4);
    paddedImg = drwnPadImage(img, margin);
    cv::rectangle(paddedImg, cv::Point(margin / 2, margin / 2),
        cvPoint(margin/2 + img.cols, margin/2 + img.rows), cv::Scalar(0, 0, 255), 2);

    drwnShowDebuggingImage(paddedImg, string("testImagePadding"), true);
}

void testImageRotation(const char *imgFilename)
{
    // load image
    cv::Mat img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, imgFilename);

    // iterate through rotations
    for (float theta = 0.0; theta < 4.0 * M_PI; theta += 0.05) {
        drwnShowDebuggingImage(drwnRotateImage(img, theta), string("testImageRotation"), false);
    }
}

void testTemplateMatcher(const char *imgFilename)
{
    // load image
    cv::Mat img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_GRAYSCALE);
    DRWN_ASSERT_MSG(img.data != NULL, imgFilename);

    cv::Mat m(img.rows, img.cols, CV_32FC1);
    img.convertTo(m, CV_32F, 1.0 / 255.0, 0.0);

    // generate templates
    static const int minWidth = 4;
    static const int minHeight = 4;
    const int maxWidth = img.cols / 4;
    const int maxHeight = img.rows / 4;

    vector<cv::Mat> templates(100);
    cv::Rect r;
    for (unsigned i = 0; i < templates.size(); i++) {

        r.width = rand() % (maxWidth - minWidth) + minWidth;
        r.height = rand() % (maxHeight - minHeight) + minHeight;
        r.x = rand() % (img.cols - r.width);
        r.y = rand() % (img.rows - r.height);

        templates[i] = cv::Mat(r.height, r.width, CV_32FC1);
        m(r).copyTo(templates[i]);
    }

    // test different methods
#if 1
    int methods[] = {CV_TM_CCORR, CV_TM_CCORR_NORMED,
                     CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED,
                     CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED};
#else
    int methods[] = {CV_TM_CCORR, CV_TM_CCOEFF_NORMED};
#endif

    // construct template matcher object
    drwnTemplateMatcher tm;
    tm.copyTemplates(templates);

    // compute matches
    for (unsigned k = 0; k < sizeof(methods) / sizeof(int); k++) {
        DRWN_LOG_MESSAGE("testing method " << k << "...");

        // template matcher
        vector<cv::Mat> responses = tm.responses(m, methods[k]);

        // check against OpenCV function
        int handle = drwnCodeProfiler::getHandle("cvMatchTemplate");
        double totalAvgDiff = 0.0;
        double maxRelDiff = 0.0;
        for (unsigned i = 0; i < templates.size(); i++) {
            drwnCodeProfiler::tic(handle);
            cv::Mat result(m.rows - templates[i].rows + 1, m.cols - templates[i].cols + 1, CV_32FC1);
            cv::matchTemplate(m, templates[i], result, methods[k]);
            drwnCodeProfiler::toc(handle);

            double d = cv::norm(result - responses[i](cv::Rect(0, 0, result.cols, result.rows)), cv::NORM_L1) /
                (double)(result.rows * result.cols);

            cv::Mat relDiff(result.rows, result.cols, CV_32FC1);            
            cv::divide(result, responses[i](cv::Rect(0, 0, result.cols, result.rows)), relDiff);
            cv::subtract(relDiff, cv::Scalar::all(1.0), relDiff);
            double maxr = cv::norm(relDiff, cv::NORM_INF);

            totalAvgDiff += d;
            maxRelDiff = std::max(maxRelDiff, maxr);

            DRWN_LOG_DEBUG("...mean abs diff for template "
                << i << " is " << d << "; max rel diff is " << maxr);
        }

        DRWN_LOG_MESSAGE("...total mean abs diff is "
            << totalAvgDiff << "; max rel diff is " << maxRelDiff);
    }
}

void testHeatMap()
{
    // populate greyscale image
    cv::Mat m(32, 256, CV_32F);
    for (int x = 0; x < m.cols; x++) {
        m.at<float>(0, x) = (float)x / (float)m.cols;
    }
    for (int y = 1; y < m.rows; y++) {
        for (int x = 0; x < m.cols; x++) {
            m.at<float>(y, x) = m.at<float>(0, x);
        }
    }

    // create heat maps
    vector<cv::Mat> heatmaps;
    heatmaps.push_back(drwnCreateHeatMap(m, DRWN_COLORMAP_RAINBOW));
    heatmaps.push_back(drwnCreateHeatMap(m, DRWN_COLORMAP_HOT));
    heatmaps.push_back(drwnCreateHeatMap(m, DRWN_COLORMAP_COOL));
    heatmaps.push_back(drwnCreateHeatMap(m, DRWN_COLORMAP_REDGREEN));
    heatmaps.push_back(drwnCreateHeatMap(m, DRWN_COLORMAP_ANU));
    heatmaps.push_back(drwnCreateHeatMap(m, CV_RGB(255, 0, 0), CV_RGB(0, 0, 255)));

    vector<cv::Scalar> colours(3);
    colours[0] = CV_RGB(255, 0, 0);
    colours[1] = CV_RGB(0, 255, 0);
    colours[2] = CV_RGB(0, 0, 255);
    heatmaps.push_back(drwnCreateHeatMap(m, colours));

    colours[0] = CV_RGB(0, 0, 0);
    colours[1] = CV_RGB(175, 30, 45);
    colours[2] = CV_RGB(148, 176, 188);
    colours.push_back(CV_RGB(255, 255, 255));
    heatmaps.push_back(drwnCreateHeatMap(m, colours));

    // show the heat maps
    cv::Mat canvas = drwnCombineImages(heatmaps, heatmaps.size(), 1);
    drwnShowDebuggingImage(canvas, string("HeatMaps"), true);
}

void testTextonFeatures(const char *filename)
{
    // load image
    cv::Mat img = cv::imread(string(filename), CV_LOAD_IMAGE_COLOR);

    // texton filterbank response images
    drwnTextonFilterBank filterBank(1.0);
    vector<cv::Mat> responses;

    filterBank.filter(img, responses);
    cv::Mat canvas = drwnCombineImages(responses);
    drwnShowDebuggingImage(canvas, string("TextonFeatures"), false);
}

void testHOGFeatures(const char *filename)
{
    // load image
    cv::Mat img = cv::imread(string(filename), CV_LOAD_IMAGE_GRAYSCALE);

    // compute featutes
    drwnHOGFeatures::DEFAULT_BLOCK_SIZE = 1;
    drwnHOGFeatures hogFeatureExtractor;

    // multi-scale
    while ((img.rows > 64) && (img.cols > 64)) {
        cv::Mat canvas = hogFeatureExtractor.visualizeCells(img);

        // show features
        int ch = drwnShowDebuggingImage(canvas, string("testHOGFeatures"), true);
        if (ch == (int)'w') {
            string outname = drwn::strBaseName(filename) + string(".hog.png");
            DRWN_LOG_VERBOSE("...saving image to " << outname);
            cv::imwrite(outname, canvas);
        }

        // free visualization
        if (ch == (int)'q')
            break;

        // scale down image
        cv::resize(img, img, cv::Size(), 0.9, 0.9);
    }
}

void testDenseHOGFeatures(const char *filename)
{
    // load image
    cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

    // compute featutes
    drwnHOGFeatures hogFeatureExtractor;

    // dense features
    cv::namedWindow("testHOGFeatures", 1);
    vector<cv::Mat> features;
    hogFeatureExtractor.computeDenseFeatures(img, features);
    for (int i = 0; i < (int)features.size(); i++) {
        DRWN_LOG_DEBUG("...features[" << i << "] has size " << features[i].cols << "-by-" << features[i].rows);
#if 1
        //! \todo make into utility function
        vector<cv::Mat> rgb(3);

        rgb[0] = img.clone();
        rgb[1] = cv::Mat(img.rows, img.cols, CV_8UC1);
        features[i].convertTo(rgb[1], CV_8U, 255.0, 0.0);
        rgb[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

        cv::Mat canvas(img.rows, img.cols, CV_8UC3);
        cv::merge(rgb, canvas);

        DRWN_LOG_DEBUG("...canvas has size " << canvas.cols << "-by-" << canvas.rows);
        cv::imshow(string("testHOGFeatures"), canvas);
        cv::waitKey(-1);
#else
        cv::imshow(string("testHOGFeatures"), features[i]);
        cv::waitKey(-1);
#endif
    }
}

void timeHOGFeatures(const char *filename, int nIterations)
{
    // load image
    cv::Mat img = cv::imread(filename);

    // compute featutes
    drwnHOGFeatures::DEFAULT_BLOCK_SIZE = 1;
    drwnHOGFeatures hogFeatureExtractor;

    // time test
    vector<cv::Mat> features;
    for (int i = 0; i < nIterations; i++) {
        hogFeatureExtractor.computeFeatures(img, features);
    }
}

void testPartsInference()
{
    class TestClass : public drwnPartsInference {
    public:
        TestClass() : drwnPartsInference(1, 1, 100) { _lambda = 0.01; }
        ~TestClass() { /* do nothing */ }

        void operator()() {
            DRWN_LOG_VERBOSE("setting up belief...");
            cv::Mat belief = cv::Mat::zeros(100, 1, CV_32FC1);
            belief.at<float>(3, 0) = 1.0f;
            belief.at<float>(15, 0) = 2.0f;
            belief.at<float>(30, 0) = 5.0f;
            belief.at<float>(75, 0) = 1.0f;
            DRWN_LOG_VERBOSE("running linear cost model...");
            cv::Mat m1 = computeLocationMessage(belief, drwnDeformationCost(10.0, 10.0, 0.0, 0.0));
            DRWN_LOG_VERBOSE("running quadratic cost model...");
            cv::Mat m2 = computeLocationMessage(belief, drwnDeformationCost(0.0, 0.0, 1.0, 1.0));

            DRWN_LOG_VERBOSE("showing results...");
            for (int i = 0; i < belief.rows; i++) {
                cout << belief.at<float>(i, 0) << "\t"
                     << m1.at<float>(i, 0) << "\t"
                     << m2.at<float>(i, 0) << "\n";
            }
        }
    };

    TestClass test;
    test();
}

void testFilterBankResponse(const char *filename)
{
    // load image
    cv::Mat img = cv::imread(filename);

    drwnFilterBankResponse filters;

    for (int i = 0; i < 100; i++) {
        // texton filterbank response images
        drwnTextonFilterBank filterBank(1.0);
        vector<cv::Mat> responses;
        filterBank.filter(img, responses);

        filters.addResponseImages(responses); // filters takes ownership
        DRWN_LOG_DEBUG("...allocated "  << (filters.memory() / (1024 * 1024))
            << "MB for " << filters.size() << " filter responses");

        // clear filters
        filters.clear();
    }
}

void testSuperpixelContainer(const char *imgFile, const char *segFile)
{
    // load image and superpixels
    cv::Mat img = cv::imread(imgFile);

    drwnSuperpixelContainer superpixels;
    superpixels.loadSuperpixels(segFile);
    superpixels.removeSmallSuperpixels(32);

    // show segments
    cvNamedWindow("testSuperpixelContainer", 1);
    for (int i = 0; i < superpixels.size(); i++) {
        cv::Mat mask = superpixels.mask(i);
        cv::Mat canvas = img.clone();
        drwnShadeRegion(canvas, mask, CV_RGB(255, 0, 0), 1.0, DRWN_FILL_DIAG);
        cv::imshow( "testSuperpixelContainer", canvas);
        DRWN_LOG_DEBUG("...superpixel " << i << " has size " << superpixels.pixels(i));
        cv::waitKey(-1);
    }

    // free memory
    cv::destroyWindow("testSuperpixelContainer");
}

void testSuperpixelContainerBasic()
{
    const int N = 8;

    drwnSuperpixelContainer superpixels;

    // add four superpixels
    cv::Mat m = cv::Mat::zeros(N, N, CV_32SC1);
    int spIndx = 0;
    for (int y = 0; y < N; y += N/2) {
        for (int x = 0; x < N; x += N/2) {
            cv::rectangle(m, cv::Point(x, y), cv::Point(x + N/2, y + N/2),
                cv::Scalar::all(spIndx), CV_FILLED);
            spIndx += 1;
        }
    }
    superpixels.addSuperpixels(m);

    m = superpixels.intersection();
    cout << m << endl;

    // add another superpixel
    m = cv::Mat::zeros(N, N, CV_32SC1);
    superpixels.addSuperpixels(m);

    m = superpixels.intersection();
    cout << m << endl;
}

void testSuperpixelContainerIO(const char *imgFile)
{
    const bool bVisualize = drwnLogger::checkLogLevel(DRWN_LL_VERBOSE);

    // load image and superpixels
    cv::Mat img = cv::imread(imgFile);

    drwnSuperpixelContainer superpixels;
    superpixels.addSuperpixels(drwnFastSuperpixels(img, 5));
    superpixels.addSuperpixels(drwnFastSuperpixels(img, 10));
    superpixels.addSuperpixels(drwnFastSuperpixels(img, 25));
    superpixels.addSuperpixels(drwnFastSuperpixels(img, 32));
    DRWN_LOG_MESSAGE("...serialization requires " << superpixels.numBytesOnDisk() << " bytes");

    // show segments
    if (bVisualize) {
        cv::Mat canvas = superpixels.visualize(img);
        drwnShowDebuggingImage(canvas, string("original superpixels"), false);
    }

    // write to disk
    ofstream ofs("_testSuperpixelContainerIO.bin", ios::binary);
    superpixels.write(ofs);
    ofs.close();

    // read from disk;
    drwnSuperpixelContainer container;
    ifstream ifs("_testSuperpixelContainerIO.bin", ios::binary);
    container.read(ifs);
    ifs.close();
    DRWN_ASSERT(container.channels() == superpixels.channels());

    // compare
    int nErrors = 0;
    for (int n = 0; n < superpixels.channels(); n++) {
        nErrors += cv::norm(superpixels[n], container[n], CV_L1);
    }
    DRWN_LOG_MESSAGE("...readback generated " << nErrors << " errors");

    if (bVisualize) {
        cv::Mat canvas = container.visualize(img);
        drwnShowDebuggingImage(canvas, string("readback superpixels"), true);
    }
}

void testImageCache(const char *directory)
{
    DRWN_FCN_TIC;
    set<const char *> extensions;
    extensions.insert(".jpg");
    extensions.insert(".png");

    vector<string> files = drwnDirectoryListing(directory, extensions);
    DRWN_LOG_MESSAGE("found " << files.size() << " images in " << directory);

    vector<unsigned> images(5 * files.size());
    for (unsigned i = 0; i < images.size(); i++) {
        images[i] = (unsigned)(drand48() * files.size());
    }

    drwnImageCache cache(files);
    for (unsigned i = 0; i < images.size(); i += 5) {
        for (unsigned j = i; j < i + 5; j++) {
            cache.lock(images[j]);
            DRWN_LOG_VERBOSE("lock " << j << ", size " << cache.size() << ", memory " << drwn::bytesToString(cache.memory()));
        }
        for (unsigned j = i; j < i + 5; j++) {
            cache.unlock(images[j]);
            DRWN_LOG_VERBOSE("unlock " << j << ", size " << cache.size() << ", memory " << drwn::bytesToString(cache.memory()));
        }
    }

    cache.clear();
    DRWN_FCN_TOC;
}

void testImagePyramidCache(const char *directory)
{
    DRWN_FCN_TIC;
    set<const char *> extensions;
    extensions.insert(".jpg");
    extensions.insert(".png");

    vector<string> files = drwnDirectoryListing(directory, extensions);
    DRWN_LOG_MESSAGE("found " << files.size() << " images in " << directory);

    vector<unsigned> images(5 * files.size());
    for (unsigned i = 0; i < images.size(); i++) {
        images[i] = (unsigned)(drand48() * files.size());
    }

    drwnImagePyramidCache cache(files, 0.71, 32);
    for (unsigned i = 0; i < images.size(); i += 5) {
        for (unsigned j = i; j < i + 5; j++) {
            cache.lock(images[j]);
#if 1
            vector<cv::Mat> pyr;
            for (unsigned k = 0; k < cache.levels(images[j]); k++) {
                pyr.push_back(cache.get(images[j], k));
            }
            cv::Mat canvas = drwnCombineImages(pyr);
            drwnShowDebuggingImage(canvas, string("pyramid"), false);
#endif
            DRWN_LOG_VERBOSE("lock " << j << ", size " << cache.size() << ", memory " << drwn::bytesToString(cache.memory()));
        }
        for (unsigned j = i; j < i + 5; j++) {
            cache.unlock(images[j]);
            DRWN_LOG_VERBOSE("unlock " << j << ", size " << cache.size() << ", memory " << drwn::bytesToString(cache.memory()));
        }
    }

    cache.clear();
    DRWN_FCN_TOC;
}

void testImageInPainting(const char *imgFile)
{
    cv::Mat img = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data, "could not read image from " << imgFile);

    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    cv::rectangle(mask, cv::Point(img.cols / 4, img.rows / 4),
        cv::Point(3 * img.cols / 4, 3 * img.rows / 4), cv::Scalar(0xff), -1);

    cv::Mat output;
    drwnInPaint::inPaint(img, output, mask);

    cv::imwrite((drwn::strBaseName(imgFile) + string("_inpainted.png")).c_str(), output);
}

void testColourHistogram(const char *imgFile)
{
    DRWN_FCN_TIC;
    cv::Mat img = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data, "could not read image from " << imgFile);

    // construct histogram for the image
    drwnColourHistogram histogram;
    DRWN_LOG_VERBOSE("colour histogram has " << histogram.size() << " bins");

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            histogram.accumulate(img.at<const Vec3b>(y, x));
        }
    }

    // display colour histogram
    drwnShowDebuggingImage(histogram.visualize(), string("histogram"), false);

    // display probability of each pixel as a heat map
    cv::Mat canvas(img.rows, img.cols, CV_32FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            canvas.at<float>(y, x) = histogram.probability(img.at<const Vec3b>(y, x));
        }
    }

    drwnScaleToRange(canvas, 0.0, 1.0);
    drwnShowDebuggingImage(drwnCreateHeatMap(canvas), string("testColourHistogram"), true);
    DRWN_FCN_TOC;
}

void testMaskedPatchMatch()
{
    // run test twice, once with gradient image and once with random image
    for (int nTest = 0; nTest < 2; nTest++) {
        DRWN_LOG_MESSAGE("Running test on " << (nTest == 0 ? "gradient" : "random") << " image...");

        // create gradient image or random image
        cv::Mat testImage(256, 256, CV_8UC1);
        for (int y = 0; y < testImage.rows; y++) {
            for (int x = 0; x < testImage.cols; x++) {
                if (nTest == 0) {
                    testImage.at<unsigned char>(y, x) = (x + y) % 256;
                } else {
                    testImage.at<unsigned char>(y, x) = rand() % 256;
                }
            }
        }

        // show the image
        drwnShowDebuggingImage(testImage, "testMaskedPatchMatch.1", false);

        // match to self
        const unsigned patchRadius = 4;
        drwnMaskedPatchMatch::TRY_IDENTITY_INIT = false;
        drwnMaskedPatchMatch pm(testImage, testImage, patchRadius);
        drwnShowDebuggingImage(pm.visualize(), "testMaskedPatchMatch.2", false);
        pm.search(10);
        drwnShowDebuggingImage(pm.visualize(), "testMaskedPatchMatch.3", false);
        
        // copy a patch and check that is matches exactly
        const int TEST_POINTS[6][2] = {{32, 32}, {1, 1}, {1, 128}, {128, 1}, {255, 255}, {254, 254}};

        vector<cv::Mat> views;
        for (int i = 0; i < 6; i++) {
            cv::Point srcPoint(TEST_POINTS[i][0], TEST_POINTS[i][1]);
            pair<cv::Rect, cv::Rect> match = pm.getMatchingPatches(srcPoint);
            DRWN_LOG_MESSAGE(match.first << " centred at " << srcPoint << " matches to " << match.second);
        
            testImage(match.second).copyTo(testImage(match.first));
            views.push_back(testImage.clone());
        }
        drwnShowDebuggingImage(views, "testMaskedPatchMatch.4", false);

        cv::waitKey(-1);
    }
}
