/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    photoMontage.cpp
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
#include "drwnPGM.h"
#include "drwnVision.h"

// project headers
// TODO: better GUI (wxWidgets)

using namespace std;
//using namespace Eigen;

// prototypes ----------------------------------------------------------------

// visualization
cv::Mat makePhotoCollection(const vector<cv::Mat>& photos, const cv::Mat& mask);
cv::Mat colourMask(const cv::Mat& mask);
cv::Mat makePhotoMontage(const vector<cv::Mat>& photos, const cv::Mat& mask);

// inference
inline double colourDifference(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Point& p);
cv::Mat graphCut(const vector<cv::Mat>& photos, const cv::Mat& mask);

// usage ---------------------------------------------------------------------

void description()
{
    cerr << "DESCRIPTION:  Creates a photo-montage from images captured from a\n"
         << "  webcam (press '1' to '4' to capture; press 'esc' or 'enter' to\n"
         << "  continue) or loaded from disk.\n\n"
         << "  After images have been captured or loaded, mark regions you want\n"
         << "  in the montage using the mouse; press 'c' to clear current markings.\n"
         << "  Press 'enter' to create montage (can be done repeatedly); press 'v'\n"
         << "  to toggle montage view; press 's' to save montage; and press 'esc'\n"
         << "  or 'q' to quit.\n\n";
}

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./photoMontage [OPTIONS]\n\n";
	description();
    cerr << "OPTIONS:\n"
         << "  -o <dir>          :: output directory for saving images\n"
         << "  -overlay          :: overlay previous image during capture\n"
         << "  -size <w> <h>     :: image size (width-by-height)\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    bool bOverlayCapture = false;
    const char *outputDir = NULL;
    cv::Size imgSize(320, 240);

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", outputDir)
        DRWN_CMDLINE_BOOL_OPTION("-overlay", bOverlayCapture)
        DRWN_CMDLINE_OPTION_BEGIN("-size", p)
          imgSize.width = atoi(p[0]);
          imgSize.height = atoi(p[1]);
        DRWN_CMDLINE_OPTION_END(2)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    DRWN_ASSERT_MSG((outputDir == NULL) || (drwnDirExists(outputDir)),
        "directory " << outputDir << " does not exist");

	description();

    // images to combine
    vector<cv::Mat> photos(4);
    for (unsigned i = 0; i < photos.size(); i++) {
        photos[i] = cv::Mat(imgSize.height, imgSize.width, CV_8UC3, cv::Scalar::all(255));
    }
    // mask with integer indicating which image to hard constrain assignment
    cv::Mat mask(imgSize.height, imgSize.width, CV_32SC1, cv::Scalar::all(-1));

    // initialize images
    int baseline;
    for (unsigned i = 0; i < photos.size(); i++) {
        string message = string("press '") + toString(i + 1) + string("' to replace");
        cv::Size textExtent = cv::getTextSize(message, CV_FONT_HERSHEY_SIMPLEX, 0.71, 2, &baseline);
        cv::rectangle(photos[i], cv::Point(0, 0), cv::Point(photos[i].cols, photos[i].rows),
            CV_RGB(255, 255, 255), CV_FILLED);
        cv::putText(photos[i], message, cv::Point((photos[i].cols - textExtent.width) / 2,
                (photos[i].rows - textExtent.height)/2), CV_FONT_HERSHEY_SIMPLEX, 0.71, CV_RGB(0, 0, 0), 2);
    }

    // initialize gui
    const string wndCapture("webcam");
    const string wndCollection("photo collection");
    const string wndOutput("montage");

    cv::namedWindow(wndCollection, cv::WINDOW_AUTOSIZE);
    cv::Mat canvas = makePhotoCollection(photos, mask);
    cv::imshow(wndCollection, canvas);

    // capture images
    cv::VideoCapture camera(0);
    DRWN_ASSERT_MSG(camera.isOpened(), "could not initialize webcam");

    while (1) {
        // grab a frame
        cv::Mat view;
        camera >> view;
        if (bOverlayCapture) {
            canvas = cv::Mat(view.rows, view.cols, CV_8UC3);
            cv::resize(photos[0], canvas, canvas.size(), 0, 0, CV_INTER_CUBIC);
            drwnGreyImageInplace(canvas);
            drwnColorImageInplace(canvas);
            drwnOverlayImages(canvas, view);
        }
        int ch = drwnShowDebuggingImage(bOverlayCapture ? canvas : view, wndCapture, false);

        // continue on <esc> or <enter>
        if (((ch & 0xff) == 27) || ((ch & 0xff) == 10) || ((ch & 0xff) == 13)) break;

        switch (ch & 0xff) {
          case '1': cv::resize(view, photos[0], photos[0].size(), 0, 0, CV_INTER_CUBIC); break;
          case '2': cv::resize(view, photos[1], photos[1].size(), 0, 0, CV_INTER_CUBIC); break;
          case '3': cv::resize(view, photos[2], photos[2].size(), 0, 0, CV_INTER_CUBIC); break;
          case '4': cv::resize(view, photos[3], photos[3].size(), 0, 0, CV_INTER_CUBIC); break;
        }

        if (ch != -1) {
            canvas = makePhotoCollection(photos, mask);
            cv::imshow(wndCollection, canvas);
        }
    }

    cv::destroyWindow(wndCapture);
    camera.release();

    // save images
    if (outputDir != NULL) {
        string filename;
        for (unsigned i = 0; i < photos.size(); i++) {
            filename = string(outputDir) + string("/img") + toString(i + 1) + string(".jpg");
            cv::imwrite(filename, photos[i]);
        }
    }

    // initialize and show montage
    cv::Mat montageMap = cv::Mat::zeros(mask.rows, mask.cols, CV_32SC1);
    bool bOverlayMap = false;
    cv::namedWindow(wndOutput, CV_WINDOW_AUTOSIZE);

    // event processing loop
    drwnMouseState mouseState;
    cv::setMouseCallback(wndCollection, drwnOnMouse, (void *)&mouseState);
    cv::Point lastPoint(0, 0);
    int lastId = 0;
    bool bQuit = false;
    while (!bQuit) {
        // update canvas
        canvas = makePhotoCollection(photos, mask);
        cv::imshow(wndCollection, canvas);

        canvas = makePhotoMontage(photos, montageMap);
        if (bOverlayMap) {
            cv::Mat overlay = colourMask(montageMap);
            drwnOverlayImages(canvas, overlay, 0.5);
        }

        cv::Mat tmp(2 * canvas.rows, 2 * canvas.cols, canvas.type());
        cv::resize(canvas, tmp, tmp.size(), 0, 0, CV_INTER_LINEAR);
        cv::imshow(wndOutput, tmp);

        // wait for next event
        int ch = -1;
        while (mouseState.event == -1) {
            ch = cv::waitKey(30);
            if (ch != -1) break;
        }

        // check event type
        switch (mouseState.event) {
        case -1: // process key
            switch (ch & 0xff) {
            case 27:
            case 'q': bQuit = true; break;
            case 'c': mask = cv::Scalar::all(-1); break;
            case 's':
                if (outputDir != NULL) {
                    string filename = string(outputDir) + string("/montage.jpg");
                    canvas = makePhotoMontage(photos, montageMap);
                    cv::imwrite(filename, canvas);
                    // TODO: save masks
                }
                break;
            case 'v': bOverlayMap = !bOverlayMap; break;
            case 10:
            case 13:
                montageMap = graphCut(photos, mask);
                break;
            }
            break;

        case CV_EVENT_MOUSEMOVE:
            if ((mouseState.flags & CV_EVENT_FLAG_LBUTTON) != 0) {
                int mouseId = (mouseState.x < mask.cols ? 0 : 1) + (mouseState.y < mask.rows ? 0 : 2);
                if (lastId == mouseId) {
                    cv::line(mask, cv::Point(mouseState.x % mask.cols, mouseState.y % mask.rows),
                        lastPoint, cv::Scalar(mouseId), 3);
                }
            }
            break;
        }

        // update mouse state
        lastPoint = cv::Point(mouseState.x % mask.cols, mouseState.y % mask.rows);
        lastId = (mouseState.x < mask.cols ? 0 : 1) + (mouseState.y < mask.rows ? 0 : 2);
        mouseState.event = -1;
    }

    // clean up
    cv::setMouseCallback(wndCollection, NULL);
    cv::destroyWindow(wndCollection);
    cv::destroyWindow(wndOutput);

    drwnCodeProfiler::print();
    return 0;
}

// helper functions ----------------------------------------------------------

cv::Mat makePhotoCollection(const vector<cv::Mat>& photos, const cv::Mat& mask)
{
    const int COLOURS[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};
    const int WIDTH = 4;

    vector<cv::Mat> views;
    cv::Mat viewMask(mask.rows, mask.cols, CV_8UC1);
    for (unsigned i = 0; i < photos.size(); i++) {
        cv::Scalar viewColour = CV_RGB(COLOURS[i % 4][0], COLOURS[i % 4][1], COLOURS[i % 4][2]);
        views.push_back(photos[i].clone());
        cv::compare(mask, cv::Scalar::all(i), viewMask, CV_CMP_EQ);
        drwnShadeRegion(views.back(), viewMask, viewColour);
        drwnDrawBoundingBox(views.back(), cv::Rect(WIDTH - 1, WIDTH - 1,
                views.back().cols - 2 * WIDTH + 1, views.back().rows - 2 * WIDTH + 1),
            viewColour, CV_RGB(0, 0, 0), WIDTH);
    }

    return drwnCombineImages(views);
}

cv::Mat colourMask(const cv::Mat& mask)
{
    const int COLOURS[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

    cv::Mat canvas(mask.rows, mask.cols, CV_8UC3, cv::Scalar::all(255));
    cv::Mat viewMask(mask.rows, mask.cols, CV_8UC1);
    for (unsigned i = 0; i < 4; i++) {
        cv::Scalar viewColour = CV_RGB(COLOURS[i % 4][0], COLOURS[i % 4][1], COLOURS[i % 4][2]);
        cv::compare(mask, cv::Scalar::all(i), viewMask, CV_CMP_EQ);
        drwnShadeRegion(canvas, viewMask, viewColour, 1.0);
    }
    return canvas;
}

cv::Mat makePhotoMontage(const vector<cv::Mat>& photos, const cv::Mat& mask)
{
    cv::Mat canvas(mask.rows, mask.cols, CV_8UC2, cv::Scalar::all(255));
    cv::Mat viewMask(mask.rows, mask.cols, CV_8UC1);
    for (unsigned i = 0; i < photos.size(); i++) {
        cv::compare(mask, cv::Scalar::all(i), viewMask, CV_CMP_EQ);
        photos[i].copyTo(canvas, viewMask);
    }

    return canvas;
}

inline double colourDifference(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Point& p)
{
    int d = 0;
    const unsigned char *pA = imgA.ptr<const unsigned char>(p.y) + 3 * p.x;
    const unsigned char *pB = imgB.ptr<const unsigned char>(p.y) + 3 * p.x;
    for (int c = 0; c < 3; c++) {
        d += (int)(pA[c] - pB[c]) * (pA[c] - pB[c]);
    }
    return sqrt((double)d);
}

cv::Mat graphCut(const vector<cv::Mat>& photos, const cv::Mat& mask)
{
    cv::Mat assignment = cv::Mat::zeros(mask.rows, mask.cols, CV_32SC1);

    // initialize assigment
    for (int y = 0; y < assignment.rows; y++) {
        for (int x = 0; x < assignment.cols; x++) {
            if (mask.at<int>(y, x) != -1) {
                assignment.at<int>(y, x) = mask.at<int>(y, x);
            }
        }
    }

    // alpha-expansion
    const int nVariables = mask.rows * mask.cols;
    drwnBKMaxFlow g(nVariables);
    g.addNodes(nVariables);

    bool bConverged = false;
    double minEnergy = numeric_limits<double>::max();
    while (!bConverged) {
        bConverged = true;
        for (int alpha = 0; alpha < (int)photos.size(); alpha++) {

            // construct graph
            g.reset();
            DRWN_ASSERT(g.numNodes() == (size_t)nVariables);

            // add unary terms
            int varIndx = 0;
            for (int y = 0; y < assignment.rows; y++) {
                for (int x = 0; x < assignment.cols; x++, varIndx++) {
                    if ((mask.at<int>(y, x) != -1) &&
                        (mask.at<int>(y, x) != alpha)) {
                        g.addTargetEdge(varIndx, DRWN_DBL_MAX);
                    }
                }
            }

            // add pairwise terms
            // X = |S_lp(p) - S_lq(p)| + |S_lp(q) - S_lq(q)|
            // Z = E_lp(p,q) + E_lq(p,q)
            // penalty = X/Z, where S is the RGB image and E is the Sobel edge strength
            // truncate to make submodular
            // TODO

#if 1
            // add horizontal pairwise terms
            for (int y = 0; y < assignment.rows; y++) {
                for (int x = 1; x < assignment.cols; x++) {
                    int u = y * assignment.cols + x;
                    int v = y * assignment.cols + x - 1;

                    int labelU = assignment.at<int>(y, x);
                    int labelV = assignment.at<int>(y, x - 1);
                    if ((labelU == alpha) && (labelV == alpha)) continue;

                    double A = 0.0;
                    double B = colourDifference(photos[alpha], photos[labelV], cv::Point(x, y)) +
                        colourDifference(photos[alpha], photos[labelV], cv::Point(x - 1, y));
                    double C = colourDifference(photos[labelU], photos[alpha], cv::Point(x, y)) +
                        colourDifference(photos[labelU], photos[alpha], cv::Point(x - 1, y));
                    double D = colourDifference(photos[labelU], photos[labelV], cv::Point(x, y)) +
                        colourDifference(photos[labelU], photos[labelV], cv::Point(x - 1, y));

                    // check for submodularity
                    if (A + D > C + B) {
                        // truncate non-submodular functions
                        //numNonSubmodular += 1;
                        double delta = A + D - C - B;
                        A -= delta / 3 - DRWN_EPSILON;
                        C += delta / 3 + DRWN_EPSILON;
                        B = A + D - C + DRWN_EPSILON;
                    }

                    g.addSourceEdge(u, D);
                    g.addTargetEdge(u, A);

                    B -= A; C -= D; B += DRWN_EPSILON; C += DRWN_EPSILON;
                    DRWN_ASSERT_MSG(B + C >= 0, "B = " << B << ", C = " << C);
                    if (B < 0) {
                        g.addTargetEdge(u, B);
                        g.addTargetEdge(v, -B);
                        g.addEdge(u, v, 0.0, B + C);
                    } else if (C < 0) {
                        g.addTargetEdge(u, -C);
                        g.addTargetEdge(v, C);
                        g.addEdge(u, v, B + C, 0.0);
                    } else {
                        g.addEdge(u, v, B, C);
                    }
                }
            }

            // add vertical pairwise terms
            for (int y = 1; y < assignment.rows; y++) {
                for (int x = 0; x < assignment.cols; x++) {
                    int u = y * assignment.cols + x;
                    int v = (y - 1) * assignment.cols + x;

                    int labelU = assignment.at<int>(y, x);
                    int labelV = assignment.at<int>(y - 1, x);
                    if ((labelU == alpha) && (labelV == alpha)) continue;

                    double A = 0.0;
                    double B = colourDifference(photos[alpha], photos[labelV], cv::Point(x, y)) +
                        colourDifference(photos[alpha], photos[labelV], cv::Point(x, y - 1));
                    double C = colourDifference(photos[labelU], photos[alpha], cv::Point(x, y)) +
                        colourDifference(photos[labelU], photos[alpha], cv::Point(x, y - 1));
                    double D = colourDifference(photos[labelU], photos[labelV], cv::Point(x, y)) +
                        colourDifference(photos[labelU], photos[labelV], cv::Point(x, y - 1));

                    // check for submodularity
                    if (A + D > C + B) {
                        // truncate non-submodular functions
                        //numNonSubmodular += 1;
                        double delta = A + D - C - B;
                        A -= delta / 3 - DRWN_EPSILON;
                        C += delta / 3 + DRWN_EPSILON;
                        B = A + D - C + DRWN_EPSILON;
                    }

                    g.addSourceEdge(u, D);
                    g.addTargetEdge(u, A);

                    B -= A; C -= D; B += DRWN_EPSILON; C += DRWN_EPSILON;
                    DRWN_ASSERT_MSG(B + C >= 0, "B = " << B << ", C = " << C);
                    if (B < 0) {
                        g.addTargetEdge(u, B);
                        g.addTargetEdge(v, -B);
                        g.addEdge(u, v, 0.0, B + C);
                    } else if (C < 0) {
                        g.addTargetEdge(u, -C);
                        g.addTargetEdge(v, C);
                        g.addEdge(u, v, B + C, 0.0);
                    } else {
                        g.addEdge(u, v, B, C);
                    }
                }
            }
#endif

            // find min-cut
            double e = g.solve();
            DRWN_LOG_MESSAGE("e = " << e << ", alpha = " << alpha);

            // extract assignment
            // TODO: compute true energy with respect to non-submodular energy
            if (e < minEnergy) {
                minEnergy = e;
                bConverged = false;

                varIndx = 0;
                for (int y = 0; y < assignment.rows; y++) {
                    for (int x = 0; x < assignment.cols; x++, varIndx++) {
                        if (g.inSetS(varIndx)) {
                            assignment.at<int>(y, x) = alpha;
                        }
                    }
                }
            }
        }
    }

    return assignment;
}
