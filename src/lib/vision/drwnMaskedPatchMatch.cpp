/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMaskedPatchMatch.cpp
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

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// drwnBasicPatchMatch -------------------------------------------------------

cv::Mat drwnBasicPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB,
    const cv::Size& patchRadius, cv::Mat& nnfA, unsigned maxIterations)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT((imgA.depth() == CV_8U) && (imgB.depth() == CV_8U));
    DRWN_ASSERT(imgA.channels() == imgB.channels());

    // determine patch size
    const cv::Size patchSize(2 * patchRadius.width + 1, 2 * patchRadius.height + 1);

    // initialize nearest neighbour field
    if (nnfA.empty()) {
        nnfA = cv::Mat(imgA.rows, imgA.cols, CV_16SC2, cv::Scalar::all(-1));
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
                nnfA.at<cv::Vec2s>(y, x) = cv::Vec2s(patchRadius.width + rand() % (imgB.cols - patchSize.width),
                    patchRadius.height + rand() % (imgB.rows - patchSize.height));
            }
        }
    }

    DRWN_ASSERT((nnfA.type() == CV_16SC2) && (nnfA.size() == imgA.size()));

    // compute initial costs
    cv::Mat costsA(nnfA.rows, nnfA.cols, CV_32FC1, cv::Scalar(0.0));

    for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
        for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
            const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);
            const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height, patchSize.width, patchSize.height);
            const cv::Rect roiB(p[0] - patchRadius.width, p[1] - patchRadius.height, patchSize.width, patchSize.height);
            costsA.at<float>(y, x) = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
        }
    }

    DRWN_LOG_DEBUG("...initial PatchMatch energy is " << (double)cv::sum(costsA)[0]);

    // search moves
    cv::Mat lastChanged = cv::Mat::zeros(nnfA.rows, nnfA.cols, CV_32SC1);
    for (unsigned i = 0; i < maxIterations; i++) {

        // forward propagation
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {

                // check if the pixel has changed
	        if (lastChanged.at<int>(y, x) < (int)i)
		    continue;

                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                // south
                if (y < nnfA.rows - patchRadius.height - 1) {
                    const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height + 1, patchSize.width, patchSize.height);
                    const cv::Rect roiB(p[0] - patchRadius.width, std::min(p[1] - patchRadius.height + 1, imgB.rows - patchSize.height - 1),
                        patchSize.width, patchSize.height);
                    const float cost = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
                    if (cost < costsA.at<float>(y + 1, x)) {
                        nnfA.at<cv::Vec2s>(y + 1, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                        costsA.at<float>(y + 1, x) = cost;
                        lastChanged.at<int>(y + 1, x) = i + 1;
                    }
                }

                // west
                if (x < nnfA.cols - patchRadius.width - 1) {
                    const cv::Rect roiA(x - patchRadius.width + 1, y - patchRadius.height, patchSize.width, patchSize.height);
                    const cv::Rect roiB(std::min(p[0] - patchRadius.width + 1, imgB.cols - patchSize.width - 1), p[1] - patchRadius.height,
                        patchSize.width, patchSize.height);
                    const float cost = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
                    if (cost < costsA.at<float>(y, x + 1)) {
                        nnfA.at<cv::Vec2s>(y, x + 1) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                        costsA.at<float>(y, x + 1) = cost;
                        lastChanged.at<int>(y, x + 1) = i + 1;
                    }
                }
            }
        }

        // backward propagation
        for (int y = nnfA.rows - patchRadius.height - 1; y >= patchRadius.height; y--) {
            for (int x = nnfA.cols - patchRadius.width - 1; x >= patchRadius.width; x--) {

                // check if the pixel has changed
                if (lastChanged.at<int>(y, x) < (int)i)
		    continue;

                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                // north
                if (y > patchRadius.height) {
                    const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height - 1, patchSize.width, patchSize.height);
                    const cv::Rect roiB(p[0] - patchRadius.width, std::max(p[1] - patchRadius.height - 1, 0), patchSize.width, patchSize.height);
                    const float cost = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
                    if (cost < costsA.at<float>(y - 1, x)) {
                        nnfA.at<cv::Vec2s>(y - 1, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                        costsA.at<float>(y - 1, x) = cost;
                        lastChanged.at<int>(y - 1, x) = i + 1;
                    }
                }

                // west
                if (x > patchRadius.width) {
                    const cv::Rect roiA(x - patchRadius.width - 1, y - patchRadius.height, patchSize.width, patchSize.height);
                    const cv::Rect roiB(std::max(p[0] - patchRadius.width - 1, 0), p[1] - patchRadius.height, patchSize.width, patchSize.height);
                    const float cost = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
                    if (cost < costsA.at<float>(y, x - 1)) {
                        nnfA.at<cv::Vec2s>(y, x - 1) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                        costsA.at<float>(y, x - 1) = cost;
                        lastChanged.at<int>(y, x - 1) = i + 1;
                    }
                }
            }
        }

        // random update
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                const int diameter = std::max<int>(i + 2 - lastChanged.at<int>(y, x), 1);
                const int x_min = std::max(patchRadius.width, p[0] - diameter);
                const int x_max = std::min(p[0] + diameter, imgB.cols - patchRadius.width - 1);
                const int y_min = std::max(patchRadius.height, p[1] - diameter);
                const int y_max = std::min(p[1] + diameter, imgB.rows - patchRadius.height - 1);

                const cv::Vec2s q = cv::Vec2s(x_min + rand() % (x_max - x_min + 1),
                    y_min + rand() % (y_max - y_min + 1));
                const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height, patchSize.width, patchSize.height);
                const cv::Rect roiB(q[0] - patchRadius.width, q[1] - patchRadius.height, patchSize.width, patchSize.height);
                const float cost = cv::norm(imgA(roiA), imgB(roiB), cv::NORM_L1);
                if (cost < costsA.at<float>(y, x)) {
                    nnfA.at<cv::Vec2s>(y, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                    costsA.at<float>(y, x) = cost;
                    lastChanged.at<int>(y, x) = i + 1;
                }
            }
        }

        DRWN_LOG_DEBUG("...at iteration " << (i + 1) << " PatchMatch energy is " << (double)cv::sum(costsA)[0]);
    }

    DRWN_FCN_TOC;
    return costsA;
}

// drwnSelfPatchMatch --------------------------------------------------------

cv::Mat drwnSelfPatchMatch(const cv::Mat& imgA, const cv::Size& patchRadius,
    cv::Mat& nnfA, double illegalOverlap, unsigned maxIterations)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT((0.0 <= illegalOverlap) && (illegalOverlap <= 1.0));

    // determine patch size
    const cv::Size patchSize(2 * patchRadius.width + 1, 2 * patchRadius.height + 1);
    const int illegalAreaOverlap = (int)std::max(1.0, illegalOverlap * patchSize.width * patchSize.height);

    // initialize nearest neighbour field
    if (nnfA.empty()) {
        nnfA = cv::Mat(imgA.rows, imgA.cols, CV_16SC2, cv::Scalar::all(-1));
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
                nnfA.at<cv::Vec2s>(y, x) = cv::Vec2s(patchRadius.width + rand() % (imgA.cols - patchSize.width),
                    patchRadius.height + rand() % (imgA.rows - patchSize.height));
            }
        }
    }

    DRWN_ASSERT((nnfA.type() == CV_16SC2) && (nnfA.size() == imgA.size()));

    // compute initial costs
    cv::Mat costsA(nnfA.rows, nnfA.cols, CV_32FC1, cv::Scalar(0.0));

    for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
        for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
            const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);
            const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height, patchSize.width, patchSize.height);
            const cv::Rect roiB(p[0] - patchRadius.width, p[1] - patchRadius.height, patchSize.width, patchSize.height);
            const double overlap = (roiA & roiB).area();
            if (overlap < illegalAreaOverlap) {
                costsA.at<float>(y, x) = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
            } else {
                costsA.at<float>(y, x) = DRWN_FLT_MAX;
            }
        }
    }

    DRWN_LOG_DEBUG("...initial PatchMatch energy is " << (double)cv::sum(costsA)[0]);

    // search moves
    cv::Mat lastChanged = cv::Mat::zeros(nnfA.rows, nnfA.cols, CV_32SC1);
    for (unsigned i = 0; i < maxIterations; i++) {

        // forward propagation
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {

                // check if the pixel has changed
	        if (lastChanged.at<int>(y, x) < (int)i)
		    continue;

                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                // south
                if (y < nnfA.rows - patchRadius.height - 1) {
                    const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height + 1, patchSize.width, patchSize.height);
                    const cv::Rect roiB(p[0] - patchRadius.width, std::min(p[1] - patchRadius.height + 1, imgA.rows - patchSize.height - 1),
                        patchSize.width, patchSize.height);
                    const double overlap = (roiA & roiB).area();
                    if (overlap < illegalAreaOverlap) {
                        const float cost = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
                        if (cost < costsA.at<float>(y + 1, x)) {
                            nnfA.at<cv::Vec2s>(y + 1, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                            costsA.at<float>(y + 1, x) = cost;
                            lastChanged.at<int>(y + 1, x) = i + 1;
                        }
                    }
                }

                // west
                if (x < nnfA.cols - patchRadius.width - 1) {
                    const cv::Rect roiA(x - patchRadius.width + 1, y - patchRadius.height, patchSize.width, patchSize.height);
                    const cv::Rect roiB(std::min(p[0] - patchRadius.width + 1, imgA.cols - patchSize.width - 1), p[1] - patchRadius.height,
                        patchSize.width, patchSize.height);
                    const double overlap = (roiA & roiB).area();
                    if (overlap < illegalAreaOverlap) {
                        const float cost = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
                        if (cost < costsA.at<float>(y, x + 1)) {
                            nnfA.at<cv::Vec2s>(y, x + 1) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                            costsA.at<float>(y, x + 1) = cost;
                            lastChanged.at<int>(y, x + 1) = i + 1;
                        }
                    }
                }
            }
        }

        // backward propagation
        for (int y = nnfA.rows - patchRadius.height - 1; y >= patchRadius.height; y--) {
            for (int x = nnfA.cols - patchRadius.width - 1; x >= patchRadius.width; x--) {

                // check if the pixel has changed
                if (lastChanged.at<int>(y, x) < (int)i)
		    continue;

                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                // north
                if (y > patchRadius.height) {
                    const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height - 1, patchSize.width, patchSize.height);
                    const cv::Rect roiB(p[0] - patchRadius.width, std::max(p[1] - patchRadius.height - 1, 0), patchSize.width, patchSize.height);
                    const double overlap = (roiA & roiB).area();
                    if (overlap < illegalAreaOverlap) {
                        const float cost = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
                        if (cost < costsA.at<float>(y - 1, x)) {
                            nnfA.at<cv::Vec2s>(y - 1, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                            costsA.at<float>(y - 1, x) = cost;
                            lastChanged.at<int>(y - 1, x) = i + 1;
                        }
                    }
                }

                // west
                if (x > patchRadius.width) {
                    const cv::Rect roiA(x - patchRadius.width - 1, y - patchRadius.height, patchSize.width, patchSize.height);
                    const cv::Rect roiB(std::max(p[0] - patchRadius.width - 1, 0), p[1] - patchRadius.height, patchSize.width, patchSize.height);
                    const double overlap = (roiA & roiB).area();
                    if (overlap < illegalAreaOverlap) {
                        const float cost = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
                        if (cost < costsA.at<float>(y, x - 1)) {
                            nnfA.at<cv::Vec2s>(y, x - 1) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                            costsA.at<float>(y, x - 1) = cost;
                            lastChanged.at<int>(y, x - 1) = i + 1;
                        }
                    }
                }
            }
        }

        // random update
        for (int y = patchRadius.height; y < nnfA.rows - patchRadius.height; y++) {
            for (int x = patchRadius.width; x < nnfA.cols - patchRadius.width; x++) {
                const cv::Vec2s p = nnfA.at<cv::Vec2s>(y, x);

                const int diameter = std::max<int>(i + 2 - lastChanged.at<int>(y, x), 1);
                const int x_min = std::max(patchRadius.width, p[0] - diameter);
                const int x_max = std::min(p[0] + diameter, imgA.cols - patchRadius.width - 1);
                const int y_min = std::max(patchRadius.height, p[1] - diameter);
                const int y_max = std::min(p[1] + diameter, imgA.rows - patchRadius.height - 1);

                const cv::Vec2s q = cv::Vec2s(x_min + rand() % (x_max - x_min + 1),
                    y_min + rand() % (y_max - y_min + 1));
                const cv::Rect roiA(x - patchRadius.width, y - patchRadius.height, patchSize.width, patchSize.height);
                const cv::Rect roiB(q[0] - patchRadius.width, q[1] - patchRadius.height, patchSize.width, patchSize.height);
                const double overlap = (roiA & roiB).area();
                if (overlap < illegalAreaOverlap) {
                    const float cost = cv::norm(imgA(roiA), imgA(roiB), cv::NORM_L1);
                    if (cost < costsA.at<float>(y, x)) {
                        nnfA.at<cv::Vec2s>(y, x) = cv::Vec2s(roiB.x + patchRadius.width, roiB.y + patchRadius.height);
                        costsA.at<float>(y, x) = cost;
                        lastChanged.at<int>(y, x) = i + 1;
                    }
                }
            }
        }

        DRWN_LOG_DEBUG("...at iteration " << (i + 1) << " PatchMatch energy is " << (double)cv::sum(costsA)[0]);
    }

    DRWN_FCN_TOC;
    return costsA;
}

// drwnNNFRepaint ------------------------------------------------------------

cv::Mat drwnNNFRepaint(const cv::Mat& img, const cv::Mat& nnf)
{
    DRWN_ASSERT(img.type() == CV_8UC3);
    DRWN_ASSERT(nnf.type() == CV_16SC2);

    cv::Mat canvas = cv::Mat::zeros(nnf.size(), img.type());
    for (int y = 0; y < nnf.rows; y++) {
        for (int x = 0; x < nnf.cols; x++) {
            const cv::Vec2s p = nnf.at<cv::Vec2s>(y, x);
            if ((p[0] < 0) || (p[1] < 0)) continue;
            canvas.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(p[1], p[0]);
        }
    }
    return canvas;
}

// drwnMaskedPatchMatch ------------------------------------------------------

bool drwnMaskedPatchMatch::TRY_IDENTITY_INIT = true;
int drwnMaskedPatchMatch::DISTANCE_MEASURE = cv::NORM_L1;
float drwnMaskedPatchMatch::HEIGHT_PENALTY = 32.0f;

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, unsigned patchRadius) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _patchRadius(patchRadius, patchRadius)
{
    DRWN_ASSERT((imgA.depth() == CV_8U) && (imgB.depth() == CV_8U));
    DRWN_ASSERT(imgA.channels() == imgB.channels());

    _maskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0xff));
    _maskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0xff));
    _invmaskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0x00));
    _invmaskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0x00));

    cacheValidPixels();
    initialize();
}

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Size& patchRadius) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _patchRadius(patchRadius)
{
    DRWN_ASSERT((imgA.depth() == CV_8U) && (imgB.depth() == CV_8U));
    DRWN_ASSERT(imgA.channels() == imgB.channels());

    _maskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0xff));
    _maskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0xff));
    _invmaskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0x00));
    _invmaskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0x00));

    cacheValidPixels();
    initialize();
}

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB,
    const cv::Mat& maskA, const cv::Mat& maskB, unsigned patchRadius) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _maskA(maskA.clone()), _maskB(maskB.clone()),
    _patchRadius(patchRadius, patchRadius)
{
    DRWN_ASSERT((imgA.depth() == CV_8U) && (imgB.depth() == CV_8U));
    DRWN_ASSERT(imgA.channels() == imgB.channels());
    DRWN_ASSERT(maskA.empty() || ((maskA.type() == CV_8UC1) && (maskA.size() == imgA.size())));
    DRWN_ASSERT(maskB.empty() || ((maskB.type() == CV_8UC1) && (maskB.size() == imgB.size())));

    if (maskA.empty()) { _maskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0xff)); }
    if (maskB.empty()) { _maskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0xff)); }
    cv::compare(_maskA, cv::Scalar(0x00), _invmaskA, CV_CMP_EQ);
    cv::compare(_maskB, cv::Scalar(0x00), _invmaskB, CV_CMP_EQ);

    cacheValidPixels();
    initialize();
}

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB,
    const cv::Mat& maskA, const cv::Mat& maskB, const cv::Size& patchRadius) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _maskA(maskA.clone()), _maskB(maskB.clone()), _patchRadius(patchRadius)
{
    DRWN_ASSERT((imgA.depth() == CV_8U) && (imgB.depth() == CV_8U));
    DRWN_ASSERT(imgA.channels() == imgB.channels());
    DRWN_ASSERT(maskA.empty() || ((maskA.type() == CV_8UC1) && (maskA.size() == imgA.size())));
    DRWN_ASSERT(maskB.empty() || ((maskB.type() == CV_8UC1) && (maskB.size() == imgB.size())));

    if (maskA.empty()) { _maskA = cv::Mat(imgA.size(), CV_8UC1, cv::Scalar(0xff)); }
    if (maskB.empty()) { _maskB = cv::Mat(imgB.size(), CV_8UC1, cv::Scalar(0xff)); }
    cv::compare(_maskA, cv::Scalar(0x00), _invmaskA, CV_CMP_EQ);
    cv::compare(_maskB, cv::Scalar(0x00), _invmaskB, CV_CMP_EQ);

    cacheValidPixels();
    initialize();
}

std::pair<cv::Rect, cv::Rect> drwnMaskedPatchMatch::getMatchingPatches(const cv::Point& ptA) const
{
    const int x = std::max(_patchRadius.width, std::min(ptA.x, _imgA.cols - _patchRadius.width - 1));
    const int y = std::max(_patchRadius.height, std::min(ptA.y, _imgA.rows - _patchRadius.height - 1));
    const int w = 2 * (_patchRadius.width - abs(ptA.x - x)) + 1;
    const int h = 2 * (_patchRadius.height - abs(ptA.y - y)) + 1;
    const int dx = 2 * std::max(0, ptA.x - x);
    const int dy = 2 * std::max(0, ptA.y - y);

    std::pair<cv::Rect, cv::Rect> match;
    match.first = cv::Rect(x - _patchRadius.width + dx, y - _patchRadius.height + dy, w, h);

    const cv::Vec2s ptB = _nnfA.at<cv::Vec2s>(y, x);
    match.second = cv::Rect(ptB[0] - _patchRadius.width + dx, ptB[1] - _patchRadius.height + dy, w, h);

    return match;
}

void drwnMaskedPatchMatch::initialize(const cv::Size& patchRadius)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(2 * patchRadius.width + 1 < std::min(_imgA.cols, _imgB.cols));
    DRWN_ASSERT(2 * patchRadius.height + 1 < std::min(_imgA.rows, _imgB.rows));

    // update valid pixels and remember patch size
    if (patchRadius != _patchRadius) {
        DRWN_LOG_WARNING("change patch radius to " << patchRadius);
        _patchRadius = patchRadius;
        cacheValidPixels();
    }

    // initialize nearest neighbour field if not already
    if ((_nnfA.rows != _imgA.rows) || (_nnfA.cols != _imgA.cols)) {
        const cv::Rect roi(_patchRadius.width, _patchRadius.height,
            _imgA.cols - 2 * _patchRadius.width, _imgA.rows - 2 * _patchRadius.height);
        _nnfA = cv::Mat(_imgA.size(), CV_16SC2, cv::Scalar(-1, -1));
        _costsA = cv::Mat::zeros(_imgA.size(), CV_32FC1);
        _costsA(roi).setTo(cv::Scalar(DRWN_FLT_MAX));
        _lastChanged = cv::Mat(_nnfA.size(), CV_32SC1);
    }

    // construct vector of allowed match points
    vector<cv::Point> allowedB;
    allowedB.reserve(_imgB.rows * _imgB.cols);
    for (int y = 0; y < _imgB.rows; y++) {
        for (int x = 0; x < _imgB.cols; x++) {
            if (_validB.at<unsigned char>(y, x) != 0x00) {
                allowedB.push_back(cv::Point(x, y));
            }
        }
    }
    DRWN_ASSERT(!allowedB.empty());

    for (int y = _patchRadius.height; y < _nnfA.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _nnfA.cols - _patchRadius.width; x++) {
            const cv::Point ptA(x, y);

            // try initialize with identity first
            if (TRY_IDENTITY_INIT) {
                if (_imgA.size() == _imgB.size()) {
                    update(ptA, ptA);
                }
                if (_costsA.at<float>(y, x) == 0.0f) {
                    continue;
                }
            }

            //! only attempt valid B
            update(ptA, allowedB[rand() % allowedB.size()]);
        }
    }

#if 0
    // check integrity of nnf
    for (int y = _patchRadius.height; y < _nnfA.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _nnfA.cols - _patchRadius.width; x++) {
            const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
            DRWN_ASSERT_MSG((p[0] >= _patchRadius.width) && (p[0] < _imgB.cols - _patchRadius.width),
                "nnf(" << x << ", " << y << ") = " << p << " with cost " << _costsA.at<float>(y, x));
            DRWN_ASSERT_MSG((p[1] >= _patchRadius.height) && (p[1] < _imgB.rows - _patchRadius.height),
                "nnf(" << x << ", " << y << ") = " << p << " with cost " << _costsA.at<float>(y, x));
        }
    }
#endif

    DRWN_LOG_DEBUG("...initial PatchMatch energy is " << energy());

    // update iterations
    _iterationCount = 0;
    _lastChanged = cv::Mat::zeros(_nnfA.rows, _nnfA.cols, CV_32SC1);
    DRWN_FCN_TOC;
}

void drwnMaskedPatchMatch::initialize(const cv::Mat& nnf)
{
    DRWN_ASSERT((nnf.size() == _imgA.size()) && (nnf.type() == CV_16SC2));

    _nnfA = nnf.clone();
    rescore();

    // update iterations
    _iterationCount = 0;
    _lastChanged = cv::Mat::zeros(_nnfA.rows, _nnfA.cols, CV_32SC1);
}

const cv::Mat& drwnMaskedPatchMatch::search(cv::Rect roiToUpdate, unsigned maxIterations)
{
    DRWN_FCN_TIC;

#if 0
    // check integrity of nnf
    for (int y = _patchRadius.height; y < _nnfA.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _nnfA.cols - _patchRadius.width; x++) {
            const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
            DRWN_ASSERT_MSG((p[0] >= _patchRadius.width) && (p[0] < _imgB.cols - _patchRadius.width),
                "nnf(" << x << ", " << y << ") = " << p << " with cost " << _costsA.at<float>(y, x));
            DRWN_ASSERT_MSG((p[1] >= _patchRadius.height) && (p[1] < _imgB.rows - _patchRadius.height),
                "nnf(" << x << ", " << y << ") = " << p << " with cost " << _costsA.at<float>(y, x));
        }
    }
#endif

    // make sure roiToUpdate is within the search region
    roiToUpdate = roiToUpdate & cv::Rect(_patchRadius.width, _patchRadius.height,
        _nnfA.cols - 2 * _patchRadius.width, _nnfA.rows - 2 * _patchRadius.height);

    // perform search moves (source centric)
    for (unsigned i = 0; i < maxIterations; i++) {

        // forward propagation
        for (int y = roiToUpdate.y; y < roiToUpdate.y + roiToUpdate.height; y++) {
            for (int x = roiToUpdate.x; x < roiToUpdate.x + roiToUpdate.width; x++) {

                // souce patch
                const cv::Point ptA(x, y);

                // north
                if ((y > _patchRadius.height) && (_lastChanged.at<int>(y - 1, x) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y - 1, x);
                    const cv::Point ptB(p[0], std::min(p[1] + 1, _imgB.rows - _patchRadius.height - 1));
                    update(ptA, ptB);
                }

                // east
                if ((x > _patchRadius.width) && (_lastChanged.at<int>(y, x - 1) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x - 1);
                    const cv::Point ptB(std::min(p[0] + 1, _imgB.cols - _patchRadius.width - 1), p[1]);
                    update(ptA, ptB);
                }
            }
        }

        // backward propagation
        for (int y = roiToUpdate.y + roiToUpdate.height - 1; y >= roiToUpdate.y; y--) {
            for (int x = roiToUpdate.x + roiToUpdate.width - 1; x >= roiToUpdate.x; x--) {

                // souce patch
                const cv::Point ptA(x, y);

                // south
                if ((y < _nnfA.rows - _patchRadius.height - 1) && (_lastChanged.at<int>(y + 1, x) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y + 1, x);
                    const cv::Point ptB(p[0], std::max(p[1] - 1, _patchRadius.height));
                    update(ptA, ptB);
                }

                // west
                if ((x < _nnfA.cols - _patchRadius.width - 1)  && (_lastChanged.at<int>(y, x + 1) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x + 1);
                    const cv::Point ptB(std::max(p[0] - 1, _patchRadius.width), p[1]);
                    update(ptA, ptB);
                }
            }
        }

        // random update
        for (int y = roiToUpdate.y; y < roiToUpdate.y + roiToUpdate.height; y++) {
            for (int x = roiToUpdate.x; x < roiToUpdate.x + roiToUpdate.width; x++) {
                // skip if we've found the optimum
                if (_costsA.at<float>(y, x) == 0.0f) {
                    continue;
                }

                // select random offset from current position
                const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
                const int diameter = std::max(_iterationCount - _lastChanged.at<int>(y, x), 1);
                const int xmin = std::max(_patchRadius.width, p[0] - diameter);
                const int xmax = std::min(_imgB.cols - _patchRadius.width - 1, p[0] + diameter);
                const int ymin = std::max(_patchRadius.height, p[1] - diameter);
                const int ymax = std::min(_imgB.rows - _patchRadius.height - 1, p[1] + diameter);

                DRWN_ASSERT((xmax >= xmin) && (ymax >= ymin));
                const cv::Vec2s q = cv::Vec2s(xmin + rand() % (xmax - xmin + 1),
                    ymin + rand() % (ymax - ymin + 1));
                if (q == p) continue;
                update(cv::Point(x, y), cv::Point(q[0], q[1]));
            }
        }

        // update iteration counter
        _iterationCount += 1;
        DRWN_LOG_DEBUG("...at iteration " << _iterationCount << " PatchMatch energy is " << energy());
    }

    DRWN_FCN_TOC;
    return _nnfA;
}

void drwnMaskedPatchMatch::modifySourceImage(const cv::Rect& roi, const cv::Mat& img, double alpha)
{
    DRWN_ASSERT((img.rows == roi.height) && (img.cols == roi.width));
    DRWN_ASSERT(img.type() == _imgA.type());
    DRWN_ASSERT((0.0 <= alpha) && (alpha <= 1.0));

    // copy patch and unmask it
    img.copyTo(_imgA(roi), _invmaskA(roi));
    if (alpha != 0.0) {
        cv::Mat tmp = _imgA(roi).clone();
        cv::addWeighted(tmp, (1.0 - alpha), img, alpha, 0.0, _imgA(roi));
    }
    _maskA(roi).setTo(cv::Scalar(0xff));
    _invmaskA(roi).setTo(cv::Scalar(0x00));
    updateValidPixels(roi);

    // update patch scores
    rescore(roi);

    // reset last changed pixels around mask
    cv::Rect affected = affectedRegion(roi);
    affected.x -= 1; affected.y -= 1; affected.width += 2; affected.height += 2;
    drwnTruncateRect(affected, _lastChanged);
    _lastChanged(affected).setTo(cv::Scalar(_iterationCount + 1));
}

void drwnMaskedPatchMatch::modifySourceImage(const cv::Rect& roiA, const cv::Rect& roiB, double alpha)
{
    modifySourceImage(roiA, _imgB(roiB), alpha);
}

void drwnMaskedPatchMatch::modifyTargetImage(const cv::Rect& roi, const cv::Mat& img, double alpha)
{
    DRWN_ASSERT((img.rows == roi.height) && (img.cols == roi.width));
    DRWN_ASSERT(img.type() == _imgB.type());
    DRWN_ASSERT((0.0 <= alpha) && (alpha <= 1.0));

    // copy patch and unmask it
    img.copyTo(_imgB(roi), _invmaskB(roi));
    if (alpha != 0.0) {
        cv::Mat tmp = _imgB(roi).clone();
        cv::addWeighted(tmp, (1.0 - alpha), img, alpha, 0.0, _imgB(roi));
    }
    _maskB(roi).setTo(cv::Scalar(0xff));
    _invmaskB(roi).setTo(cv::Scalar(0x00));

    // update valid pixels
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1),
        cv::Point(_patchRadius.width, _patchRadius.height));
    cv::erode(_maskB, _validB, element);

    _validB(cv::Rect(0, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(_validB.cols - _patchRadius.width, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, 0, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, _validB.rows - _patchRadius.height, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));

    // rescore any region that matches into this patch (and the masked pixels have changed)
    if (alpha != 0.0) {
        for (int y = _patchRadius.height; y < _nnfA.rows - _patchRadius.height; y++) {
            for (int x = _patchRadius.width; x < _nnfA.cols - _patchRadius.width; x++) {
                const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
                const cv::Rect roiB(p[0] - _patchRadius.width, p[1] - _patchRadius.height,
                    2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1);
                if ((roiB & roi).area() > 0) {
                    _costsA.at<float>(y, x) = score(cv::Point(x, y), cv::Point(p[0], p[1]));
                    _lastChanged.at<int>(y, x) = _iterationCount + 1;
                }
            }
        }
    }

    // mark as updated any patch adjacent to the updated region
    cv::Rect adjROI(roi.x - 1, roi.y - 1, roi.width + 2, roi.height + 2);
    for (int y = 0; y < _nnfA.rows; y++) {
        for (int x = 0; x < _nnfA.cols; x++) {
            const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
            if (cv::Point(p[0], p[1]).inside(adjROI)) {
                _lastChanged.at<int>(y, x) = _iterationCount + 1;
            }
        }
    }
}

void drwnMaskedPatchMatch::modifyTargetImage(const cv::Rect& roiA, const cv::Rect& roiB, double alpha)
{
    modifyTargetImage(roiA, _imgB(roiB), alpha);
}

void drwnMaskedPatchMatch::expandTargetMask(unsigned radius)
{
    // mark as updated any matches adjacent to the boundary of the old mask
    cv::Mat adj;
    cv::dilate(_maskB, adj, cv::Mat());
    for (int y = 0; y < _nnfA.rows; y++) {
        for (int x = 0; x < _nnfA.cols; x++) {
            const cv::Vec2s p = _nnfA.at<cv::Vec2s>(y, x);
            if (adj.at<unsigned char>(p[1], p[0]) != 0x00) {
                _lastChanged.at<int>(y, x) = _iterationCount + 1;
            }
        }
    }

    // dilate the mask/erode the inverse mask
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * radius + 1, 2 * radius + 1), cv::Point(radius, radius));
    cv::dilate(_maskB, _maskB, element);
    cv::compare(_maskB, cv::Scalar(0x00), _invmaskB, CV_CMP_EQ);

    // update valid pixels
    element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1),
        cv::Point(_patchRadius.width, _patchRadius.height));
    cv::erode(_maskB, _validB, element);

    _validB(cv::Rect(0, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(_validB.cols - _patchRadius.width, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, 0, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, _validB.rows - _patchRadius.height, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));

}

cv::Mat drwnMaskedPatchMatch::visualize() const
{
    vector<cv::Mat> views(_imgA.type() == CV_8UC3 ? 7 : 5);

    // nnf (x and y)
    cv::split(_nnfA, &views[0]);
    views[0] = drwnCreateHeatMap(views[0], DRWN_COLORMAP_RAINBOW);
    views[1] = drwnCreateHeatMap(views[1], DRWN_COLORMAP_RAINBOW);

    // cost
    views[2] = _costsA.clone();
    drwnScaleToRange(views[2], 0.0, 1.0);
    views[2] = drwnCreateHeatMap(views[2], DRWN_COLORMAP_REDGREEN);

    // masks
    views[3] = drwnColorImage(_overlapA);
    drwnDrawRegionBoundaries(views[3], _maskA, CV_RGB(255, 0, 0), 1);
    views[4] = drwnColorImage(_validB);
    drwnDrawRegionBoundaries(views[4], _maskB, CV_RGB(255, 0, 0), 1);

    if (views.size() == 7) {
        // masked source
        views[5] = _imgA.clone();
        drwnDrawRegionBoundaries(views[5], _maskA, CV_RGB(0, 255, 0), 2);

        // masked target
        views[6] = _imgB.clone();
        drwnShadeRegion(views[6], _invmaskB, CV_RGB(255, 0, 0), 1.0, DRWN_FILL_DIAG, 1);
        drwnDrawRegionBoundaries(views[6], _maskB, CV_RGB(255, 0, 0), 2);
    }

    return drwnCombineImages(views, 1);
}

bool drwnMaskedPatchMatch::update(const cv::Point& ptA, const cv::Point& ptB)
{
    if (_validB.at<unsigned char>(ptB.y, ptB.x) == 0x00)
        return false;

    const float cost = score(ptA, ptB);
    if (cost < _costsA.at<float>(ptA.y, ptA.x)) {
        _nnfA.at<cv::Vec2s>(ptA.y, ptA.x) = cv::Vec2s(ptB.x, ptB.y);
        _costsA.at<float>(ptA.y, ptA.x) = cost;
        _lastChanged.at<int>(ptA.y, ptA.x) = _iterationCount + 1;
        return true;
    }

    return false;
}

float drwnMaskedPatchMatch::score(const cv::Point& ptA, const cv::Point& ptB) const
{
#if 0
    //! \todo add no-self option
    if (ptA == ptB) return DRWN_FLT_MAX;
#endif

    // compute distance between patch features
    const cv::Rect roiA(ptA.x - _patchRadius.width, ptA.y - _patchRadius.height,
        2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1);
    const cv::Rect roiB(ptB.x - _patchRadius.width, ptB.y - _patchRadius.height,
        2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1);

    float cost;
    if (_overlapA.at<unsigned char>(ptA.y, ptA.x) == 0x00) {
        cost = cv::norm(_imgA(roiA), _imgB(roiB), DISTANCE_MEASURE);
    } else {
        cost = cv::norm(_imgA(roiA), _imgB(roiB), DISTANCE_MEASURE, _maskA(roiA));
    }

    // add height prior
    cost += HEIGHT_PENALTY * _patchRadius.height * _patchRadius.width *
        fabs((float)ptA.y / (float)_imgA.rows - (float)ptB.y / (float)_imgB.rows);

    return cost;
}

void drwnMaskedPatchMatch::rescore(const cv::Rect& roi)
{
    DRWN_FCN_TIC;
    const cv::Rect region = affectedRegion(roi);
    for (int y = region.y; y < region.y + region.height; y++) {
        for (int x = region.x; x < region.x + region.width; x++) {
            const cv::Vec2s ptB = _nnfA.at<cv::Vec2s>(y, x);
            _costsA.at<float>(y, x) = score(cv::Point(x, y), cv::Point(ptB[0], ptB[1]));
            _lastChanged.at<int>(y, x) = _iterationCount + 1;
        }
    }
    DRWN_FCN_TOC;
}

void drwnMaskedPatchMatch::cacheValidPixels()
{
#if 0
    // find centre of any patch that overlap with invmaskA
    _overlapA = cv::Mat::zeros(_maskA.size(), CV_8UC1);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1),
        cv::Point(_patchRadius.width, _patchRadius.height));
    cv::dilate(_invmaskA, _overlapA, element);


    cv::Mat maskSum;
    /*
    cv::integral(_invmaskA, maskSum, CV_32S);

    for (int y = _patchRadius.height; y < _overlapA.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _overlapA.cols - _patchRadius.width; x++) {
            const int count = maskSum.at<int>(y - _patchRadius.height, x - _patchRadius.width) +
                maskSum.at<int>(y + _patchRadius.height + 1, x + _patchRadius.width + 1) -
                maskSum.at<int>(y - _patchRadius.height, x + _patchRadius.width + 1) -
                maskSum.at<int>(y + _patchRadius.height + 1, x - _patchRadius.width);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(y, x) = 0xff;
            }
        }
    }
    */

     // find patches that don't overlap with maskB
    _validB = cv::Mat::zeros(_maskB.size(), CV_8UC1);

    maskSum.release();
    cv::integral(_invmaskB, maskSum, CV_32S);

    for (int y = _patchRadius.height; y < _validB.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _validB.cols - _patchRadius.width; x++) {
            const int count = maskSum.at<int>(y - _patchRadius.height, x - _patchRadius.width) +
                maskSum.at<int>(y + _patchRadius.height + 1, x + _patchRadius.width + 1) -
                maskSum.at<int>(y - _patchRadius.height, x + _patchRadius.width + 1) -
                maskSum.at<int>(y + _patchRadius.height + 1, x - _patchRadius.width);
            if (count == 0x00) {
                _validB.at<unsigned char>(y, x) = 0xff;
            }
        }
    }

    cv::Mat tmp;
    cv::erode(_maskB, tmp, element);
    drwnShowDebuggingImage(tmp, "A", false);
    drwnShowDebuggingImage(_validB, "B", true);

#else
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1),
        cv::Point(_patchRadius.width, _patchRadius.height));

    cv::dilate(_invmaskA, _overlapA, element);
    _overlapA(cv::Rect(0, 0, _patchRadius.width, _overlapA.rows)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(_overlapA.cols - _patchRadius.width, 0, _patchRadius.width, _overlapA.rows)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(0, 0, _overlapA.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(0, _overlapA.rows - _patchRadius.height, _overlapA.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));

    cv::erode(_maskB, _validB, element);
    _validB(cv::Rect(0, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(_validB.cols - _patchRadius.width, 0, _patchRadius.width, _validB.rows)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, 0, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
    _validB(cv::Rect(0, _validB.rows - _patchRadius.height, _validB.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
#endif
}

void drwnMaskedPatchMatch::updateValidPixels(const cv::Rect& roi)
{
    DRWN_FCN_TIC;

#if 1
    const cv::Rect region = affectedRegion(roi);
    const cv::Rect iregion(region.x - _patchRadius.width, region.y - _patchRadius.height,
        region.width + 2 * _patchRadius.width, region.height + 2 * _patchRadius.height);

    cv::Mat maskSum;
    cv::integral(_invmaskA(iregion), maskSum, CV_32S);

    for (int y = 0; y < region.height; y++) {
        for (int x = 0; x < region.width; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + 2 * _patchRadius.height + 1, x + 2 * _patchRadius.width + 1) -
                maskSum.at<int>(y, x + 2 * _patchRadius.width + 1) -
                maskSum.at<int>(y + 2 * _patchRadius.height + 1, x);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(region.y + y, region.x + x) = 0xff;
            } else {
                _overlapA.at<unsigned char>(region.y + y, region.x + x) = 0x00;
            }
        }
    }
#elif 0
    cv::Mat maskSum;
    cv::integral(_invmaskA, maskSum, CV_32S);

    for (int y = _patchRadius.height; y < _overlapA.rows - _patchRadius.height; y++) {
        for (int x = _patchRadius.width; x < _overlapA.cols - _patchRadius.width; x++) {
            const int count = maskSum.at<int>(y - _patchRadius.height, x - _patchRadius.width) +
                maskSum.at<int>(y + _patchRadius.height + 1, x + _patchRadius.width + 1) -
                maskSum.at<int>(y - _patchRadius.height, x + _patchRadius.width + 1) -
                maskSum.at<int>(y + _patchRadius.height + 1, x - _patchRadius.width);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(y, x) = 0xff;
            }
        }
    }
#else
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1),
        cv::Point(_patchRadius.width, _patchRadius.height));
    cv::dilate(_invmaskA, _overlapA, element);
    _overlapA(cv::Rect(0, 0, _patchRadius.width, _overlapA.rows)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(_overlapA.cols - _patchRadius.width, 0, _patchRadius.width, _overlapA.rows)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(0, 0, _overlapA.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
    _overlapA(cv::Rect(0, _overlapA.rows - _patchRadius.height, _overlapA.cols, _patchRadius.height)).setTo(cv::Scalar(0x00));
#endif

    DRWN_FCN_TOC;
}

// drwnMaskedPatchMatchConfig -----------------------------------------------
//! \addtogroup drwnConfigSettings
//! \section drwnMaskedPatchMatch
//! \b distance      :: distance measure (1, 2 or 4 for L_{\\infty}, L1 and L2 norm, resp.)\n
//! \b heightPenalty :: strength of prior for matching to same image row\n

class drwnMaskedPatchMatchConfig : public drwnConfigurableModule {
public:
    drwnMaskedPatchMatchConfig() : drwnConfigurableModule("drwnMaskedPatchMatch") { }
    ~drwnMaskedPatchMatchConfig() { }

    void usage(ostream &os) const {
        os << "      distance        :: distance measure (default: " << drwnMaskedPatchMatch::DISTANCE_MEASURE << ")\n";
        os << "      heightPenalty   :: strength of prior for matching to same image row (default: " << drwnMaskedPatchMatch::HEIGHT_PENALTY << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "distance")) {
            drwnMaskedPatchMatch::DISTANCE_MEASURE = atoi(value);
        } else if (!strcmp(name, "heightPenalty")) {
            drwnMaskedPatchMatch::HEIGHT_PENALTY = std::max(0.0, atof(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnMaskedPatchMatchConfig gMaskedPatchMatchConfig;
