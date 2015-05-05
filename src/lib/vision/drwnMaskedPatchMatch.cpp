/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
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

// drwnMaskedPatchMatch ------------------------------------------------------

int drwnMaskedPatchMatch::DISTANCE_MEASURE = cv::NORM_L1;
float drwnMaskedPatchMatch::HEIGHT_PENALTY = 32.0f;

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, unsigned patchSize) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _patchSize(patchSize, patchSize)
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

drwnMaskedPatchMatch::drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Size& patchSize) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _patchSize(patchSize)
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
    const cv::Mat& maskA, const cv::Mat& maskB, unsigned patchSize) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _maskA(maskA.clone()), _maskB(maskB.clone()),
    _patchSize(patchSize, patchSize)
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
    const cv::Mat& maskA, const cv::Mat& maskB, const cv::Size& patchSize) :
    _imgA(imgA.clone()), _imgB(imgB.clone()), _maskA(maskA.clone()), _maskB(maskB.clone()), _patchSize(patchSize)
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

void drwnMaskedPatchMatch::initialize(const cv::Size& patchSize)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(patchSize.width < std::min(_imgA.cols, _imgB.cols));
    DRWN_ASSERT(patchSize.height < std::min(_imgA.rows, _imgB.rows));

    // update valid pixels and remember patch size
    if (patchSize != _patchSize) {
        DRWN_LOG_WARNING("change patch size to " << patchSize);
        _patchSize = patchSize;
        cacheValidPixels();
    }

    // initialize nearest neighbour field if not already
    if ((_nnfA.rows != _imgA.rows - _patchSize.height + 1) || (_nnfA.rows != _imgA.rows - _patchSize.height + 1)) {
        _nnfA = cv::Mat(_imgA.rows - _patchSize.height + 1, _imgA.cols - _patchSize.width + 1, CV_16SC2, cv::Scalar(0, 0));
        _costsA = cv::Mat(_nnfA.rows, _nnfA.cols, CV_32FC1, cv::Scalar(DRWN_FLT_MAX));
        _lastChanged = cv::Mat(_nnfA.rows, _nnfA.cols, CV_32SC1);
    }

    for (int y = 0; y < _nnfA.rows; y++) {
        for (int x = 0; x < _nnfA.cols; x++) {
            const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);
#if 1
            // try initialize with identity first
            if (_imgA.size() == _imgB.size()) {
                update(roiA, roiA);
            }
            if (_costsA.at<float>(y, x) == 0.0f)
                continue;
#endif
            //! \todo only attempt valid B
            const cv::Rect roiB(rand() % (_imgB.cols - _patchSize.width),
                rand() % (_imgB.rows - _patchSize.height), _patchSize.width, _patchSize.height);
            update(roiA, roiB);
        }
    }

    DRWN_LOG_DEBUG("...initial PatchMatch energy is " << energy());

    // update iterations
    _iterationCount = 0;
    _lastChanged = cv::Mat::zeros(_nnfA.rows, _nnfA.cols, CV_32SC1);
    DRWN_FCN_TOC;
}

void drwnMaskedPatchMatch::initialize(const cv::Mat& nnf)
{
    DRWN_ASSERT(nnf.type() == CV_16SC2);

    cv::Size patchSize(_imgA.cols - nnf.cols + 1, _imgA.rows - nnf.rows + 1);
    _nnfA = nnf.clone();
    rescore();
    initialize(patchSize);
}

const cv::Mat& drwnMaskedPatchMatch::search(const cv::Rect& roiToUpdate, unsigned maxIterations)
{
    DRWN_ASSERT_MSG((roiToUpdate.x >= 0) && (roiToUpdate.x + roiToUpdate.width <= _nnfA.cols) &&
        (roiToUpdate.y >= 0) && (roiToUpdate.y + roiToUpdate.height <= _nnfA.rows),
        roiToUpdate << " outside of " << toString(_nnfA));
    DRWN_FCN_TIC;

    // perform search moves (source centric)
    for (unsigned i = 0; i < maxIterations; i++) {

        // forward propagation
        for (int y = roiToUpdate.y; y < roiToUpdate.y + roiToUpdate.height; y++) {
            for (int x = roiToUpdate.x; x < roiToUpdate.x + roiToUpdate.width; x++) {

                // souce patch
                const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);

                // north
                if ((y > 0) && (_lastChanged.at<int>(y - 1, x) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<Vec2s>(y - 1, x);
                    const cv::Rect roiB(p[0], std::min(p[1] + 1, _imgB.rows - _patchSize.height),
                        _patchSize.width, _patchSize.height);
                    update(roiA, roiB);
                }

                // east
                if ((x > 0) && (_lastChanged.at<int>(y, x - 1) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<Vec2s>(y, x - 1);
                    const cv::Rect roiB(std::min(p[0] + 1, _imgB.cols - _patchSize.width), p[1],
                        _patchSize.width, _patchSize.height);
                    update(roiA, roiB);
                }
            }
        }

        // backward propagation
        for (int y = roiToUpdate.y + roiToUpdate.height - 1; y >= roiToUpdate.y; y--) {
            for (int x = roiToUpdate.x + roiToUpdate.width - 1; x >= roiToUpdate.x; x--) {

                // souce patch
                const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);

                // south
                if ((y < _nnfA.rows - 1) && (_lastChanged.at<int>(y + 1, x) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<Vec2s>(y + 1, x);
                    const cv::Rect roiB(p[0], std::max(p[1] - 1, 0), _patchSize.width, _patchSize.height);
                    update(roiA, roiB);
                }

                // west
                if ((x < _nnfA.cols - 1)  && (_lastChanged.at<int>(y, x + 1) >= _iterationCount)) {
                    const cv::Vec2s p = _nnfA.at<Vec2s>(y, x + 1);
                    const cv::Rect roiB(std::max(p[0] - 1, 0), p[1], _patchSize.width, _patchSize.height);
                    update(roiA, roiB);
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
                const cv::Vec2s p = _nnfA.at<Vec2s>(y, x);
                const int diameter = std::max(_iterationCount - _lastChanged.at<int>(y, x), 1);
                const int xmin = std::max(0, p[0] - diameter);
                const int xmax = std::min(_imgB.cols - _patchSize.width, p[0] + diameter);
                const int ymin = std::max(0, p[1] - diameter);
                const int ymax = std::min(_imgB.rows - _patchSize.height, p[1] + diameter);

                //! \todo only attempt valid B
                const cv::Vec2s q = cv::Vec2s(xmin + rand() % (xmax - xmin + 1),
                    ymin + rand() % (ymax - ymin + 1));
                if (q == p) continue;
                const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);
                const cv::Rect roiB(q[0], q[1], _patchSize.width, _patchSize.height);
                update(roiA, roiB);
            }
        }

#if 0
        // local hill climb
        for (int y = roiToUpdate.y; y < roiToUpdate.y + roiToUpdate.height; y++) {
            for (int x = roiToUpdate.x; x < roiToUpdate.x + roiToUpdate.width; x++) {

                // check if the pixel has changed
                if (_lastChanged.at<int>(y, x) < _iterationCount)
                    continue;

                const cv::Vec2s p = _nnfA.at<Vec2s>(y, x);
                const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if ((dx == 0) && (dy == 0)) continue;
                        const cv::Rect roiB(std::min(std::max(p[0] + dx, 0), _imgB.cols - _patchSize.width),
                            std::min(std::max(p[1] + dy, 0), _imgB.rows - _patchSize.height),
                            _patchSize.width, _patchSize.height);
                        update(roiA, roiB);
                    }
                }
            }
        }
#endif

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
    //! \todo speedup
    cv::Mat maskSum;
    cv::integral(_invmaskB, maskSum, CV_32S);

    for (int y = 0; y < _validB.rows; y++) {
        for (int x = 0; x < _validB.cols; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            _validB.at<unsigned char>(y, x) = (count == 0x00) ? 0xff : 0x00;
        }
    }

    // rescore any region that matches into this patch (and the masked pixels have changed)
    if (alpha != 0.0) {
        for (int y = 0; y < _nnfA.rows; y++) {
            for (int x = 0; x < _nnfA.cols; x++) {
                const cv::Vec2s p = _nnfA.at<Vec2s>(y, x);
                const cv::Rect roiB(p[0], p[1], _patchSize.width, _patchSize.height);
                if ((roiB & roi).area() > 0) {
                    _costsA.at<float>(y, x) = score(cv::Rect(x, y, _patchSize.width, _patchSize.height), roiB);
                    _lastChanged.at<int>(y, x) = _iterationCount + 1;
                }
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
    // dilate the mask/erode the inverse mask
    cv::Mat element = cv::getStructuringElement(MORPH_CROSS,
        cv::Size(2 * radius + 1, 2 * radius + 1), cv::Point(radius, radius));
    cv::dilate(_maskB, _maskB, element);
    cv::compare(_maskB, cv::Scalar(0x00), _invmaskB, CV_CMP_EQ);

    // update valid pixels
    //! \todo speedup
    cv::Mat maskSum;
    cv::integral(_invmaskB, maskSum, CV_32S);

    for (int y = 0; y < _validB.rows; y++) {
        for (int x = 0; x < _validB.cols; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            _validB.at<unsigned char>(y, x) = (count == 0x00) ? 0xff : 0x00;
        }
    }

    //! \todo mark as updated any matches adjacent to the boundary
}

cv::Mat drwnMaskedPatchMatch::visualize() const
{
    vector<cv::Mat> views(_imgA.type() == CV_8UC3 ? 5 : 3);

    // nnf (x and y)
    cv::split(_nnfA, &views[0]);
    views[0] = drwnCreateHeatMap(views[0], DRWN_COLORMAP_RAINBOW);
    views[1] = drwnCreateHeatMap(views[1], DRWN_COLORMAP_RAINBOW);

    // cost
    views[2] = _costsA.clone();
    drwnScaleToRange(views[2], 0.0, 1.0);
    views[2] = drwnCreateHeatMap(views[2], DRWN_COLORMAP_REDGREEN);

    if (views.size() == 5) {
        // masked source
        views[3] = _imgA.clone();
        drwnDrawRegionBoundaries(views[3], _maskA, CV_RGB(0, 255, 0), 2);

        // masked target
        views[4] = _imgB.clone();
        drwnShadeRegion(views[4], _invmaskB, CV_RGB(255, 0, 0), 1.0, DRWN_FILL_DIAG, 1);
        drwnDrawRegionBoundaries(views[4], _maskB, CV_RGB(255, 0, 0), 2);
    }

    return drwnCombineImages(views, 1);
}

bool drwnMaskedPatchMatch::update(const cv::Rect& roiA, const cv::Rect& roiB)
{
    if (_validB.at<unsigned char>(roiB.y, roiB.x) == 0x00)
        return false;

    const float cost = score(roiA, roiB);
    if (cost < _costsA.at<float>(roiA.y, roiA.x)) {
        _nnfA.at<cv::Vec2s>(roiA.y, roiA.x) = cv::Vec2s(roiB.x, roiB.y);
        _costsA.at<float>(roiA.y, roiA.x) = cost;
        _lastChanged.at<int>(roiA.y, roiA.x) = _iterationCount + 1;
        return true;
    }

    return false;
}

float drwnMaskedPatchMatch::score(const cv::Rect& roiA, const cv::Rect& roiB) const
{
#if 0
    //! \todo add no-self option
    if (roiA == roiB) return DRWN_FLT_MAX;
#endif

    // compute distance between patch features
    float cost;
    if (_overlapA.at<unsigned char>(roiA.y, roiA.x) == 0x00) {
        cost = cv::norm(_imgA(roiA), _imgB(roiB), DISTANCE_MEASURE);
    } else {
        cost = cv::norm(_imgA(roiA), _imgB(roiB), DISTANCE_MEASURE, _maskA(roiA));
    }

    // add height prior
    cost += HEIGHT_PENALTY * _patchSize.height * _patchSize.width *
        fabs((float)roiA.y / (float)(_imgA.rows - _patchSize.height) -
            (float)roiB.y / (float)(_imgB.rows - _patchSize.height));

    return cost;
}

void drwnMaskedPatchMatch::rescore(const cv::Rect& roi)
{
    DRWN_FCN_TIC;
    const cv::Rect region = affectedRegion(roi);
    for (int y = region.y; y < region.y + region.height; y++) {
        for (int x = region.x; x < region.x + region.width; x++) {
            const cv::Rect roiA(x, y, _patchSize.width, _patchSize.height);
            const cv::Rect roiB(getBestMatch(roiA.tl()));
            _costsA.at<float>(roiA.y, roiA.x) = score(roiA, roiB);
            _lastChanged.at<int>(roiA.y, roiA.x) = _iterationCount + 1;
        }
    }
    DRWN_FCN_TOC;
}

void drwnMaskedPatchMatch::cacheValidPixels()
{
    // find top-left of patches that overlap with invmaskA
    _overlapA = cv::Mat::zeros(_maskA.rows - _patchSize.height + 1, _maskA.cols - _patchSize.width + 1, CV_8UC1);

    cv::Mat maskSum;
    cv::integral(_invmaskA, maskSum, CV_32S);

    for (int y = 0; y < _overlapA.rows; y++) {
        for (int x = 0; x < _overlapA.cols; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(y, x) = 0xff;
            }
        }
    }

     // find top-left of patches that don't overlap with maskB
    _validB = cv::Mat::zeros(_maskB.rows - _patchSize.height + 1, _maskB.cols - _patchSize.width + 1, CV_8UC1);

    maskSum.release();
    cv::integral(_invmaskB, maskSum, CV_32S);

    for (int y = 0; y < _validB.rows; y++) {
        for (int x = 0; x < _validB.cols; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            if (count == 0x00) {
                _validB.at<unsigned char>(y, x) = 0xff;
            }
        }
    }
}

void drwnMaskedPatchMatch::updateValidPixels(const cv::Rect& roi)
{
    DRWN_FCN_TIC;
#if 1
    const cv::Rect region = affectedRegion(roi);
    cv::Rect iregion(region);
    iregion.width += _patchSize.width - 1;
    iregion.height += _patchSize.height - 1;

    cv::Mat maskSum;
    cv::integral(_invmaskA(iregion), maskSum, CV_32S);

    for (int y = 0; y < region.height; y++) {
        for (int x = 0; x < region.width; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(region.y + y, region.x + x) = 0xff;
            } else {
                _overlapA.at<unsigned char>(region.y + y, region.x + x) = 0x00;
            }
        }
    }
#else
    cv::Mat maskSum;
    cv::integral(_invmaskA, maskSum, CV_32S);

    for (int y = 0; y < _overlapA.rows; y++) {
        for (int x = 0; x < _overlapA.cols; x++) {
            const int count = maskSum.at<int>(y, x) +
                maskSum.at<int>(y + _patchSize.height, x + _patchSize.width) -
                maskSum.at<int>(y, x + _patchSize.width) -
                maskSum.at<int>(y + _patchSize.height, x);
            if (count != 0x00) {
                _overlapA.at<unsigned char>(y, x) = 0xff;
            }
        }
    }
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
