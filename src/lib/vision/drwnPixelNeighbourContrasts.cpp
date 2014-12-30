/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelNeighbourContrasts.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <iostream>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnOpenCVUtils.h"
#include "drwnPixelNeighbourContrasts.h"

// drwnPixelNeighbourContrasts class ----------------------------------------

drwnPixelNeighbourContrasts::drwnPixelNeighbourContrasts()
{
    // do nothing
}

drwnPixelNeighbourContrasts::drwnPixelNeighbourContrasts(const cv::Mat& img)
{
    initialize(img);
}

drwnPixelNeighbourContrasts::~drwnPixelNeighbourContrasts()
{
    // do nothing
}

// i/o
void drwnPixelNeighbourContrasts::clear()
{
    _horzContrast = MatrixXd::Zero(0, 0);
    _vertContrast = MatrixXd::Zero(0, 0);
    _nwContrast = MatrixXd::Zero(0, 0);
    _swContrast = MatrixXd::Zero(0, 0);
}

void drwnPixelNeighbourContrasts::initialize(const cv::Mat& img)
{
    DRWN_ASSERT_MSG((img.channels() == 3) && (img.depth() == CV_8U), toString(img));
    DRWN_LOG_DEBUG("calculating pairwise contrast for " << toString(img) << "...");
    DRWN_FCN_TIC;

    // mean contrast
    double meanContrast = 0.0;
    for (int y = 0; y < img.rows - 1; y++) {
        for (int x = 0; x < img.cols - 1; x++) {
            meanContrast += pixelContrast(img, cv::Point(x, y), cv::Point(x + 1, y));
            meanContrast += pixelContrast(img, cv::Point(x, y), cv::Point(x, y + 1));
        }
    }
    meanContrast /= 2.0 * (double)((img.rows - 1) * (img.cols - 1));

    // cache pixel contrast
    _horzContrast = MatrixXd::Zero(img.cols, img.rows);
    _vertContrast = MatrixXd::Zero(img.cols, img.rows);
    _nwContrast = MatrixXd::Zero(img.cols, img.rows);
    _swContrast = MatrixXd::Zero(img.cols, img.rows);

    const double beta = -0.5 / meanContrast;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            // west (left)
            if (x > 0) {
                _horzContrast(x, y) = exp(beta *
                    pixelContrast(img, cv::Point(x, y), cv::Point(x - 1, y)));
            }
            // north (top)
            if (y > 0) {
                _vertContrast(x, y) = exp(beta *
                    pixelContrast(img, cv::Point(x, y), cv::Point(x, y - 1)));
            }
            // north-west (top-left)
            if ((x > 0) && (y > 0)) {
                _nwContrast(x, y) = M_SQRT1_2 * exp(beta *
                    pixelContrast(img, cv::Point(x, y), cv::Point(x - 1, y - 1)));
            }
            // south-west (bottom-left)
            if ((x > 0) && (y < height() - 1)) {
                _swContrast(x, y) = M_SQRT1_2 * exp(beta *
                    pixelContrast(img, cv::Point(x, y), cv::Point(x - 1, y + 1)));
            }
        }
    }

    DRWN_FCN_TOC;
}

bool drwnPixelNeighbourContrasts::save(drwnXMLNode& xml) const
{
    // serialize matrices
    MatrixXd m(4 * _horzContrast.rows(), _horzContrast.cols());
    m << _horzContrast, _vertContrast, _nwContrast, _swContrast;
    drwnXMLUtils::serialize(xml, m);

    return true;
}

bool drwnPixelNeighbourContrasts::load(drwnXMLNode& xml)
{
    // de-serialize matrices
    MatrixXd m;
    drwnXMLUtils::deserialize(xml, m);
    DRWN_ASSERT(m.rows() % 4 == 0);

    int h = m.rows() / 4;
    _horzContrast = m.block(0, 0, h, m.cols());
    _vertContrast = m.block(h, 0, h, m.cols());
    _nwContrast = m.block(2 * h, 0, h, m.cols());
    _swContrast = m.block(3 * h, 0, h, m.cols());

    return true;
}

cv::Mat drwnPixelNeighbourContrasts::visualize(bool bComposite) const
{
    vector<cv::Mat> views(9);
    for (unsigned i = 0; i < views.size(); i++) {
        views[i] = cv::Mat::zeros(height(), width(), CV_32FC1);
    }

    for (int y = 0; y < height(); y++) {
        for (int x = 0; x < width(); x++) {
            views[0].at<float>(y, x) = contrastNW(x, y);
            views[1].at<float>(y, x) = contrastN(x, y);
            views[2].at<float>(y, x) = contrastNW(x, y);
            views[3].at<float>(y, x) = contrastW(x, y);
            views[5].at<float>(y, x) = contrastE(x, y);
            views[6].at<float>(y, x) = contrastSW(x, y);
            views[7].at<float>(y, x) = contrastS(x, y);
            views[8].at<float>(y, x) = contrastSE(x, y);
        }
    }

    cv::add(views[0], views[4], views[4]);
    cv::add(views[1], views[4], views[4]);
    cv::add(views[2], views[4], views[4]);
    cv::add(views[3], views[4], views[4]);
    cv::add(views[5], views[4], views[4]);
    cv::add(views[6], views[4], views[4]);
    cv::add(views[7], views[4], views[4]);
    cv::add(views[8], views[4], views[4]);

    for (unsigned i = 0; i < views.size(); i++) {
        drwnScaleToRange(views[i], 0.0, 1.0);
    }

    cv::Mat m;
    if (bComposite) {
        m = views[4];
    } else {
        m = drwnCombineImages(views, 3, 3);
    }

    cv::Mat canvas(m.rows, m.cols, CV_8UC1);
    m.convertTo(canvas, CV_8UC1, 255.0);

    return canvas;
}


double drwnPixelNeighbourContrasts::pixelContrast(const cv::Mat& img, const cv::Point &p, const cv::Point& q)
{
    const cv::Vec3b a = img.at<cv::Vec3b>(p.y, p.x);
    const cv::Vec3b b = img.at<cv::Vec3b>(q.y, q.x);
    return (double)((int(a[0]) - int(b[0])) * (int(a[0]) - int(b[0])) +
        (int(a[1]) - int(b[1])) * (int(a[1]) - int(b[1])) +
        (int(a[2]) - int(b[2])) * (int(a[2]) - int(b[2])));
}
