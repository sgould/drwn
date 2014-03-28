/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextonFilterBank.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <vector>

#include "cv.h"

#include "drwnBase.h"
#include "drwnTextonFilterBank.h"

// drwnTextonFilterBank -----------------------------------------------------

drwnTextonFilterBank::drwnTextonFilterBank(double k) :
    _kappa(k)
{
    // do nothing
}

drwnTextonFilterBank::~drwnTextonFilterBank()
{
    // do nothing
}

void drwnTextonFilterBank::filter(const cv::Mat& img, std::vector<cv::Mat>& response) const
{
    // check input
    DRWN_ASSERT(img.data != NULL);
    if (response.empty()) {
        response.resize(NUM_FILTERS);
    }
    DRWN_ASSERT((int)response.size() == NUM_FILTERS);
    DRWN_ASSERT((img.channels() == 3) && (img.depth() == CV_8U));
    for (int i = 0; i < NUM_FILTERS; i++) {
        if ((response[i].rows != img.rows) || (response[i].cols != img.cols)) {
            response[i] = cv::Mat(img.rows, img.cols, CV_32FC1);
	}
        DRWN_ASSERT((response[i].channels() == 1) && (response[i].depth() == CV_32F));
    }

    int k = 0;

    // color convert
    DRWN_LOG_DEBUG("Color converting image...");
    cv::Mat imgCIELab8U(img.rows, img.cols, CV_8UC3);
    cv::cvtColor(img, imgCIELab8U, CV_BGR2Lab);
    cv::Mat imgCIELab(img.rows, img.cols, CV_32FC3);
    imgCIELab8U.convertTo(imgCIELab, CV_32F, 1.0 / 255.0);

    cv::Mat greyImg(img.rows, img.cols, CV_32FC1);
    const int from_to[] = {0, 0};
    cv::mixChannels(&imgCIELab, 1, &greyImg, 1, from_to, 1);

    // gaussian filter on all color channels
    DRWN_LOG_DEBUG("Generating gaussian filter responses...");
    cv::Mat gImg32f(img.rows, img.cols, CV_32FC3);
    for (double sigma = 1.0; sigma <= 4.0; sigma *= 2.0) {
        const int h = 2 * (int)(_kappa * sigma) + 1;
        cv::GaussianBlur(imgCIELab, gImg32f, cv::Size(h, h), 0);
        cv::split(gImg32f, &response[k]);
        k += 3;
    }

    // derivatives of gaussians on just greyscale image
    DRWN_LOG_DEBUG("Generating derivative of gaussian filter responses...");
    for (double sigma = 2.0; sigma <= 4.0; sigma *= 2.0) {
        // x-direction
        cv::Sobel(greyImg, response[k++], CV_32F, 1, 0, 1);
        cv::GaussianBlur(response[k - 1], response[k - 1],
            cv::Size(2 * (int)(_kappa * sigma) + 1, 2 * (int)(3.0 * _kappa * sigma) + 1), 0);

        // y-direction
        cv::Sobel(greyImg, response[k++], CV_32F, 0, 1, 1);
        cv::GaussianBlur(response[k - 1], response[k - 1],
            cv::Size(2 * (int)(3.0 * _kappa * sigma) + 1, 2 * (int)(_kappa * sigma) + 1), 0);
    }

    // laplacian of gaussian on just greyscale image
    DRWN_LOG_DEBUG("Generating laplacian of gaussian filter responses...");
    cv::Mat tmpImg(img.rows, img.cols, CV_32FC1);
    for (double sigma = 1.0; sigma <= 8.0; sigma *= 2.0) {
        const int h = 2 * (int)(_kappa * sigma) + 1;
        cv::GaussianBlur(greyImg, tmpImg, cv::Size(h, h), 0);
        cv::Laplacian(tmpImg, response[k++], CV_32F, 3);
    }

    DRWN_ASSERT_MSG(k == NUM_FILTERS, k << " != " << NUM_FILTERS);
}
