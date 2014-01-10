/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLBPFilterBank.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <vector>

#include "cv.h"

#include "drwnBase.h"
#include "drwnLBPFilterBank.h"

// drwnLBPFilterBank --------------------------------------------------------

drwnLBPFilterBank::drwnLBPFilterBank(bool b8Neighbourbood) :
    _b8Neighbourhood(b8Neighbourbood)
{
    // do nothing
}

drwnLBPFilterBank::~drwnLBPFilterBank()
{
    // do nothing
}

void drwnLBPFilterBank::filter(const cv::Mat& img, std::vector<cv::Mat>& response)
{
    // check input
    DRWN_ASSERT(img.data != NULL);
    if (response.empty()) {
        response.resize(this->numFilters());
    }
    DRWN_ASSERT(response.size() == this->numFilters());

    if (img.channels() != 1) {
        cv::Mat tmp(img.rows, img.cols, img.depth());
        cv::cvtColor(img, tmp, CV_RGB2GRAY);
        return filter(tmp, response);
    }
    DRWN_ASSERT_MSG(img.depth() == CV_8U, "image must be 8-bit");

    // allocate output channels as 32-bit floating point
    for (unsigned i = 0; i < response.size(); i++) {
        if ((response[i].rows == img.rows) && (response[i].cols == img.cols) &&
            (response[i].depth() == CV_32F) && (response[i].channels() == 1)) {
            response[i].setTo(0.0f);
        } else {
            response[i] = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
        }
    }

    for (int y = 0; y < img.rows; y++) {

        const unsigned char *p = img.ptr<const unsigned char>(y);
        const unsigned char *p_prev = (y == 0) ? p : img.ptr<const unsigned char>(y - 1);
        const unsigned char *p_next = (y == img.rows - 1) ? p : img.ptr<const unsigned char>(y + 1);

        // 4-connected neighbourhood
        for (int x = 0; x < img.cols; x++) {
            if (p[x] > p_prev[x]) response[0].at<float>(y, x) = 1.0f;
            if ((x < img.cols - 1) && (p[x] > p[x + 1])) response[1].at<float>(y, x) = 1.0f;
            if (p[x] > p_next[x]) response[2].at<float>(y, x) = 1.0f;
            if ((x > 0) && (p[x] > p[x - 1])) response[3].at<float>(y, x) = 1.0f;
        }

        // 8-connected neighbourhood
        if (_b8Neighbourhood) {
            for (int x = 0; x < img.cols; x++) {
                if ((p[x] > p_prev[x]) && (x < img.cols - 1) && (p[x] > p[x + 1])) response[4].at<float>(y, x) = 1.0f;
                if ((x < img.cols - 1) && (p[x] > p[x + 1]) && (p[x] > p_next[x])) response[5].at<float>(y, x) = 1.0f;
                if ((p[x] > p_next[x]) && (x > 0) && (p[x] > p[x - 1])) response[6].at<float>(y, x) = 1.0f;
                if ((x > 0) && (p[x] > p[x - 1]) && (p[x] > p_prev[x])) response[7].at<float>(y, x) = 1.0f;
            }
        }
    }
}

void drwnLBPFilterBank::regionFeatures(const cv::Mat& regions,
    const std::vector<cv::Mat>& response, vector<vector<double> >& features)
{
    DRWN_ASSERT(!response.empty() && (response[0].rows == regions.rows) &&
        (response[0].cols == regions.cols) && (regions.type() == CV_32SC1));

    cv::Mat codewords = response[0].clone();
    float multiplier = 2.0;
    for (unsigned i = 1; i < response.size(); i++) {
        codewords += multiplier * response[i];
        multiplier *= 2.0f;
    }

    const int numRegions = cv::norm(regions, cv::NORM_INF) + 1;
    DRWN_ASSERT(numRegions > 0);

    // accumulate histograms
    features.clear();
    features.resize(numRegions, vector<double>((int)multiplier, 0.0));
    
    for (int y = 0; y < regions.rows; y++) {
        const int *segId = regions.ptr<int>(y);
        const float *p = codewords.ptr<const float>(y);
        for (int x = 0; x < regions.cols; x++) {
            if (segId[x] < 0) continue;
            features[segId[x]][(int)p[x]] += 1.0;
        }
    }
    
    // normalize histograms
    for (unsigned segId = 0; segId < features.size(); segId++) {
        double Z = 0.0;
        for (unsigned i = 0; i < features[segId].size(); i++) {
            Z += features[segId][i];
        }
        if (Z > 0.0) {
            for (unsigned i = 0; i < features[segId].size(); i++) {
                features[segId][i] /= Z;
            }
        }
    }
}

void drwnLBPFilterBank::regionFeatures(const drwnSuperpixelContainer& regions, 
    const std::vector<cv::Mat>& response, vector<vector<double> >& features)
{
    DRWN_ASSERT(!response.empty() && (response[0].rows == regions.height()) &&
        (response[0].cols == regions.width()));

    cv::Mat codewords = response[0].clone();
    float multiplier = 2.0;
    for (unsigned i = 1; i < response.size(); i++) {
        codewords += multiplier * response[i];
        multiplier *= 2.0f;
    }

    // accumulate histograms
    features.clear();
    features.resize(regions.size(), vector<double>((int)multiplier, 0.0));

    for (int c = 0; c < regions.channels(); c++) {
        for (int y = 0; y < regions.height(); y++) {
            const int *segId = regions[c].ptr<int>(y);
            const float *p = codewords.ptr<const float>(y);
            for (int x = 0; x < regions.width(); x++) {
                if (segId[x] < 0) continue;
                features[segId[x]][(int)p[x]] += 1.0;
            }
        }
    }
    
    // normalize histograms
    for (unsigned segId = 0; segId < features.size(); segId++) {
        double Z = 0.0;
        for (unsigned i = 0; i < features[segId].size(); i++) {
            Z += features[segId][i];
        }
        if (Z > 0.0) {
            for (unsigned i = 0; i < features[segId].size(); i++) {
                features[segId][i] /= Z;
            }
        }
    }
}
