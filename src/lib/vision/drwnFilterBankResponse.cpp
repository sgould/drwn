/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFilterBankResponse.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnFilterBankResponse.h"
#include "drwnOpenCVUtils.h"

using namespace std;
using namespace Eigen;

// drwnFilterBankResponse ------------------------------------------------------

drwnFilterBankResponse::drwnFilterBankResponse()
{
    // do nothing
}

drwnFilterBankResponse::drwnFilterBankResponse(const drwnFilterBankResponse& f)
{
    // deep copy
    _responses.resize(f._responses.size());
    _sum.resize(f._sum.size());
    _sqSum.resize(f._sqSum.size());
    for (unsigned i = 0; i < f._responses.size(); i++) {
        _responses[i] = f._responses[i].clone();
        _sum[i] = f._sum[i].clone();
        _sqSum[i] = f._sqSum[i].clone();
    }
}

drwnFilterBankResponse::~drwnFilterBankResponse()
{
    // do nothing
}

void drwnFilterBankResponse::clear()
{
    _responses.clear();
    _sum.clear();
    _sqSum.clear();
}

void drwnFilterBankResponse::addResponseImage(cv::Mat& r)
{
    DRWN_ASSERT(r.depth() == CV_32F);
    DRWN_ASSERT(empty() || ((r.cols == width()) && (r.rows == height())));

    // add response image
    _responses.push_back(r);

    // allocate integral response images
    _sum.push_back(cv::Mat(r.rows + 1, r.cols + 1, CV_64FC1));
    _sqSum.push_back(cv::Mat(r.rows + 1, r.cols + 1, CV_64FC1));

    cv::integral(_responses.back(), _sum.back(), _sqSum.back());
}

void drwnFilterBankResponse::addResponseImages(vector<cv::Mat>& r)
{
    for (unsigned i = 0; i < r.size(); i++) {
        addResponseImage(r[i]);
    }
}

void drwnFilterBankResponse::copyResponseImage(const cv::Mat& r)
{
    cv::Mat c = r.clone();
    addResponseImage(c);
}

void drwnFilterBankResponse::copyResponseImages(const vector<cv::Mat>& r)
{
    for (unsigned i = 0; i < r.size(); i++) {
        copyResponseImage(r[i]);
    }
}

const cv::Mat& drwnFilterBankResponse::getResponseImage(int i) const
{
    DRWN_ASSERT((i >= 0) && (i < size()));
    return _responses[i];
}

void drwnFilterBankResponse::deleteResponseImage(int i)
{
    DRWN_ASSERT((i >= 0) && (i < size()));

    _responses.erase(_responses.begin() + i);
    _sum.erase(_sum.begin() + i);
    _sqSum.erase(_sqSum.begin() + i);
}

// transformations
void drwnFilterBankResponse::exponentiateResponses(double alpha)
{
    for (int i = 0; i < size(); i++) {
        for (int y = 0; y < height(); y++) {
            float *p = _responses[i].ptr<float>(y);
            for (int x = 0; x < width(); x++) {
                p[x] = (float)exp(alpha * p[x]);
            }
        }
        cv::integral(_responses[i], _sum[i], _sqSum[i]);
    }
}

void drwnFilterBankResponse::normalizeResponses()
{
    if (empty()) return;

    vector<float *> p(size());
    for (int y = 0; y < height(); y++) {
        for (int i = 0; i < size(); i++) {
            p[i] = _responses[i].ptr<float>(y);
        }
        for (int x = 0; x < width(); x++) {
            float z = 0.0;
            for (int i = 0; i < size(); i++) {
                z += fabs(p[i][x]);
            }
            if (z > 0.0) {
                for (int i = 0; i < size(); i++) {
                    p[i][x] /= z;
                }
            }
        }
    }

    for (int i = 0; i < size(); i++) {
        cv::integral(_responses[i], _sum[i], _sqSum[i]);
    }
}

void drwnFilterBankResponse::expAndNormalizeResponses(double alpha)
{
    if (empty()) return;

    vector<float *> p(size());
    for (int y = 0; y < height(); y++) {
        for (int i = 0; i < size(); i++) {
            p[i] = _responses[i].ptr<float>(y);
        }

        for (int x = 0; x < width(); x++) {
            float maxValue = p[0][x];
            if (alpha > 0.0) {
                for (int i = 1; i < size(); i++) {
                    maxValue = std::max(maxValue, p[i][x]);
                }
            } else {
                for (int i = 1; i < size(); i++) {
                    maxValue = std::min(maxValue, p[i][x]);
                }
            }

            float z = 0.0;
            for (int i = 0; i < size(); i++) {
                p[i][x] = exp(alpha * (p[i][x] - maxValue));
                z += p[i][x];
            }
            for (int i = 0; i < size(); i++) {
                p[i][x] /= z;
            }
        }
    }

    for (int i = 0; i < size(); i++) {
        cv::integral(_responses[i], _sum[i], _sqSum[i]);
    }
}

// pixel and region features --- no index checking
VectorXd drwnFilterBankResponse::value(int x, int y) const
{
    //DRWN_ASSERT((x >= 0) && (x < width()) && (y >= 0) && (y < height()));
    VectorXd v(size());

    for (int i = 0; i < size(); i++) {
        v[i] = (double)_responses[i].at<float>(y, x);
    }

    return v;
}

VectorXd drwnFilterBankResponse::mean(int x, int y, int w, int h) const
{
    VectorXd m(size());

    for (int i = 0; i < size(); i++) {
        m[i] = _sum[i].at<double>(y, x) +
            _sum[i].at<double>(y + h, x + w) -
            _sum[i].at<double>(y + h, x) -
            _sum[i].at<double>(y, x + w);
    }

    m /= (double)(w * h);
    return m;
}

VectorXd drwnFilterBankResponse::energy(int x, int y, int w, int h) const
{
    VectorXd e(size());

    for (int i = 0; i < size(); i++) {
        e[i] = _sqSum[i].at<double>(y, x) +
            _sqSum[i].at<double>(y + h, x + w) -
            _sqSum[i].at<double>(y + h, x) -
            _sqSum[i].at<double>(y, x + w);
    }

    return e;
}

VectorXd drwnFilterBankResponse::variance(int x, int y, int w, int h) const
{
    return (energy(x, y, w, h).array() / (double)(w * h) - mean(x, y, w, h).array().square()).max(0.0);
}

VectorXd drwnFilterBankResponse::mean(const list<cv::Point>& pixels) const
{
    VectorXd m = VectorXd::Zero(size());

    if (pixels.empty()) {
        return m;
    }

    for (list<cv::Point>::const_iterator ip = pixels.begin(); ip != pixels.end(); ip++) {
        m += value(ip->x, ip->y);
    }

    m /= (double)pixels.size();
    return m;
}

VectorXd drwnFilterBankResponse::energy(const list<cv::Point>& pixels) const
{
    VectorXd e = VectorXd::Zero(size());

    if (pixels.empty()) {
        return e;
    }

    for (list<cv::Point>::const_iterator ip = pixels.begin(); ip != pixels.end(); ip++) {
        e.array() += value(ip->x, ip->y).array().square();
    }

    return e;
}

VectorXd drwnFilterBankResponse::variance(const list<cv::Point>& pixels) const
{
    return (energy(pixels).array() / (double)pixels.size() - mean(pixels).array().square()).max(0.0);
}

VectorXd drwnFilterBankResponse::mean(const cv::Mat& mask) const
{
    DRWN_ASSERT(mask.depth() == CV_8UC1);
    DRWN_ASSERT((mask.rows == height()) && (mask.cols == width()));

    VectorXd m = VectorXd::Zero(size());

    int count = 0;
    for (int y = 0; y < height(); y++) {
        const unsigned char *p = mask.ptr<const unsigned char>(y);
        int xStart = 0;
        for (int x = 0; x < width(); x++, p++) {
            if (*p == 0) {
                if (xStart != x) {
                    if (xStart == x - 1) {
                        // singleton case
                        m += value(x - 1, y);
                    } else {
                        // line case
                        for (int i = 0; i < size(); i++) {
                            const double *q = _sum[i].ptr<const double>(y);
                            m[i] +=  q[xStart] + q[x + width() + 1] - q[xStart + width() + 1] - q[x];
                        }
                    }
                }
                count += x - xStart;
                xStart = x + 1;
            }
        }

        // special case of region ends at boundary
        if (xStart != width()) {
            if (xStart == width() - 1) {
                // singleton case
                m += value(width() - 1, y);
            } else {
                // line case
                for (int i = 0; i < size(); i++) {
                    const double *q = _sum[i].ptr<const double>(y);
                    m[i] +=  q[xStart] + q[2 * width() + 1] - q[xStart + width() + 1] - q[width()];
                }
            }
            count += width() - xStart;
        }
    }

    // normalize
    if (count != 0) {
        m /= (double)count;
    }

    return m;
}

VectorXd drwnFilterBankResponse::energy(const cv::Mat& mask) const
{
    DRWN_ASSERT(mask.depth() == CV_8UC1);
    DRWN_ASSERT((mask.rows == height()) && (mask.cols == width()));

    VectorXd e = VectorXd::Zero(size());

    int count = 0;
    for (int y = 0; y < height(); y++) {
        const unsigned char *p = mask.ptr<const unsigned char>(y);
        int xStart = 0;
        for (int x = 0; x < width(); x++, p++) {
            if (*p == 0) {
                if (xStart != x) {
                    if (xStart == x - 1) {
                        // singleton case
                        e.array() += value(x - 1, y).array().square();
                    } else {
                        // line case
                        for (int i = 0; i < size(); i++) {
                            const double *q = _sqSum[i].ptr<const double>(y);
                            e[i] +=  q[xStart] + q[x + width() + 1] - q[xStart + width() + 1] - q[x];
                        }
                    }
                }
                count += x - xStart;
                xStart = x + 1;
            }
        }

        // special case of region ends at boundary
        if (xStart != width()) {
            if (xStart == width() - 1) {
                // singleton case
                e.array() += value(width() - 1, y).array().square();
            } else {
                // line case
                for (int i = 0; i < size(); i++) {
                    const double *q = _sqSum[i].ptr<const double>(y);
                    e[i] +=  q[xStart] + q[2 * width() + 1] - q[xStart + width() + 1] - q[width()];
                }
            }
            count += width() - xStart;
        }
    }

    return e;
}

VectorXd drwnFilterBankResponse::variance(const cv::Mat& mask) const
{
    DRWN_ASSERT(mask.depth() == CV_8UC1);
    DRWN_ASSERT((mask.rows == height()) && (mask.cols == width()));

    VectorXd m = VectorXd::Zero(size());
    VectorXd e = VectorXd::Zero(size());

    int count = 0;
    for (int y = 0; y < height(); y++) {
        const unsigned char *p = mask.ptr<const unsigned char>(y);
        int xStart = 0;
        for (int x = 0; x < width(); x++, p++) {
            if (*p == 0) {
                if (xStart != x) {
                    if (xStart == x - 1) {
                        // singleton case
                        m += value(x - 1, y);
                        e.array() += value(x - 1, y).array().square();
                    } else {
                        // line case
                        for (int i = 0; i < size(); i++) {
                            const double *q = _sum[i].ptr<const double>(y);
                            m[i] +=  q[xStart] + q[x + width() + 1] - q[xStart + width() + 1] - q[x];
                            q = _sqSum[i].ptr<const double>(y);
                            e[i] +=  q[xStart] + q[x + width() + 1] - q[xStart + width() + 1] - q[x];
                        }
                    }
                }
                count += x - xStart;
                xStart = x + 1;
            }
        }

        // special case of region ends at boundary
        if (xStart != width()) {
            if (xStart == width() - 1) {
                // singleton case
                m += value(width() - 1, y);
                e.array() += value(width() - 1, y).array().square();
            } else {
                // line case
                for (int i = 0; i < size(); i++) {
                    const double *q = _sum[i].ptr<const double>(y);
                    m[i] +=  q[xStart] + q[2 * width() + 1] - q[xStart + width() + 1] - q[width()];
                    q = _sqSum[i].ptr<const double>(y);
                    e[i] +=  q[xStart] + q[2 * width() + 1] - q[xStart + width() + 1] - q[width()];
                }
            }
            count += width() - xStart;
        }
    }

    if (count == 0) {
        return VectorXd::Zero(size());
    }

    return (e.array()/(double)count - (m / (double)count).array().square()).max(0.0);
}

cv::Mat drwnFilterBankResponse::visualize() const
{
    return drwnCombineImages(_responses);
}
