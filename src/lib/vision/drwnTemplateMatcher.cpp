/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTemplateMatcher.cpp
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
#include "drwnTemplateMatcher.h"
#include "drwnOpenCVUtils.h"

using namespace std;
using namespace Eigen;

// drwnTemplateMatcher ----------------------------------------------------------

drwnTemplateMatcher::drwnTemplateMatcher() :
    _largestTemplate(cv::Size(0, 0)), _imgSize(cv::Size(0, 0))
{
    // do nothing
}

drwnTemplateMatcher::drwnTemplateMatcher(const drwnTemplateMatcher& tm) :
    _templateStats(tm._templateStats), _largestTemplate(tm._largestTemplate), _imgSize(cv::Size(0, 0))
{
    _templates.resize(tm._templates.size());
    for (unsigned i = 0; i < tm._templates.size(); i++) {
        _templates[i] = tm._templates[i].clone();
    }
}

drwnTemplateMatcher::~drwnTemplateMatcher()
{
    clear();
}

void drwnTemplateMatcher::clear()
{
    // clear buffered memory
    reset();

    // clear templates
    _templates.clear();
    _templateStats.clear();
    _largestTemplate = cv::Size(0, 0);
}

void drwnTemplateMatcher::reset()
{
    _dftTemplates.clear();
    _imgSize = cvSize(0, 0);
    _dftImage = cv::Mat();
    _dftResponse = cv::Mat();
    _imgSum = cv::Mat();
    _imgSqSum = cv::Mat();
}

// template definitions
void drwnTemplateMatcher::addTemplate(cv::Mat& t)
{
    DRWN_ASSERT(t.data != NULL);
    reset();

    _templates.push_back(t);

    double mu = cv::mean(t)[0];
    double energy = cv::norm(t, cv::NORM_L2);
    _templateStats.push_back(make_pair(mu, energy));

    _largestTemplate.width = std::max(_largestTemplate.width, t.cols);
    _largestTemplate.height = std::max(_largestTemplate.height, t.rows);
}

void drwnTemplateMatcher::addTemplates(vector<cv::Mat>& t)
{
    for (vector<cv::Mat>::iterator it = t.begin(); it != t.end(); it++) {
        this->addTemplate(*it);
    }
}

void drwnTemplateMatcher::copyTemplate(const cv::Mat& t)
{
    cv::Mat m = t.clone();
    this->addTemplate(m);
}

void drwnTemplateMatcher::copyTemplates(const vector<cv::Mat>& t)
{
    for (vector<cv::Mat>::const_iterator it = t.begin(); it != t.end(); it++) {
        this->copyTemplate(*it);
    }
}

// template responses
vector<cv::Mat> drwnTemplateMatcher::responses(const cv::Mat& img, int method)
{
    DRWN_ASSERT(img.data != NULL);
    cacheDFTAndIntegrals(img);
    return this->responses(method);
}

vector<cv::Mat> drwnTemplateMatcher::responses(int method)
{
    DRWN_ASSERT(_dftImage.data != NULL);

    vector<cv::Mat> r(_templates.size());
    for (unsigned i = 0; i < _templates.size(); i++) {
        r[i] = this->response(i, method);
    }

    return r;
}

cv::Mat drwnTemplateMatcher::response(const cv::Mat& img, unsigned tid, int method)
{
    DRWN_ASSERT(img.data != NULL);
    cacheDFTAndIntegrals(img);
    return this->response(tid, method);
}

cv::Mat drwnTemplateMatcher::response(unsigned tid, int method)
{
    DRWN_ASSERT((tid < _templates.size()) && (_dftImage.data != NULL));
    DRWN_FCN_TIC;

    cv::Mat results(_imgSize.height, _imgSize.width, CV_32FC1);
    cv::Mat subResults = results(cv::Rect(0, 0, 
            _imgSize.width - _templates[tid].cols + 1,
            _imgSize.height - _templates[tid].rows + 1));

    // zero pad
    results(cv::Rect(_imgSize.width - _templates[tid].cols + 1, 0,
            _templates[tid].cols - 1, _imgSize.height)) = 0.0f;
    results(cv::Rect(0, _imgSize.height - _templates[tid].rows + 1,
            _imgSize.width, _templates[tid].rows - 1)) = 0.0f;

    // cross-correlation
    cv::mulSpectrums(_dftImage, _dftTemplates[tid], _dftResponse, 0, true);

    // inverse DFT
    cv::dft(_dftResponse, _dftResponse, CV_DXT_INV_SCALE, _imgSize.height - _templates[tid].rows + 1);
    _dftResponse(cv::Rect(0, 0, _imgSize.width - _templates[tid].cols + 1,
            _imgSize.height - _templates[tid].rows + 1)).copyTo(subResults);

    // return if CV_TM_CCORR
    if (method == CV_TM_CCORR) {
        DRWN_FCN_TOC;
        return results;
    }

    // post-processing for non-CV_TM_CCORR methods
    double tmplEnergy = _templateStats[tid].second;
    double tmplMean = _templateStats[tid].first;
    double tmplEnergy2 = tmplEnergy * tmplEnergy;
    if (method == CV_TM_CCOEFF_NORMED) {
        tmplEnergy = sqrt(std::max(tmplEnergy2 -
                tmplMean * tmplMean * (double)(_templates[tid].rows * _templates[tid].cols), 0.0));
        tmplEnergy2 = tmplEnergy * tmplEnergy;
    }

    const double *p = _imgSum.ptr<const double>(0);
    const double *q = _imgSum.ptr<const double>(_templates[tid].rows);
    const double *p2 = _imgSqSum.ptr<const double>(0);
    const double *q2 = _imgSqSum.ptr<const double>(_templates[tid].rows);
    float *r = results.ptr<float>(0);
    for (int y = 0; y < _imgSize.height - _templates[tid].rows + 1; y++) {
        for (int x = 0; x < _imgSize.width - _templates[tid].cols + 1; x++) {
            double imgEnergy2 = p2[x] - p2[x + _templates[tid].cols] + q2[x + _templates[tid].cols] - q2[x];
            double imgEnergy = sqrt(imgEnergy2);
            double imgSum = p[x] - p[x + _templates[tid].cols] + q[x + _templates[tid].cols] - q[x];

            switch (method) {
            case CV_TM_SQDIFF:
                r[x] = imgEnergy2 + tmplEnergy2 - 2.0 * r[x];
                break;
            case CV_TM_SQDIFF_NORMED:
                r[x] = imgEnergy2 + tmplEnergy2 - 2.0 * r[x];
                if (fabs(r[x]) < imgEnergy * tmplEnergy) {
                    r[x] /= imgEnergy * tmplEnergy;
                } else if (fabs(r[x]) < 1.125 * imgEnergy * tmplEnergy) {
                    r[x] = (r[x] > 0) ? 1.0f : -1.0f;
                } else {
                    r[x] = 1.0f;
                }
                break;
            case CV_TM_CCORR_NORMED:
                r[x] /= imgEnergy * tmplEnergy;
                break;
            case CV_TM_CCOEFF:
                r[x] -= tmplMean * imgSum;
                break;
            case CV_TM_CCOEFF_NORMED:
                r[x] -= tmplMean * imgSum;
                imgEnergy = sqrt(std::max(imgEnergy2 -
                        imgSum * imgSum / (double)(_templates[tid].rows * _templates[tid].cols), 0.0));
                if (fabs(r[x]) < imgEnergy * tmplEnergy) {
                    r[x] /= imgEnergy * tmplEnergy;
                } else if (fabs(r[x]) < 1.125 * imgEnergy * tmplEnergy) {
                    r[x] = (r[x] > 0) ? 1.0f : -1.0f;
                } else {
                    r[x] = 0.0f;
                }
                break;
            default:
                DRWN_LOG_FATAL("unrecognized method " << method);
            }
        }

        p += _imgSum.cols;
        q += _imgSum.cols;
        p2 += _imgSqSum.cols;
        q2 += _imgSqSum.cols;
        r += results.cols;
    }

    DRWN_FCN_TOC;
    return results;
}

// operators
drwnTemplateMatcher& drwnTemplateMatcher::operator=(const drwnTemplateMatcher& tm)
{
    if (&tm == this) {
        return *this;
    }

    clear();
    _templates.resize(tm._templates.size());
    for (unsigned i = 0; i < tm._templates.size(); i++) {
        _templates[i] = tm._templates[i].clone();
    }
    _templateStats = tm._templateStats;
    _largestTemplate = tm._largestTemplate;

    return *this;
}

const cv::Mat& drwnTemplateMatcher::operator[](unsigned i) const
{
    DRWN_ASSERT(i < _templates.size());
    return _templates[i];
}

void drwnTemplateMatcher::cacheDFTAndIntegrals(const cv::Mat& image)
{
    // reset of image size doesn't match previous size
    if ((image.cols != _imgSize.width) || (image.rows != _imgSize.height)) {
        DRWN_LOG_DEBUG("resetting drwnTemplateMatcher cache to " <<
            image.cols << "-by-" << image.rows);
        reset();
        _imgSize = image.size();
    }

    // get optimal DFT size
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(_imgSize.width + _largestTemplate.width - 1);
    dftSize.height = cv::getOptimalDFTSize(_imgSize.height + _largestTemplate.height - 1);

    // cache DFTs of templates
    if (_dftTemplates.empty()) {
        _dftTemplates.resize(_templates.size());
        for (unsigned i = 0; i < _templates.size(); i++) {
            cv::Mat dft(dftSize.height, dftSize.width, CV_32FC1);
            if (_templates[i].depth() == CV_32F) {
                _templates[i].copyTo(dft(cv::Rect(0, 0, _templates[i].cols, _templates[i].rows)));
            } else {
                _templates[i].convertTo(dft(cv::Rect(0, 0, _templates[i].cols, _templates[i].rows)), CV_32F);
            }
            dft(cv::Rect(_templates[i].cols, 0, dftSize.width - _templates[i].cols, _templates[i].rows)) = 0.0f;
            cv::dft(dft, dft, CV_DXT_FORWARD, _templates[i].rows);
            _dftTemplates[i] = dft;
        }
    }

#if 0
    for (unsigned i = 0; i < _templates.size(); i++) {
        for (int y = 0; y < _templates[i].rows; y++) {
            for (int x = 0; x < _templates[i].cols; x++) {
                cout << " " << _templates[i].at<float>(y, x);
            }
            cout << "\n";
        }
    }
#endif

    // cache DFT of image
    if (_dftImage.data == NULL) {
        _dftImage = cv::Mat(dftSize.height, dftSize.width, CV_32FC1);
    }

    image.convertTo(_dftImage(cv::Rect(0, 0, image.cols, image.rows)), CV_32F);
    _dftImage(cv::Rect(image.cols, 0, dftSize.width - image.cols, image.rows)) = 0.0f;
    cv::dft(_dftImage, _dftImage, CV_DXT_FORWARD, image.rows);

#if 0
    for (int y = 0; y < _dftImage.rows; y++) {
        for (int x = 0; x < _dftImage.cols; x++) {
            cout << " " << _dftImage.at<float>(y, x);
        }
        cout << "\n";
    }
#endif

    // create response buffer
    if (_dftResponse.data == NULL) {
        _dftResponse = cv::Mat(dftSize.height, dftSize.width, CV_32FC1);
    }

    // compute integral images
    if (_imgSum.data == NULL) {
        _imgSum = cv::Mat(image.rows + 1, image.cols + 1, CV_64FC1);
        _imgSqSum = cv::Mat(image.rows + 1, image.cols + 1, CV_64FC1);
    }
    cv::integral(image, _imgSum, _imgSqSum);
}
