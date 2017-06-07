/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImageInstance.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;

// drwnSegImageInstance class ---------------------------------------------

drwnSegImageInstance::drwnSegImageInstance(const char *imgFilename, const char *baseName)
{
    _img = cv::imread(string(imgFilename), CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(_img.data != NULL, imgFilename);
    if (baseName != NULL) {
        _baseName = string(baseName);
    } else {
        _baseName = drwn::strBaseName(imgFilename);
    }
    initInstance();
}

drwnSegImageInstance::drwnSegImageInstance(const cv::Mat& img, const char *baseName)
{
    DRWN_ASSERT(img.data != NULL);
    _img = img.clone();
    if (baseName != NULL)
        _baseName = string(baseName);
    initInstance();
}

drwnSegImageInstance::drwnSegImageInstance(const drwnSegImageInstance& instance) :
    _baseName(instance._baseName), _img(instance._img.clone()),
    _grayImg(instance._grayImg.clone()), _softEdgeImg(instance._softEdgeImg.clone()),
    unaries(instance.unaries), contrast(instance.contrast), auxEdges(instance.auxEdges),
    superpixels(instance.superpixels), auxiliaryData(instance.auxiliaryData),
    pixelLabels(instance.pixelLabels)
{
    // do nothing
}

drwnSegImageInstance::~drwnSegImageInstance()
{
    // do nothing
}

//
// image features
//
void drwnSegImageInstance::clearPixelFeatures()
{
    unaries.clear();
    unaries.resize(this->size());
}

void drwnSegImageInstance::appendPixelFeatures(drwnSegImagePixelFeatures &featureGenerator)
{
    featureGenerator.cacheInstanceData(*this);
    featureGenerator.appendAllPixelFeatures(unaries);
    featureGenerator.clearInstanceData();
}

drwnSegImageInstance& drwnSegImageInstance::operator=(const drwnSegImageInstance& instance)
{
    if (&instance == this) {
        return *this;
    }

    _baseName = instance._baseName;
    unaries = instance.unaries;
    contrast = instance.contrast;
    auxEdges = instance.auxEdges;
    superpixels = instance.superpixels;
    auxiliaryData = instance.auxiliaryData;
    pixelLabels = instance.pixelLabels;

    _img = instance._img.clone();
    _grayImg = instance._grayImg.clone();
    _softEdgeImg = instance._softEdgeImg.clone();

    return *this;
}

// protected member functions --------------------------------------------------------

void drwnSegImageInstance::initInstance()
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(_img.data != NULL);
    DRWN_LOG_DEBUG("initializing instance " << _baseName);

    // convert image to greyscale
    _grayImg = cv::Mat(_img.rows, _img.cols, CV_8UC1);
    cv::cvtColor(_img, _grayImg, CV_BGR2GRAY);

    // create edgemap
    _softEdgeImg = cv::Mat(_img.rows, _img.cols, CV_32FC1);

    // horizontal and vertical edge filters
    cv::Mat hEdgeBuffer(_img.rows, _img.cols, CV_32FC1);
    cv::Mat vEdgeBuffer(_img.rows, _img.cols, CV_32FC1);
    cv::Sobel(_grayImg, hEdgeBuffer, CV_32F, 1, 0, 3);
    cv::Sobel(_grayImg, vEdgeBuffer, CV_32F, 0, 1, 3);

    // combine and rescale
    for (int y = 0; y < _softEdgeImg.rows; y++) {
        const float *pGx = hEdgeBuffer.ptr<const float>(y);
        const float *pGy = vEdgeBuffer.ptr<const float>(y);
        float *p = _softEdgeImg.ptr<float>(y);
        for (int x = 0; x < _softEdgeImg.cols; x++) {
            p[x] = M_SQRT1_2 * sqrt(pGx[x] * pGx[x] + pGy[x] * pGy[x]);
        }
    }

    // initialize random variables
    pixelLabels = MatrixXi::Constant(height(), width(), -1);

    // cache pixel contrast
    contrast.initialize(_img);

    DRWN_FCN_TOC;
}
