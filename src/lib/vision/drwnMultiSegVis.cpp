/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiSegVis.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"

#include "drwnOpenCVUtils.h"
#include "drwnMultiSegVis.h"

using namespace std;

// visualization rountines
cv::Mat drwnMultiSegVis::visualizeInstance(const drwnSegImageInstance &instance)
{
    cv::Mat canvas(instance.height(), 3 * instance.width(), CV_8UC3);

    instance.image().copyTo(canvas(cv::Rect(0, 0, instance.width(), instance.height())));

    cv::Mat tmp(instance.image().clone());
    visualizePixelLabels(instance, tmp, 1.0);
    tmp.copyTo(canvas(cv::Rect(instance.width(), 0, instance.width(), instance.height())));

    tmp = drwnGreyImage(instance.image());
    drwnColorImageInplace(tmp);
    visualizePixelLabels(instance, tmp, 0.7);
    tmp.copyTo(canvas(cv::Rect(2 * instance.width(), 0, instance.width(), instance.height())));

    return canvas;
}

void drwnMultiSegVis::visualizePixelLabels(const drwnSegImageInstance &instance, cv::Mat& canvas, double alpha)
{
    DRWN_ASSERT((canvas.data != NULL) && (canvas.channels() == 3) && (canvas.depth() == CV_8U));

    // copy pixel assignments
    cv::Mat pixels(instance.height(), instance.width(), CV_32SC1);
    for (int y = 0; y < pixels.rows; y++) {
        int *p = pixels.ptr<int>(y);
        for (int x = 0; x < pixels.cols; x++) {
            p[x] = instance.pixelLabels(y, x);
        }
    }

    // resize pixel assignments
    drwnResizeInPlace(pixels, canvas.rows, canvas.cols, CV_INTER_NN);

    // overlay regions
    for (int y = 0; y < canvas.rows; y++) {
        const int *q = pixels.ptr<const int>(y);
        unsigned char *p = canvas.ptr<unsigned char>(y);
        for (int x = 0; x < canvas.cols; x++, q++) {
            const unsigned int c = gMultiSegRegionDefs.color(*q);
            unsigned char red = gMultiSegRegionDefs.red(c);
            unsigned char green = gMultiSegRegionDefs.green(c);
            unsigned char blue = gMultiSegRegionDefs.blue(c);

            p[3 * x + 2] = (unsigned char)((1.0 - alpha) * p[3 * x + 2] + alpha * red);
            p[3 * x + 1] = (unsigned char)((1.0 - alpha) * p[3 * x + 1] + alpha * green);
            p[3 * x + 0] = (unsigned char)((1.0 - alpha) * p[3 * x + 0] + alpha * blue);
        }
    }
}

cv::Mat drwnMultiSegVis::visualizePixelFeatures(const drwnSegImageInstance &instance)
{
    DRWN_ASSERT((int)instance.unaries.size() == instance.size());

    const int nFeatures = (int)instance.unaries[0].size();
    vector<cv::Mat> views(nFeatures);

    // create one image per feature
    cv::Mat m(instance.height(), instance.width(), CV_32FC1);
    float *p = m.ptr<float>(0);
    for (int f = 0; f < nFeatures; f++) {
        for (int i = 0; i < instance.size(); i++) {
            p[i] = instance.unaries[i][f];
        }
        drwnScaleToRange(m, 0.0, 1.0);
        views[f] = drwnCreateHeatMap(m, DRWN_COLORMAP_RAINBOW);
    }

    // create composite image
    return drwnCombineImages(views);
}
