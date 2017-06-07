/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPatchMatchUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <set>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

#include "drwnPatchMatch.h"

using namespace std;

// drwnPatchMatchUtils -------------------------------------------------------

namespace drwnPatchMatchUtils {

    //! map a pixel from an image of one size to an image of another size
    inline cv::Point transformPixel(const cv::Point& srcPixel,
        const cv::Size& srcSize, const cv::Size& dstSize)
    {
        return cv::Point(srcPixel.x * dstSize.width / srcSize.width,
            srcPixel.y * dstSize.height / dstSize.height);
    }

    //! map a region of interest from an image of one size to an image of another size
    inline cv::Rect transformRect(const cv::Rect& srcRect,
        const cv::Size& srcSize, const cv::Size& dstSize)
    {
        cv::Rect dstRect;

        dstRect.x = srcRect.x * dstSize.width / srcSize.width;
        dstRect.y = srcRect.y * dstSize.height / srcSize.height;
        dstRect.width = srcRect.width * dstSize.width / srcSize.width;
        dstRect.height = srcRect.height * dstSize.height / srcSize.height;

        return dstRect;
    }

    //! load labels for images appearing in the dataset
    void loadLabels(const set<string>& baseNames, map<string, MatrixXi>& labels);

    //! sorted list of pixels by patch variance
    vector<cv::Point> sortPixelsByVariance(const cv::Mat& img, const cv::Size& patchSize);

    //! counts the number of pixels being processed (i.e., those with matches)
    size_t countMatchablePixels(const drwnPatchMatchGraph& graph);

    //! compute overlap between two nearest neighbour sets
    double overlap(const drwnPatchMatchEdgeList& edgesA,
        const drwnPatchMatchEdgeList& edgesB);

    //! debugging routine to determine whether a given node is valid within a graph
    bool isValidNode(const drwnPatchMatchGraph& graph, const drwnPatchMatchNode& node);
};
