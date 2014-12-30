/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPatchMatchUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

#include "drwnPatchMatchUtils.h"

using namespace std;

// drwnPatchMatchUtils -------------------------------------------------------

void drwnPatchMatchUtils::loadLabels(const set<string>& baseNames, map<string, MatrixXi>& labels)
{
    for (set<string>::const_iterator it = baseNames.begin(); it != baseNames.end(); ++it) {
        if (labels.find(*it) != labels.end()) {
            DRWN_LOG_WARNING("overwriting labels for " << *it);
        }

        const string lblFilename = gMultiSegConfig.filename("lblDir", *it, "lblExt");
        DRWN_LOG_DEBUG("loading groundtruth labels from " << lblFilename << "...");
        if (!drwnFileExists(lblFilename.c_str())) {
            DRWN_LOG_WARNING("groundtruth does not exist for " << *it);
        } else {
            MatrixXi L;
            drwnReadUnknownMatrix(L, lblFilename.c_str());
            labels[*it] = L;
        }
    }
}

vector<cv::Point> drwnPatchMatchUtils::sortPixelsByVariance(const cv::Mat& img, const cv::Size& patchSize)
{
    drwnFilterBankResponse greyFilter;
    cv::Mat greyImg = drwnGreyImage(img);
    greyFilter.addResponseImage(greyImg);

    multimap<double, cv::Point, std::greater<double> > scoredPixels;
    for (int y = 0; y < img.rows - patchSize.height + 1; y++) {
        for (int x = 0; x < img.cols - patchSize.width + 1; x++) {
            const double v = greyFilter.variance(x, y, patchSize.width, patchSize.height)[0];
            scoredPixels.insert(make_pair(v, cv::Point(x, y)));
        }
    }

    vector<cv::Point> pixels;
    pixels.reserve(scoredPixels.size());
    for (multimap<double, cv::Point, std::greater<double> >::const_iterator it = scoredPixels.begin();
         it != scoredPixels.end(); ++it) {
        pixels.push_back(it->second);
    }

    return pixels;
}

size_t drwnPatchMatchUtils::countMatchablePixels(const drwnPatchMatchGraph& graph)
{
    size_t count = 0;
    for (unsigned imgIndx = 0; imgIndx < graph.size(); imgIndx++) {
        for (unsigned lvlIndx = 0; lvlIndx < graph[imgIndx].levels(); lvlIndx++) {
            for (unsigned pixIndx = 0; pixIndx < graph[imgIndx][lvlIndx].size(); pixIndx++) {
                if (!graph[imgIndx][lvlIndx][pixIndx].empty()) {
                    count += 1;
                }
            }
        }
    }

    return count;
}

double drwnPatchMatchUtils::overlap(const drwnPatchMatchEdgeList& edgesA,
    const drwnPatchMatchEdgeList& edgesB)
{
    double area = 0.0;
    for (drwnPatchMatchEdgeList::const_iterator ia = edgesA.begin(); ia != edgesA.end(); ++ia) {
        for (drwnPatchMatchEdgeList::const_iterator ib = edgesB.begin(); ib != edgesB.end(); ++ib) {
            if (ia->targetNode.imgIndx != ib->targetNode.imgIndx) continue;
            //! \todo handle pyramid scale

            // find region of overlap
            const int iw = std::min(ia->targetNode.xPosition, ib->targetNode.xPosition) -
                std::max(ia->targetNode.xPosition, ib->targetNode.xPosition) + drwnPatchMatchGraph::PATCH_WIDTH;
            const int ih = std::min(ia->targetNode.yPosition, ib->targetNode.yPosition) -
                std::max(ia->targetNode.yPosition, ib->targetNode.yPosition) + drwnPatchMatchGraph::PATCH_HEIGHT;

            if ((iw > 0) && (ih > 0))
                area += (double)(iw * ih);
        }
    }

    // normalize area
    //area /= (double)std::min(edgesA.size(), edgesB.size());
    area /= (double)(edgesA.size() + edgesB.size()) / 2.0;
    //area /= (double)std::max(edgesA.size(), edgesB.size());

    area /= (drwnPatchMatchGraph::PATCH_WIDTH * drwnPatchMatchGraph::PATCH_HEIGHT);
    return area;
}

bool drwnPatchMatchUtils::isValidNode(const drwnPatchMatchGraph& graph, const drwnPatchMatchNode& node)
{
    if (node.imgIndx >= graph.size()) {
        DRWN_LOG_WARNING("node has invalid image index " << node.imgIndx);
        return false;
    }

    if (node.imgScale >= graph[node.imgIndx].levels()) {
        DRWN_LOG_WARNING("node has invalid image scale " << node.imgScale);
        return false;
    }

    if (node.xPosition > graph[node.imgIndx][node.imgScale].width() - graph.patchWidth()) {
        DRWN_LOG_WARNING("node has invalid x-position " << node.xPosition);
        return false;
    }

    if (node.yPosition > graph[node.imgIndx][node.imgScale].height() - graph.patchHeight()) {
        DRWN_LOG_WARNING("node has invalid y-position " << node.yPosition);
        return false;
    }

    return true;
}
