/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphVis.cpp
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

#include "drwnNNGraph.h"
#include "drwnNNGraphVis.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphVis ------------------------------------------------------------

void drwnNNGraphVis::loadImageData(const drwnNNGraph& graph, vector<drwnNNGraphImageData>& images)
{
    DRWN_ASSERT(images.empty());
    images.reserve(graph.numImages());

    for (unsigned imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
        DRWN_LOG_VERBOSE("...loading image data for " << graph[imgIndx].name());
        images.push_back(drwnNNGraphImageData(graph[imgIndx].name()));
    }
}

cv::Mat drwnNNGraphVis::visualizeDataset(const vector<drwnNNGraphImageData>& images)
{
    // accumulate views
    vector<cv::Mat> views;
    views.resize(images.size());

    for (unsigned i = 0; i < images.size(); i++) {
        views[i] = images[i].image().clone();
        drwnAverageRegions(views[i], images[i].segments()[0]);
        drwnDrawRegionBoundaries(views[i], images[i].segments()[0], CV_RGB(0, 0, 0));
    }

    // combine views
    return drwnCombineImages(views);
}

cv::Mat drwnNNGraphVis::visualizeMatchQuality(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph)
{
    // accumulate views
    vector<cv::Mat> views(images.size());

    for (unsigned i = 0; i < images.size(); i++) {
        int imgIndx = graph.findImage(images[i].name());
        DRWN_ASSERT_MSG(imgIndx >= 0, images[i].name());

        cv::Mat m(images[imgIndx].height(), images[imgIndx].width(), CV_32FC1);
        for (unsigned y = 0; y < images[imgIndx].height(); y++) {
            for (unsigned x = 0; x < images[imgIndx].width(); x++) {
                const int segId = images[imgIndx].segments()[0].at<int>(y, x);
                if (segId < 0) {
                    m.at<float>(y, x) = 0.0f;
                } else {
                    const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                    if (e.empty()) {
                        m.at<float>(y, x) = 0.0f;
                    } else {
                        m.at<float>(y, x) = e.front().weight;
                    }
                }
            }
        }

        views[imgIndx] = drwnCreateHeatMap(m);
        drwnDrawRegionBoundaries(views[imgIndx], images[imgIndx].segments()[0], CV_RGB(0, 0, 0));
    }

    // combine views
    return drwnCombineImages(views);
}

cv::Mat drwnNNGraphVis::visualizeRetarget(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph)
{
    // accumulate views
    vector<cv::Mat> views(images.size());

    for (unsigned i = 0; i < images.size(); i++) {
        int imgIndx = graph.findImage(images[i].name());
        DRWN_ASSERT_MSG(imgIndx >= 0, images[i].name());

        views[i] = cv::Mat(images[i].height(), images[i].width(), CV_8UC3);
        for (unsigned y = 0; y < images[i].height(); y++) {
            for (unsigned x = 0; x < images[i].width(); x++) {
                cv::Scalar accRGB = cv::Scalar::all(0);
                unsigned count = 0;
                for (int c = 0; c < images[i].segments().channels(); c++) {
                    const int segId = images[i].segments()[0].at<int>(y, x);
                    if (segId < 0) continue;
                    const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                    if (e.empty()) continue;

                    const cv::Scalar segRGB = images[e.front().targetNode.imgIndx].rgbColour(e.front().targetNode.segId);
                    accRGB += segRGB;
                    count += 1;
                }

                if (count > 0) {
                    accRGB[0] /= count;
                    accRGB[1] /= count;
                    accRGB[2] /= count;
                }

                views[i].at<unsigned char>(y, 3 * x + 0) = (unsigned char)accRGB[0];
                views[i].at<unsigned char>(y, 3 * x + 1) = (unsigned char)accRGB[1];
                views[i].at<unsigned char>(y, 3 * x + 2) = (unsigned char)accRGB[2];
            }
        }

        //drwnDrawRegionBoundaries(views[i], images[i].segments()[0], CV_RGB(0, 0, 0));
    }

    // combine views
    return drwnCombineImages(views);
}

cv::Mat drwnNNGraphVis::visualizeImageIndex(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph)
{
    // accumulate views
    vector<cv::Mat> views(images.size());

    for (unsigned i = 0; i < images.size(); i++) {
        int imgIndx = graph.findImage(images[i].name());
        DRWN_ASSERT_MSG(imgIndx >= 0, images[i].name());

        cv::Mat m(images[i].height(), images[i].width(), CV_32FC1);
        for (unsigned y = 0; y < images[i].height(); y++) {
            for (unsigned x = 0; x < images[i].width(); x++) {
                const int segId = images[i].segments()[0].at<int>(y, x);
                if (segId < 0) {
                    m.at<float>(y, x) = 0.0f;
                } else {
                    const drwnNNGraphEdgeList& e = graph[imgIndx][segId].edges;
                    if (e.empty()) {
                        m.at<float>(y, x) = 0.0f;
                    } else {
                        m.at<float>(y, x) = (float)e.front().targetNode.imgIndx / graph.numImages();
                    }
                }
            }
        }

        views[i] = drwnCreateHeatMap(m);
        drwnDrawRegionBoundaries(views[i], images[i].segments()[0], CV_RGB(0, 0, 0));
    }

    // combine views
    return drwnCombineImages(views);
}
