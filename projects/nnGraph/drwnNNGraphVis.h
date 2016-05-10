/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraphVis.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

// opencv headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnNNGraph.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphVis ------------------------------------------------------------
//! Visualization routines for a drwnNNGraph.

namespace drwnNNGraphVis {
    //! load data necessary for visualization
    void loadImageData(const drwnNNGraph& graph, vector<drwnNNGraphImageData>& images);

    //! visualize images and superpixels
    cv::Mat visualizeDataset(const vector<drwnNNGraphImageData>& images);
    //! visualize match quality
    cv::Mat visualizeMatchQuality(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph);
    //! visualize match retarget
    cv::Mat visualizeRetarget(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph);
    //! visualize image index for matching segment
    cv::Mat visualizeImageIndex(const vector<drwnNNGraphImageData>& images, const drwnNNGraph& graph);
};
