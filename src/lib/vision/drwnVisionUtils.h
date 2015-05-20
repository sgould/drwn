/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVisionUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Jimmy Lin <JimmyLin@utexas.edu>
**
*****************************************************************************/

/*!
** \file drwnVisionUtils.h
** \anchor drwnVisionUtils
** \brief Computer vision utility functions (not dependent on OpenCV
** functions, but may use OpenCV data structures).
*/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>

#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;
using namespace Eigen;

//! Weighted undirected arc between pixels in an image
class drwnWeightedPixelEdge {
 public:
    double w;     //!< weight
    cv::Point p;  //!< first pixel
    cv::Point q;  //!< second pixel

 public:
    drwnWeightedPixelEdge() : w(0.0), p(-1, -1), q(-1, -1) { /* do nothing */ };
    drwnWeightedPixelEdge(const cv::Point& _p, const cv::Point& _q, double _w = 0.0) :
        w(_w), p(_p), q(_q) { /* do nothing */ };
    ~drwnWeightedPixelEdge() { /* do nothing */ };
};

//! Image rescaling factor for \p lambda levels per octave
inline double drwnPyramidScale(int lambda) {
    return exp(-1.0 * log(2.0) / (double)lambda);
}

//! Maps a region of interest from one image scale to another.
cv::Rect drwnTransformROI(const cv::Rect& roi, const cv::Size& srcSize, const cv::Size& dstSize);

//! Load an over-segmentation (superpixel image) or pixel labeling
//! from a .png or .txt file. Checks size if \p pixelLabels is not
//! empty.
void drwnLoadPixelLabels(cv::Mat& pixelLabels, const char *filename);

//! Load an over-segmentation (superpixel image) or pixel labeling
//! from a .png or .txt file. Checks size if \p pixelLabels is not
//! empty. Replaces any value greater than or equal to \p numLabels
//! with -1.
void drwnLoadPixelLabels(cv::Mat& pixelLabels, const char *filename, int numLabels);

//! Load an over-segmentation (superpixel image) or pixel labeling
//! from a .png or .txt file. Checks size if \p pixelLabels is not
//! empty. Replaces any value greater than or equal to \p numLabels with -1.
void drwnLoadPixelLabels(MatrixXi &pixelLabels, const char *filename,
    int numLabels = DRWN_INT_MAX);

//! Finds connected components in an over-segmentation and renumbers
//! superpixels contiguously from 0. Second argument controls whether
//! connectivity is defined on a 4-connected or 8-connected neighborhood.
//! Returns the number of connected components.
int drwnConnectedComponents(cv::Mat& segments, bool b8Connected = false);
//! See above.
int drwnConnectedComponents(MatrixXi& segments, bool b8Connected = false);

//! Generates an over-segmentation (superpixels) of an image. The
//! parameter \p gridSize controls the number of superpixels. A
//! value of 10 will produce about 100 superpixels.
cv::Mat drwnFastSuperpixels(const cv::Mat& img, unsigned gridSize);

//! Generates an over-segmentation (superpixels) of an image as a set
//! of disconnected regions. The parameter \p numCentroids controls the
//! number of regions. A value of 10 will produce up to 10 regions.
cv::Mat drwnKMeansSegments(const cv::Mat& img, unsigned numCentroids);

//! Generates an over-segmentation (superpixels) of an image based on
//! the SLIC algorithm (Achanta et al., PAMI 2012). The parameter \p
//! nClusters controls the number of superpixels, the parameter \p
//! spatialWeight controls the relative weight of the spatial term,
//! and the parameter \p threshold (between 0 and 1) defines a
//! stopping criteria. The image schould be provided in CIELAB format.
cv::Mat drwnSLICSuperpixels(const cv::Mat& img, unsigned nClusters, 
    double spatialWeight = 200.0, double threshold = 1.0e-3);

//! Merges small superpixels into neighbours until at most \p maxSegs
//! remain.
void drwnMergeSuperpixels(const cv::Mat& img, cv::Mat& seg, unsigned maxSegs);
