/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImageInstance.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <list>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

#include "drwnPixelNeighbourContrasts.h"
#include "drwnSuperpixelContainer.h"
#include "drwnSegImagePixelFeatures.h"
#include "drwnVisionUtils.h"

using namespace std;
using namespace Eigen;

// drwnSegImageInstance class ----------------------------------------------
//! Encapsulates a single instance of an image for multi-class pixel labeling
//! problems (i.e., image segmentation).
//!
//! The data-structure includes basic image features and can be annotated
//! with pixelwise class labels. Negative labels are assumed to be unknown.
//!
//! \sa drwnPixelSegModel
//! \sa \ref drwnProjMultiSeg

class drwnSegImageInstance {
 protected:
    string _baseName;                   //!< image identifier
    cv::Mat _img;                       //!< 3-channel RGB image of the scene
    cv::Mat _grayImg;                   //!< 1-channel greyscale image of the scene
    cv::Mat _softEdgeImg;               //!< 1-channel edgemap image of the scene

 public:
    // cached pixel features
    vector<vector<double> > unaries;         //!< cached unary potentials or pixel feature vectors
    drwnPixelNeighbourContrasts contrast;    //!< neighborhood contrast for pairwise smoothness

    // long range edges
    vector<drwnWeightedPixelEdge> auxEdges;  //!< auxiliary (long range) edges

    // superpixels and auxiliary data
    drwnSuperpixelContainer superpixels;     //!< superpixels for features or consistency terms
    vector<cv::Mat> auxiliaryData;           //!< auxiliary data for certain models (future extensions)

    // class labels
    MatrixXi pixelLabels;                    //!< pixel labels (0 to K-1) and -1 for unknown

 public:
    //! create a drwnSegImageInstance from file
    drwnSegImageInstance(const char *imgFilename, const char *baseName = NULL);
    //! create an existing image
    drwnSegImageInstance(const cv::Mat& img, const char *baseName = NULL);
    //! copy constructor
    drwnSegImageInstance(const drwnSegImageInstance& instance);
    //! destructor
    virtual ~drwnSegImageInstance();

    //! returns the colour image
    inline const cv::Mat& image() const { return _img; }
    //! returns a greyscale version of the image
    inline const cv::Mat& greyImage() const { return _grayImg; }
    //! returns the edge magnitude of the image
    inline const cv::Mat& edgeMap() const { return _softEdgeImg; }
    //! returns the name of the image (if available)
    inline const string &name() const { return _baseName; }

    //! returns the width of the image in pixels
    inline int width() const { return _img.cols; }
    //! returns the height of the image in pixels
    inline int height() const { return _img.rows; }
    //! returns the number of pixels in the image
    inline int size() const { return (_img.cols * _img.rows); }
    //void resize(int w, int h);

    //! convert from image coordinates to pixel index
    inline int pixel2Indx(const cv::Point& p) const { return pixel2Indx(p.x, p.y); }
    //! convert from image coordinates to pixel index
    inline int pixel2Indx(int x, int y) const { return y * _img.cols + x; }
    //! convert from pixel index to image coordinates
    inline cv::Point indx2Pixel(int indx) const {
        return cv::Point(indx % _img.cols, indx / _img.cols);
    }

    //! clear cached pixel features or unary terms
    void clearPixelFeatures();
    //! append standard pixel features to cached pixel feature vectors
    inline void appendPixelFeatures() {
        drwnSegImageStdPixelFeatures g;
        appendPixelFeatures(g);
    }
    //! append pixel features to cached pixel feature vectors
    void appendPixelFeatures(drwnSegImagePixelFeatures &featureGenerator);

    //! assignment operator
    drwnSegImageInstance& operator=(const drwnSegImageInstance& instance);

 protected:
    //! initialization
    void initInstance();
};
