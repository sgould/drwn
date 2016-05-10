/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelNeighbourContrasts.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once
#include "Eigen/Core"

#include "cv.h"

using namespace std;
using namespace Eigen;

// drwnPixelNeighbourContrasts class ----------------------------------------
//! \brief Convenience class for holding pixel contrast weights.
//!
//! These are used in various CRF image segmentation models (see drwnSegImageInstance)
//! with pairwise neighbourhood penalties taking the form
//! \f$\frac{1}{d_{pq}} exp\left(-\frac{\beta}{2} * \left|p - q\right|^2\right)\f$
//! where \f$\beta\f$ is the mean square-difference between neighbouring
//! pixel colours and \f$d_{pq}\f$ is the distance between pixels p and q.

class drwnPixelNeighbourContrasts : public drwnStdObjIface {
protected:
    MatrixXd _horzContrast;  //!< contrast between (x, y) and (x - 1, y)
    MatrixXd _vertContrast;  //!< contrast between (x, y) and (x, y - 1)
    MatrixXd _nwContrast;    //!< (north-west) contrast between (x, y) and (x - 1, y - 1)
    MatrixXd _swContrast;    //!< (south-west) contrast between (x, y) and (x - 1, y + 1)

public:
    //! default constructor
    drwnPixelNeighbourContrasts();
    //! construct and cache pixel contrast weights for the given image
    drwnPixelNeighbourContrasts(const cv::Mat& img);
    virtual ~drwnPixelNeighbourContrasts();

    // i/o
    const char *type() const { return "drwnPixelNeighbourContrasts"; }
    drwnPixelNeighbourContrasts *clone() const { return new drwnPixelNeighbourContrasts(*this); }

    //! clear all cached contrast weights
    virtual void clear();
    //! cache contrast weights for the given image
    virtual void initialize(const cv::Mat& img);

    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    //! width of the image (in pixels)
    int width() const { return _horzContrast.rows(); }
    //! height of the image (in pixels)
    int height() const { return _horzContrast.cols(); }

    // access
    //! return the contrast weight between pixels (x,y) and (x,y-1)
    double contrastN(int x, int y) const { return _vertContrast(x, y); }
    //! return the contrast weight between pixels (x,y) and (x,y+1)
    double contrastS(int x, int y) const { return _vertContrast(x, y + 1); }
    //! return the contrast weight between pixels (x,y) and (x+1,y)
    double contrastE(int x, int y) const { return _horzContrast(x + 1, y); }
    //! return the contrast weight between pixels (x,y) and (x-1,y)
    double contrastW(int x, int y) const { return _horzContrast(x, y); }
    //! return the contrast weight between pixels (x,y) and (x-1,y-1)
    double contrastNW(int x, int y) const { return _nwContrast(x, y); }
    //! return the contrast weight between pixels (x,y) and (x-1,y+1)
    double contrastSW(int x, int y) const { return _swContrast(x, y); }
    //! return the contrast weight between pixels (x,y) and (x+1,y-1)
    double contrastNE(int x, int y) const { return _swContrast(x + 1, y - 1); }
    //! return the contrast weight between pixels (x,y) and (x+1,y+1)
    double contrastSE(int x, int y) const { return _nwContrast(x + 1, y + 1); }

    // visualization
    //! Visualize the strength of the contrast term. If \p bComposite is \p false
    //! then the contrast is visualized in each direction separately.
    cv::Mat visualize(bool bComposite = true) const;

protected:
    static double pixelContrast(const cv::Mat& img, const cv::Point &p, const cv::Point& q);
};
