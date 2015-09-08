/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImageInPainter.h
** AUTHOR(S):   Robin Liang <robin.gnail@gmail.com>
**              Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// opencv library headers
#include "cv.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// drwnImageInPainter --------------------------------------------------------
//! Performs exemplar-based image inpainting.
//!
//! Completes a masked part of an image with patches copied from other regions
//! in the image. Motivated by the image completion algorithm of Criminisi et al.,
//! IEEE TIP, 2004 using the PatchMatch algorithm of Barnes et al. for finding
//! patch candidates. An example code snippet is provided below:
//!
//!
//! \code
//!   cv::Mat img = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);
//!   DRWN_ASSERT_MSG(img.data, "could not read image from " << imageFilename);
//!    
//!   cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
//!   cv::rectangle(mask, cv::Point(img.cols / 4, img.rows / 4),
//!       cv::Point(3 * img.cols / 4, 3 * img.rows / 4), cv::Scalar(0xff), -1);
//!
//!   drwnImageInPainter inpainter(3);
//!   inpainter.bVisualize = true;
//!   cv::Mat output = inpainter.fill(img, mask);
//!
//!   cv::imwrite(outputFilename, output);
//! \endcode

class drwnImageInPainter {
public:
    static unsigned UPDATE_STEPS; //!< patch match update steps between inpaints
    static bool PIXELWISE;        //!< pixelwise update versus patch update
    static bool PRIORITY_FILLING; //!< use priority filling scheme versus onion peeling

public:
    unsigned patchRadius;   //!< size of the patch (2 * radius + 1) for comparison
    bool bAllowFromFilled;  //!< allow copying from already infilled regions
    double alphaBlend;      //!< alpha-blend with un-masked region
    bool bVisualize;        //!< show progress using the monitor function

public:
    //! construct an image inpainter object
    drwnImageInPainter(unsigned _patchRadius = 3, bool _bAllowFromFilled = false) :
        patchRadius(_patchRadius), bAllowFromFilled(_bAllowFromFilled),
        alphaBlend(0.125), bVisualize(false) {
        // do nothing
    }
    //! destructor
    virtual ~drwnImageInPainter() { /* do nothing */ }

    //! inpaint the pixels within \p fillMask
    cv::Mat fill(const cv::Mat& image, const cv::Mat& fillMask) const {
        return fill(image, fillMask, fillMask == 0);
    }
    //! inpaint the pixels within \p fillMask using only pixels from \p copyMask
    cv::Mat fill(const cv::Mat& image, const cv::Mat& fillMask, const cv::Mat& copyMask) const {
        return fill(image, fillMask, copyMask, fillMask);
    }
    //! inpaint the pixels within \p fillMask using only pixels from \p copyMask
    //! and ignoring matching against pixels in \p ignoreMask
    cv::Mat fill(const cv::Mat& image, const cv::Mat& fillMask, const cv::Mat& copyMask,
        const cv::Mat& ignoreMask) const;

    //! monitor function called each iteration (can be used to show progress)
    virtual void monitor(const cv::Mat& image, const cv::Mat& mask) const;

protected:
    //! compute pixelwise features for matching
    virtual cv::Mat featurizeImage(const cv::Mat& image) const;
    //! compute pixel priority
    double computePixelPriority(const cv::Point& p, const cv::Mat& image, const cv::Mat& mask) const;
    //! extract pixels within the mask adjacent to the boundary
    static cv::Mat extractMaskBoundary(const cv::Mat& mask);
    //! alpha-blend images on masked area
    static void blendMaskedImage(cv::Mat img, const cv::Mat& src, const cv::Mat& mask, double alpha);
};
