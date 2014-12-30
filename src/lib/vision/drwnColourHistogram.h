/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnColourHistogram.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>

#include "cv.h"

// drwnColourHistogram ------------------------------------------------------

//! \brief Specialized histogram for quantized 3-channel colour values (e.g., RGB).
//!
//! The following example shows how to construct a colour histogram for all pixels
//! in a given image:
//! /code
//!    cv::Mat img = cv::imread("testImage.jpg", CV_LOAD_IMAGE_COLOR);
//!
//!    // construct histogram for the image
//!    drwnColourHistogram histogram;
//!
//!    for (int y = 0; y < img.rows; y++) {
//!        for (int x = 0; x < img.cols; x++) {
//!            histogram.accumulate(img.at<const Vec3b>(y, x));
//!        }
//!    }
//!
//!    // display the histogram
//!    drwnShowDebuggingImage(histogram.visualize(), string("histogram"), true);
//! \endcode
//!

class drwnColourHistogram {
 protected:
    unsigned _channelBits;        //!< number of bits for each colour channel
    unsigned char _mask;          //!< mask for channel bits
    double _pseudoCounts;         //!< psuedo counts for dirchelet prior
    vector<double> _histogram;    //!< the histogram counts (with interpolation)
    double _totalCounts;          //!< sum of histogram counts

 public:
    //! constructor
    drwnColourHistogram(double pseudoCounts = 1.0, unsigned channelBits = 3);
    //! destructor
    ~drwnColourHistogram() { /* do nothing */ }

    //! returns histogram size
    size_t size() const { return _histogram.size(); }

    //! clear the histogram counts
    void clear() { clear(_pseudoCounts); }

    //! clear the histogram and assign a new prior
    void clear(double pseudoCounts);

    //! accumulate an RGB colour sample
    void accumulate(unsigned char red, unsigned char green, unsigned char blue);
    //! accumulate a cv::Vec3b colour sample
    void accumulate(const cv::Vec3b& colour) {
        accumulate(colour.val[2], colour.val[1], colour.val[0]);
    }
    //! accumulate a cv::Scalar colour sample
    void accumulate(const cv::Scalar& colour) {
        accumulate(colour.val[2], colour.val[1], colour.val[0]);
    }

    //! return probability of an RGB colour sample
    double probability(unsigned char red, unsigned char green, unsigned char blue) const;
    //! return probability of a cv::Vec3b colour sample
    double probability(const cv::Vec3b& colour) const {
        return probability(colour.val[2], colour.val[1], colour.val[0]);
    }
    //! return probability of a cv::Scalar colour sample
    double probability(const cv::Scalar& colour) const { 
        return probability(colour.val[2], colour.val[1], colour.val[0]); 
    }

    //! visualization
    cv::Mat visualize() const;

    // operators
    const double& operator[](size_t indx) const { return _histogram[indx]; } 
};
