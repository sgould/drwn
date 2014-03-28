/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFilterBankResponse.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <list>
#include <vector>

#include "Eigen/Core"

#include "cv.h"

using namespace std;
using namespace Eigen;

// drwnFilterBankResponse ------------------------------------------------------
//! Holds the results of running an image through a bank of filters and allows
//! for computation of features over rectangular regions.
//!
//! The difference between \p addResponseImage and \p copyResponseImage functions
//! is that the former does not clone the images --- the calling function must not
//! modify the image since this will corrupt data internal to drwnFilterBankResponse.
//! Response images should be 32-bit floating point.
//!
//! The class uses integral images to allow quick computation of sums over
//! rectangular regions. For example, the following code snippet computes
//! the mean image intensity over a few small rectangular patches.
//! \code
//!   // load the image and convert to 32-bit greyscale
//!   cv::Mat img = cv::imread( ... );
//!   cv::Mat grey = drwnGreyImage(img);
//!
//!   // create filterbank response object and add greyscale image
//!   drwnFilterBankResponse filterbank;
//!   filterbank.addResponseImage(grey);
//!
//!   // compute average intensity over random rectangular regions
//!   for (int i = 0; i < 100; i++) {
//!       const int x = (int)(drand48() * filterbank.width());
//!       const int y = (int)(drand48() * filterbank.height());
//!       const int w = (int)(drand48() * (filterbank.width() - x)) + 1;
//!       const int h = (int)(drand48() * (filterbank.height() - y)) + 1;
//!
//!       double meanIntensity = filterbank.mean(x, y, w, h)[0];
//!       DRWN_LOG_VERBOSE("mean intensity over " << w << "-by-" << h << " patch at ("
//!         << x << ", " << y << ") is " << meanIntensity);
//!   }
//! \endcode
//!

class drwnFilterBankResponse {
 protected:
    vector<cv::Mat> _responses;     //!< filter responses
    vector<cv::Mat> _sum;           //!< sum of filter responses (integral image)
    vector<cv::Mat> _sqSum;         //!< sum of filter responses squared

 public:
    //! default constructor
    drwnFilterBankResponse();
    //! copy constructor
    drwnFilterBankResponse(const drwnFilterBankResponse& f);
    ~drwnFilterBankResponse();

    //! clears the filter bank responses (and releases memory)
    void clear();
    //! returns true if their are no filter responses
    inline bool empty() const { return _responses.empty(); }
    //! returns the number of responses from the filter bank
    inline int size() const { return (int)_responses.size(); }
    //! returns the width of each filter response (i.e., image width)
    inline int width() const { return _responses.empty() ? 0 : _responses[0].cols; }
    //! returns the height of each filter response (i.e., image height)
    inline int height() const { return _responses.empty() ? 0 : _responses[0].rows; }
    //! returns number of bytes stored
    inline size_t memory() const { return size() * width() * height() * sizeof(float) +
            2 * size() * (width() + 1) * (height() + 1) * sizeof(double); }

    // add or copy 32-bit floating-point response images to the filter bank
    //! add a filter response image to the filterbank (takes ownership)
    void addResponseImage(cv::Mat& r);
    //! add a number of filter response images to the filterbank (takes ownership)
    void addResponseImages(vector<cv::Mat>& r);
    //! copies a filter response image to the filterbank (called retains ownership)
    void copyResponseImage(const cv::Mat& r);
    //! copies a number of filter response images to the filterbank (called retains ownership)
    void copyResponseImages(const vector<cv::Mat>& r);
    //! return the i-th filter response image
    const cv::Mat& getResponseImage(int i) const;
    //! delete the i-th filter response image (all responses above \p i are renumbered)
    void deleteResponseImage(int i);

    //! transform responses by exponentiation (useful, for example, when approximating
    //! max over a region, via log-sum-exp)
    void exponentiateResponses(double alpha = 1.0);
    //! transform responses by pixelwise normalization (e.g., following an exponentiation)
    void normalizeResponses();
    //! transform responses by pixelwise exponentiation and normalization. More numerically
    //! stable than exponentiateResponses followed by normalizeResponses.
    void expAndNormalizeResponses(double alpha = 1.0);

    //! value of each filter at given pixel
    VectorXd value(int x, int y) const;
    //! mean of each filter in rectangular region <x, y, x+w, y+h>
    VectorXd mean(int x, int y, int w, int h) const;
    //! sum of squared values for each filter in rectangular region <x, y, x+w, y+h>
    VectorXd energy(int x, int y, int w, int h) const;
    //! variance of each filter in rectangular region <x, y, x+w, y+h>
    VectorXd variance(int x, int y, int w, int h) const;

    //! mean of each filter over given pixels
    VectorXd mean(const list<cv::Point>& pixels) const;
    //! energy of each filter over given pixels
    VectorXd energy(const list<cv::Point>& pixels) const;
    //! variance of each filter over given pixels
    VectorXd variance(const list<cv::Point>& pixels) const;

    //! mean of each filter over masked pixels
    VectorXd mean(const cv::Mat& mask) const;
    //! energy of each filter over masked pixels
    VectorXd energy(const cv::Mat& mask) const;
    //! variance of each filter over masked pixels
    VectorXd variance(const cv::Mat& mask) const;

    //! mean of each filter over entire image
    inline VectorXd mean() const { return mean(0, 0, width(), height()); }
    //! energy of each filter over entire image
    inline VectorXd energy() const { return energy(0, 0, width(), height()); }
    //! variance of each filter over entire image
    inline VectorXd variance() const { return variance(0, 0, width(), height()); }

    //! visualize the feature responses
    cv::Mat visualize() const;
};



