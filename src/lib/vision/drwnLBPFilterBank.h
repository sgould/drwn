/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLBPFilterBank.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>

#include "cv.h"

#include "drwnSuperpixelContainer.h"

// drwnLBPFilterBank --------------------------------------------------------

//! \brief Implements filter bank for encoding local binary patterns.
//!
//! Aggregating the responses over regions and normalizing give the standard
//! LBP feature.
//!
//! \sa drwnFilterBankResponse

class drwnLBPFilterBank {
 protected:
    bool _b8Neighbourhood; //!< true of 8-connected neighbourhood, otherwise 
                           //!< a 4-connected is used

 public:
    //! construct an LBP filterbank
    drwnLBPFilterBank(bool b8Neighbourbood = false);
    virtual ~drwnLBPFilterBank();

    //! returns the number of response channels
    size_t numFilters() const { return _b8Neighbourhood ? 8 : 4; }

    //! Filtering function. The caller must provide a vector of CV32F destination
    //! matrices (or empty). The source image should be a 3-channel RGB color or 
    //! 1-channel greyscale image.
    void filter(const cv::Mat& img, std::vector<cv::Mat>& response) const;

    //! Compute LBP histograms over given superpixel regions. The \p response input
    //! comes from running the \p filter member function. Features are output in
    //! \p features.
    void regionFeatures(const cv::Mat& regions, const std::vector<cv::Mat>& response,
        vector<vector<double> >& features);

    //! Compute LBP histograms over given superpixel regions. The \p response input
    //! comes from running the \p filter member function. Features are output in
    //! \p features.
    void regionFeatures(const drwnSuperpixelContainer& regions, 
        const std::vector<cv::Mat>& response, vector<vector<double> >& features);
};
