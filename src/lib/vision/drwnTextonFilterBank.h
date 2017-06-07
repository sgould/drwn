/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextonFilterBank.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>

#include "cv.h"

// drwnTextonFilterBank -----------------------------------------------------

//! \brief Implements a 17-dimensional filter bank.
//!
//! The filter bank is defined in:
//! \li J. Shotton, J. Winn, C. Rother, A. Criminisi.
//!     "TextonBoost for Image Understanding: Multi-Class Object Recognition
//!     and Segmentation by Jointly Modeling Texture, Layout, and Context,"
//!     IJCV 2008.
//! \li J. Winn, A. Criminisi, and T. Minka.
//!     "Categorization by learned universal visual dictionary," ICCV 2005.
//!
//! The features produced by this filterbank are used in multi-class image
//! segmentation (see \ref drwnProjMultiSeg).
//!
//! \sa drwnFilterBankResponse

class drwnTextonFilterBank {
 protected:
    double _kappa; //!< base filter bandwidth

 public:
    static const int NUM_FILTERS = 17; //!< number of filters in the filterbank

    //! construct a filterbank with bandwidth \p k
    drwnTextonFilterBank(double k = 1.0);
    virtual ~drwnTextonFilterBank();

    //! Filtering function. The caller must provide a vector of CV32F destination
    //! matrices. The source image should be a 3-channel RGB color image (it is
    //! automatically converted to CIELab).
    void filter(const cv::Mat& img, std::vector<cv::Mat>& response) const;
};
