/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTemplateMatcher.h
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

// drwnTemplateMatcher ---------------------------------------------------------
//! Utility class for computing multiple template matches.
//!
//! The instantiated object pre-computes template (and image) DFTs to allow for
//! faster matching on multiple images (of the same size). However, the code is
//! very memory intensive. Call the drwnTemplateMatcher::reset function to release
//! the internal buffers.

class drwnTemplateMatcher {
 protected:
    vector<cv::Mat> _templates;    //!< vector of templates to match
    vector<pair<double, double> > _templateStats; //!< template mean and energy
    cv::Size _largestTemplate;     //!< dimensions of the largest template

    // buffered data (clear with reset())
    cv::Size _imgSize;             //!< size of image being buffered
    vector<cv::Mat> _dftTemplates; //!< buffered templates in fourier space
    cv::Mat _dftImage;             //!< buffered image in fourier space
    cv::Mat _dftResponse;          //!< buffer for response image
    cv::Mat _imgSum;               //!< buffer for integral image
    cv::Mat _imgSqSum;             //!< buffer for integral square image

 public:
    //! default constructor
    drwnTemplateMatcher();
    //! copy constructor
    drwnTemplateMatcher(const drwnTemplateMatcher& tm);
    ~drwnTemplateMatcher();

    //! clears all templates and buffered storage
    void clear();
    //! returns true if there are no templates
    inline bool empty() const { return _templates.empty(); }
    //! returns the number of templates
    inline int size() const { return (int)_templates.size(); }
    //! returns the width of the largest template
    inline int width() const { return _largestTemplate.width; }
    //! returns the height of the largest template
    inline int height() const { return _largestTemplate.height; }

    //! release internal memory (keeps templates)
    void reset();

    //! Add templates to the object. The difference between addTemplate and 
    //! copyTemplate is that the former takes does not make a copy so changes
    //! by the calling function will be corrupt the data. Templates should be
    //! single channel 32-bit floating point.
    void addTemplate(cv::Mat& t);
    //! See above.
    void addTemplates(vector<cv::Mat>& t);
    //! See above.
    void copyTemplate(const cv::Mat& t);
    //! See above.
    void copyTemplates(const vector<cv::Mat>& t);

    //! Response images are all generate to be the same size as the
    //! original image. Boundaries are zero-padded. Caller is responsible
    //! for freeing memory. Functions without image argument use the
    //! previously loaded image.
    vector<cv::Mat> responses(const cv::Mat& img, int method = CV_TM_CCORR);
    //! See above.
    vector<cv::Mat> responses(int method = CV_TM_CCORR);
    //! See above.
    cv::Mat response(const cv::Mat& img, unsigned tid, int method = CV_TM_CCORR);
    //! See above.
    cv::Mat response(unsigned tid, int method = CV_TM_CCORR);

    // operators
    //! assignment operator
    drwnTemplateMatcher& operator=(const drwnTemplateMatcher& tm);
    //! returns the i-th template
    const cv::Mat& operator[](unsigned i) const;

 protected:
    //! caches all buffered data for a given image
    void cacheDFTAndIntegrals(const cv::Mat& image);
};
