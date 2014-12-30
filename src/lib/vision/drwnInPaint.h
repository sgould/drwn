/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInPaint.h
** AUTHOR(S):   Robin Liang <robin.gnail@gmail.com>
**
*****************************************************************************/

#pragma once

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>

// opencv library headers
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;
using namespace cv;

// drwnInPaint -----------------------------------------------------------------
//! Performs exemplar-based image inpainting.
//!
//! Completes a masked part of an image with patches copied from other regions
//! in the image. Implements the inpainting algorithm of Criminisi et al.,
//! IEEE TIP, 2004. Uses multi-threading to accelerate running time. An example
//! code snippet is provided below:
//!
//! \code
//!   cv::Mat img = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);
//!   DRWN_ASSERT_MSG(img.data, "could not read image from " << imageFilename);
//!    
//!   cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
//!   cv::rectangle(mask, cv::Point(img.cols / 4, img.rows / 4),
//!       cv::Point(3 * img.cols / 4, 3 * img.rows / 4), cv::Scalar(0xff), -1);
//!
//!   cv::Mat output;
//!   drwnInPaint::inPaint(img, output, mask);
//!
//!   cv::imwrite(outputFilename, output);
//! \endcode

class drwnInPaint {
 private:
    //! data structure for holding information about pixels to be inpainted
    class _Pixel : public Point {
    public:
        float confidence, data, priority;

    public:
        _Pixel() : confidence(0.0f), data(0.0f), priority(0.0f) { /* do nothing */ };
        _Pixel(const Point& pt) : confidence(0.0f), data(0.0f), priority(0.0f) {
            this->x = pt.x;
            this->y = pt.y;
        }

        bool operator<(const _Pixel& px) const {
            return (this->priority < px.priority);
        }
    };

    //! drwnThreadJob worker to find the exemplar with the smallest difference
    class _findExemplarJob : public drwnThreadJob {
    private:
        Mat _source;
        Mat _fillMask;
        Mat _sourceMask;
        Mat _validMask;
        float _lambda;

    public:
        _Pixel px;       //!< target location for patch
        _Pixel sourcePx; //!< point to copy patch from (return value)
        Mat replacement;
        float difference;

    public:
        _findExemplarJob(const _Pixel& p, const Mat& source, const Mat& fillMask,
            const Mat& sourceMask, const Mat& validMask, const float& lambda) :
            _source(source), _fillMask(fillMask), _sourceMask(sourceMask), 
                _validMask(validMask), _lambda(lambda), px(p) { /* do nothing */ }

        void operator()();
    };

    public:
    friend class drwnInPaintConfig;

    //! inPaint() overloaded method that assumes the \p sourceMask is the
    //! negation of \p fillMask and \p validMask is the whole image
    static Mat inPaint(const Mat& source, Mat& output, const Mat& fillMask);
    //! inPaint() overloaded method that assumes that \p validMask is the whole image
    static Mat inPaint(const Mat& source, Mat& output, const Mat& fillMask,
        const Mat& sourceMask);
    //! Main inPaint() function that performs a variant of the (Criminisi et al., 2004) 
    //! inpainting algorithm. The \p source parameter is the original image,
    //! the \p output parameter is the inpainted image, the \p fillMask parameter
    //! specifies the area to be painted (non-zero entries), the \p sourceMask
    //! parameter specifies valid locations for sourcing patches, and the \p validMask
    //! limits the area of the image being inpainted (for example, if we wish to control
    //! what part of \p fillMask gets painted first).
    static Mat inPaint(const Mat& source, Mat& output, const Mat& fillMask,
        const Mat& sourceMask, const Mat& validMask);

 private:
    // Whether to stop at each iteration, write out intermediate results
    // and wait for user input to continue
    static bool STEP_ITERATION;

    // Whether to write the output at each iteration to progress/ to
    // build a time lapse of the inpainting process
    static bool WRITE_PROGRESS;

    // Maximum fraction of points to keep from low res comparison
    static float CULL_FACTOR;

    // WindowSize limits for searching. Disables variable window size
    // if they are equal
    static int MIN_WIN_SIZE;
    static int MAX_WIN_SIZE;

    // Tuning factor that determines the importance of regions that were
    // inpainted when downscaled previously
    static float LAMBDA;

    // Minimum ratio between minWindowSize and image dimension to accept
    static float MIN_SIZE_FAC;

    // Scharr convolution kernels used to find gradient
    static const Mat G_X;
    // y kernel is just the transpose of x
    static const Mat G_Y;

 private:

    // Given the location from px and the replacement patch, fills the target patch
    // in output and fillMask and updates the confidence map subject to \p validMask.
    static void inPaintPatch(const _Pixel& px, const Mat& replacement, const Mat& validMask,
        Mat& output, Mat& fillMask, Mat& confidence, const float& diff);
    // Extract a mat region with border extrapolation if the region exceeds
    // the source's bounds
    static Mat extract(const Mat& source, const int& x, const int& y,
        const int& width, const int& height);
    // Calculates the mean of squared differences (MSD) of two images given
    // a mask of valid comparisons
    static float calcDifference(const Mat& source1, const Mat& source2, const Mat& mask, const float& lambda, const Mat& validMask);
    // Finds the edge of the fillMask and writes it to a vector<Point>
    // for further calculations. It can be limited to a specific area
    // using the roi parameter to cull the vector of unchanged pixels
    // that do not require recalculation
    static void findFillFront(const Mat& fillMask, const Mat& valid, vector<_Pixel>& fillFront);
    // Calculates the confidence term (per the Criminisi algorithm)
    static float findConfidence(const Point& px, Mat& confidence);
    // Finds the fill front normal at a coordinate
    static Vec2f findN_p(const Point& px, const Mat& fillMask);
    // Finds the orthogonal vector to the image gradient.
    // Because we have masked out terms it tries to find the gradient of
    // a nearby patch that has the most number of completely known terms
    static Vec2f findI_p(const Point& coord, const Mat& fillMask, const Mat& source);
    // Finds the data term using N_p and I_p
    static float findData(const Point& px, const Mat& fillMask, const Mat& source);
    // Actual worker unit for finding best exemplar patches using multiple threads
    static void* findExemplarWorker(void* arguments);
    // Returns a colour from blue (0) to red (1) given an input range [0,1]
    static Vec3b heatMap(const double& val);
};
