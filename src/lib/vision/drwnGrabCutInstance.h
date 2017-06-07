/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGrabCutInstance.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**		Kevin Guo <Kevin.Guo@nicta.com.au>
**
*****************************************************************************/

#pragma once
#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

#include "drwnColourHistogram.h"
#include "drwnPixelNeighbourContrasts.h"

using namespace std;
using namespace Eigen;

// drwnGrabCutInstance ------------------------------------------------------
//! Implements the grabCut algorithm of Rother et al., SIGGRAPH 2004 for
//! figure/ground segmentation.
//!
//! All images are assumed to be 3-channel, 8-bit. All masks are assumed to
//! be 8-bit unsigned (single channel) with values: 0 to mean background,
//! 64 unknown (initialize as background), 128 unknown (initialize as foreground),
//! 255 foreground. Other values are treated as unknown but not used to
//! learn colour models.
//!
//! When learning foreground colour models pixels marked as MASK_FG, MASK_C_FG
//! and MASK_C_BOTH are used. Likewise for background colour models (MASK_BG,
//! MASK_C_BG and MASK_C_BOTH). During inference pixels marked as MASK_FG or
//! MASK_BG are forced to take foreground or background labels, respectively.
//!
//! \sa \ref drwnAppGrabCut "grabCut Application"

class drwnGrabCutInstance {
public:
    static const unsigned char MASK_FG = 0xff;       //!< foreground mask
    static const unsigned char MASK_BG = 0x00;       //!< background mask
    static const unsigned char MASK_C_FG = 0x80;     //!< foreground colour mask
    static const unsigned char MASK_C_BG = 0x40;     //!< background colour mask
    static const unsigned char MASK_C_BOTH = 0xc0;   //!< both colour mask
    static const unsigned char MASK_C_NONE = 0x20;   //!< neither colour mask

    static bool bVisualize;    //!< visualize output
    static int maxIterations;  //!< maximum number of inference iterations

public:
    // meta parameters
    string name;               //!< instance name (if available)

protected:
    // instance definition
    cv::Mat _img;              //!< the image
    cv::Mat _trueMask;         //!< ground-truth segmentation (trimap)
    cv::Mat _mask;             //!< segmentation mask (quadmap)
    int _numUnknown;           //!< number of unknown pixels in _mask

    // cached data
    cv::Mat _unary;            //!< unary potentials, \psi_i(y_i == background)
    drwnPixelNeighbourContrasts *_pairwise; //!< pairwise potentials

    // model parameters
    double _unaryWeight;       //!< weight for unary term in grabCut model
    double _pottsWeight;       //!< weight for potts smoothness term
    double _pairwiseWeight;    //!< weight for contrast-sensitive pairwise term

public:
    //! default constructor
    drwnGrabCutInstance();
    //! copy constructor
    drwnGrabCutInstance(const drwnGrabCutInstance& instance);
    //! destructor
    virtual ~drwnGrabCutInstance();

    //! width of the image
    int width() const { return _img.cols; }
    //! height of the image
    int height() const { return _img.rows; }
    //! number of pixels in the image
    int size() const { return _img.cols * _img.rows; }
    //! number of unknown pixels in the inference mask
    int numUnknown() const { return _numUnknown; }

    // caller does not need to free returned objects
    //! return the image
    const cv::Mat& image() const { return _img; }
    //! return the mask for the true segmentation (if known)
    const cv::Mat& trueSegmentation() const { return _trueMask; }
    //! return the mask for the inferred segmentation
    const cv::Mat& segmentationMask() const { return _mask; }

    //! returns mask of all pixels not marked as MASK_FG (caller must free)
    cv::Mat knownForeground() const;
    //! returns mask of all pixels not marked as MASK_BG (caller must free)
    cv::Mat knownBackground() const;
    //! returns mask of all pixels not marked as MASK_FG or MASK_BG (caller must free)
    cv::Mat unknownPixels() const;
    //! returns mask of all pixels marked as MASK_FG or MASK_FGC (caller must free)
    cv::Mat foregroundColourMask() const;
    //! returns mask of all pixels marked as MASK_BG or MASK_BGC (caller must free)
    cv::Mat backgroundColourMask() const;

    //! returns true if the pixel at (x, y) is not foreground or background in \p mask
    bool isUnknownPixel(int x, int y, const cv::Mat& mask) const {
        const unsigned char p = mask.at<unsigned char>(y, x);
        return ((p != MASK_FG) && (p != MASK_BG));
    }
    //! returns true if the pixel at (x, y) is not foreground or background in the initialized inference mask
    bool isUnknownPixel(int x, int y) const {
        return isUnknownPixel(x, y, _mask);
    }

    //! initialize with image and bounding box region; ground-truth segmentation unknown
    //! uses \p colorModelFile to initialize colour models if given
    void initialize(const cv::Mat& img, const cv::Rect& rect, const char *colorModelFile = NULL);
    //! initialize with image and mask of inference region; ground-truth segmentation unknown
    void initialize(const cv::Mat& img, const cv::Mat& inferMask, const char *colorModelFile = NULL);
    //! initialize with image and bounding box region
    void initialize(const cv::Mat& img, const cv::Rect& rect, const cv::Mat& trueMask,
        const char *colorModelFile = NULL);
    //! initialize with image and mask of inference region
    void initialize(const cv::Mat& img, const cv::Mat& inferMask, const cv::Mat& trueMask,
        const char *colorModelFile = NULL);

    //! load colour models
    virtual void loadColourModels(const char *filename) = 0;
    //! save colour models
    virtual void saveColourModels(const char *filename) const = 0;

    //! sets unary and pairwise weights
    void setBaseModelWeights(double u, double p, double c);

    //! get unary potentials
    const cv::Mat& unaryPotentials() const { return _unary; }

    //! get pairwise potentials
    const drwnPixelNeighbourContrasts *pairwisePotentials() { return _pairwise; }

    //! compute energy contribution from the unary term
    double unaryEnergy(const cv::Mat& seg) const;

    //! compute energy contribution from the potts pairwise term
    double pottsEnergy(const cv::Mat& seg) const;

    //! compute energy contribution from the contrast-sensitive pairwise term
    double pairwiseEnergy(const cv::Mat& seg) const;

    //! compute the energy of a given segmentation
    virtual double energy(const cv::Mat& seg) const;

    //! compute the percentage of unknown pixels labeled as foreground
    double foregroundRatio(const cv::Mat& seg) const;

    //! compute the percentage of unknown pixels labeled as background
    inline double backgroundRatio(const cv::Mat& seg) const {
        return 1.0 - foregroundRatio(seg);
    }

    //! computes the loss between ground-truth and given segmentation
    double loss(const cv::Mat& seg) const;

    //! inference
    virtual cv::Mat inference();
    //! loss augmented inference (using ground-truth segmentation)
    virtual cv::Mat lossAugmentedInference();

    //! visualization
    virtual cv::Mat visualize(const cv::Mat& seg) const;

protected:
    //! free all memory
    void free();

    //! extract pixel colour as a 3-vector
    inline vector<double> pixelColour(int y, int x) const;

    //! learn a gaussian mixture model for pixels in masked region
    virtual void learnColourModel(const cv::Mat& mask, bool bForeground) = 0;

    //! update unary potential from colour models
    virtual void updateUnaryPotentials() = 0;

    //! graph-cut inference
    virtual cv::Mat graphCut(const cv::Mat& unary) const;
};

// drwnGrabCutInstanceGMM ---------------------------------------------------

class drwnGrabCutInstanceGMM : public drwnGrabCutInstance {
 public:
    static size_t maxSamples;  //!< maximum samples to use for colour models
    static int numMixtures;    //!< number of mixture components in colour models

 protected:
    drwnGaussianMixture _fgColourModel; //!< foreground colour model
    drwnGaussianMixture _bgColourModel; //!< background colour model

 public:
    //! default constructor
    drwnGrabCutInstanceGMM();
    //! destructor
    virtual ~drwnGrabCutInstanceGMM();

    //! load colour models
    void loadColourModels(const char *filename);
    //! save colour models
    void saveColourModels(const char *filename) const;

protected:
    //! learn a gaussian mixture model for pixels in masked region
    void learnColourModel(const cv::Mat& mask, bool bForeground);

    //! update unary potential from colour models
    void updateUnaryPotentials();
};

// drwnGrabCutInstanceHistogram ---------------------------------------------

class drwnGrabCutInstanceHistogram : public drwnGrabCutInstance {
 public:
    static double pseudoCounts;   //!< pseudocounts in colour histogram model
    static unsigned channelBits;  //!< number of bits per RGB colour channel

 protected:
    drwnColourHistogram _fgColourModel; //!< foreground colour model
    drwnColourHistogram _bgColourModel; //!< background colour model

 public:
    //! default constructor
    drwnGrabCutInstanceHistogram();
    //! destructor
    virtual ~drwnGrabCutInstanceHistogram();

    //! load colour models
    void loadColourModels(const char *filename);
    //! save colour models
    void saveColourModels(const char *filename) const;

protected:
    //! learn a gaussian mixture model for pixels in masked region
    void learnColourModel(const cv::Mat& mask, bool bForeground);

    //! update unary potential from colour models
    void updateUnaryPotentials();
};
