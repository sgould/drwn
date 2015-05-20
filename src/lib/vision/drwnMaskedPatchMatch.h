/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMaskedPatchMatch.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// eigen matrix library headers
#include "Eigen/Core"

// opencv library headers
#include "cv.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// drwnBasicPatchMatch -------------------------------------------------------
//! Implements the basic PatchMatch algorithm of Barnes et al., SIGGRAPH 2009. 
//!
//! The input and output images can have different sizes but must be CV_8U and
//! have the same number of channels (features). Nearest neighbor field is returned 
//! in \p nnfA the same size as \p imgA (but not valid around the boundary) has
//! type CV_16SC2. If empty the nearest neighbor field is initialized randomly. Call
//! repeatedly to continue the search. Returns matching costs for patch centred
//! at pixel (x, y).
//!
//! \code
//! Vec2s p = nnf.at<Vec2s>(y, x);
//! cv::Rect rA(x - patchRadius.width, y - patchRadius.height,
//!   2 * patchRadius.width + 1, 2 * patchRadius.height + 1);
//! cv::Rect rA(p[0] - patchRadius.width, p[1] - patchRadius.height,
//!   2 * patchRadius.width + 1, 2 * patchRadius.height + 1);
//! cout << rA << " in imageA matches " << rB << " in imageB\n";
//! \endcode

cv::Mat drwnBasicPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB,
    const cv::Size& patchRadius, cv::Mat& nnfA, unsigned maxIterations = 2);

// drwnSelfPatchMatch --------------------------------------------------------
//! Same as drwnBasicPatchMatch but applies a penalty to avoid matching
//! a patch to itself.

cv::Mat drwnSelfPatchMatch(const cv::Mat& imgA, const cv::Size& patchRadius,
    cv::Mat& nnfA, double illegalOverlap = 0.0, unsigned maxIterations = 2);

// drwnNNFRetarget -----------------------------------------------------------
//! Performs simple image retargetting given a nearest neighbour field (e.g.,
//! from running a PatchMatch algorithm.
//!
//! \code
//! cv::Mat img = ...;
//! cv::Mat nnf;
//! drwnSelfPatchMatch(img, nnf, cv::Size(4, 4));
//! drwnShowDebuggingImage(drwnNNFRetarget(img, nnf), "wnd", true);
//! \endcode

cv::Mat drwnNNFRetarget(const cv::Mat& img, const cv::Mat& nnf);

// drwnMaskedPatchMatch ------------------------------------------------------
//! Implements the basic PatchMatch algorithm of Barnes et al., SIGGRAPH 2009 on
//! masked images. 
//!
//! The source and target images can have different sizes but must both be CV_8U 
//! and have the same number of channels (features). The source image/feautures
//! and mask can be adjusted between calls to \p search(). Masked regions in the
//! source image are ignored. Matching to masked regions in the target image has
//! infinite cost.
//!
//! Each pixel in the returned nearest neighbour field represents the offset to
//! the optimal match for a patch of size 2 * \p patchRadius + 1 and centred on 
//! that pixel. Use the \p getMatchingPatches() to get matching regions with size
//! adjusted for pixels within \p patchRadius pixel of the boundary.
//!
//! \code
//! cv::Mat imgA;
//! cv::Mat imgB;
//! cv::Size patchRadius;
//!
//! ...
//!
//! drwnMaskedPatchMatch pm(imgA, imgB, patchRadius);
//! pm.search(10);
//! Vec2s p = pm.nnf().at<Vec2s>(y, x);
//! cv::Rect rA(x - patchRadius.width, y - patchRadius.height, 2 * patchRadius.width + 1, 2 * patchRadius.height + 1);
//! cv::Rect rB(p[0] - patchRadius.width, p[1] - patchRadius.height, 2 * patchRadius.width + 1, 2 * patchRadius.height + 1);
//! cout << rA << " in imageA matches " << rB << " in imageB\n";
//! \endcode
//!
//! Alternatively you can use the \p getMatchingPatches() function, as in:
//!
//! \code
//! pair<cv::Rect, cv::Rect> r = pm.getMatchingPatches(cv::Point(x, y));
//! cout << r.first << " in imageA matches " << r.second << " in imageB\n";
//! \endcode
//!
//! \sa \ref drwnInPaint

class drwnMaskedPatchMatch {
 public:
    static bool TRY_IDENTITY_INIT; //!< attempt identity match during initialization (default: true)
    static int DISTANCE_MEASURE;   //!< norm type for comparing patches (cv::NORM_L1 or cv::NORM_L2)
    static float HEIGHT_PENALTY;   //!< bias term added for row/height difference

 protected:
    cv::Mat _imgA;          //!< CV_8U multi-channel image (source)
    cv::Mat _imgB;          //!< CV_8U multi-channel image (target)
    cv::Mat _maskA;         //!< CV_8UC1 mask for valid source pixels
    cv::Mat _maskB;         //!< CV_8UC1 mask for valid target pixels
    cv::Mat _invmaskA;      //!< CV_8UC1 inverse of _maskA
    cv::Mat _invmaskB;      //!< CV_8UC1 inverse of _maskB

    cv::Mat _overlapA;      //!< pixels in _imgA whose patch overlaps with _invmaskA
    cv::Mat _validB;        //!< pixels in _imgB whose patch doesn't overlap with _maskB

    cv::Size _patchRadius;  //!< size of patch is 2 * _patchRadius + 1
    cv::Mat _nnfA;          //!< CV_16S2 nearest neighbour field (size _imgA)
    cv::Mat _costsA;        //!< CV_32FC1 field of match costs (same size as _nnfA)

    int _iterationCount;    //!< total number of iterations since initialization
    cv::Mat _lastChanged;   //!< last iteration that given patch was changed

 public:
    //! construct without masks and square patches
    drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, unsigned patchRadius);
    //! construct without masks and arbitrary patches
    drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Size& patchRadius);
    //! costruct with matching masks (empty for no mask) and square patches
    drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Mat& maskA,
        const cv::Mat& maskB, unsigned patchRadius);
    //! costruct with matching masks (empty for no mask) and arbitrary patches
    drwnMaskedPatchMatch(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Mat& maskA,
        const cv::Mat& maskB, const cv::Size& patchRadius);
    //! destructor
    virtual ~drwnMaskedPatchMatch() { /* do nothing */ }

    //! return the source image
    const cv::Mat& getSourceImage() const { return _imgA; }
    //! return the target image
    const cv::Mat& getTargetImage() const { return _imgB; }
    //! return the source mask
    const cv::Mat& getSourceMask() const { return _maskA; }
    //! return the target mask
    const cv::Mat& getTargetMask() const { return _maskB; }
    //! return the patch size
    cv::Size getPatchSize() const { return cv::Size(2 * _patchRadius.width + 1, 2 * _patchRadius.height + 1); }
    //! return the size of the nearest neighbour field
    cv::Size getFieldSize() const { return _nnfA.size(); }
    //! Return the best matching pair of patches (source and destination) for source patch
    //! centred at \p ptA. Points close to the boundary have their size adjusted.
    std::pair<cv::Rect, cv::Rect> getMatchingPatches(const cv::Point& ptA) const;

    //! initialize the matches with default patch size
    void initialize() { initialize(_patchRadius); }
    //! initialize the matches given patchRadius
    void initialize(const cv::Size& patchRadius);
    //! initialize the matches given nearest neighbour field (patch size is unchanged)
    void initialize(const cv::Mat& nnf);

    //! Search for better matches and return updated nearest neighbour field. The
    //! nearest neighbour field maps is indexed by the centre pixel of the patch.
    const cv::Mat& search(unsigned maxIterations = 1) {
        return search(cv::Rect(0, 0, _nnfA.cols, _nnfA.rows), maxIterations);
    }
    //! Search for better matches on subregion of the image and return updated
    //! nearest neighbour field. Only regions whose centre pixel appear in
    //! \p roiToUpdate will be changed.
    const cv::Mat& search(cv::Rect roiToUpdate, unsigned maxIterations = 1);

    //! return the current nearest neighbour field
    const cv::Mat& nnf() const { return _nnfA; }
    //! return (matching) costs (CV_32FC1) associated with the current nearest neighbour field
    const cv::Mat& costs() const { return _costsA; }
    //! return the current match energy
    double energy() const { return cv::norm(_costsA, cv::NORM_L1, _maskA); }

    //! Copy over a region of the source image and unmask the region copied into.
    //! Alpha-blends with the masked area.
    void modifySourceImage(const cv::Rect& roi, const cv::Mat& img, double alpha = 0.0);
    //! Copy over a region of the source image from the target image and unmask
    //! the region copied into. Alpha-blends with the masked area.
    void modifySourceImage(const cv::Rect& roiA, const cv::Rect& roiB, double alpha = 0.0);
    //! Copy over a region of the target image (_imgB) and unmask the region
    //! copied into. Alpha-blends with the masked area.
    void modifyTargetImage(const cv::Rect& roi, const cv::Mat& img, double alpha = 0.0);
    //! Copy over a region of the target image (_imgB) from itself and unmask the 
    //! region copied into. Alpha-blends with the masked area.
    void modifyTargetImage(const cv::Rect& roiA, const cv::Rect& roiB, double alpha = 0.0);
    //! Expand the target mask (_maskB) by dilating pixels around the boundary
    //! with the unmasked region. The expansion kernel is a rectangle of size
    //! 2 * \p radius + 1.
    void expandTargetMask(unsigned radius = 1);

    //! \todo Add modifySourceImage and modifyTargetImage functions based on a mask

    //! visualize the nearest neighbour field
    cv::Mat visualize() const;

 protected:
    //! attempt and update a match candidate
    bool update(const cv::Point& ptA, const cv::Point& ptB);
    //! scores a match
    float score(const cv::Point& ptA, const cv::Point& ptB) const;

    //! calculate nnf pixels affected by a mask or image modification
    cv::Rect affectedRegion(const cv::Rect& modified) const {
        const int y1 = std::max(modified.y - _patchRadius.height, _patchRadius.height);
        const int x1 = std::max(modified.x - _patchRadius.width, _patchRadius.width);
        const int y2 = std::min(modified.y + modified.height + _patchRadius.height, _nnfA.rows - _patchRadius.height);
        const int x2 = std::min(modified.x + modified.width + _patchRadius.width, _nnfA.cols - _patchRadius.width);
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }

    //! update scores on whole image
    void rescore() { rescore(cv::Rect(0, 0, _imgA.rows, _imgA.cols)); }
    //! Update scores on given region. All affected scores are updated, that is,
    //! any score that was calculated from a patch that overlaps with \p roi.
    void rescore(const cv::Rect& roi);

    //! compute valid pixels
    void cacheValidPixels();
    //! Update the validity of pixels that occur because of a change to the
    //! source mask on a given region (i.e., any pixel whose corresponding
    //! patch overlaps with the region).
    void updateValidPixels(const cv::Rect& roi);
};
