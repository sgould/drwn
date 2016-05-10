/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSuperpixelContainer.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <set>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;

// drwnSuperpixelContainer -----------------------------------------------------
//! Holds multiple oversegmentations for a given image.
//!
//! The container is populated with segment maps with the same dimensions as
//! the image. A map is an integer array, each entry corresponding to a pixel
//! in the image. Maps can define more than one segment/superpixel. A segment
//! is defined as all pixels (in a map) labeled with the same non-negative
//! integer. Pixels labeled with negative integers are ignored. Maps are 
//! renumbered when added to the container. Member functions allow you to
//! retrieve the mask of each segment/superpixel or the (renumbered) maps.
//!
//! The following code snippet shows how the container can be used to iterate
//! through different superpixels:
//! \code
//!     // load an image three different oversegmentations
//!     cv::Mat img = cv::imread("image.jpg");
//!
//!     drwnSuperpixelContainer container;
//!     container.loadSuperpixels("oversegmentation_1.txt");
//!     container.loadSuperpixels("oversegmentation_2.txt");
//!     container.loadSuperpixels("oversegmentation_3.txt");
//!
//!     // iterate through the superpixels and display them
//!     cv::Mat segMask;
//!     for (int sedId = 0; segId < container.size(); segId++) {
//!         // get segmentation mask for the superpixel
//!         segMask = container.mask(segId, segMask);
//!         // draw a white and red boundary around the superpixel on a temporary image
//!         cv::Mat canvas = img.clone();
//!         drwnDrawRegionBoundaries(canvas, segMask, CV_RGB(255, 255, 255), 3);
//!         drwnDrawRegionBoundaries(canvas, segMask, CV_RGB(255, 0, 0), 1);
//!         // show the image and superpixel
//!         drwnShowDebuggingImage(canvas, string("superpixel"), true);
//!     }
//! \endcode
//!
//! The class provides a drwnPersistentStorage interface. Superpixel region data
//! is stored run-length encoded in binary format. This is useful for caching
//! superpixel calculations. The following code snippet shows an example of this
//! functionality.
//! \code
//!     // load an image three different oversegmentations
//!     cv::Mat img = cv::imread("image.jpg");
//!
//!     drwnSuperpixelContainer container;
//!     if (drwnFileExists("cache.bin")) {
//!         // load cached superpixels
//!         ifstream ifs("cache.bin", ios::binary);
//!         container.read(ifs);
//!         ifs.close();
//!     } else {
//!         // compute superpixels
//!         container.addSuperpixels(drwnFastSuperpixels(img, 32));
//!         container.addSuperpixels(drwnFastSuperpixels(img, 16));
//!
//!         // cache to file
//!         ofstream ofs("cache.bin", ios::binary);
//!         container.write(ofs);
//!         ofs.close();
//!    }
//! \endcode
//!
//! The superpixels in stored drwnSuperpixelContainer objects can be
//! accessed in Matlab via the mexLoadSuperpixels and mexSaveSuperpixels
//! application wrappers.

class drwnSuperpixelContainer : public drwnPersistentRecord {
 protected:
    vector<int> _start;        //!< start index for each _map
    vector<int> _nsegs;        //!< number of segments in each _map
    vector<cv::Mat> _maps;     //!< superpixel maps (-1 if pixel is not part of a map)
    vector<int> _pixels;       //!< size of each superpixel
    vector<cv::Rect> _bboxes;  //!< bounding boxes for each superpixel

 public:
    //! default constructor
    drwnSuperpixelContainer();
    //! copy constructor
    drwnSuperpixelContainer(const drwnSuperpixelContainer& container);
    ~drwnSuperpixelContainer();

    //! clears all oversegmentations (and releases memory)
    void clear();
    //! returns true if their are no superpixels
    inline bool empty() const { return _maps.empty(); }
    //! returns the number of superpixels
    inline int size() const { return _maps.empty() ? 0 : _start.back() + _nsegs.back(); }
    //! returns the number of pixels within the given superpixel
    inline int pixels(unsigned segId) const { return _pixels[segId]; }
    //! returns the image width
    inline int width() const { return _maps.empty() ? 0 : _maps[0].cols; }
    //! returns the image height
    inline int height() const { return _maps.empty() ? 0 : _maps[0].rows; }
    //! returns the maximum number of superpixels to which any pixel can belong
    inline int channels() const { return (int)_maps.size(); }
    //! returns number of bytes stored (approximately)
    inline size_t memory() const {
        return _maps.size() * width() * height() * sizeof(int) + 3 * size() * sizeof(int);
    }

    // persistent storage interface
    virtual size_t numBytesOnDisk() const;
    virtual bool write(ostream& os) const;
    virtual bool read(istream& is);

    //! load superpixels from file (.png or .txt)
    void loadSuperpixels(const char *filename);
    //! add 32-bit oversegmentation (container does not clone)
    void addSuperpixels(cv::Mat superpixels);
    //! copy 32-bit oversegmentation (container clones argument)
    void copySuperpixels(const cv::Mat& superpixels);

    //! return 8-bit mask for given superpixel
    cv::Mat mask(unsigned segId) const;
    //! return 8-bit mask for given superpixel (use supplied matrix if given)
    cv::Mat& mask(unsigned segId, cv::Mat& m) const;

    //! returns set of superpixel indices to which the given pixel belongs
    set<unsigned> parents(int x, int y) const;
    //! returns all overlapping or 8-connected superpixels to a given superpixel
    set<unsigned> neighbours(unsigned segId) const;
    //! returns the bounding box for a superpixel
    const cv::Rect& boundingBox(unsigned segId) const { return _bboxes[segId]; }
    //! calculates and returns centroid for a superpixel
    cv::Point centroid(unsigned segId) const;

    //! return 32-bit intersection of superpixels
    cv::Mat intersection() const;

    //! filters small superpixels and returns number of superpixels removed
    int removeSmallSuperpixels(unsigned minSize = 2);
    //! filters superpixels that do not overlap with supplied mask
    int removeUnmaskedSuperpixels(const cv::Mat& mask, double areaOverlap = 0.5);
    //! filters superpixels matching the given set of segIds
    int removeSuperpixels(const std::set<unsigned>& segIds);

    //! visualize superpixels
    cv::Mat visualize(const cv::Mat& img, bool bColourById = false) const;
    //! visualize superpixels with specific colours
    cv::Mat visualize(const cv::Mat& img, const vector<cv::Scalar>& colors, double alpha = 1.0) const;

    //! assignment operator
    drwnSuperpixelContainer& operator=(const drwnSuperpixelContainer& container);
    //! access the \p i-th superpixel map as CV_32SC1
    const cv::Mat& operator[](unsigned int indx) const { return _maps[indx]; }
};



