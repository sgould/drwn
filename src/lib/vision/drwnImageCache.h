/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImageCache.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <map>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;

// drwnImageCache ------------------------------------------------------------
//! Caches images in memory up to a maximum number of images or memory limit.
//!
//! The cache is initialized with a list of filenames. Images can then be access
//! by index (fast) or by filename (slower). Images must be locked before being
//! used and then unlocked so that the cache can free them if memory limits
//! are exceeded. Images are removed under a least-recently-used (i.e., unlocked)
//! policy. The same image can be locked multiple times. Each lock() must have a
//! corresponding unlock(). The following code shows example usage:
//! \code
//!   vector<string> filenames = drwnDirectoryListing(directoryName, ".jpg");
//!   drwnImageCache cache(filenames);
//!
//!   while (stillRunning()) {
//!      unsigned int indxA = getFirstImageIndex();
//!      unsigned int indxB = getSecondImageIndex();
//!
//!      cache.lock(indxA);
//!      cache.lock(indxB);
//!
//!      const cv::Mat imgA = cache.get(indxA);
//!      const cv::Mat imgB = cache.get(indxB);
//!      processImages(imgA, imgB);
//!
//!      cache.unlock(indxB);
//!      cache.unlock(indxA);
//!   }
//! \endcode
//!
//! The class can be initialized in big memory mode. In this mode all images are
//! pre-loaded and memory/size limits are ignored, resulting in much faster calls
//! to the lock() and unlock() functions, but may consume all available memory.
//! Big memory mode can be enabled/disabled explicitly via the class constructor.
//! Its default setting is controlled via the \p BIG_MEMORY static data member.
//!
//! \note This class is thread-safe (under Linux).
//!
//! \sa drwnImagePyramidCache

class drwnImageCache {
protected:
    vector<string> _filenames; //!< image filenames

    vector<cv::Mat> _images;   //!< loaded images
    vector<list<unsigned>::iterator> _free_list_map; //!< ref to location in _free_list
    vector<unsigned> _lock_counter; //!< allows for multiple locks

    //! index of images that can be safely released in least-recently-used order
    list<unsigned> _free_list;

    bool _bGreyImages;     //!< store images in greyscale (instead of RGB)
    bool _bBigMemoryModel; //!< load all images into memory (ignores MAX_IMAGES and MAX_MEMORY)
    size_t _imagesLoaded;  //!< number of images loaded
    size_t _memoryUsed;    //!< bytes used by loaded images

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_t _mutex; // thread safety
#endif

    // debug statistics
    unsigned _dbImagesLocked;
    unsigned _dbImagesLoaded;
    size_t _dbMaxMemUsed;

public:
    static size_t MAX_IMAGES; //!< maximum number of images in memory at any one time
    static size_t MAX_MEMORY; //!< maximum number of bytes used at any one time
    static bool GREY_IMAGES;  //!< default setting for caching images in greyscale
    static bool BIG_MEMORY;   //!< default setting for big memory mode

public:
    //! constructor
    drwnImageCache(bool bGreyImages = GREY_IMAGES, bool bBigMemory = BIG_MEMORY);
    //! constructor with filename initialization
    drwnImageCache(const vector<string>& filenames, bool bGreyImages = GREY_IMAGES,
        bool bBigMemory = BIG_MEMORY);
    //! destructor
    virtual ~drwnImageCache();

    //! initializes the cache with a list of filenames (cache must be clear)
    void initialize(const vector<string>& filenames);

    //! adds a filename to the list of filenames managed by the cache
    void append(const string& filename);

    //! Clear all images from the cache (cannot be called if some images are still
    //! locked). Does nothing if in big memory mode.
    void clear();

    //! returns true if the cache is empty (but may still be initialized with filenames)
    bool empty() const { return (_imagesLoaded == 0); }
    //! returns number of images stored in the cache
    size_t size() const { return _imagesLoaded; }
    //! returns number of locked images in the cache
    size_t numLocked() const { return _imagesLoaded - _free_list.size(); }
    //! returns memory used by in-memory images
    size_t memory() const { return _memoryUsed; }

    //! return the filename for image \p indx
    const string& filename(unsigned indx) const { return _filenames[indx]; }

    //! returns the index for filename (slow)
    int index(const string& filename) const {
        vector<string>::const_iterator it = find(_filenames.begin(), _filenames.end(), filename);
        if (it == _filenames.end()) return -1;
        return (int)(it - _filenames.begin());
    }

    //! lock an image for use (loads if not already in the cache)
    void lock(unsigned indx);
    //! lock an image for use (loads if not already in the cache)
    void lock(const string& filename) { lock((unsigned)index(filename)); }

    //! returns an image (which must first be locked)
    inline const cv::Mat& get(unsigned indx) const { return _images[indx]; }
    //! returns an image (which must first be locked)
    inline const cv::Mat& get(const string& filename) const {
        return _images[(unsigned)index(filename)];
    }

    //! marks an image as free
    void unlock(unsigned indx);
    //! marks an image as free
    void unlock(const string& filename) { unlock((unsigned)index(filename)); }

    //! copies an image without locking it (caller must free the image)
    cv::Mat copy(unsigned indx);
    //! copies an image without locking it (caller must free the image)
    cv::Mat copy(const string& filename) { return copy((unsigned)index(filename)); }

protected:
    //! checks memory and count limits and frees images if exceeded
    void enforceLimits();

    //! loads an image (can be overrided to perform some pre-processing)
    virtual cv::Mat load(const string& filename) const;
};

