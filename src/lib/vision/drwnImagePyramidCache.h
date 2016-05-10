/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImagePyramidCache.h
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

// drwnImagePyramidCache -----------------------------------------------------
//! Caches image pyramids in main memory up to a maximum number of images or
//! memory limit.
//!
//! \sa drwnImageCache.

class drwnImagePyramidCache {
protected:
    vector<string> _filenames; //!< image filenames

    vector<vector<cv::Mat> > _images; //!< image pyramids
    vector<list<unsigned>::iterator> _free_list_map; //!< ref to location in _free_list
    vector<unsigned> _lock_counter; //!< allows for multiple locks

    //! index of images that can be safely released in least-recently-used order
    list<unsigned> _free_list;

    bool _bGreyImages;     //!< store images in greyscale (instead of RGB)
    bool _bBigMemoryModel; //!< load all images into memory (ignores MAX_IMAGES and MAX_MEMORY)
    size_t _imagesLoaded;  //!< number of images loaded
    size_t _memoryUsed;    //!< bytes used by loaded images

    double _downSampleRate; //!< downsampling rate
    int _minImageSize;      //!< minimum image dimension
    int _maxLevels;         //!< maximum pyramid height


#ifdef DRWN_USE_PTHREADS
    pthread_mutex_t _mutex; //!< for thread safety
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
    drwnImagePyramidCache(double downSampleRate = 0.5, int minSize = 32,
        int maxLevels = DRWN_INT_MAX, bool bGreyImages = GREY_IMAGES,
        bool bBigMemory = BIG_MEMORY);
    //! constructor with filename initialization
    drwnImagePyramidCache(const vector<string>& filenames, double downSampleRate = 0.5,
        int minSize = 32, int maxLevels = DRWN_INT_MAX, bool bGreyImages = GREY_IMAGES,
        bool bBigMemory = BIG_MEMORY);
    //! destructor
    ~drwnImagePyramidCache();

    //! initializes the cache with a list of filenames (cache must be clear)
    void initialize(const vector<string>& filenames);

    //! adds a filename to the list of filenames managed by the cache
    void append(const string& filename);

    //! Clear all images from the cache (cannot be called if some images are still
    //! locked). Does nothing if in big memory mode.
    void clear();

    //! returns true if the cache is empty (but may still be initialized with filenames)
    bool empty() const { return (_imagesLoaded == 0); }
    //! returns number of image pyramids stored in the cache
    size_t size() const { return _imagesLoaded; }
    //! returns number of locked image pyramids in the cache
    size_t numLocked() const { return _imagesLoaded - _free_list.size(); }
    //! returns memory used by in-memory image pyramids
    size_t memory() const { return _memoryUsed; }

    //! return the filename for image \p indx
    const string& filename(unsigned indx) const { return _filenames[indx]; }

    //! returns the index for filename (slow)
    int index(const string& filename) const {
        vector<string>::const_iterator it = find(_filenames.begin(), _filenames.end(), filename);
        if (it == _filenames.end()) return -1;
        return (int)(it - _filenames.begin());
    }

    //! lock an image pyramid for use (loads if not already in the cache)
    void lock(unsigned indx);
    //! lock an image pyramid for use (loads if not already in the cache)
    void lock(const string& filename) { lock((unsigned)index(filename)); }

    //! returns the number of levels in an image pyramid (which must first be locked)
    inline size_t levels(unsigned indx) const { return _images[indx].size(); }
    //! returns the number of levels in an image pyramid (which must first be locked)
    inline size_t levels(const string& filename) const {
        return _images[(unsigned)index(filename)].size();
    }

    //! returns an image (which must first be locked)
    inline const cv::Mat& get(unsigned indx, unsigned level = 0) const {
        return _images[indx][level];
    }
    //! returns an image (which must first be locked)
    inline const cv::Mat& get(const string& filename, unsigned level = 0) const {
        return _images[(unsigned)index(filename)][level];
    }

    //! marks an image as free
    void unlock(unsigned indx);
    //! marks an image as free
    void unlock(const string& filename) { unlock((unsigned)index(filename)); }

    //! copies an image without locking it (caller must free the image)
    cv::Mat copy(unsigned indx, unsigned level = 0);
    //! copies an image without locking it (caller must free the image)
    cv::Mat copy(const string& filename, unsigned level = 0) {
        return copy((unsigned)index(filename), level);
    }

protected:
    //! checks memory and count limits and frees images if exceeded
    void enforceLimits();

    //! compute memory used by image pyramid
    size_t memory(const vector<cv::Mat>& images) const;

    //! loads an image pyramid
    vector<cv::Mat> load(const string& filename) const;
};

