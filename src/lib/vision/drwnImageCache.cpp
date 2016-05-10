/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImageCache.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnIO.h"

#include "drwnImageCache.h"

using namespace std;

// drwnImageCache ------------------------------------------------------------

size_t drwnImageCache::MAX_IMAGES = 10000;
size_t drwnImageCache::MAX_MEMORY = 500000000;
bool drwnImageCache::GREY_IMAGES = false;
bool drwnImageCache::BIG_MEMORY = false;

drwnImageCache::drwnImageCache(bool bGreyImages, bool bBigMemory) :
    _bGreyImages(bGreyImages), _bBigMemoryModel(bBigMemory), _imagesLoaded(0), _memoryUsed(0),
    _dbImagesLocked(0), _dbImagesLoaded(0), _dbMaxMemUsed(0)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
}

drwnImageCache::drwnImageCache(const vector<string>& filenames, bool bGreyImages, bool bBigMemory) :
    _bGreyImages(bGreyImages), _bBigMemoryModel(bBigMemory), _imagesLoaded(0), _memoryUsed(0),
    _dbImagesLocked(0), _dbImagesLoaded(0), _dbMaxMemUsed(0)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
    initialize(filenames);
}

drwnImageCache::~drwnImageCache()
{
    clear();

    if (_dbImagesLocked != 0) {
        DRWN_LOG_METRICS("drwnImageCache locked " << _dbImagesLocked << " and loaded "
            << _dbImagesLoaded << " images (ratio: "
            << (double)_dbImagesLoaded / (double)_dbImagesLocked << ")");
        DRWN_LOG_METRICS("drwnImageCache used " << drwn::bytesToString(_dbMaxMemUsed));
    }
}

void drwnImageCache::initialize(const vector<string>& filenames)
{
    DRWN_ASSERT_MSG(_imagesLoaded == 0, "cache is not empty");
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    _filenames = filenames;
    _images.resize(filenames.size());
    _free_list_map.resize(filenames.size());
    _lock_counter.resize(filenames.size());
    DRWN_ASSERT(_free_list.empty());

    fill(_images.begin(), _images.end(), cv::Mat());
    fill(_free_list_map.begin(), _free_list_map.end(), _free_list.end());
    fill(_lock_counter.begin(), _lock_counter.end(), 0);

    if (_bBigMemoryModel) {
        for (unsigned indx = 0; indx < _filenames.size(); indx++) {
            // load image
            _images[indx] = load(_filenames[indx]);

            // update memory usage
            _imagesLoaded += 1;
            _memoryUsed += _images[indx].channels() * _images[indx].cols *
                _images[indx].rows * sizeof(unsigned char);

            _dbImagesLoaded += 1;
            _dbImagesLocked += 1;
        }

        _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImageCache::append(const string& filename)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    _filenames.push_back(filename);
    _images.push_back(cv::Mat());
    _free_list_map.push_back(_free_list.end());
    _lock_counter.push_back(0);

    if (_bBigMemoryModel) {
        const unsigned indx = _filenames.size() - 1;
        // load image
        _images[indx] = load(_filenames[indx]);

        // update memory usage
        _imagesLoaded += 1;
        _memoryUsed += _images[indx].channels() * _images[indx].cols *
            _images[indx].rows * sizeof(unsigned char);

        _dbImagesLoaded += 1;
        _dbImagesLocked += 1;
        _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImageCache::clear()
{
    if (_bBigMemoryModel) return;

    DRWN_ASSERT_MSG(_free_list.size() == _imagesLoaded,
        "images are still locked (" << _free_list.size() << ", " << _imagesLoaded << ")");

    for (list<unsigned>::iterator it = _free_list.begin(); it != _free_list.end(); ++it) {
        _images[*it] = cv::Mat();
        _free_list_map[*it] = _free_list.end();
    }
    _free_list.clear();

    _imagesLoaded = 0;
    _memoryUsed = 0;
}

void drwnImageCache::lock(unsigned indx)
{
    DRWN_ASSERT(indx < _filenames.size());
    if (_bBigMemoryModel) return;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    _dbImagesLocked += 1;
    _lock_counter[indx] += 1;

    // check if image is already in the cache
    if (_images[indx].data != NULL) {
        // if image is not already locked add it to the free list
        if (_free_list_map[indx] != _free_list.end()) {
            _free_list.erase(_free_list_map[indx]);
            _free_list_map[indx] = _free_list.end();
        }

        // return the image
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_unlock(&_mutex);
#endif
        return;
    }

    // image is not in the cache so load it
    _dbImagesLoaded += 1;
    DRWN_LOG_DEBUG("adding " << _filenames[indx] << " to drwnImageCache");
    _images[indx] = load(_filenames[indx]);
    _free_list_map[indx] = _free_list.end();

    // update memory usage
    _imagesLoaded += 1;
    _memoryUsed += _images[indx].channels() * _images[indx].cols *
        _images[indx].rows * sizeof(unsigned char);
    enforceLimits();

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImageCache::unlock(unsigned indx)
{
    DRWN_ASSERT(indx < _filenames.size());
    if (_bBigMemoryModel) return;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    DRWN_ASSERT_MSG(_lock_counter[indx] > 0, "image " << _filenames[indx] << " is not locked");
    DRWN_ASSERT((_images[indx].data != NULL) && (_free_list_map[indx] == _free_list.end()));

    _lock_counter[indx] -= 1;
    if (_lock_counter[indx] == 0) {
        _free_list.push_front(indx);
        _free_list_map[indx] = _free_list.begin();

        enforceLimits();
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

cv::Mat drwnImageCache::copy(unsigned indx)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    cv::Mat img;
    if (_images[indx].data != NULL) {
        img = _images[indx].clone();
    } else {
        img = load(_filenames[indx]);
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif

    return img;
}

void drwnImageCache::enforceLimits()
{
    _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);

    while (!_free_list.empty()) {
        if ((_imagesLoaded <= MAX_IMAGES) && (_memoryUsed <= MAX_MEMORY)) break;

        unsigned indx = _free_list.back();
        DRWN_LOG_DEBUG("removing " << _filenames[indx] << " from drwnImageCache");
        _memoryUsed -= _images[indx].channels() * _images[indx].cols *
            _images[indx].rows * sizeof(unsigned char);
        _imagesLoaded -= 1;

        _images[indx] = cv::Mat();
        _free_list_map[indx] = _free_list.end();

        _free_list.pop_back();
    }
}

cv::Mat drwnImageCache::load(const string& filename) const
{
    cv::Mat img = cv::imread(filename, _bGreyImages ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, "could not read image from " << filename);
    return img;
}

// drwnImageCacheConfig -----------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnImageCache
//! \b maxImages :: maximum images stored in cache (default: 1000)\n
//! \b maxMemory :: maximum memory (in bytes) used by cache (default: 500MB)\n
//! \b greyImages :: store images in greyscale (default: false)\n
//! \b bigMemory :: big memory mode (default: false)\n

class drwnImageCacheConfig : public drwnConfigurableModule {
public:
    drwnImageCacheConfig() : drwnConfigurableModule("drwnImageCache") { }
    ~drwnImageCacheConfig() { }

    void usage(ostream &os) const {
        os << "      maxImages       :: maximum images stored in cache (default: "
           << drwnImageCache::MAX_IMAGES << ")\n";
        os << "      maxMemory       :: maximum memory (in bytes) used by cache (default: "
           << drwnImageCache::MAX_MEMORY << ")\n";
        os << "      greyImages      :: greyscale images (default: "
           << (drwnImageCache::GREY_IMAGES ? "true" : "false") << ")\n";
        os << "      bigMemory       :: big memory mode (default: "
           << (drwnImageCache::BIG_MEMORY ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxImages")) {
            drwnImageCache::MAX_IMAGES = (size_t)std::max(0.0, atof(value));
        } else if (!strcmp(name, "maxMemory")) {
            drwnImageCache::MAX_MEMORY = (size_t)std::max(0.0, atof(value));
        } else if (!strcmp(name, "greyImages") || !strcmp(name, "grayImages")) {
            drwnImageCache::GREY_IMAGES = drwn::trueString(value);
        } else if (!strcmp(name, "bigMemory")) {
            drwnImageCache::BIG_MEMORY = drwn::trueString(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnImageCacheConfig gImageCacheConfig;
