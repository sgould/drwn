/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImagePyramidCache.cpp
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

#include "drwnOpenCVUtils.h"
#include "drwnImagePyramidCache.h"

using namespace std;

// drwnImagePyramidCache -----------------------------------------------------

size_t drwnImagePyramidCache::MAX_IMAGES = 10000;
size_t drwnImagePyramidCache::MAX_MEMORY = 500000000;
bool drwnImagePyramidCache::GREY_IMAGES = false;
bool drwnImagePyramidCache::BIG_MEMORY = false;

drwnImagePyramidCache::drwnImagePyramidCache(double downSampleRate, int minSize,
    int maxLevels, bool bGreyImages, bool bBigMemory) :
    _bGreyImages(bGreyImages), _bBigMemoryModel(bBigMemory), _imagesLoaded(0), _memoryUsed(0),
    _downSampleRate(downSampleRate), _minImageSize(minSize), _maxLevels(maxLevels),
    _dbImagesLocked(0), _dbImagesLoaded(0), _dbMaxMemUsed(0)
{
    DRWN_ASSERT_MSG((downSampleRate > 0.0) && (downSampleRate < 1.0),
        "invalid downSampleRate " << downSampleRate);
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
}

drwnImagePyramidCache::drwnImagePyramidCache(const vector<string>& filenames,
    double downSampleRate, int minSize, int maxLevels, bool bGreyImages, bool bBigMemory) :
    _bGreyImages(bGreyImages), _bBigMemoryModel(bBigMemory), _imagesLoaded(0), _memoryUsed(0),
    _downSampleRate(downSampleRate), _minImageSize(minSize), _maxLevels(maxLevels),
    _dbImagesLocked(0), _dbImagesLoaded(0), _dbMaxMemUsed(0)
{
    DRWN_ASSERT_MSG((downSampleRate > 0.0) && (downSampleRate < 1.0),
        "invalid downSampleRate " << downSampleRate);
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
    // initialize
    initialize(filenames);
}

drwnImagePyramidCache::~drwnImagePyramidCache()
{
    clear();

    if (_dbImagesLocked != 0) {
        DRWN_LOG_METRICS("drwnImagePyramidCache locked " << _dbImagesLocked << " and loaded "
            << _dbImagesLoaded << " images (ratio: "
            << (double)_dbImagesLoaded / (double)_dbImagesLocked << ")");
        DRWN_LOG_METRICS("drwnImageCache used " << bytesToString(_dbMaxMemUsed));
    }
}

void drwnImagePyramidCache::initialize(const vector<string>& filenames)
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

    fill(_images.begin(), _images.end(), vector<cv::Mat>());
    fill(_free_list_map.begin(), _free_list_map.end(), _free_list.end());
    fill(_lock_counter.begin(), _lock_counter.end(), 0);

    if (_bBigMemoryModel) {
        for (unsigned indx = 0; indx < _filenames.size(); indx++) {
            // load image pyramid
            _images[indx] = load(_filenames[indx]);
            // update memory usage
            _imagesLoaded += 1;
            _memoryUsed += memory(_images[indx]);

            _dbImagesLoaded += 1;
            _dbImagesLocked += 1;
        }

        _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImagePyramidCache::append(const string& filename)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    _filenames.push_back(filename);
    _images.push_back(vector<cv::Mat>());
    _free_list_map.push_back(_free_list.end());
    _lock_counter.push_back(0);

    if (_bBigMemoryModel) {
        const unsigned indx = _filenames.size() - 1;
        // load image pyramid
        _images[indx] = load(_filenames[indx]);
        // update memory usage
        _imagesLoaded += 1;
        _memoryUsed += memory(_images[indx]);

        _dbImagesLoaded += 1;
        _dbImagesLocked += 1;
        _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImagePyramidCache::clear()
{
    if (_bBigMemoryModel) return;

    DRWN_ASSERT_MSG(_free_list.size() == _imagesLoaded,
        "images are still locked (" << _free_list.size() << ", " << _imagesLoaded << ")");

    for (list<unsigned>::iterator it = _free_list.begin(); it != _free_list.end(); ++it) {
        _images[*it].clear();
        _free_list_map[*it] = _free_list.end();
    }
    _free_list.clear();

    _imagesLoaded = 0;
    _memoryUsed = 0;
}

void drwnImagePyramidCache::lock(unsigned indx)
{
    DRWN_ASSERT(indx < _filenames.size());
    if (_bBigMemoryModel) return;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    _dbImagesLocked += 1;
    _lock_counter[indx] += 1;

    // check if image is already in the cache
    if (!_images[indx].empty()) {
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
    DRWN_LOG_DEBUG("adding " << _filenames[indx] << " to drwnImagePyramidCache");
    _images[indx] = load(_filenames[indx]);
    _free_list_map[indx] = _free_list.end();

    // update memory usage
    _imagesLoaded += 1;
    _memoryUsed += memory(_images[indx]);
    enforceLimits();

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}

void drwnImagePyramidCache::unlock(unsigned indx)
{
    DRWN_ASSERT(indx < _filenames.size());
    if (_bBigMemoryModel) return;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    DRWN_ASSERT_MSG(_lock_counter[indx] > 0, "image " << _filenames[indx] << " is not locked");
    DRWN_ASSERT((!_images[indx].empty()) && (_free_list_map[indx] == _free_list.end()));

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

cv::Mat drwnImagePyramidCache::copy(unsigned indx, unsigned level)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    cv::Mat img;
    if (!_images[indx].empty()) {
        DRWN_ASSERT(level < _images[indx].size());
        img = _images[indx][level].clone();
    } else {
        vector<cv::Mat> pyr = load(_filenames[indx]);
        DRWN_ASSERT(level < pyr.size());
        img = pyr[level];
    }

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif

    return img;
}

void drwnImagePyramidCache::enforceLimits()
{
    _dbMaxMemUsed = std::max(_dbMaxMemUsed, _memoryUsed);

    while (!_free_list.empty()) {
        if ((_imagesLoaded <= MAX_IMAGES) && (_memoryUsed <= MAX_MEMORY)) break;

        unsigned indx = _free_list.back();
        DRWN_LOG_DEBUG("removing " << _filenames[indx] << " from drwnImagePyramidCache");
        _memoryUsed -= memory(_images[indx]);
        _imagesLoaded -= 1;

        _images[indx].clear();
        _free_list_map[indx] = _free_list.end();

        _free_list.pop_back();
    }
}

size_t drwnImagePyramidCache::memory(const vector<cv::Mat>& images) const
{
    size_t m = 0;
    for (unsigned i = 0; i < images.size(); i++) {
        if (images[i].data != NULL) {
            m += images[i].channels() * images[i].rows * images[i].cols * sizeof(unsigned char);
        }
    }

    return m;
}

vector<cv::Mat> drwnImagePyramidCache::load(const string& filename) const
{
    cv::Mat img = cv::imread(filename, _bGreyImages ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, "could not read image from " << filename);
    DRWN_ASSERT_MSG((img.cols >= _minImageSize) && (img.rows >= _minImageSize),
        "image " << filename << " is too small " << toString(img));

    vector<cv::Mat> pyramid(1, img);    
    for (int i = 0; i < _maxLevels; i++) {
        cv::Size s(_downSampleRate * img.cols, _downSampleRate * img.rows);
        if ((s.width < _minImageSize) || (s.height < _minImageSize))
            break;

        cv::Mat downSampledImage;
        cv::resize(img, downSampledImage, s, 0, 0, CV_INTER_LINEAR);
        img = downSampledImage;
        pyramid.push_back(img);
    }

    return pyramid;
}

// drwnImagePyramidCacheConfig ----------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnImagePyramidCache
//! \b maxImages :: maximum images stored in cache (default: 1000)\n
//! \b maxMemory :: maximum memory (in bytes) used by cache (default: 500MB)\n
//! \b greyImages :: store images in greyscale (default: false)\n
//! \b bigMemory :: big memory mode (default: false)\n

class drwnImagePyramidCacheConfig : public drwnConfigurableModule {
public:
    drwnImagePyramidCacheConfig() : drwnConfigurableModule("drwnImagePyramidCache") { }
    ~drwnImagePyramidCacheConfig() { }

    void usage(ostream &os) const {
        os << "      maxImages       :: maximum images stored in cache (default: "
           << drwnImagePyramidCache::MAX_IMAGES << ")\n";
        os << "      maxMemory       :: maximum memory (in bytes) used by cache (default: "
           << drwnImagePyramidCache::MAX_MEMORY << ")\n";
        os << "      greyImages      :: greyscale images (default: "
           << (drwnImagePyramidCache::GREY_IMAGES ? "true" : "false") << ")\n";
        os << "      bigMemory       :: big memory mode (default: "
           << (drwnImagePyramidCache::BIG_MEMORY ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxImages")) {
            drwnImagePyramidCache::MAX_IMAGES = (size_t)std::max(0.0, atof(value));
        } else if (!strcmp(name, "maxMemory")) {
            drwnImagePyramidCache::MAX_MEMORY = (size_t)std::max(0.0, atof(value));
        } else if (!strcmp(name, "greyImages") || !strcmp(name, "grayImages")) {
            drwnImagePyramidCache::GREY_IMAGES = trueString(value);
        } else if (!strcmp(name, "bigMemory")) {
            drwnImagePyramidCache::BIG_MEMORY = trueString(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnImagePyramidCacheConfig gImagePyramidCacheConfig;
