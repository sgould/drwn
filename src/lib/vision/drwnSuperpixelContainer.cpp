/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSuperpixelContainer.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

#include "cv.h"

#include "drwnBase.h"
#include "drwnVision.h"

using namespace std;

// drwnSuperpixelContainer -----------------------------------------------------

drwnSuperpixelContainer::drwnSuperpixelContainer()
{
    // do nothing
}

drwnSuperpixelContainer::drwnSuperpixelContainer(const drwnSuperpixelContainer& container) :
    _start(container._start), _nsegs(container._nsegs), _pixels(container._pixels),
    _bboxes(container._bboxes)
{
    // deep copy
    _maps.resize(container._maps.size());
    for (unsigned i = 0; i < container._maps.size(); i++) {
        _maps[i] = container._maps[i].clone();
    }
}

drwnSuperpixelContainer::~drwnSuperpixelContainer()
{
    clear();
}

void drwnSuperpixelContainer::clear()
{
    _start.clear();
    _nsegs.clear();
    _maps.clear();
    _pixels.clear();
    _bboxes.clear();
}

size_t drwnSuperpixelContainer::numBytesOnDisk() const
{
    // number of maps, width and height
    size_t n = 3 * sizeof(uint32_t);

    // run-length encoded data
    for (unsigned i = 0; i < _maps.size(); i++) {
        for (int y = 0; y < _maps[i].rows; y++) {
            int x = 0;
            const int *p = _maps[i].ptr<const int>(y);
            while (x < _maps[i].cols) {
                const int id = p[x++];
                unsigned count = 0;
                while ((count < 255) && (x < _maps[i].cols) && (p[x] == id)) {
                    x += 1; count += 1;
                }
                n += sizeof(uint32_t);
            }
        }
    }

    return n;
}

bool drwnSuperpixelContainer::write(ostream& os) const
{
    DRWN_FCN_TIC;

    // write header
    uint32_t n = _maps.size();
    os.write((char *)&n, sizeof(uint32_t));
    n = (uint32_t)this->width();
    os.write((char *)&n, sizeof(uint32_t));
    n = (uint32_t)this->height();
    os.write((char *)&n, sizeof(uint32_t));

    // write run-length encoded maps
    for (unsigned i = 0; i < _maps.size(); i++) {
        for (int y = 0; y < _maps[i].rows; y++) {
            int x = 0;
            const int *p = _maps[i].ptr<const int>(y);
            while (x < _maps[i].cols) {
                const int id = p[x++];
                unsigned count = 0;
                while ((count < 255) && (x < _maps[i].cols) && (p[x] == id)) {
                    x += 1; count += 1;
                }
                n = (count << 24) | (id & 0x00ffffff);
                os.write((char *)&n, sizeof(uint32_t));
            }
        }
    }

    DRWN_FCN_TOC;
    return true;
}

bool drwnSuperpixelContainer::read(istream& is)
{
    DRWN_FCN_TIC;

    // clear current data and read header
    clear();
    uint32_t n, w, h;
    is.read((char *)&n, sizeof(uint32_t));
    is.read((char *)&w, sizeof(uint32_t));
    is.read((char *)&h, sizeof(uint32_t));
    DRWN_LOG_DEBUG("drwnSuperpixelContainer reading " << n << " maps of size " << w << "-by-" << h);

    // read run-length encoded maps
    while (n-- > 0) {
        cv::Mat m((int)h, (int)w, CV_32SC1);
        for (int y = 0; y < m.rows; y++) {
            int *p = m.ptr<int>(y);
            int x = 0;
            while (x < m.cols) {
                uint32_t data;
                is.read((char *)&data, sizeof(uint32_t));
                const unsigned count = ((data >> 24) & 0x000000ff);
                const int id = (int)(((data & 0x00ffffff) == 0x00ffffff) ? -1 : (data & 0x00ffffff));
                for (unsigned i = 0; i <= count; i++) {
                    p[x + i] = id;
                }
                x += count + 1;
            }
        }
        addSuperpixels(m);
    }

    DRWN_FCN_TOC;
    return true;
}

void drwnSuperpixelContainer::loadSuperpixels(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    string ext = strExtension(string(filename));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    cv::Mat superpixels;
    if (ext.compare("bin") == 0) {
        // load binary data
        // TODO: need width and height
        DRWN_TODO;
    } else if (ext.compare("png") == 0) {
        // load 8- or 16-bit png
        cv::Mat tmp = cv::imread(string(filename), CV_LOAD_IMAGE_ANYDEPTH);
        DRWN_LOG_DEBUG(filename << " is " << toString(tmp));
        superpixels = cv::Mat(tmp.rows, tmp.cols, CV_32SC1);
        tmp.convertTo(superpixels, CV_32S);
    } else if (ext.compare("txt") == 0) {
        ifstream ifs(filename);
        DRWN_ASSERT_MSG(!ifs.fail(), filename);
        int w = drwnCountFields(&ifs);
        DRWN_LOG_DEBUG(filename << " has width " << w);
        list<int> values;
        while (1) {
            int v;
            ifs >> v;
            if (ifs.fail()) break;
            values.push_back(v);
        }
        int h = values.size() / w;
        DRWN_LOG_DEBUG(filename << " has height " << h);
        superpixels = cv::Mat(h, w, CV_32SC1);
        int *p = superpixels.ptr<int>(0);
        list<int>::const_iterator it = values.begin();
        while (it != values.end()) {
            *p++ = *it++;
        }
    } else {
        DRWN_LOG_FATAL("unrecognized extension");
    }

    // renumber and add superpixels
    addSuperpixels(superpixels);
}

// renumbers superpixels contiguously from 0
void drwnSuperpixelContainer::addSuperpixels(cv::Mat superpixels)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(superpixels.type() == CV_32SC1);
    DRWN_ASSERT(_maps.empty() || ((superpixels.rows == _maps[0].rows) &&
            (superpixels.cols == _maps[0].cols)));

    // compute renumbering
    map<int, int> renumbering;
    for (int y = 0; y < superpixels.rows; y++) {
        const int *p = superpixels.ptr<const int>(y);
        for (int x = 0; x < superpixels.cols; x++) {
            if (p[x] < 0) continue;
            if (renumbering.find(p[x]) == renumbering.end()) {
                renumbering.insert(make_pair(p[x], (int)renumbering.size()));
            }
        }
    }
    DRWN_ASSERT_MSG(!renumbering.empty(), "no superpixels");

    // update start index
    _start.push_back(_start.empty() ? 0 : _start.back() + _nsegs.back());

    // update number of segments
    _nsegs.push_back((int)renumbering.size());

    // renumber superpixels
    _pixels.resize(_pixels.size() + renumbering.size(), 0);
    for (int y = 0; y < superpixels.rows; y++) {
        int *p = superpixels.ptr<int>(y);
        for (int x = 0; x < superpixels.cols; x++) {
            if (p[x] < 0) continue;
            p[x] = _start.back() + renumbering[p[x]];
            _pixels[p[x]] += 1;
        }
    }

    // compute bounding boxes
    _bboxes.resize(_bboxes.size() + renumbering.size(), cv::Rect(superpixels.cols, superpixels.rows, 0, 0));
    for (int y = 0; y < superpixels.rows; y++) {
        int *p = superpixels.ptr<int>(y);
        for (int x = 0; x < superpixels.cols; x++) {
            if (p[x] < 0) continue;
            _bboxes[p[x]].x = std::min(_bboxes[p[x]].x, x);
            _bboxes[p[x]].width = std::max(_bboxes[p[x]].width, x);
            _bboxes[p[x]].y = std::min(_bboxes[p[x]].y, y);
            _bboxes[p[x]].height = std::max(_bboxes[p[x]].height, y);
        }
    }

    for (unsigned i = _start.back(); i < _bboxes.size(); i++) {
        _bboxes[i].width -= _bboxes[i].x - 1;
        _bboxes[i].height -= _bboxes[i].y - 1;
    }

    // add superpixels
    _maps.push_back(superpixels);
    DRWN_FCN_TOC;
}

void drwnSuperpixelContainer::copySuperpixels(const cv::Mat& superpixels)
{
    addSuperpixels(superpixels.clone());
}

// masks
cv::Mat drwnSuperpixelContainer::mask(unsigned segId) const
{
    DRWN_ASSERT(!_maps.empty() && (segId < (unsigned)(_start.back() + _nsegs.back())));
    cv::Mat m(height(), width(), CV_8UC1);
    return this->mask(segId, m);
}

cv::Mat& drwnSuperpixelContainer::mask(unsigned segId, cv::Mat& m) const
{
    if ((m.rows == 0) || (m.cols == 0)) {
        m = cv::Mat(height(), width(), CV_8UC1);
    }

    DRWN_ASSERT((m.rows == height()) && (m.cols == width()));

    unsigned indx = 0;
    while ((int)segId >= _start[indx] + _nsegs[indx]) {
        indx += 1;
    }
    DRWN_ASSERT(indx < _maps.size());

    cv::compare(_maps[indx], cv::Scalar::all(segId), m, CV_CMP_EQ);
    return m;
}

set<unsigned> drwnSuperpixelContainer::parents(int x, int y) const
{
    set<unsigned> par;

    for (unsigned i = 0; i < _maps.size(); i++) {
        const int id = _maps[i].at<int>(y, x);
        if (id >= 0) {
            par.insert((unsigned)id);
        }
    }

    return par;
}

set<unsigned> drwnSuperpixelContainer::neighbours(unsigned segId) const
{
    cv::Mat m(this->mask(segId));
    cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));
    cv::dilate(m, m, kernel);

    set<unsigned> nbrs;
    for (int y = std::max(0, _bboxes[segId].y - 1); y < std::min(m.rows, _bboxes[segId].y + _bboxes[segId].height + 1); y++) {
        for (int x = std::max(0, _bboxes[segId].x - 1); x < std::min(m.cols, _bboxes[segId].x + _bboxes[segId].width + 1); x++) {
            if (m.at<unsigned char>(y, x) == 0x00)
                continue;
            for (unsigned i = 0; i < _maps.size(); i++) {
                const int id = _maps[i].at<int>(y, x);
                if ((id >= 0) && ((unsigned)id != segId)) {
                    nbrs.insert((unsigned)id);
                }
            }
        }
    }

    return nbrs;
}

cv::Point drwnSuperpixelContainer::centroid(unsigned segId) const
{
    // find the channel for this segId
    unsigned indx = 0;
    while ((int)segId >= _start[indx] + _nsegs[indx]) {
        indx += 1;
    }
    DRWN_ASSERT(indx < _maps.size());

    // compute centroid
    cv::Point p(0, 0);
    for (int y = _bboxes[segId].y; y < _bboxes[segId].y + _bboxes[segId].height; y++) {
        for (int x = _bboxes[segId].x; x < _bboxes[segId].x + _bboxes[segId].width; x++) {
            const int id = _maps[indx].at<int>(y, x);
            if (id == (int)segId) {
                p.x += x;
                p.y += y;
            }
        }
    }

    const int n = pixels(segId);
    p.x /= n; p.y /= n;
    return p;
}

cv::Mat drwnSuperpixelContainer::intersection() const
{
    DRWN_ASSERT(!this->empty());
    cv::Mat segments = cv::Mat::zeros(height(), width(), CV_32SC1);

    // compute unique intersection identifiers
    cv::Mat m(height(), width(), CV_8UC1);
    cv::Mat shiftedSegments(height(), width(), CV_32SC1);
    for (unsigned i = 0; i < _maps.size(); i++) {
        DRWN_LOG_DEBUG("...processing over-segmentation " << i);
        cv::add(_maps[i], cv::Scalar::all(-1.0 * _start[i]), shiftedSegments);
        cv::compare(_maps[i], cv::Scalar::all(0.0), m, CV_CMP_LT);
        shiftedSegments.setTo(0.0, m);
        segments = segments + (_start[i] + 1) * shiftedSegments;
    }

    // relabel identifiers
    map<int, int> renumbering;
    for (int y = 0; y < segments.rows; y++) {
        const int *p = segments.ptr<const int>(y);
        for (int x = 0; x < segments.cols; x++) {
            if (p[x] < 0) continue;
            if (renumbering.find(p[x]) == renumbering.end()) {
                renumbering.insert(make_pair(p[x], (int)renumbering.size()));
            }
        }
    }

    // renumber segments
    for (int y = 0; y < segments.rows; y++) {
        int *p = segments.ptr<int>(y);
        for (int x = 0; x < segments.cols; x++) {
            if (p[x] < 0) continue;
            p[x] = renumbering[p[x]];
        }
    }

    DRWN_LOG_DEBUG("intersection of over-segmentations has "
        << renumbering.size() << " superpixels");
    return segments;
}

// filtering
int drwnSuperpixelContainer::removeSmallSuperpixels(unsigned minSize)
{
    if (this->empty()) return 0;
    DRWN_FCN_TIC;

    const int nInitialSize = size();

    for (unsigned i = 0; i < _maps.size(); i++) {
        for (int y = 0; y < _maps[i].rows; y++) {
            int *p = _maps[i].ptr<int>(y);
            for (int x = 0; x < _maps[i].cols; x++) {
                if (p[x] < 0) continue;
                if (_pixels[p[x]] <= (int)minSize) {
                    p[x] = -1;
                }
            }
        }
    }

    // copy maps and reassign
    vector<cv::Mat> maps(_maps);
    clear();
    for (unsigned i = 0; i < maps.size(); i++) {
        addSuperpixels(maps[i]);
    }

    DRWN_FCN_TOC;
    return nInitialSize - size();
}

int drwnSuperpixelContainer::removeUnmaskedSuperpixels(const cv::Mat& mask, double areaOverlap)
{
    DRWN_ASSERT((mask.data != NULL) && (mask.depth() == CV_8U));
    DRWN_ASSERT((mask.rows == height()) && (mask.cols == width()));
    areaOverlap = std::max(0.0, std::min(areaOverlap, 1.0));

    // count number of pixels inside and outside the mask for each superpixel
    const int nInitialSize = this->size();
    vector<pair<int, int> > counts(nInitialSize, make_pair<int, int>(0, 0));
    for (int y = 0; y < mask.rows; y++) {
        const unsigned char *p = mask.ptr<const unsigned char>(y);
        for (int x = 0; x < mask.cols; x++) {
            for (unsigned c = 0; c < _maps.size(); c++) {
                const int segId = _maps[c].at<const int>(y, x);
                if (segId < 0) continue;
                if (p[x] != 0x00) {
                    counts[segId].first += 1;
                }
                counts[segId].second += 1;
            }
        }
    }

    vector<bool> remove(nInitialSize, false);
    for (unsigned i = 0; i < remove.size(); i++) {
        if (counts[i].first < areaOverlap * counts[i].second) {
            remove[i] = true;
        }
    }

    // remove pixels with too many outside pixels
    for (unsigned i = 0; i < _maps.size(); i++) {
        for (int y = 0; y < _maps[i].rows; y++) {
            int *p = _maps[i].ptr<int>(y);
            for (int x = 0; x < _maps[i].cols; x++) {
                if (p[x] < 0) continue;
                if (remove[p[x]]) p[x] = -1;
            }
        }
    }

    // copy maps and reassign
    vector<cv::Mat> maps(_maps);
    clear();
    for (unsigned i = 0; i < maps.size(); i++) {
        addSuperpixels(maps[i]);
    }

    DRWN_FCN_TOC;
    return nInitialSize - size();
}

int drwnSuperpixelContainer::removeSuperpixels(const set<unsigned>& segIds)
{
    if (this->empty()) return 0;
    DRWN_FCN_TIC;

    const int nInitialSize = size();

    for (set<unsigned>::const_iterator it = segIds.begin(); it != segIds.end(); ++it) {
        for (unsigned i = 0; i < _maps.size(); i++) {
            for (int y = 0; y < _maps[i].rows; y++) {
                int *p = _maps[i].ptr<int>(y);
                for (int x = 0; x < _maps[i].cols; x++) {
                    if (p[x] == (int)*it) p[x] = -1;
                }
            }
        }
    }

    // copy maps and reassign
    vector<cv::Mat> maps(_maps);
    clear();
    for (unsigned i = 0; i < maps.size(); i++) {
        addSuperpixels(maps[i]);
    }

    DRWN_FCN_TOC;
    return nInitialSize - size();
}

// visualization
cv::Mat drwnSuperpixelContainer::visualize(const cv::Mat& img, bool bColorById) const
{
    DRWN_ASSERT(img.data != NULL);
    if (empty()) return img.clone();

    vector<cv::Mat> views;
    cv::Mat m(height(), width(), CV_8UC1);
    for (unsigned i = 0; i < _maps.size(); i++) {
        if (bColorById) {
            views.push_back(drwnCreateHeatMap(_maps[i]));
        } else {
            views.push_back(img.clone());
        }
        cv::compare(_maps[i], cv::Scalar::all(0.0), m, CV_CMP_LT);
        views.back().setTo(cv::Scalar::all(0.0), m);
        drwnShadeRegion(views.back(), m, CV_RGB(255, 0, 0), 1.0, DRWN_FILL_DIAG, 1);
        drwnDrawRegionBoundaries(views.back(), _maps[i], CV_RGB(255, 255, 255), 3);
        drwnDrawRegionBoundaries(views.back(), _maps[i], CV_RGB(255, 0, 0), 1);
    }

    return drwnCombineImages(views);
}

drwnSuperpixelContainer& drwnSuperpixelContainer::operator=(const drwnSuperpixelContainer& container)
{
    if (this != &container) {
        clear();
        _start = container._start;
        _nsegs = container._nsegs;
        _pixels = container._pixels;
        _bboxes = container._bboxes;

        // deep copy
        _maps.resize(container._maps.size());
        for (unsigned i = 0; i < container._maps.size(); i++) {
            _maps[i] = container._maps[i].clone();
        }
    }

    return *this;
}
