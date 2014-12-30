/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiSegConfig.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "highgui.h"

#include "drwnBase.h"
#include "drwnMultiSegConfig.h"

using namespace std;

// globals ------------------------------------------------------------------

drwnMultiSegConfig gMultiSegConfig;

//! \todo make singleton object so don't need to create if not used
drwnMultiSegRegionDefinitions gMultiSegRegionDefs;

// drwnMultiSegConfig -------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnMultiSegConfig
//! \b baseDir      :: prepended to all other directory paths (default: )\n
//! \b imgDir       :: subdirectory containing images (default: data/images/)\n
//! \b lblDir       :: subdirectory containing groundtruth labels (default: data/labels/)\n
//! \b segDir       :: subdirectory containing over-segmentations (default: data/superpixels/)\n
//! \b cacheDir     :: subdirectory for caching intermediate calculations (default: cached/)\n
//! \b modelsDir    :: subdirectory storing models (default: models/)\n
//! \b outputDir    :: subdirectory generating output (default: output/)\n
//! \b imgExt       :: image extension (default: .jpg)\n
//! \b lblExt       :: label extension (default: .txt)\n
//! \b segExt       :: superpixel container extension (default: <none>)\n
//! \b useCache     :: use feature cache (default: 1)\n
//! \b compressedCache :: compress the feature cache (default: 0)

drwnMultiSegConfig::drwnMultiSegConfig() :
    drwnProperties(), drwnConfigurableModule("drwnMultiSegConfig"),
    _baseDir(""), _imgDir("data/images/"), _lblDir("data/labels/"),
    _segDir("data/superpixels/"), _cacheDir("cached/"), _modelsDir("models/"),
    _outputDir("output/"), _imgExt(".jpg"), _lblExt(".txt"), _segExt(""),
    _bUseCache(true), _bCompressedCache(false)
{
    // declare default data options
    declareProperty("baseDir", new drwnStringProperty(&_baseDir));
    declareProperty("imgDir", new drwnStringProperty(&_imgDir));
    declareProperty("lblDir", new drwnStringProperty(&_lblDir));
    declareProperty("segDir", new drwnStringProperty(&_segDir));
    declareProperty("cacheDir", new drwnStringProperty(&_cacheDir));
    declareProperty("modelsDir", new drwnStringProperty(&_modelsDir));
    declareProperty("outputDir", new drwnStringProperty(&_outputDir));

    declareProperty("imgExt", new drwnStringProperty(&_imgExt));
    declareProperty("lblExt", new drwnStringProperty(&_lblExt));
    declareProperty("segExt", new drwnStringProperty(&_segExt));

    declareProperty("useCache", new drwnBooleanProperty(&_bUseCache));
    declareProperty("compressedCache", new drwnBooleanProperty(&_bCompressedCache));
}

drwnMultiSegConfig::~drwnMultiSegConfig()
{
    // do nothing
}

// options i/o
void drwnMultiSegConfig::readConfiguration(drwnXMLNode& node)
{
    // read options
    drwnProperties::readProperties(node, "option");

    // read region definitions
    drwnXMLNode *childNode = node.first_node("regionDefinitions");
    if (childNode != NULL) {
        gMultiSegRegionDefs.read(*childNode);
    } else {
        DRWN_LOG_WARNING("drwnMultiSegConfig is missing regionDefinitions");
    }
}

void drwnMultiSegConfig::usage(std::ostream& os) const
{
    os << "      baseDir      :: prepended to all other directory paths (default: "
       << getPropertyAsString(findProperty("baseDir")) << ")\n"
       << "      imgDir       :: subdirectory containing images (default: "
       << getPropertyAsString(findProperty("imgDir")) << ")\n"
       << "      lblDir       :: subdirectory containing groundtruth labels (default: "
       << getPropertyAsString(findProperty("lblDir")) << ")\n"
       << "      segDir       :: subdirectory containing over-segmentations (default: "
       << getPropertyAsString(findProperty("segDir")) << ")\n"
       << "      cacheDir     :: subdirectory for caching intermediate calculations (default: "
       << getPropertyAsString(findProperty("cacheDir")) << ")\n"
       << "      modelsDir    :: subdirectory storing models (default: "
       << getPropertyAsString(findProperty("modelsDir")) << ")\n"
       << "      outputDir    :: subdirectory generating output (default: "
       << getPropertyAsString(findProperty("outputDir")) << ")\n"
       << "      imgExt       :: image extension (default: "
       << getPropertyAsString(findProperty("imgExt")) << ")\n"
       << "      lblExt       :: label extension (default: "
       << getPropertyAsString(findProperty("lblExt")) << ")\n"
       << "      segExt       :: superpixel container extension (default: "
       << getPropertyAsString(findProperty("segExt")) << ")\n"
       << "      useCache     :: use feature cache (default: "
       << getPropertyAsString(findProperty("useCache")) << ")\n"
       << "      compressedCache :: compress the feature cache (default: "
       << getPropertyAsString(findProperty("compressedCache")) << ")\n";
}

// utility functions
string drwnMultiSegConfig::filebase(const char *dirKey, const char *baseName)
{
    return _baseDir + getPropertyAsString(findProperty(dirKey)) + string(baseName);
}

string drwnMultiSegConfig::filebase(const char *dirKey, const string& baseName)
{
    return _baseDir + getPropertyAsString(findProperty(dirKey)) + baseName;
}

string drwnMultiSegConfig::filename(const char *dirKey,
    const char *baseName, const char *extKey)
{
    return (filebase(dirKey, baseName) + getPropertyAsString(findProperty(extKey)));
}

string drwnMultiSegConfig::filename(const char *dirKey,
    const string &baseName, const char *extKey)
{
    return (filebase(dirKey, baseName) + getPropertyAsString(findProperty(extKey)));
}

// drwnMultiSegRegionDefinitions --------------------------------------------

drwnMultiSegRegionDefinitions::drwnMultiSegRegionDefinitions()
{
    initDefaultRegions();
}

drwnMultiSegRegionDefinitions::drwnMultiSegRegionDefinitions(const char *filename)
{
    read(filename);
}

drwnMultiSegRegionDefinitions::~drwnMultiSegRegionDefinitions()
{
    // do nothing
}

// i/o
void drwnMultiSegRegionDefinitions::clear()
{
    _keys.clear();
    _colors.clear();
    _names.clear();
}

void drwnMultiSegRegionDefinitions::read(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    clear();

    // if no filename is given then reset to default definitions
    if (strlen(filename) == 0) {
        initDefaultRegions();
        return;
    }

    drwnXMLDoc root;
    drwnXMLNode *node = root.first_node("regionDefinitions");
    DRWN_ASSERT(node != NULL);
    read(*node);
}

void drwnMultiSegRegionDefinitions::read(drwnXMLNode& root)
{
    DRWN_ASSERT(!drwnIsXMLEmpty(root));
    clear();

    unsigned char red, green, blue;

    for (drwnXMLNode *node = root.first_node("region"); node != NULL; node = node->next_sibling("region")) {
        _keys[atoi(drwnGetXMLAttribute(*node, "id"))] = (int)_names.size();
        _names.push_back(string(drwnGetXMLAttribute(*node, "name")));
#if defined(_WIN32)
        int ir, ig, ib;
        if (sscanf(drwnGetXMLAttribute(*node, "color"), "%d %d %d", &ir, &ig, &ib) != 3) {
            DRWN_LOG_FATAL("could not parse color for \"" << _names.back() << "\"");
        }
        if ((ir < 0) || (ir > 255) || (ig < 0) || (ig > 255) || (ib < 0) || (ib > 255)) {
            DRWN_LOG_FATAL("could not parse color for \"" << _names.back() << "\"");
        }
        red = (unsigned char)ir;
        green = (unsigned char)ig;
        blue = (unsigned char)ib;
#else
        if (sscanf(drwnGetXMLAttribute(*node, "color"), "%hhu %hhu %hhu",
                &red, &green, &blue) != 3) {
            DRWN_LOG_FATAL("could not parse color for \"" << _names.back() << "\"");
        }
#endif
        _colors.push_back((red << 16) | (green << 8) | blue);
    }
}

int drwnMultiSegRegionDefinitions::maxKey() const
{
    return (_keys.empty() ? -1 : std::max(_keys.begin()->first, _keys.rbegin()->first));
}

set<int> drwnMultiSegRegionDefinitions::keys() const
{
    set<int> k;
    for (map<int, int>::const_iterator it = _keys.begin(); it != _keys.end(); ++it) {
        k.insert(it->first);
    }

    return k;
}

string drwnMultiSegRegionDefinitions::name(int key) const
{
    map<int, int>::const_iterator it = _keys.find(key);
    if (it == _keys.end()) {
        return string("unknown");
    }

    DRWN_ASSERT((unsigned)it->second < _names.size());
    return _names[it->second];
}

unsigned int drwnMultiSegRegionDefinitions::color(int key) const
{
    map<int, int>::const_iterator it = _keys.find(key);
    if (it == _keys.end()) {
        return 0x00000000;
    }

    DRWN_ASSERT((unsigned)it->second < _colors.size());
    return _colors[it->second];
}

void drwnMultiSegRegionDefinitions::initializeForDataset(const drwnStandardMultiSegRegionDatasets& dataset)
{
    // clear existing definitions
    clear();

    // add definitions for a given dataset
    switch (dataset) {
    case DRWN_DS_FGBG: // foreground/background
        {
            _keys[0] = 0;
            _keys[1] = 1;
            _colors.push_back(0x00000000);
            _colors.push_back(0x00ffffff);
            _names.push_back(string("background"));
            _names.push_back(string("foreground"));
        }
    case DRWN_DS_MSRC: // msrc 21 definitions
        {
            int KEYS[22] = {
                -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20};
            unsigned int COLORS[22] = {
                0x00000000, 0x00800000, 0x00008000, 0x00808000,
                0x00000080, 0x00008080, 0x00808080, 0x00c00000,
                0x00408000, 0x00c08000, 0x00400080, 0x00c00080,
                0x00408080, 0x00c08080, 0x00004000, 0x00804000,
                0x0000c000, 0x00804080, 0x0000c080, 0x0080c080,
                0x00404000, 0x00c04000};
            const char *NAMES[22] = {
                "void", "building", "grass", "tree", "cow", "sheep", "sky", "airplane",
                "water", "face", "car", "bicycle", "flower", "sign", "bird", "book",
                "chair", "road", "cat", "dog", "body", "boat"};

            for (int i = 0; i < (int)(sizeof(KEYS) / sizeof(int)); i++) {
                _keys[KEYS[i]] = i;
                _colors.push_back(COLORS[i]);
                _names.push_back(string(NAMES[i]));
            }
        }
        break;

    case DRWN_DS_STANFORD:
        {
            int KEYS[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            unsigned int COLORS[8] = {
                0x00808080, 0x00808000, 0x00804080, 0x00008000,
                0x00000080, 0x00800000, 0x00803200, 0x00ff8000};
            const char *NAMES[8] = {
                "sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"
            };

            for (int i = 0; i < (int)(sizeof(KEYS) / sizeof(int)); i++) {
                _keys[KEYS[i]] = i;
                _colors.push_back(COLORS[i]);
                _names.push_back(string(NAMES[i]));
            }
        }
        break;

    case DRWN_DS_PASCALVOC:
        {
            int KEYS[21] = {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20};
            unsigned int COLORS[21] = {
                0x00000000, 0x00800000, 0x00008000, 0x00808000,
                0x00000080, 0x00800080, 0x00008080, 0x00808080,
                0x00400000, 0x00c00000, 0x00408000, 0x00c08000,
                0x00400080, 0x00c00080, 0x00408080, 0x00c08080,
                0x00004000, 0x00804000, 0x0000c000, 0x0080c000,
                0x00004080
            };
            const char *NAMES[21] = {
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                "tvmonitor"
            };

            for (int i = 0; i < (int)(sizeof(KEYS) / sizeof(int)); i++) {
                _keys[KEYS[i]] = i;
                _colors.push_back(COLORS[i]);
                _names.push_back(string(NAMES[i]));
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized semantic segmentation dataset");
    }
}

MatrixXi drwnMultiSegRegionDefinitions::convertImageToLabels(const cv::Mat& img) const
{
    DRWN_ASSERT((img.channels() == 3) && (img.depth() == CV_8U));

    // build reverse lookup table
    map<unsigned, int> lookup;
    for (map<int, int>::const_iterator it = _keys.begin(); it != _keys.end(); ++it) {
        lookup.insert(make_pair(_colors[it->second], it->first));
    }

    MatrixXi labels = MatrixXi::Constant(img.rows, img.cols, -1);

    for (int y = 0; y < img.rows; y++) {
        const unsigned char *p = img.ptr(y);
        for (int x = 0; x < img.cols; x++) {
            const unsigned colour = this->rgb(p[3*x + 2], p[3*x + 1], p[3*x + 0]);
            map<unsigned, int>::const_iterator it = lookup.find(colour);
            if (it != lookup.end()) {
                labels(y, x) = it->second;
            }
        }
    }

    return labels;
}

MatrixXi drwnMultiSegRegionDefinitions::convertImageToLabels(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);
    cv::Mat img = cv::imread(string(filename));
    DRWN_ASSERT_MSG(img.data != NULL, filename);
    const MatrixXi labels = convertImageToLabels(img);
    return labels;
}

MatrixXi drwnMultiSegRegionDefinitions::loadLabelFile(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);
    string ext = drwn::strExtension(string(filename));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // convert from an image file
    if (ext.compare("png") == 0) {
        return convertImageToLabels(filename);
    }

    // assume text file
    DRWN_ASSERT(ext.compare("txt") == 0);

    ifstream ifs(filename);
    DRWN_ASSERT_MSG(!ifs.fail(), filename);
    const int w = drwnCountFields(&ifs);
    DRWN_LOG_DEBUG(filename << " has width " << w);
    list<int> values;
    while (1) {
        int v;
        ifs >> v;
        if (ifs.fail()) break;
        values.push_back(v);
    }
    const int h = values.size() / w;
    DRWN_LOG_DEBUG(filename << " has height " << h);

    MatrixXi L(h, w);

    list<int>::const_iterator it = values.begin();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            L(i,j) = *it++;
        }
    }

    return L;
}

// operators
unsigned int& drwnMultiSegRegionDefinitions::operator[](int key)
{
    return _colors[_keys[key]];
}

void drwnMultiSegRegionDefinitions::initDefaultRegions()
{
    initializeForDataset(DRWN_DS_MSRC);
}
