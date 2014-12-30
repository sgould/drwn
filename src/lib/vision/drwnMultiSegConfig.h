/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiSegConfig.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <list>
#include <map>

#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"

using namespace std;

// drwnMultiSegConfig -------------------------------------------------------
//! Manages configuration settings for multiple image segmentation.
//!
//! The settings can be changed via an XML file (using \p -config) or
//! separate command line options (using \p -set).
//!
//! \sa \ref drwnProjMultiSeg

class drwnMultiSegConfig : public drwnProperties, drwnConfigurableModule {
 protected:
    string _baseDir;        //!< base directory for data, models and results
    string _imgDir;         //!< directory containing images (relative to _baseDir)
    string _lblDir;         //!< directory conatining labels (relative to _baseDir)
    string _segDir;         //!< directory containing over-segmentations (relative to _baseDir)
    string _cacheDir;       //!< directory to store pre-computed features (relative to _baseDir)
    string _modelsDir;      //!< directory to read and write model files (relative to _baseDir)
    string _outputDir;      //!< directory to write results (relative to _baseDir)
    string _imgExt;         //!< file extension of images
    string _lblExt;         //!< file extension of image
    string _segExt;         //!< file extension for over-segmentations (drwnSuperpixelContainer)
    bool _bUseCache;        //!< set true to store features in the _cacheDir
    bool _bCompressedCache; //!< set true to compress cached data

 public:
    drwnMultiSegConfig();
    ~drwnMultiSegConfig();

    // options i/o
    void readConfiguration(drwnXMLNode& node);
    void setConfiguration(const char *name, const char *value) {
        drwnProperties::setProperty(drwnProperties::findProperty(name), value);
    }

    void usage(std::ostream& os) const;

    // utility functions
    //! returns a filename consisting of _baseDir, the directory corresponding
    //! to \p dirKey and \p baseName
    string filebase(const char *dirKey, const char *baseName);
    //! returns a filename consisting of _baseDir, the directory corresponding
    //! to \p dirKey and \p baseName
    string filebase(const char *dirKey, const string& baseName);
    //! returns a filename consisting of _baseDir, the directory corresponding
    //! to \p dirKey, \p baseName, and the extension corresponding to \p extKey
    string filename(const char *dirKey, const char *baseName, const char *extKey);
    //! returns a filename consisting of _baseDir, the directory corresponding
    //! to \p dirKey, \p baseName, and the extension corresponding to \p extKey
    string filename(const char *dirKey, const string& baseName, const char *extKey);
};

extern drwnMultiSegConfig gMultiSegConfig;

// drwnStandardMultiSegRegionDatasets ----------------------------------------
//! Provides definitions for various standard multi-class image labeling
//! (semantic segmentation) datasets.

typedef enum _drwnStandardMultiSegRegionDatasets {
    DRWN_DS_FGBG,     //!< 2-class foreground/background segmentation
    DRWN_DS_MSRC,     //!< 21-class MSRC dataset
    DRWN_DS_STANFORD, //!< 8-class Stanford background dataset
    DRWN_DS_PASCALVOC //!< 20-class PASCAL VOC dataset plus background
} drwnStandardMultiSegRegionDatasets;

// drwnMultiSegRegionDefinitions ---------------------------------------------
//! Provides a mechanism for mapping region IDs to colours and class names. Can
//! be initialized from an XML configuration file or programmatically for a
//! number of standard datasets.
//!
//! For example, the following XML file provides region IDs for the 21-class
//! MSRC dataset:
//! \code
//!    <regionDefinitions>
//!      <region id="-1" name="void" color="0 0 0"/>
//!      <region id="0" name="building" color="128 0 0"/>
//!      <region id="1" name="grass" color="0 128 0"/>
//!      <region id="2" name="tree" color="128 128 0"/>
//!      <region id="3" name="cow" color="0 0 128"/>
//!      <region id="4" name="sheep" color="0 128 128"/>
//!      <region id="5" name="sky" color="128 128 128"/>
//!      <region id="6" name="airplane" color="192 0 0"/>
//!      <region id="7" name="water" color="64 128 0"/>
//!      <region id="8" name="face" color="192 128 0"/>
//!      <region id="9" name="car" color="64 0 128"/>
//!      <region id="10" name="bicycle" color="192 0 128"/>
//!      <region id="11" name="flower" color="64 128 128"/>
//!      <region id="12" name="sign" color="192 128 128"/>
//!      <region id="13" name="bird" color="0 64 0"/>
//!      <region id="14" name="book" color="128 64 0"/>
//!      <region id="15" name="chair" color="0 192 0"/>
//!      <region id="16" name="road" color="128 64 128"/>
//!      <region id="17" name="cat" color="0 192 128"/>
//!      <region id="18" name="dog" color="128 192 128"/>
//!      <region id="19" name="body" color="64 64 0"/>
//!      <region id="20" name="boat" color="192 64 0"/>
//!    </regionDefinitions>
//! \endcode
//!

class drwnMultiSegRegionDefinitions {
 protected:
    map<int, int> _keys;          //!< mapping from keys to colour and name index
    vector<unsigned int> _colors; //!< colour for i-th key
    vector<string> _names;        //!< name of class for i-th key

 public:
    //! default region definitions (constructed from initDefaultRegions)
    drwnMultiSegRegionDefinitions();
    //! construct region definitions from file
    drwnMultiSegRegionDefinitions(const char *filename);
    virtual ~drwnMultiSegRegionDefinitions();

    // i/o
    //! clear region definitions
    void clear();
    //! read region definitions from file
    void read(const char *filename);
    //! read region definitions from XML node
    void read(drwnXMLNode& root);

    //! returns \p true if no regions have been definied
    bool empty() const { return _colors.empty(); }
    //! returns the number of defined regions
    int size() const { return (int)_colors.size(); }
    //! returns the highest region ID
    int maxKey() const;

    //! returns the set of region IDs
    set<int> keys() const;
    //! returns the name of the region with ID \p key
    string name(int key) const;
    //! returns the colour of the region with ID \p kep
    unsigned int color(int key) const;

    //! converts from 32-bit colour to 8-bit red
    static unsigned char red(unsigned int c) {
        return ((c >> 16) & 0xff);
    }
    //! converts from 32-bit colour to 8-bit green
    static unsigned char green(unsigned int c) {
        return ((c >> 8) & 0xff);
    }
    //! converts from 32-bit colour to 8-bit blue
    static unsigned char blue(unsigned int c) { 
        return (c & 0xff);
    }
    //! converts 8-bit red, green and blue into 32-bit colour
    static unsigned int rgb(unsigned char r, unsigned char g, unsigned char b) {
        return ((unsigned int)(r << 16) | (unsigned int)(g << 8) | (unsigned int)b);
    }

    //! re-initialize with a standard multi-class image labeling dataset
    virtual void initializeForDataset(const drwnStandardMultiSegRegionDatasets& dataset);

    //! utility function for converting an image from colours to labels
    MatrixXi convertImageToLabels(const cv::Mat& img) const;
    //! utility function for reading an image and converting it from colours to labels
    MatrixXi convertImageToLabels(const char *filename) const;

    //! utility to load labels from file (.txt read as is; .png converted via convertImageToLabels)
    MatrixXi loadLabelFile(const char *filename) const;

    // operators
    //! returns the colour of the region with ID \p key
    unsigned int& operator[](int key);

 protected:
    //! initializes the defitions with a default set of regions
    virtual void initDefaultRegions();
};

extern drwnMultiSegRegionDefinitions gMultiSegRegionDefs;
