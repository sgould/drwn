/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPatchMatch.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
#include <cstdint>
#endif
#include <cassert>
#include <list>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// drwnPatchMatchStatus ------------------------------------------------------
//! Status of a drwnPatchMatchEdge object indicating whether the edge is new
//! or has been processed by a propagation move.

typedef enum { DRWN_PM_DIRTY, //!< the match is new
               DRWN_PM_PASS1, //!< the match has been through one propagation pass
               DRWN_PM_PASS2  //!< the match has been through two propagation passes
} drwnPatchMatchStatus;

// drwnPatchMatchTransform ---------------------------------------------------
//! Patch transformations. Currently only supports horizontal and vertical
//! flipping.
//! \todo: make into class to support future expansion and arbitrary transforms

typedef unsigned char drwnPatchMatchTransform;

#define DRWN_PM_TRANSFORM_NONE 0x00
#define DRWN_PM_TRANSFORM_HFLIP 0x01
#define DRWN_PM_TRANSFORM_VFLIP 0x02

//! compose two transforms f(g(.))
inline drwnPatchMatchTransform drwnComposeTransforms(const drwnPatchMatchTransform& f,
    const drwnPatchMatchTransform& g) {
    return g ^ f; // xor transformations (horizontal and vertical flips)
}

// drwnPatchMatchNode --------------------------------------------------------
//! Represents a node in the drwnPatchMatchGraph.
//!
//! Includes image index, scale (i.e., pyramid level), and index of the top
//! left pixel. The patch size is stored in the drwnPatchMatchGraph data
//! structure.
//!
//! The precision of the data members allows for up to 64 thousand images of
//! size 64k-by-64k.

class drwnPatchMatchNode {
 public:
    uint16_t imgIndx;    //!< index of image (in drwnPatchMatchGraph) for this match
    uint8_t imgScale;    //!< level in the image pyramid for the match
    uint16_t xPosition;  //!< x location of the patch in the target pyramid level
    uint16_t yPosition;  //!< y location of the patch in the target pyramid level

 public:
    drwnPatchMatchNode() : imgIndx(-1), imgScale(0), xPosition(-1), yPosition(-1)
        { /* do nothing */ };
    drwnPatchMatchNode(uint16_t idx, uint8_t sc, uint16_t x, uint16_t y) :
        imgIndx(idx), imgScale(sc), xPosition(x), yPosition(y) { /* do nothing */ };
    drwnPatchMatchNode(uint16_t idx, uint8_t sc, const cv::Point& p) :
        imgIndx(idx), imgScale(sc), xPosition(p.x), yPosition(p.y) { /* do nothing */ };
    ~drwnPatchMatchNode() { /* do nothing */ };

    //! default comparison for sorting
    bool operator<(const drwnPatchMatchNode& node) const {
        if (imgIndx < node.imgIndx) return true;
        if (imgIndx > node.imgIndx) return false;
        if (imgScale < node.imgScale) return true;
        if (imgScale > node.imgScale) return false;
        if (yPosition < node.yPosition) return true;
        if (yPosition > node.yPosition) return false;
        return (xPosition < node.xPosition);
    }
};

// drwnPatchMatchEdge --------------------------------------------------------
//! Represents an edge in the drwnPatchMatchGraph.
//!
//! This class stores a reference to a target patch in another image. The source
//! of the patch is implicit in how the edge is stored in drwnPatchMatchGraph.

class drwnPatchMatchEdge {
 public:
    float matchScore;              //!< score of this match
    drwnPatchMatchNode targetNode; //!< index of image, pyramid level and location of patch
    drwnPatchMatchTransform xform; //!< transformation applied to the patch

    drwnPatchMatchStatus status;   //!< status flag

 public:
    //! default constructor
    inline drwnPatchMatchEdge() : matchScore(DRWN_FLT_MAX),
        xform(DRWN_PM_TRANSFORM_NONE), status(DRWN_PM_DIRTY) { /* do nothing */ };
    //! constructor
    inline drwnPatchMatchEdge(float score, const drwnPatchMatchNode& node,
        const drwnPatchMatchTransform& patchXform) : matchScore(score), targetNode(node),
        xform(patchXform), status(DRWN_PM_DIRTY) { /* do nothing */ };

    //! number of bytes needed on disk
    size_t numBytesOnDisk() const;
    //! read from a stream (deserialization)
    bool read(istream& is);
    //! write to a stream (serialization)
    bool write(ostream& os) const;

    //! default comparison for sorting
    bool operator<(const drwnPatchMatchEdge& e) const {
        return (matchScore < e.matchScore);
    }
};

// drwnPatchMatchEdge Utility Functions --------------------------------------

inline bool drwnPatchMatchSortByScore(const drwnPatchMatchEdge& a, const drwnPatchMatchEdge& b)
{
    return (a.matchScore < b.matchScore);
}

inline bool drwnPatchMatchSortByImage(const drwnPatchMatchEdge& a, const drwnPatchMatchEdge& b)
{
    return (a.targetNode.imgIndx < b.targetNode.imgIndx) ||
        ((a.targetNode.imgIndx == b.targetNode.imgIndx) && (a.matchScore < b.matchScore));
}

string toString(const drwnPatchMatchEdge& e);

// drwnPatchMatchEdgeList ----------------------------------------------------
//! Record of matches for each pixel maintained as a sorted list.
//! \todo this data-structure uses a lot of memory, can we change to another
//! datatype (like a singly-linked list?)

typedef list<drwnPatchMatchEdge> drwnPatchMatchEdgeList;

// drwnPatchMatchImageRecord ------------------------------------------------
//! Records matches for one level in an image pyramid.

class drwnPatchMatchImageRecord {
 protected:
    uint16_t  _width;      //!< width of the image at this pyramid level
    uint16_t  _height;     //!< height of the image at this pyrmaid level

    //!< row-wise storage of pixel matches for this pyramid level
    vector<drwnPatchMatchEdgeList> _matches;

 public:
    //! default constructor
    drwnPatchMatchImageRecord() : _width(0), _height(0) {
        // do nothing
    }
    //! constructor
    drwnPatchMatchImageRecord(uint16_t width, uint16_t height) : _width(width), _height(height) {
        DRWN_ASSERT((width > 0) && (height > 0));
        _matches.resize(width * height);
    }
    //! destructor
    ~drwnPatchMatchImageRecord() {
        // do nothing
    }

    //! image width
    uint16_t width() const { return _width; }
    //! image height
    uint16_t height() const { return _height; }
    //! number of pixels in image (width * height)
    size_t size() const { return _width * _height; }

    //! clears all matches for this image
    void clear();

    //! convert from index to x pixel coordinate
    inline uint16_t x(int indx) const { return indx % _width; }
    //! convert from index to x pixel coordinate
    inline uint16_t y(int indx) const { return indx / _width; }
    //! convert from index to pixel coordinates
    inline cv::Point index2pixel(int indx) const { return cv::Point(indx % _width, indx / _width); }
    //! convert from pixel coordinates to index
    inline int pixel2index(uint16_t x, uint16_t y) const { return y * _width + x; }

    //! updates edge list for pixel at index \p indx with a candidate match and
    //! returns true if match is better than existing matches
    bool update(int indx, const drwnPatchMatchEdge& match);
    //! updates edge list for pixel \p (x,y) with a candidate match and
    //! returns true if match is better than existing matches
    inline bool update(uint16_t x, uint16_t y, const drwnPatchMatchEdge& match) {
        return update(pixel2index(x, y), match);
    }

    //! reference edges at location \p (x, y)
    drwnPatchMatchEdgeList& operator()(uint16_t x, uint16_t y) { return _matches[y * _width + x]; }
    //! reference edges at location \p (x, y)
    const drwnPatchMatchEdgeList& operator()(uint16_t x, uint16_t y) const { return _matches[y * _width + x]; }
    //! reference edges at index \p indx
    drwnPatchMatchEdgeList& operator[](int indx) { return _matches[indx]; }
    //! reference edges at index \p indx
    const drwnPatchMatchEdgeList& operator[](int indx) const { return _matches[indx]; }
};

// drwnPatchMatchImagePyramid -------------------------------------------------
//! Record of patch matches for mutliple levels each image.
//! \todo make cacheable so drwnPatchMatchGraph can handle thousands of images

class drwnPatchMatchImagePyramid : public drwnPersistentRecord {
 public:
    static unsigned MAX_LEVELS; //!< maximum level in the image pyramid
    static unsigned MAX_SIZE;   //!< maximum image size (if larger will be resized in first level)
    static unsigned MIN_SIZE;   //!< minimum image size (if smaller than size at MAX_LEVELS)
    static double PYR_SCALE;    //!< pyramid downsampling rate

 public:
    //! Include this image during update (initialize, propagate, local, search, and
    //! enrichment). If false the image can still be matched to but it's own matches
    //! are not updated.
    bool bActive;

    //! Equivalence class for this image. If negative (which is the default) the image 
    //! is considered to be in its own equivalence class. Images in the same equivalence
    //! class will not be matched when learning the PatchMatchGraph.
    int eqvClass;

 protected:
    string _name;      //!< image identifier
    uint16_t _width;   //!< width of the image (may differ from first pyramid level)
    uint16_t _height;  //!< height of the image (may differ from first pyramid level)

    //! pyramid levels holding image matches
    vector<drwnPatchMatchImageRecord> _levels;

 public:
    //! default constructor
    drwnPatchMatchImagePyramid() : drwnPersistentRecord(),
        bActive(true), eqvClass(-1), _name(""), _width(0), _height(0) {
        // do nothing
    }
    //! constructor an image record with a given name (and size)
    drwnPatchMatchImagePyramid(const string& name, uint16_t width, uint16_t height) :
        drwnPersistentRecord(), bActive(true), eqvClass(-1),
        _name(name), _width(width), _height(height) {
        DRWN_ASSERT((width > 0) && (height > 0));
        constructPyramidLevels();
    }
    //! destructor
    ~drwnPatchMatchImagePyramid() {
        // do nothing
    }

    //! image name
    const string& name() const { return _name; }
    //! image width
    uint16_t width() const { return _width; }
    //! image height
    uint16_t height() const { return _height; }
    //! image pyramid levels
    size_t levels() const { return _levels.size(); }

    //! clear all matches
    void clear();

    // i/o (from drwnPersistentRecord)
    size_t numBytesOnDisk() const;
    bool write(ostream& os) const;
    bool read(istream& is);

    //! map a pixel from one pyramid level to another
    inline cv::Point mapPixel(const cv::Point& p, int srcLevel, int dstLevel) const {
        return cv::Point(p.x * _levels[dstLevel].width() / _levels[srcLevel].width(),
            p.y * _levels[dstLevel].height() / _levels[srcLevel].height());
    }

    //! sample a patch (i.e., pyramid level and pixel position) in proportion to
    //! best scoring match for the patch
    bool samplePatchByScore(drwnPatchMatchNode& sample) const;

    //! reference pyramid level \p indx
    drwnPatchMatchImageRecord& operator[](uint8_t indx) { return _levels[indx]; }
    //! reference pyramid level \p indx
    const drwnPatchMatchImageRecord& operator[](uint8_t indx) const { return _levels[indx]; }

 protected:
    //! constructs the image pyramid
    void constructPyramidLevels();
};

// drwnPatchMatchGraph -------------------------------------------------------
//! Each image maintains a W-by-H-by-K array of match records referencing the
//! (approximate) best K matches to other images.
//!
//! That is, stores a PatchMatchGraph as described in Gould and Zhang,
//! ECCV 2012.
//!
//! \sa \ref drwnProjPatchMatch

class drwnPatchMatchGraph {
 public:
    static unsigned int PATCH_WIDTH;   //!< default patch width (at base scale)
    static unsigned int PATCH_HEIGHT;  //!< default patch height (at base scale)
    static unsigned int K;             //!< default number of matches per pixel

    string imageDirectory;             //!< directory to prepend to instance basename
    string imageExtension;             //!< extension to append to instance basename

 protected:
    unsigned int _patchWidth;          //!< patch width (at base scale in this graph)
    unsigned int _patchHeight;         //!< patch height (at base scale in this graph)
    vector<drwnPatchMatchImagePyramid *> _images; //!< image pyramid matches

 public:
    //! default constructor
    drwnPatchMatchGraph();
    //! copy constructor
    drwnPatchMatchGraph(const drwnPatchMatchGraph& graph);
    //! destructor
    ~drwnPatchMatchGraph();

    //! write to persistent storage
    bool write(const char *filestem) const;
    //! read from persistent storage
    bool read(const char *filestem);

    //! number of images in the graph
    size_t size() const { return _images.size(); }

    //! patch width used for constructing this graph
    unsigned int patchWidth() const { return _patchWidth; }
    //! patch height used for constructing this graph
    unsigned int patchHeight() const { return _patchHeight; }

    //! returns the index for an image or -1 if not in graph
    int findImage(const string& baseName) const;    
    //! add a single image (do not use while the graph is being learned)
    void appendImage(const string& baseName);
    //! add a single image of known size
    void appendImage(const string& baseName, const cv::Size& imgSize);
    //! add a set of images
    void appendImages(const vector<string>& baseNames);
    //! delete an image from the graph (do not use while the graph is being learned)
    //! and returns the number of edges removed which terminated at patches within the 
    //! deleted image
    int removeImage(unsigned imgIndx);

    //! returns (a reference to) the list of outgoing edges for a given node
    drwnPatchMatchEdgeList& edges(const drwnPatchMatchNode& node) {
        DRWN_ASSERT(node.xPosition < (*_images[node.imgIndx])[node.imgScale].width());
        DRWN_ASSERT(node.yPosition < (*_images[node.imgIndx])[node.imgScale].height());
        return (*_images[node.imgIndx])[node.imgScale](node.xPosition, node.yPosition);
    }
    //! returns (a constant reference to) the list of outgoing edges for a given node
    const drwnPatchMatchEdgeList& edges(const drwnPatchMatchNode& node) const {
        DRWN_ASSERT(node.xPosition < (*_images[node.imgIndx])[node.imgScale].width());
        DRWN_ASSERT(node.yPosition < (*_images[node.imgIndx])[node.imgScale].height());
        return (*_images[node.imgIndx])[node.imgScale](node.xPosition, node.yPosition);
    }

    //! sum of total match scores (first) and best match scores (second)
    pair<double, double> energy() const;

    //! full image filename of a given image
    string imageFilename(int indx) const { return imageFilename(_images[indx]->name()); }
    //! full image filename from an image basename
    string imageFilename(const string& baseName) const {
        return imageDirectory.empty() ? baseName + imageExtension :
            imageDirectory + DRWN_DIRSEP + baseName + imageExtension;
    }

    //! reference image pyramid \p indx
    inline drwnPatchMatchImagePyramid& operator[](unsigned indx) { return *_images[indx]; }
    //! reference image pyramid level \p indx
    inline const drwnPatchMatchImagePyramid& operator[](unsigned indx) const { return *_images[indx]; }
};

// drwnPatchMatchGraphLearner ------------------------------------------------
//! Learns a PatchMatchGraph by iteratively performing search moves over the
//! space of matches.
//!
//! \sa \ref drwnProjPatchMatch

class drwnPatchMatchGraphLearner {
 public:
    static double SEARCH_DECAY_RATE;   //!< decay rate during search in range [0, 1)
    static int FORWARD_ENRICHMENT_K;   //!< forward enrichment search depth
    static bool DO_INVERSE_ENRICHMENT; //!< perform inverse enrichment
    static bool DO_LOCAL_SEARCH;       //!< run neighbourhood search on dirty pixels
    static bool DO_EXHAUSTIVE;         //!< run exhustive search on random pixel

    static drwnPatchMatchTransform ALLOWED_TRANSFORMATIONS; //!< allowable patch transformations

    static double TOP_VAR_PATCHES;     //!< initialize all patches below this with only one match

#ifdef DRWN_DEBUG_STATISTICS
    static unsigned _dbPatchesScored;
#endif

 protected:
    drwnPatchMatchGraph &_graph;           //!< the graph that is being learned
    vector<vector<cv::Mat> > _features;    //!< image features indexed by (image, pyramid level)

 public:
    //! construct an object to optimize the given patch match graph
    drwnPatchMatchGraphLearner(drwnPatchMatchGraph& graph);
    virtual ~drwnPatchMatchGraphLearner();

    //! randomly initialize matches for all active images (keeps existing
    //! matches unless an improvement is found)
    void initialize();
    //! randomly initialize matches for a given image
    virtual void initialize(unsigned imgIndx);

    //! rescore matches if scoring function or features have changed (e.g., greyscale to rgb)
    void rescore();

    //! perform one update iteration (including enrichment and exhuastive)
    //! on all active images
    virtual void update();
    //! perform one update iteration (excluding enrichment) on a single image
    virtual void update(unsigned imgIndx);

 protected:
    // --- feature calculation ---

    //! \todo allow a pixel feature generator to be provided
    //! caches features for all image pyramids
    void cacheImageFeatures();
    //! caches features for given image pyramid
    virtual void cacheImageFeatures(unsigned imgIndx);

    //! helper functions for feature computation
    //! \todo move these elsewhere
    int appendCIELabFeatures(const cv::Mat& img, cv::Mat& features, int nChannel = 0) const;
    int appendVerticalFeatures(const cv::Mat& img, cv::Mat& features, int nChannel = 0) const;
    int appendEdgeFeatures(const cv::Mat& img, cv::Mat& features, int nChannel = 0) const;

    // --- move-making steps ---

    //! propagate good matches from patch \p u to its neighbours
    //! (returns true if a better match was found)
    bool propagate(const drwnPatchMatchNode& u, bool bDirection);
    //! random exponentially decaying neighbourhood search for patch \p pixIndx
    //! in image \p imgIndx (returns true if a better match was found)
    bool search(const drwnPatchMatchNode& u);
    //! local neighbourhood search around patch \p pixIndx in
    //! image \p imgIndx (returns true if a better match was found)
    bool local(const drwnPatchMatchNode& u);

    //! enrichment: inverse (from target to source) and forward (from
    //! source to target's target) (returns true if a better match was found)
    bool enrichment();

    //! exhaustive search for best match across entire graph --- use sparingly
    //! (returns true if a better match was found)
    bool exhaustive(const drwnPatchMatchNode& u);

    // --- utility routines ---

    //! match scoring function (\p maxValue allows for early termination)
    float scoreMatch(const drwnPatchMatchNode &u, const drwnPatchMatchNode& v,
        const drwnPatchMatchTransform& xform, float maxValue = DRWN_FLT_MAX) const;

    //! map a patch from one image pyramid level to another
    cv::Point mapPatch(const cv::Point& p, int imgIndx, int srcScale, int dstScale) const;
};

// drwnPatchMatchRetarget ----------------------------------------------------
//! Class for retargetting an image from matches within the PatchMatchGraph.
//!
//! \sa \ref drwnProjPatchMatch

class drwnPatchMatchGraphRetarget {
 protected:
    const drwnPatchMatchGraph &_graph;    //!< the graph that is being learned
    vector<cv::Mat> _labels;              //!< the labels that will be used for retargetting

 public:
    drwnPatchMatchGraphRetarget(const drwnPatchMatchGraph& graph);
    virtual ~drwnPatchMatchGraphRetarget();

    //! retargets an image based on matched patches
    virtual cv::Mat retarget(unsigned imgIndx) const;

 protected:
    //! caches labels for all images
    void cacheImageLabels();
    //! caches features for given image
    virtual void cacheImageLabels(unsigned imgIndx);
};

// drwnPatchMatchVisualization -----------------------------------------------
//! Visualization routines.

namespace drwnPatchMatchVis {
    //! visualize matches for a given pixel
    cv::Mat visualizeMatches(const drwnPatchMatchGraph &graph,
        int imgIndx, const cv::Point& p);

    //! visualize match quality for a given image
    cv::Mat visualizeMatchQuality(const drwnPatchMatchGraph &graph,
        int imgIndx, float maxScore = 0.0);

    //! visualize match quality over all active images
    cv::Mat visualizeMatchQuality(const drwnPatchMatchGraph &graph);

    //! visualize best match transformations for a given image
    cv::Mat visualizeMatchTransforms(const drwnPatchMatchGraph &graph,
        int imgIndx);

    //! visualize best match transformations for all active images
    cv::Mat visualizeMatchTransforms(const drwnPatchMatchGraph &graph);

    //! visualize best match target image for a given image
    cv::Mat visualizeMatchTargets(const drwnPatchMatchGraph &graph,
        int imgIndx);

    //! visualize best match targets for all active images
    cv::Mat visualizeMatchTargets(const drwnPatchMatchGraph &graph);
};
