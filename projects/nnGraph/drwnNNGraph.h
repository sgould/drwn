/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraph.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
#include <cstdint>
#endif
#include <stdint.h>
#include <cassert>
#include <list>

#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphEdgeStatus -----------------------------------------------------
//! Status of a drwnNNGraphEdge object indicating whether the edge is new or
//! has been processed by a propagation move.

typedef enum { DRWN_NNG_DIRTY,           //!< the edge is new
               DRWN_NNG_PROCESSED_ONCE,  //!< the edge has been through one propagation pass
               DRWN_NNG_PROCESSED_TWICE  //!< the edge has been through two propagation passes
} drwnNNGraphEdgeStatus;

// drwnNNGraphNodeIndex ------------------------------------------------------

class drwnNNGraphNodeIndex {
 public:
    uint16_t imgIndx;                    //!< image index for this node
    uint16_t segId;                      //!< superpixel identifier this node

 public:
    //! default constructor
    drwnNNGraphNodeIndex() : imgIndx(0), segId(0) { /* do nothing */ }
    //! constructor
    drwnNNGraphNodeIndex(uint16_t i, uint16_t s) : imgIndx(i), segId(s) { /* do nothing */ }

    //! default comparison for sorting
    bool operator<(const drwnNNGraphNodeIndex& n) const {
        if (imgIndx < n.imgIndx) return true;
        if (imgIndx > n.imgIndx) return false;
        return (segId < n.segId);
    }

    //! comparison operator
    bool operator==(const drwnNNGraphNodeIndex& n) const {
        return ((imgIndx == n.imgIndx) && (segId == n.segId));
    }

    //! not-equal-to operator
    bool operator!=(const drwnNNGraphNodeIndex& n) const {
        return ((imgIndx != n.imgIndx) || (segId != n.segId));
    }

    //! stream output operator
    friend ostream& operator<<(ostream& os, const drwnNNGraphNodeIndex& n) { 
        os << "(" << n.imgIndx << ", " << n.segId << ")"; return os;
    }
};

// drwnNNGraphEdge -----------------------------------------------------------
//! Encapsulates an outgoing edge in a drwnNNGraph

class drwnNNGraphEdge {
 public:
    drwnNNGraphNodeIndex targetNode;     //!< index of the target node
    float weight;                        //!< weight ot score for this edge

    drwnNNGraphEdgeStatus status;        //!< status of this edge

 public:
    //! default constructor
    drwnNNGraphEdge() :
        weight(DRWN_FLT_MAX), status(DRWN_NNG_DIRTY) { /* do nothing */ }
    //! constructor
    drwnNNGraphEdge(const drwnNNGraphNodeIndex& tgtIndx, float w) :
        targetNode(tgtIndx), weight(w), status(DRWN_NNG_DIRTY) { /* do nothing */ }

    //! default comparison for sorting
    bool operator<(const drwnNNGraphEdge& e) const {
        if (weight < e.weight) return true;
        if (weight > e.weight) return false;
        return (targetNode < e.targetNode);
    }
};

// drwnNNGraphEdge sorting functions -----------------------------------------

class drwnNNGraphSortByScore : public std::binary_function<drwnNNGraphEdge, drwnNNGraphEdge, bool> {
 public:
    bool operator()(const drwnNNGraphEdge& a, const drwnNNGraphEdge& b) const {
        return (a.weight < b.weight);
    }
};

class drwnNNGraphSortByImage : public std::binary_function<drwnNNGraphEdge, drwnNNGraphEdge, bool> {
 public:
    bool operator()(const drwnNNGraphEdge& a, const drwnNNGraphEdge& b) const {
        return (a.targetNode.imgIndx < b.targetNode.imgIndx) ||
            ((a.targetNode.imgIndx == b.targetNode.imgIndx) && (a.weight < b.weight));
    }
};

// drwnNNGraphEdgeList -------------------------------------------------------
//! List of (out-going) edges.

typedef list<drwnNNGraphEdge> drwnNNGraphEdgeList;

// drwnNNGraphNode -----------------------------------------------------------
//! Encapsulates a superpixel node in a drwnNNGraph.

class drwnNNGraphNode : public drwnPersistentRecord {
 public:
    VectorXf features;                //!< features for this node
    int32_t label;                    //!< label for this node

    drwnNNGraphEdgeList edges;        //!< sorted outgoing edges (match neighbours)
    set<uint16_t> spatialNeighbours;  //!< set of spatial neighbours (for search)

 public:
    //! default constructor
    drwnNNGraphNode() : label(-1) { /* do nothing */ }
    //! constructor
    drwnNNGraphNode(const VectorXf& x, int32_t y = -1) :
        features(x), label(-1) { /* do nothing */ }

    //! clear features, label, edges and spatial neighbours
    void clear();

    //! update node with a potential edge (removing previous worst edge)
    bool insert(const drwnNNGraphEdge& e);

    //! number of bytes needed on disk
    size_t numBytesOnDisk() const;
    //! read from a stream (deserialization)
    bool read(istream& is);
    //! write to a stream (serialization)
    bool write(ostream& os) const;
};

// drwnNNGraphImageData ------------------------------------------------------
//! Holds image, segments and other housekeeping information for an image.

class drwnNNGraphImageData {
 public:
    static string imgDir;              //!< directory to prepend to instance basename for images
    static string imgExt;              //!< extension to append to instance basename for images
    static string lblDir;              //!< directory to prepend to instance basename for labels
    static string lblExt;              //!< extension to append to instance basename for labels
    static string segDir;              //!< directory to prepend to instance basename for segments
    static string segExt;              //!< extension to append to instance basename for segments

 protected:
    string _name;                       //!< basename for this image
    cv::Mat _img;                       //!< the RGB image
    MatrixXi _labels;                   //!< pixel labels
    drwnSuperpixelContainer _segments;  //!< superpixel maps
    vector<unsigned> _colours;          //!< cached 24-bit colour for visualization
    vector<cv::Point> _centroids;       //!< superpixel centroids for visualization

 public:
    //! construct from basename
    drwnNNGraphImageData(const string &name);
    //! construct from image (no labels or name)
    drwnNNGraphImageData(const cv::Mat& img, const drwnSuperpixelContainer& segments);

    //! return the name of the image
    const string& name() const { return _name; }
    //! return the image
    const cv::Mat& image() const { return _img; }
    //! return the labels
    const MatrixXi& labels() const { return _labels; }
    //! return the superpixel container
    const drwnSuperpixelContainer& segments() const { return _segments; }
    //! return the superpixel container
    drwnSuperpixelContainer& segments() { return _segments; }

    //! return the number of superpixels
    size_t numSegments() const { return _segments.size(); }
    //! return image height
    size_t height() const { return _img.rows; }
    //! return image width
    size_t width() const { return _img.cols; }

    //! return average colour (as 24-bit unsigned) for a given superpixel
    unsigned colour(unsigned segId) const { return _colours[segId]; }
    //! return average colour (as 8-bit rgb) for a given superpixle
    cv::Scalar rgbColour(unsigned segId) const { return CV_RGB(_colours[segId] & 0x0000ff,
        (_colours[segId] >> 8) & 0x0000ff, (_colours[segId] >> 16) & 0x0000ff); }

    //! return centroid for a given superpixel
    cv::Point centroid(unsigned segId) const { return _centroids[segId]; }

    //! set new labels for this image
    void setLabels(const MatrixXi& labels);

    //! calculate segment label distributions
    vector<VectorXd> getSegmentLabelMarginals(int numLabels = -1) const;

 protected:
    //! cache segments and neighbourhood
    void cacheSegmentData();
};

// drwnNNGraphImage ----------------------------------------------------------
//! Holds nodes (superpixels) for a single image.

class drwnNNGraphImage : public drwnPersistentRecord {
 public:
    //! Include this image during update (initialize, propagate, local, search, and
    //! enrichment). If false the outgoing edges from all nodes in this image are fixed.
    bool bSourceMatchable;

    //! Allow other image to match to this one. If false no new edges will be created
    //! that end at any node in this image (but existing incoming edges may be removed).
    //! During updates this has the effect of putting this image in the same equivalence
    //! class as all any other image.
    bool bTargetMatchable;

    //! Equivalence class for this image. If negative (which is the default) the image
    //! is considered to be in its own equivalence class. Images in the same equivalence
    //! class will not be matched when learning the superpixel graph.
    int32_t eqvClass;

 protected:
    string _name;                     //!< basename for this image
    vector<drwnNNGraphNode> _nodes;   //!< nodes associated with this image

 public:
    //! default constructor
    drwnNNGraphImage() : drwnPersistentRecord(), bSourceMatchable(true), bTargetMatchable(true),
        eqvClass(-1), _name("") { /* do nothing */ }
    //! constructor with n nodes
    drwnNNGraphImage(const string& name, unsigned n = 0);
    //! constructor from image data
    drwnNNGraphImage(const drwnNNGraphImageData& image);
    //! destructor
    ~drwnNNGraphImage() { /* do nothing */ }

    //! return the name of the image
    const string& name() const { return _name; }
    //! return the number of nodes (superpixels)
    size_t numNodes() const { return _nodes.size(); }

    //! clear all nodes (including features and labels) and out-going edges
    void clearNodes();
    //! clears all out-going edges (matches) for this image
    void clearEdges();

    //! initialize number of nodes (no features or labels)
    void initialize(const string& name, unsigned n = 0);
    //! initialize nodes with features and labels from image data
    virtual void initialize(const drwnNNGraphImageData& image);

    //! arbitrary feature transformation
    void transformNodeFeatures(const drwnFeatureTransform& xform);
    //! append node features derived from pixel features
    void appendNodeFeatures(const drwnNNGraphImageData& image, const cv::Mat& features);
    //! append a set of node features derived from pixel features
    void appendNodeFeatures(const drwnNNGraphImageData& image, const vector<cv::Mat>& features);

    //! number of bytes needed on disk
    size_t numBytesOnDisk() const;
    //! read from a stream (deserialization)
    bool read(istream& is);
    //! write to a stream (serialization)
    bool write(ostream& os) const;

    //! clone the image (with or without node features)
    drwnNNGraphImage clone(bool bWithFeatures = true) const;

    //! returns const reference to the node for superpixel \p segId
    inline const drwnNNGraphNode& operator[](unsigned segId) const { return _nodes[segId]; }
    //! returns reference to the node for superpixel \p segId
    inline drwnNNGraphNode& operator[](unsigned segId) { return _nodes[segId]; }

 protected:
    //! cache the spatial neighbours for each node
    virtual void cacheNodeNeighbourhoods(const drwnNNGraphImageData& image);
    //! cache (standard) node features
    virtual void cacheNodeFeatures(const drwnNNGraphImageData& image);
    //! cache node labels
    virtual void cacheNodeLabels(const drwnNNGraphImageData& image);
};

// drwnNNGraph ---------------------------------------------------------------
//! Class for maintaining a nearest neighbour graph over superpixel images.
//! Search moves are implemented by templated functions in the drwnNNGraphMoves
//! namespace.
//!
class drwnNNGraph {
 public:
    static unsigned int K;             //!< default number of matches per node
    static bool DO_PROPAGATE;          //!< execute propagate move
    static bool DO_LOCAL;              //!< execute local move
    static bool DO_SEARCH;             //!< execute random search move
    static int  DO_RANDPROJ;           //!< execute random projection move to given horizon
    static bool DO_ENRICHMENT;         //!< execute enrichment moves
    static int DO_EXHAUSTIVE;          //!< execute n exhaustive search moves per iteration 

 protected:
    vector<drwnNNGraphImage> _images;  //!< graph images containing nodes
    map<string, unsigned> _names;      //!< image name lookup

 public:
    //! default constructor
    drwnNNGraph() { /* do nothing */ }
    //! destructor
    virtual ~drwnNNGraph() { /* do nothing */ }

    //! write to persistent storage
    bool write(const char *filestem) const;
    //! read from persistent storage
    bool read(const char *filestem);

    //! clear the entire graph
    void clear() { _images.clear(); }
    //! reserve a given number of images (reduced memory allocation)
    void reserve(size_t n) { _images.reserve(_images.size() + n); }

    //! clone the graph (with or without node features)
    drwnNNGraph clone(bool bWithFeatures = true) const;

    //! number of images in the graph
    size_t numImages() const { return _images.size(); }
    //! number of nodes (superpixels) in the graph
    size_t numNodes() const;
    //! number of edges in the graph
    size_t numEdges() const;

    //! count the number of nodes (superpixels) with a given label
    size_t numNodesWithLabel(int label) const;

    //! returns the index for an image or -1 if not in graph
    int findImage(const string& baseName) const;
    //! add a single image and return its index (do not use while the graph is being learned)
    int appendImage(const string& baseName, unsigned numNodes = 0);
    //! add a drwnNNGraphImage and return its index (do not use while the graph is being learned)
    int appendImage(const drwnNNGraphImage& image);
    //! delete an image from the graph (do not use while the graph is being learned)
    //! and returns the number of edges removed which terminated at nodes within the
    //! deleted image
    int removeImage(unsigned imgIndx);

    //! sum of all edge weights (first) and best edge weights (second)
    pair<double, double> energy() const;

    //! return true if two images are in the same equivalence class
    inline bool inSameEqvClass(unsigned imgIndxA, unsigned imgIndxB) const {
        if (imgIndxA == imgIndxB) return true;
        if (_images[imgIndxA].eqvClass < 0) return false;
        return (_images[imgIndxA].eqvClass == _images[imgIndxB].eqvClass);
    }

    //! returns const reference to an image (and its nodes)
    inline const drwnNNGraphImage& operator[](size_t indx) const { return _images[indx]; }
    //! returns reference an image (and its nodes)
    inline drwnNNGraphImage& operator[](size_t indx) { return _images[indx]; }

    //! returns const reference to an image (and its nodes)
    inline const drwnNNGraphImage& operator[](const string& baseName) const {
        const map<string, unsigned>::const_iterator it = _names.find(baseName);
        return _images[it->second]; 
    }
    //! returns reference an image (and its nodes)
    inline drwnNNGraphImage& operator[](const string& baseName) {
        const map<string, unsigned>::const_iterator it = _names.find(baseName);
        return _images[it->second]; 
    }

    //! returns const reference to a node
    inline const drwnNNGraphNode& operator[](const drwnNNGraphNodeIndex& indx) const {
        return _images[indx.imgIndx][indx.segId];
    }
    //! returns reference a node
    inline drwnNNGraphNode& operator[](const drwnNNGraphNodeIndex& indx) {
        return _images[indx.imgIndx][indx.segId];
    }
};

// drwnNNGraphNodeAnnotation -------------------------------------------------
//! Templated utility class for holding annotations for every node in a graph.
//! See learning code for example use.
//!
template <typename T>
class drwnNNGraphNodeAnnotation {
 protected:
    vector<vector<T> > _tags;

 public:
    //! default constructor (use initialize after constructing)
    drwnNNGraphNodeAnnotation() { /* do nothing */ }
    //! construct for a given graph
    drwnNNGraphNodeAnnotation(const drwnNNGraph& graph, const T& initValue) {
        initialize(graph, initValue);
    }
    //! destructor
    ~drwnNNGraphNodeAnnotation() { /* do nothing */ }

    //! Initialize annotation for a given graph. Sets all tags to \p initValue.
    void initialize(const drwnNNGraph& graph, const T& initValue) {
        _tags.clear();
        _tags.resize(graph.numImages());
        for (size_t imgIndx = 0; imgIndx < graph.numImages(); imgIndx++) {
            _tags[imgIndx].resize(graph[imgIndx].numNodes(), initValue);
        }
    }

    //! reset all tags to \p initValue.
    void reset(const T& initValue) {
        for (size_t imgIndx = 0; imgIndx < _tags.size(); imgIndx++) {
            for (size_t segId = 0; segId < _tags[imgIndx].size(); segId++) {
                _tags[imgIndx][segId] = initValue;
            }
        }
    }

    //! returns const reference to annotations for an entire image
    inline const vector<T>& operator[](unsigned imgIndx) const {
        return _tags[imgIndx];
    }
    //! returns reference an annotation for an entire image
    inline vector<T>& operator[](unsigned imgIndx) {
        return _tags[imgIndx];
    }

    //! returns const reference to an annotation for an individual node
    inline const T& operator[](const drwnNNGraphNodeIndex& indx) const {
        return _tags[indx.imgIndx][indx.segId];
    }
    //! returns reference an annotation for an individual node
    inline T& operator[](const drwnNNGraphNodeIndex& indx) {
        return _tags[indx.imgIndx][indx.segId];
    }
};
