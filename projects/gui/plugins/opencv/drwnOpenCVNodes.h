/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements plugin for OpenCV nodes.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnOpenCVImageSourceNode -------------------------------------------------

class drwnOpenCVImageSourceNode : public drwnNode {
 protected:
    string _directory;      // source directory
    string _extension;      // source extension

 public:
    drwnOpenCVImageSourceNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVImageSourceNode(const drwnOpenCVImageSourceNode& node);
    virtual ~drwnOpenCVImageSourceNode();

    // i/o
    const char *type() const { return "drwnOpenCVImageSourceNode"; }
    drwnOpenCVImageSourceNode *clone() const { return new drwnOpenCVImageSourceNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();
};

// drwnOpenCVImageSinkNode ---------------------------------------------------

class drwnOpenCVImageSinkNode : public drwnNode {
 protected:
    string _directory;      // sink directory
    string _extension;      // sink extension

 public:
    drwnOpenCVImageSinkNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVImageSinkNode(const drwnOpenCVImageSinkNode& node);
    virtual ~drwnOpenCVImageSinkNode();

    // i/o
    const char *type() const { return "drwnOpenCVImageSinkNode"; }
    drwnOpenCVImageSinkNode *clone() const { return new drwnOpenCVImageSinkNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();

 protected:
    void exportImage(const string &filename, const drwnDataRecord *dataRecIn) const;
};

// drwnOpenCVResizeNode ------------------------------------------------------

class drwnOpenCVResizeNode : public drwnMultiIONode {
 protected:
    int _defaultWidth;      // default resize width (if newSizeIn not given)
    int _defaultHeight;     // default resize height (if newSizeIn not given)

    static vector<string> _interpolationMethods;
    int _interpolation;

 public:
    drwnOpenCVResizeNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVResizeNode(const drwnOpenCVResizeNode& node);
    virtual ~drwnOpenCVResizeNode();

    // i/p
    const char *type() const { return "drwnOpenCVResizeNode"; }
    drwnOpenCVResizeNode *clone() const { return new drwnOpenCVResizeNode(*this); }

 protected:
    // processing
    virtual bool forwardFunction(const string& key,
        const vector<const drwnDataRecord *>& src,
        const vector<drwnDataRecord *>& dst);
    virtual bool backwardGradient(const string& key,
        const vector<drwnDataRecord *>& src,
        const vector<const drwnDataRecord *>& dst);
};

// drwnOpenCVFilterBankNode --------------------------------------------------

class drwnOpenCVFilterBankNode : public drwnSimpleNode {
 protected:
    double _kappa;          // filter bandwidth

 public:
    drwnOpenCVFilterBankNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVFilterBankNode(const drwnOpenCVFilterBankNode& node);
    virtual ~drwnOpenCVFilterBankNode();

    // i/o
    const char *type() const { return "drwnOpenCVFilterBankNode"; }
    drwnOpenCVFilterBankNode *clone() const { return new drwnOpenCVFilterBankNode(*this); }

 protected:
    // processing
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// drwnOpenCVIntegralImageNode -----------------------------------------------

class drwnOpenCVIntegralImageNode : public drwnSimpleNode {
 protected:

 public:
    drwnOpenCVIntegralImageNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVIntegralImageNode(const drwnOpenCVIntegralImageNode& node);
    virtual ~drwnOpenCVIntegralImageNode();

    // i/o
    const char *type() const { return "drwnOpenCVIntegralImageNode"; }
    drwnOpenCVIntegralImageNode *clone() const { return new drwnOpenCVIntegralImageNode(*this); }

 protected:
    // processing
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// drwnOpenCVNeighborhoodFeaturesNode ----------------------------------------

class drwnOpenCVNeighborhoodFeaturesNode : public drwnSimpleNode {
 protected:
    int _cellSize;
    bool _bIncludeRow;
    bool _bIncludeCol;

 public:
    drwnOpenCVNeighborhoodFeaturesNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnOpenCVNeighborhoodFeaturesNode(const drwnOpenCVNeighborhoodFeaturesNode& node);
    virtual ~drwnOpenCVNeighborhoodFeaturesNode();

    // i/o
    const char *type() const { return "drwnOpenCVNeighborhoodFeaturesNode"; }
    drwnOpenCVNeighborhoodFeaturesNode *clone() const { return new drwnOpenCVNeighborhoodFeaturesNode(*this); }

 protected:
    // processing
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVImageSourceNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVImageSinkNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVResizeNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVFilterBankNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVIntegralImageNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnOpenCVNeighborhoodFeaturesNode);
