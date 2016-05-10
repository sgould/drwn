/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextFileNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements text file data source and sink nodes. Assumes the first column
**  of the text file indexes the record (key) and the remaining columns are
**  the data.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnTextFileSourceNode ----------------------------------------------------

class drwnTextFileSourceNode : public drwnNode {
 protected:
    string _filename;       // source filename
    string _delimiter;      // field seperator
    int _numHeaderLines;    // number of header lines
    bool _bIncludesKey;     // includes key field
    string _keyPrefix;      // key prefix (even if key field is included) 

 public:
    drwnTextFileSourceNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnTextFileSourceNode(const drwnTextFileSourceNode& node);
    virtual ~drwnTextFileSourceNode();

    // i/o
    const char *type() const { return "drwnTextFileSourceNode"; }
    drwnTextFileSourceNode *clone() const { return new drwnTextFileSourceNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();
};

// drwnTextFileSinkNode ------------------------------------------------------

class drwnTextFileSinkNode : public drwnNode {
 protected:
    string _filename;       // source filename
    string _delimiter;      // field seperator
    bool _bIncludeHeader;   // include header line
    bool _bIncludeKey;      // include key field

 public:
    drwnTextFileSinkNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnTextFileSinkNode(const drwnTextFileSinkNode& node);
    virtual ~drwnTextFileSinkNode();

    // i/o
    const char *type() const { return "drwnTextFileSinkNode"; }
    drwnTextFileSinkNode *clone() const { return new drwnTextFileSinkNode(*this); }

    // processing
    void evaluateForwards();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnTextFileSourceNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnTextFileSinkNode);

