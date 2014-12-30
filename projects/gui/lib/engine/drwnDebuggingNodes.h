/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDebuggingNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements nodes for code debugging and regression testing. These nodes
**  should not be shipped in any release.
**
*****************************************************************************/

#pragma once

using namespace std;

// drwnRandomSourceNode ------------------------------------------------------
// Generates random source records.

class drwnRandomSourceNode : public drwnNode {
 protected:
    int _numRecords;        // number of records to generate
    int _numFeatures;       // number of features per record
    int _minObservations;   // minimum number of observations
    int _maxObservations;   // maximum number of observations

 public:
    drwnRandomSourceNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnRandomSourceNode(const drwnRandomSourceNode& node);
    virtual ~drwnRandomSourceNode();

    // i/o
    const char *type() const { return "drwnRandomSourceNode"; }
    drwnRandomSourceNode *clone() const { return new drwnRandomSourceNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();
};

// drwnStdOutSinkNode --------------------------------------------------------
// Writes records to standard output.

class drwnStdOutSinkNode : public drwnNode {
 public:
    drwnStdOutSinkNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnStdOutSinkNode(const drwnStdOutSinkNode& node);
    virtual ~drwnStdOutSinkNode();

    // i/o
    const char *type() const { return "drwnStdOutSinkNode"; }
    drwnStdOutSinkNode *clone() const { return new drwnStdOutSinkNode(*this); }

    // processing
    void evaluateForwards();
};

