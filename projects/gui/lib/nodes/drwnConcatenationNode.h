/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConcatenationNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Combines two (or more inputs). The inputs must all have the same number
**  of observations (or a single observation), i.e. N-by-D1, N-by-D2 or 1-by-D2.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnConcatenationNode -----------------------------------------------------

class drwnConcatenationNode : public drwnMultiIONode {
 protected:
    int _nInputs;        // number of input ports
    bool _bAddOnes;      // append a column of ones to the output

 public:
    drwnConcatenationNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnConcatenationNode(const drwnConcatenationNode& node);
    virtual ~drwnConcatenationNode();

    // i/o
    const char *type() const { return "drwnConcatenationNode"; }
    drwnConcatenationNode *clone() const { return new drwnConcatenationNode(*this); }
    bool load(drwnXMLNode& xml);

 protected:
    virtual bool forwardFunction(const string& key,
        const vector<const drwnDataRecord *>& src,
        const vector<drwnDataRecord *>& dst);
    virtual bool backwardGradient(const string& key,
        const vector<drwnDataRecord *>& src,
        const vector<const drwnDataRecord *>& dst);

    void propertyChanged(const string& name);
    void updatePorts(); // TODO: move this to drwnMultiIONode class as updatePorts(int nIn, int nOut)
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnConcatenationNode);

