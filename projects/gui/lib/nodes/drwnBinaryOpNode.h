/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBinaryOpNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements component-wise binary operations between two inputs. The inputs
**  must be N-by-D, N-by-1, 1-by-D or 1-by-1. The dimensions are extended in
**  the obvious way.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnBinaryOpNode ----------------------------------------------------------

class drwnBinaryOpNode : public drwnMultiIONode {
 protected:
    static vector<string> _operations;
    int _selection;

 public:
    drwnBinaryOpNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnBinaryOpNode(const drwnBinaryOpNode& node);
    virtual ~drwnBinaryOpNode();

    // i/o
    const char *type() const { return "drwnBinaryOpNode"; }
    drwnBinaryOpNode *clone() const { return new drwnBinaryOpNode(*this); }

 protected:
    virtual bool forwardFunction(const string& key,
        const vector<const drwnDataRecord *>& src,
        const vector<drwnDataRecord *>& dst);
    virtual bool backwardGradient(const string& key,
        const vector<drwnDataRecord *>& src,
        const vector<const drwnDataRecord *>& dst);
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnBinaryOpNode);

