/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnUnaryOpNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements component-wise math operations: sqr, sqrt, inv, exp, log,
**  norm, exp-norm, pow-norm, etc.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnUnaryOpNode -----------------------------------------------------------

class drwnUnaryOpNode : public drwnSimpleNode {
 protected:
    static vector<string> _operations;
    int _selection;
    double _argument;

 public:
    drwnUnaryOpNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnUnaryOpNode(const drwnUnaryOpNode& node);
    virtual ~drwnUnaryOpNode();

    // i/o
    const char *type() const { return "drwnUnaryOpNode"; }
    drwnUnaryOpNode *clone() const { return new drwnUnaryOpNode(*this); }

 protected:
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnUnaryOpNode);

