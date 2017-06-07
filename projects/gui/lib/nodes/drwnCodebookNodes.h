/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCodebookNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements different codebook nodes: LUT (lookup table), k-Means, etc.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnLUTDecoderNode --------------------------------------------------------

class drwnLUTDecoderNode : public drwnSimpleNode {
 protected:
    MatrixXd _lut;                // the lookup table

 public:
    drwnLUTDecoderNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnLUTDecoderNode(const drwnLUTDecoderNode& node);
    virtual ~drwnLUTDecoderNode();

    // i/o
    const char *type() const { return "drwnLUTDecoderNode"; }
    drwnLUTDecoderNode *clone() const { return new drwnLUTDecoderNode(*this); }

 protected:
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// drwnLUTEncoderNode --------------------------------------------------------

class drwnLUTEncoderNode : public drwnSimpleNode {
 protected:
    static vector<string> _modeProperties;
    int _mode;                    // exact, nearest (L2), nearest (L1)
    MatrixXd _lut;                // the lookup table

 public:
    drwnLUTEncoderNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnLUTEncoderNode(const drwnLUTEncoderNode& node);
    virtual ~drwnLUTEncoderNode();

    // i/o
    const char *type() const { return "drwnLUTEncoderNode"; }
    drwnLUTEncoderNode *clone() const { return new drwnLUTEncoderNode(*this); }

 protected:
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnLUTDecoderNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnLUTEncoderNode);

