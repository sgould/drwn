/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearTransformNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements linear transformation nodes (including principal component
**  analysis node and multi-class linear discriminant analysis (aka canonical
**  variates)).
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnLinearTransformNode ---------------------------------------------------

class drwnLinearTransformNode : public drwnSimpleNode {
 protected:
    VectorXd _translation;
    MatrixXd _projection;

 public:
    drwnLinearTransformNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnLinearTransformNode(const drwnLinearTransformNode& node);
    virtual ~drwnLinearTransformNode();

    // i/o
    const char *type() const { return "drwnLinearTransformNode"; }
    drwnLinearTransformNode *clone() const { return new drwnLinearTransformNode(*this); }

 protected:
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// drwnRescaleNode -----------------------------------------------------------

class drwnRescaleNode : public drwnLinearTransformNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)

 public:
    drwnRescaleNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnRescaleNode(const drwnRescaleNode& node);
    virtual ~drwnRescaleNode();

    // i/o
    const char *type() const { return "drwnRescaleNode"; }
    drwnRescaleNode *clone() const { return new drwnRescaleNode(*this); }

    // learning
    void resetParameters();
    void initializeParameters();
};

// drwnPCANode ---------------------------------------------------------------

class drwnPCANode : public drwnLinearTransformNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)
    int _numOutputDims;    // number of output dimensions required

 public:
    drwnPCANode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnPCANode(const drwnPCANode& node);
    virtual ~drwnPCANode();

    // i/o
    const char *type() const { return "drwnPCANode"; }
    drwnPCANode *clone() const { return new drwnPCANode(*this); }

    // learning
    void resetParameters();
    void initializeParameters();
};

// drwnMultiClassLDANode -----------------------------------------------------

class drwnMultiClassLDANode : public drwnLinearTransformNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)
    double _lambda;        // regularization

 public:
    drwnMultiClassLDANode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnMultiClassLDANode(const drwnMultiClassLDANode& node);
    virtual ~drwnMultiClassLDANode();

    // i/o
    const char *type() const { return "drwnMultiClassLDANode"; }
    drwnMultiClassLDANode *clone() const { return new drwnMultiClassLDANode(*this); }

    // learning
    void resetParameters();
    void initializeParameters();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnLinearTransformNode)
DRWN_DECLARE_AUTOREGISTERNODE(drwnRescaleNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnPCANode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnMultiClassLDANode);


