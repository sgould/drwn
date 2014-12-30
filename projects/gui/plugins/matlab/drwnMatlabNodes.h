/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Plugin for interfacing to Matlab. Implements source, sink and generic data
**  processing nodes. The Matlab function prototypes are:
**
**    [recOut1, recOut2, ...] = fwdFcn(recIn1, recIn2, ..., parameters);
**    [recIn1, recIn2, ...] = bckFcn(recIn1, recIn2, ..., recOu1, recOut2, ..., parameters);
**    [parameters] = initFcn(recIn1, recIn2, ..., parameters, options);
**
**  TODO: global learning functions
**
**  where recIn and recOut are structures with fields:
**    key (string)
**    colour (integer)
**    data (N-by-D)
**    objective (empty or N-by-1)
**    gradient (empty or N-by-D)
**
**  If the number of input ports is zero, then the node acts like a source
**  node. The prototype is
**
**    [recOut1, recOut2, ...] = fwdFcn(index, parameters);
**
**  and the function is called until recOut1.key is empty.
**
*****************************************************************************/

#pragma once

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// Matlab library
#include "engine.h"
#include "mat.h"

using namespace std;

// drwnMatlabNode ------------------------------------------------------------

class drwnMatlabNode : public drwnNode {
 protected:
    static string STARTCMD; // matlab start command (and arguments
    static Engine *_matlabEngine; // shared interface to matlab
    static int _refCount;   // matlab engine reference count

    int _nInputs;           // number of input ports
    int _nOutputs;          // number of output ports
    string _fwdFcnName;     // filename of forward evaluation function
    string _bckFcnName;     // filename of backward propagation function
    string _initFcnName;    // filename of initialization function
    string _estFcnName;     // filename of parameter estimation function

    int _colour;            // colour for parameter updates
    MatrixXd _theta;        // model parameters

 public:
    drwnMatlabNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnMatlabNode(const drwnMatlabNode& node);
    virtual ~drwnMatlabNode();

    // i/o
    const char *type() const { return "drwnMatlabNode"; }
    drwnMatlabNode *clone() const { return new drwnMatlabNode(*this); }
    bool load(drwnXMLNode& xml);

    // gui
    virtual void showWindow();
    virtual void updateWindow();

    // processing
    void evaluateForwards();
    void updateForwards();

 protected:
    void sourceUpdateForwards();

    void propertyChanged(const string& name);
    void updatePorts();
};

// drwnMATFileSourceNode -----------------------------------------------------

class drwnMATFileSourceNode : public drwnNode {
 protected:
    string _filename;
    bool _bFullRecord;

 public:
    drwnMATFileSourceNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnMATFileSourceNode(const drwnMATFileSourceNode& node);
    virtual ~drwnMATFileSourceNode();

    // i/o
    const char *type() const { return "drwnMATFileSourceNode"; }
    drwnMATFileSourceNode *clone() const { return new drwnMATFileSourceNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();
};

// drwnMATFileSinkNode -------------------------------------------------------

class drwnMATFileSinkNode : public drwnNode {
 protected:
    string _filename;
    bool _bFullRecord;

 public:
    drwnMATFileSinkNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnMATFileSinkNode(const drwnMATFileSinkNode& node);
    virtual ~drwnMATFileSinkNode();

    // i/o
    const char *type() const { return "drwnMATFileSinkNode"; }
    drwnMATFileSinkNode *clone() const { return new drwnMATFileSinkNode(*this); }

    // processing
    void evaluateForwards();
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnMatlabNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnMATFileSourceNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnMATFileSinkNode);
