/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLuaNodes.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Plugin for interfacing to Lua. Implements source, sink and generic data
**  processing nodes.
**
*****************************************************************************/

#pragma once

// darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// lua library
#include "lua.hpp"

using namespace std;

// drwnLuaNode ---------------------------------------------------------------

class drwnLuaNode : public drwnNode {
 protected:
    int _nInputs;           // number of input ports
    int _nOutputs;          // number of output ports
    string _fwdFcnName;     // filename of forward evaluation function
    string _bckFcnName;     // filename of backward propagation function
    string _initFcnName;    // filename of initialization function
    string _estFcnName;     // filename of parameter estimation function

    int _colour;            // colour for parameter updates
    MatrixXd _theta;        // model parameters

    lua_State *_luaEngine;  // the lua interpreter

 public:
    drwnLuaNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnLuaNode(const drwnLuaNode& node);
    virtual ~drwnLuaNode();

    // i/o
    const char *type() const { return "drwnLuaNode"; }
    drwnLuaNode *clone() const { return new drwnLuaNode(*this); }
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

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnLuaNode);

