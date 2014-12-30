/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLuaNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ Standard Libraries
#include <cstdlib>

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// Lua library
#include "lua.hpp"

// Plugin library
#include "drwnLuaNodes.h"
#include "drwnLuaShell.h"

using namespace std;
using namespace Eigen;

// drwnLuaNode ------------------------------------------------------------

drwnLuaNode::drwnLuaNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _nInputs(1), _nOutputs(1), _colour(-1), _luaEngine(NULL)
{
    _nVersion = 100;
    _desc = string("Interface to Lua (") + string(LUA_RELEASE) + string(")");

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D data matrix"));

    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("numOutputs", new drwnIntegerProperty(&_nOutputs));
    declareProperty("fwdFunction", new drwnFilenameProperty(&_fwdFcnName));
    declareProperty("bckFunction", new drwnFilenameProperty(&_bckFcnName));
    declareProperty("initFunction", new drwnFilenameProperty(&_initFcnName));
    declareProperty("estFunction", new drwnFilenameProperty(&_estFcnName));
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("theta", new drwnMatrixProperty(&_theta));
}

drwnLuaNode::drwnLuaNode(const drwnLuaNode& node) :
    drwnNode(node), _nInputs(node._nInputs), _nOutputs(node._nOutputs),
    _fwdFcnName(node._fwdFcnName), _bckFcnName(node._bckFcnName),
    _initFcnName(node._initFcnName), _estFcnName(node._estFcnName),
    _colour(node._colour), _theta(node._theta), _luaEngine(NULL)
{
    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("numOutputs", new drwnIntegerProperty(&_nOutputs));
    declareProperty("fwdFunction", new drwnFilenameProperty(&_fwdFcnName));
    declareProperty("bckFunction", new drwnFilenameProperty(&_bckFcnName));
    declareProperty("initFunction", new drwnFilenameProperty(&_initFcnName));
    declareProperty("estFunction", new drwnFilenameProperty(&_estFcnName));
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("theta", new drwnMatrixProperty(&_theta));
}

drwnLuaNode::~drwnLuaNode()
{
    // close the lua engine
    if (_luaEngine != NULL) {
        lua_close(_luaEngine);
    }
}

// i/o
bool drwnLuaNode::load(drwnXMLNode& xml)
{
    drwnNode::load(xml);
    updatePorts();
    return true;
}

// gui
void drwnLuaNode::showWindow()
{
    if (_luaEngine == NULL) {
        _luaEngine = lua_open();
        luaL_openlibs(_luaEngine);
    }
    DRWN_ASSERT_MSG(_luaEngine != NULL, "can't initialize the Lua interpreter");

    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnLuaShell(this, _luaEngine);
        _window->Show();
    }
    updateWindow();
}

/*
void drwnLuaNode::hideWindow()
{
    if (_luaEngine != NULL) {
        lua_close(_luaEngine);
        _luaEngine = NULL;
    }

    if (_window != NULL) {
        delete _window;
        _window = NULL;
    }
}
*/

void drwnLuaNode::updateWindow()
{
    if (_luaEngine == NULL)
        return;
    if ((_window == NULL) || (!_window->IsShown()))
        return;

    // TODO: copy parameters?
}

void drwnLuaNode::evaluateForwards()
{
    // clear output tables and then update forwards
    clearOutput();
    updateForwards();
}

void drwnLuaNode::updateForwards()
{
    // error checking
    if (_fwdFcnName.empty()) {
        DRWN_LOG_ERROR("no forward function defined for node \"" << getName() << "\"");
        return;
    }

    if (!drwnFileExists(_fwdFcnName.c_str())) {
        DRWN_LOG_ERROR("can't find forward function \"" << _fwdFcnName << "\" required for node \"" << getName() << "\"");
        return;
    }

    // if no inputs, then execute as source node
    if (_nInputs == 0) {
        sourceUpdateForwards();
        return;
    }

    // check input tables
    for (int i = 0; i < _nInputs; i++) {
        drwnDataTable *tblIn = _inputPorts[i]->getTable();
        if (tblIn == NULL) {
            DRWN_LOG_WARNING("node " << getName() << " requires " << _nInputs << " inputs");
            return;
        }
    }

    // run Lua script on input records
    if (_luaEngine == NULL) {
        _luaEngine = lua_open();
        luaL_openlibs(_luaEngine);
    }

    // interate over input records
    vector<string> keys = _inputPorts[0]->getTable()->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;

        // don't overwrite existing output records
        bool bKeyExists = true;
        for (int i = 0; i < _nOutputs; i++) {
            drwnDataTable *tblOut = _outputPorts[i]->getTable();
            if (!tblOut->hasKey(*it)) {
                bKeyExists = false;
                break;
            }
        }
        if (bKeyExists) continue;

        // TODO
        DRWN_TODO;
    }
    DRWN_END_PROGRESS;
}

void drwnLuaNode::sourceUpdateForwards()
{
    // TODO: run Lua script
    DRWN_TODO;
}

void drwnLuaNode::propertyChanged(const string& name)
{
    if ((name == string("numInputs")) || (name == string("numOutputs"))) {
        _nInputs = std::max(0, _nInputs);
        _nOutputs = std::max(0, _nOutputs);
        updatePorts();
    } else {
        drwnNode::propertyChanged(name);
    }
}

void drwnLuaNode::updatePorts()
{
    // re-assign input ports
    if (_nInputs != (int)_inputPorts.size()) {
        for (vector<drwnInputPort *>::iterator it = _inputPorts.begin();
             it != _inputPorts.end(); it++) {
            delete *it;
        }
        _inputPorts.clear();

        if (_nInputs == 1) {
            _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
        } else {
            for (int i = 0; i < _nInputs; i++) {
                string portName = string("dataIn") + toString(i);
                _inputPorts.push_back(new drwnInputPort(this, portName.c_str(), "N-by-D data matrix"));
            }
        }
    }

    // re-assign input ports
    if (_nOutputs != (int)_outputPorts.size()) {
        for (vector<drwnOutputPort *>::iterator it = _outputPorts.begin();
             it != _outputPorts.end(); it++) {
            delete *it;
        }
        _outputPorts.clear();

        if (_nOutputs == 1) {
            _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D data matrix"));
        } else {
            for (int i = 0; i < _nOutputs; i++) {
                string portName = string("dataOut") + toString(i);
                _outputPorts.push_back(new drwnOutputPort(this, portName.c_str(), "N-by-D data matrix"));
            }
        }
    }
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Custom", drwnLuaNode);
