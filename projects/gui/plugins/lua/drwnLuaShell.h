/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLuaShell.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Window for interfacing to the Lua shell.
**
*****************************************************************************/

#pragma once

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// Lua library
#include "lua.hpp"

using namespace std;

// drwnLuaShell --------------------------------------------------------------

class drwnLuaShell : public drwnGUIShellWindow {
 protected:
    lua_State *_luaEngine; // lua engine

 public:
    drwnLuaShell(drwnNode *owner, lua_State *L);
    ~drwnLuaShell();

 protected:
    bool executeCommand(const string& cmd); 
};
