/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLuaShell.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard libraries
#include <cstdlib>

// Darwin library
#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

// Lua library
#include "lua.hpp"

#include "drwnLuaShell.h"

using namespace std;

// drwnLuaShell --------------------------------------------------------------

drwnLuaShell::drwnLuaShell(drwnNode *owner, lua_State *L) :
    drwnGUIShellWindow(owner), _luaEngine(L)
{
    // do nothing
}

drwnLuaShell::~drwnLuaShell()
{
    // do nothing
}

bool drwnLuaShell::executeCommand(const string& cmd)
{
    static size_t BUFFER_SIZE = 8192;

    // ensure that Lua has been initialized
    DRWN_ASSERT(_luaEngine != NULL);

    // intercept special commands
    if ((strcasecmp(cmd.c_str(), "exit") == 0) || 
        (strcasecmp(cmd.c_str(), "exit;") == 0)) {
        Close(); // close the window
        return true;
    }

    char *buffer = new char[BUFFER_SIZE];
    buffer[0] = '\0';
    fflush(stdout);
    setvbuf(stdout, buffer, _IOFBF, BUFFER_SIZE);

    // execute command
    int status = luaL_loadbuffer(_luaEngine, cmd.c_str(), cmd.size(), _owner->getName().c_str());
    if (status == 0) {
        //status = lua_pcall(_luaEngine, 0, LUA_MULTRET, 0);
        status = lua_pcall(_luaEngine, 0, 0, lua_gettop(_luaEngine));
    }

    if (status != 0) {
        // show error message
        wxFont font = _shellOutput->GetDefaultStyle().GetFont();
        font.SetWeight(wxFONTWEIGHT_BOLD);
        _shellOutput->SetDefaultStyle(wxTextAttr(*wxRED, wxNullColour, font));

        _shellOutput->AppendText(lua_tostring(_luaEngine, -1));
        _shellOutput->AppendText("\n");
        DRWN_LOG_ERROR(lua_tostring(_luaEngine, -1) << " (status: " << status << ")");
        lua_pop(_luaEngine, 1); // remove error message

        font.SetWeight(wxFONTWEIGHT_NORMAL);
        _shellOutput->SetDefaultStyle(wxTextAttr(*wxBLACK, wxNullColour, font));
    } else {
        // show any output
        _shellOutput->AppendText(buffer);
    }

    setbuf(stdout, NULL);
    delete[] buffer;

    return true;
}
