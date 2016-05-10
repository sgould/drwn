/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabShell.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Window for interfacing to the Matlab shell.
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

// drwnMatlabShell -----------------------------------------------------------

class drwnMatlabShell : public drwnGUIShellWindow {
 protected:
    Engine *_matlabEngine; // matlab engine

 public:
    drwnMatlabShell(drwnNode *owner, Engine *ep);
    ~drwnMatlabShell();

 protected:
    bool executeCommand(const string& cmd); 
};
