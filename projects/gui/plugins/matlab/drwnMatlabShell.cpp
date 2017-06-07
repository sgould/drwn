/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabShell.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard libraries
#include <cstdlib>

// Darwin library
#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

// Matlab library
#include "engine.h"
#include "mat.h"

#include "drwnMatlabShell.h"

using namespace std;

// drwnMatlabShell -----------------------------------------------------------

drwnMatlabShell::drwnMatlabShell(drwnNode *owner, Engine *ep) :
    drwnGUIShellWindow(owner), _matlabEngine(ep)
{
    // do nothing
}

drwnMatlabShell::~drwnMatlabShell()
{
    // do nothing
}

bool drwnMatlabShell::executeCommand(const string& cmd)
{
    static size_t BUFFER_SIZE = 8192;

    // check for matlab connection
    if (_matlabEngine == NULL) {
        _shellOutput->AppendText("ERROR: no connection to Matlab\n");
        return false;
    }

    // intercept special commands
    if ((strcasecmp(cmd.c_str(), "exit") == 0) || 
        (strcasecmp(cmd.c_str(), "exit;") == 0)) {
        Close(); // close the window
        return true;
    }

    // execute command
    char *buffer = new char[BUFFER_SIZE];
    engOutputBuffer(_matlabEngine, buffer, BUFFER_SIZE);
    int result = engEvalString(_matlabEngine, cmd.c_str());
    buffer[BUFFER_SIZE - 1] = '\0';
    engOutputBuffer(_matlabEngine, NULL, 0);

    // display response
    if (buffer[0] != '\0') {
        const char *p = strchr(buffer, '>');
        if ((p != NULL) && (*(p + 1) == '>')) {
            p += 2;
            while (*p == ' ') p++;
            _shellOutput->AppendText(p);
        } else {
            p = buffer;
            while (*p == ' ') p++;
            _shellOutput->AppendText(p);
        }

        if (strlen(buffer) == BUFFER_SIZE - 1) {
            _shellOutput->AppendText(" ...\n");
        }
    }
    delete[] buffer;

    return (result == 0);
}
