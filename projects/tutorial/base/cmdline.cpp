/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    cmdline.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"

using namespace std;

// configuration manager -----------------------------------------------------

class MyConfigureableModule : public drwnConfigurableModule {
public:
    string INPUT_STRING;

public:
    MyConfigureableModule() : drwnConfigurableModule("MyModule") { }
    ~MyConfigureableModule() { }

    void usage(ostream &os) const {
        os << "    This is an example configurable module\n";
        os << "      input         :: message string\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "input")) {
            INPUT_STRING = string(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static MyConfigureableModule gMyConfig;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./cmdline [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << "  -i <message>      :: input message\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char* argv[])
{
    const char *input = NULL;

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-i", input)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (input != NULL)
        gMyConfig.INPUT_STRING = string(input);
    
    cout << "Input message is \"" << gMyConfig.INPUT_STRING << "\"\n";
    
    return 0;
}
