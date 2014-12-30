/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    helloworld.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./helloworld [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // print hello world and exit
    DRWN_LOG_MESSAGE("hello world");
    DRWN_LOG_VERBOSE("hello world again");
    return 0;
}
