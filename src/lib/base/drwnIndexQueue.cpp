/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnIndexQueue.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard library
#include <cstdio>
#include <vector>

#include "drwnLogger.h"
#include "drwnIndexQueue.h"

using namespace std;

// drwnIndexQueue -----------------------------------------------------------

#ifdef __APPLE__
const int drwnIndexQueue::TERMINAL;
#endif

string toString(const drwnIndexQueue& q)
{
    drwnIndexQueue qq(q);
    std::stringstream s;
    while (qq.front() != -1) {
        s << " " << qq.front();
        qq.pop_front();
    }
    return s.str();
}
