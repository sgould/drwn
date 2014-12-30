/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnExporter.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Classes for exporting reports (e.g. HTML) or code.
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <map>
#include <list>
#include <vector>

#include "drwnGraph.h"
#include "drwnNode.h"

using namespace std;

// drwnHTMLReport ------------------------------------------------------------

class drwnHTMLReport {
 protected:

 public:
    drwnHTMLReport();
    ~drwnHTMLReport();

    void write(const char *filename, const drwnGraph *graph);
};
