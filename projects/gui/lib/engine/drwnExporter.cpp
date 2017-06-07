/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnExporter.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <fstream>

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnExporter.h"

using namespace std;

// drwnHTMLReport ------------------------------------------------------------

drwnHTMLReport::drwnHTMLReport()
{
    // do nothing
}

drwnHTMLReport::~drwnHTMLReport()
{
    // do nothing
}

void drwnHTMLReport::write(const char *filename, const drwnGraph *graph)
{
    DRWN_ASSERT((filename != NULL) && (graph != NULL));

    ofstream ofs(filename);
    DRWN_ASSERT(!ofs.fail());

    // header
    ofs << "<html>\n"
        << "<head>\n"
        << "<title>" << graph->getTitle() << "</title>\n"
        << "</head>\n";

    // body
    ofs << "<body>\n";
    ofs << "<h1>" << graph->getTitle() << "</h1>\n";
    ofs << graph->getNotes() << "<p>\n\n";

    for (int i = 0; i < graph->numNodes(); i++) {
        const drwnNode *node = graph->getNode(i);
        ofs << "<h2>" << node->getName() << "</h2>\n";
        ofs << "TODO\n";
        ofs << "<p>\n";
    }

    // footer
    ofs << "<hr><center><small>"
        << DRWN_TITLE << " (Version: " DRWN_VERSION ")<br>\n" << DRWN_COPYRIGHT 
        << "</small></center>\n";
    ofs << "</body>\n"
        << "</html>\n";
    
    ofs.close();
}
