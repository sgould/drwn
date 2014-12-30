/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInterfaces.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard library
#include <cstdio>
#include <cassert>

// Darwin library
#include "drwnConstants.h"
#include "drwnLogger.h"
#include "drwnInterfaces.h"
#include "drwnXMLUtils.h"

// drwnWriteable ------------------------------------------------------------

bool drwnWriteable::write(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);
    drwnXMLDoc xml;

    // xml declaration
    drwnXMLNode* decl = xml.allocate_node(rapidxml::node_declaration);
    xml.append_node(decl);
    drwnAddXMLAttribute(*decl, "version", "1.0", false);
    drwnAddXMLAttribute(*decl, "encoding", "utf-8", false);

    // root node
    drwnXMLNode *node = drwnAddXMLRootNode(xml, this->type(), true);
    bool bSuccess = this->save(*node);

    if (bSuccess) {
        drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);
        ofstream ofs(filename);
        ofs << xml;
        DRWN_ASSERT_MSG(!ofs.fail(), "failed to write " << this->type() << " to \"" << filename << "\"");
        ofs << "\n";
        ofs.close();
    } else {
        DRWN_LOG_ERROR("failed to write " << this->type() << " to \"" << filename << "\"");
    }

    return bSuccess;
}

bool drwnWriteable::read(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, filename, this->type());
    if (node == NULL) {
        DRWN_LOG_ERROR("failed to read " << this->type() << " from \"" << filename << "\"");
        return false;
    }

    if (drwnGetXMLAttribute(*node, "drwnVersion") == NULL) {
        DRWN_LOG_WARNING(filename << " is missing Darwin version number");
    } else if (strncmp(drwnGetXMLAttribute(*node, "drwnVersion"), DRWN_VERSION, strlen(DRWN_VERSION))) {
        DRWN_LOG_WARNING(filename << " was created by a different version of Darwin");
    }

    return this->load(*node);
}

void drwnWriteable::dump() const
{
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLRootNode(xml, this->type(), true);
    drwnXMLEncoderType oldEncoder = drwnXMLUtils::DEFAULT_ENCODER;
    drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;
    this->save(*node);
    drwnXMLUtils::DEFAULT_ENCODER = oldEncoder;
    cout << xml << endl;
}
