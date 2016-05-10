/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnXMLParser.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include "drwnFileUtils.h"
#include "drwnLogger.h"
#include "drwnXMLParser.h"

void rapidxml::parse_error_handler(const char *what, void *where)
{
    DRWN_LOG_FATAL("XMLParseError: " << what);
}

drwnXMLNode *drwnParseXMLFile(drwnXMLDoc& xml, const char* filename, const char *tag)
{
    DRWN_ASSERT(filename != NULL);

    string buffer;
    buffer.reserve(drwnFileSize(filename));

    ifstream ifs(filename);
    DRWN_ASSERT_MSG(!ifs.fail(), "error parsing XML file " << filename);
    while (!ifs.eof()) {
        string str;
        getline(ifs, str);
        if (ifs.fail()) break;
        if (str.empty()) continue;
        buffer += str;
    }
    ifs.close();

    char *data = xml.allocate_string(buffer.c_str());

    //xml.clear();
    xml.parse<rapidxml::parse_no_data_nodes>(data);

    if (tag != NULL) {
        drwnXMLNode *node = xml.first_node(tag);
        DRWN_ASSERT_MSG(node != NULL, "could not find " << tag << " in XML file \"" 
            << filename << "\"");
        return node;
    }

    return xml.first_node();
}

bool drwnIsXMLEmpty(drwnXMLNode& xml)
{
    return (xml.name() == NULL) && (xml.first_node() == NULL);
}

int drwnCountXMLChildren(drwnXMLNode& xml, const char *name)
{
    int n = 0;
    for (const drwnXMLNode *node = xml.first_node(name); node != NULL; node = node->next_sibling(name)) {
        n += 1;
    }
    return n;
}

drwnXMLAttr *drwnAddXMLAttribute(drwnXMLNode& xml, const char *name, const char *value,
    bool bCopyName)
{
    DRWN_ASSERT((name != NULL) && (value != NULL));

    drwnXMLAttr *a = xml.first_attribute(name);
    if (a != NULL) {
        a->value(xml.document()->allocate_string(value));
    } else {
        a = xml.document()->allocate_attribute(
            bCopyName ? xml.document()->allocate_string(name) : name,
            xml.document()->allocate_string(value));
        xml.append_attribute(a);
    }

    return a;
}

drwnXMLNode *drwnAddXMLText(drwnXMLNode& xml, const char *value)
{
    DRWN_ASSERT(value != NULL);

#if 1
    drwnXMLNode *node = xml.document()->allocate_node(rapidxml::node_data);
    xml.append_node(node);
    node->value(xml.document()->allocate_string(value));
#else
    xml.value(xml.document()->allocate_string(value));
#endif

    return &xml;
}

drwnXMLNode *drwnAddXMLChildNode(drwnXMLNode& xml, const char *name, const char *value,
    bool bCopyName)
{
    DRWN_ASSERT(name != NULL);

    drwnXMLNode *node = xml.document()->allocate_node(rapidxml::node_element,
        bCopyName ? xml.document()->allocate_string(name) : name);
    xml.append_node(node);

    if (value != NULL) {
        drwnXMLNode *child = xml.document()->allocate_node(rapidxml::node_data);
        node->append_node(child);
        child->value(xml.document()->allocate_string(value));
    }

    return node;
}

drwnXMLNode *drwnAddXMLRootNode(drwnXMLDoc& xml, const char *name, bool bCopyName)
{
    drwnXMLNode *node = xml.allocate_node(rapidxml::node_element, 
        bCopyName ? xml.document()->allocate_string(name) : name);
    xml.append_node(node);

    return node;
}
