/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnXMLParser.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnXMLParser.h
** \anchor drwnXMLParser
** \brief Provides XML parsing functionality for serializing and deserializing
** objects and containers of objects.
**
** Version 1.2 of Darwin and above use the Boost and MIT-licensed \b RapidXML
** code by Marcin Kalicinski (see http://rapidxml.sourceforge.net/). See also
** http://www.ffuts.org/blog/a-rapidxml-gotcha/.
**
** Previous versions used the BSD-licensed \b xmlParser code by Frank Vanden
** Berghen (see http://www.applied-mathematics.net/tools/xmlParser.html).
**
** \sa \ref drwnXMLUtils
** \sa \ref drwnXMLUtilsDoc
*/

#pragma once

#define RAPIDXML_NO_EXCEPTIONS
#include "../../../external/rapidxml/rapidxml.hpp"
#include "../../../external/rapidxml/rapidxml_utils.hpp"
#include "../../../external/rapidxml/rapidxml_print.hpp"

#define drwnXMLDoc rapidxml::xml_document<char>
#define drwnXMLNode rapidxml::xml_node<char>
#define drwnXMLAttr rapidxml::xml_attribute<char>

// essential utilities

//! parse an xml file into \p xml (loads all data into memory) and return a
//! pointer to the first node (with tag \p tag if provided)
drwnXMLNode *drwnParseXMLFile(drwnXMLDoc& xml, const char* filename, const char *tag = NULL);
//! checks whether an xml document is empty
bool drwnIsXMLEmpty(drwnXMLNode& xml);
//! counts the number of children with name \p name
int drwnCountXMLChildren(drwnXMLNode& xml, const char *name = NULL);
//! Adds/changes an attribute for a given node (set \p bCopyName to \p false if
//! it's a static string) and returns a reference to the attribute. The node must
//! belong to a document for string copying to work.
drwnXMLAttr *drwnAddXMLAttribute(drwnXMLNode& xml, const char *name, const char *value,
    bool bCopyName = true);
//! adds/changes an text for a given node
drwnXMLNode *drwnAddXMLText(drwnXMLNode& xml, const char *value);
//! adds a child node for a given parent (set \p bCopyName to \p false if it's a
//! static string) and returns a reference to the node
drwnXMLNode *drwnAddXMLChildNode(drwnXMLNode& parent, const char *name,
    const char *value = NULL, bool bCopyName = true);
//! adds a root node to an xml document (set \p bCopyName to \p false if it's a
//! static string) and returns a reference to the node
drwnXMLNode *drwnAddXMLRootNode(drwnXMLDoc& xml, const char *name, bool bCopyName = false);
//! returns an attribute string (or NULL if the string does not exist)
inline const char *drwnGetXMLAttribute(const drwnXMLNode& node, const char *name)
{
    drwnXMLAttr *a = node.first_attribute(name);
    return (a == NULL) ? NULL : a->value();
}
//! returns node text (or NULL if the node has no text)
inline const char *drwnGetXMLText(const drwnXMLNode &node)
{
    if ((node.value() != NULL) && (node.value()[0] != '\0'))
        return node.value();

    const drwnXMLNode *child = node.first_node();
    while (child != NULL) {
        if (child->type() == rapidxml::node_data) {
            return child->value();
        }
        child = child->next_sibling();
    }

    return node.value();
}
