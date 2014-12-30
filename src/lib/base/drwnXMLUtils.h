/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnXMLUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnXMLUtils.h
** \anchor drwnXMLUtils
** \brief Provides utility functions for XML parsing.
** \sa \ref drwnXMLParser
** \sa \ref drwnXMLUtilsDoc
*/

#pragma once

#include "Eigen/Core"

#include "drwnLogger.h"
#include "drwnXMLParser.h"

// drwnBase64 ----------------------------------------------------------------

namespace drwnBase64 {
    //! base 64 encoding lookup table
    extern const char *LUT;
    //! base 64 filler
    extern const char FILL;
    //! insert newlines to produce 80-column output when encoding
    extern bool INSERT_NEWLINES;

    //! Encodes a binary sequence as a NULL terminated base64 string. The calling
    //! function takes ownership and must delete the string.
    char *encode(const unsigned char *data, size_t length);
    //! Decode a NULL terminated base64 string to binary. The calling function takes
    //! ownership and must delete the binary array.
    unsigned char *decode(const char *data);
};

// drwnXMLUtils --------------------------------------------------------------

typedef enum _drwnXMLEncoderType {
    DRWN_XML_TEXT,   //!< encode binary objects as ASCII text
    DRWN_XML_BASE64  //!< encode binary objects using Base-64
} drwnXMLEncoderType;

namespace drwnXMLUtils {
    //! default encoding mode for binary objects
    extern drwnXMLEncoderType DEFAULT_ENCODER;
    //! converts string into encoder type
    drwnXMLEncoderType getEncoderType(const char *str);

    //! xml serialization of a byte array
    drwnXMLNode& serialize(drwnXMLNode& xml, const char *buffer, size_t length);
    //! xml serialization of a double-precision vector
    drwnXMLNode& serialize(drwnXMLNode& xml, const Eigen::VectorXd& v);
    //! xml serialization of a double-precision matrix
    drwnXMLNode& serialize(drwnXMLNode& xml, const Eigen::MatrixXd& m);

    //! xml de-serialization of a byte array
    drwnXMLNode& deserialize(drwnXMLNode& xml, char *buffer, size_t length);
    //! xml de-serialization of a double-precision vector
    drwnXMLNode& deserialize(drwnXMLNode& xml, Eigen::VectorXd& v);
    //! xml de-serialization of a double-precision matrix
    drwnXMLNode& deserialize(drwnXMLNode& xml, Eigen::MatrixXd& m);

    //! helper class for saving objects to xml nodes
    template<typename T>
    struct save_node { void operator()(const T& o, drwnXMLNode& node) { o.save(node); }; };

    //! helper class for saving pointers to objects to xml nodes
    template<typename T>
    struct save_node<T *> { void operator()(T* const& o, drwnXMLNode& node) { o->save(node); }; };

    //! save container contents in range [\p first, \p last). Contained type must
    //! itself have, or be a pointer to a type with, a save(drwnXMLNode&) method
    template<class RandomAccessIterator>
    void save(drwnXMLNode& xml, const char *tag,
        RandomAccessIterator first, RandomAccessIterator last) {
        save_node<typename RandomAccessIterator::value_type> functor;
        while (first != last) {
            drwnXMLNode *node = xml.document()->allocate_node(rapidxml::node_element,
                xml.document()->allocate_string(tag));
            xml.append_node(node);
            functor(*first, *node);
            ++first;
        }
    }

    //! serialize entire container to xml node. Contained type must itself have,
    //! or be a pointer to a class with, a save(drwnXMLNode&) const method.
    template<class Container>
    void save(drwnXMLNode& xml, const char *tag, const Container& container) {
        save(xml, tag, container.begin(), container.end());
    }

    //! serialize entire container to an xml file. Contained type must itself
    //! have, or be a pointer to a class with, a save(drwnXMLNode&) const method.
    template<class Container>
    void write(const char *filename, const char *root, const char *tag,
        const Container& container) {
        DRWN_ASSERT(filename != NULL);
        drwnXMLDoc xml;
        drwnXMLNode *node = xml.allocate_node(rapidxml::node_element, xml.allocate_string(root));
        xml.append_node(node);
        drwnXMLUtils::save(*node, tag, container);
        ofstream ofs(filename);
        ofs << xml << endl;
        ofs.close();
    }


    //! helper class for loading xml nodes into containers of objects
    template<typename T>
    struct load_node {
        T operator()(drwnXMLNode& node) { T o; o.load(node); return o; };
    };

    //! helper class for (creating and) loading xml nodes into
    //! containers of pointers to objects
    template<typename T>
    struct load_node<T *> {
        T* operator()(drwnXMLNode& node) { T *o = new T(); o->load(node); return o; };
    };

    //! de-serialize an xml node into \p container. The container type must
    //! itself be, or be a pointer to a type that is, default constructable
    //! and have a load(drwnXMLNode&) method.
    template<class Container>
    void load(drwnXMLNode& xml, const char *tag, Container& container) {
        typename Container::iterator it(container.end());
        load_node<typename Container::value_type> functor;

        for (drwnXMLNode *node = xml.first_node(tag); node != NULL; node = node->next_sibling(tag)) {
            // assumes default constructable value type
            it = container.insert(it, functor(*node));
            ++it;
        }
    }

    //! de-serialize an xml file into \p container. The container type must
    //! be default constructable and have a load(drwnXMLNode&) method.
    template<class Container>
    void read(const char *filename, const char *root, const char *tag,
        Container& container) {
        DRWN_ASSERT(filename != NULL);
        drwnXMLDoc xml;
        drwnParseXMLFile(xml, filename, root);
        DRWN_ASSERT(!drwnIsXMLEmpty(xml));
        load(xml, tag, container);
    }

    //! prints an XML node to standard output for debugging
    void dump(drwnXMLNode& xml);
};
