/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInterfaces.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>

#include "drwnXMLParser.h"

using namespace std;

// drwnTypeable -------------------------------------------------------------

//! interface for an object that returns its own type as a string
class drwnTypeable {
 public:
    virtual ~drwnTypeable() { /* do nothing */ }

    //! returns object type as a string (e.g., Foo::type() { return "Foo"; })
    virtual const char *type() const = 0;
};

// drwnCloneable ------------------------------------------------------------

//! interface for cloning object (i.e., virtual copy constructor)
class drwnCloneable {
 public:
    virtual ~drwnCloneable() { /* do nothing */ }

    //! returns a copy of the class
    //! usually implemented as virtual Foo* clone() { return new Foo(*this); }
    virtual drwnCloneable *clone() const = 0;
};

// drwnWriteable ------------------------------------------------------------

//! interface for objects that can serialize and de-serialize themselves
class drwnWriteable : public drwnTypeable {
 public:
    virtual ~drwnWriteable() { /* do nothing */ }

    //! write object to file (calls \ref save)
    bool write(const char *filename) const;
    //! read object from file (calls \ref load)
    bool read(const char *filename);

    //! write object to XML node (see also \ref write)
    virtual bool save(drwnXMLNode& xml) const = 0;
    //! read object from XML node (see also \ref read)
    virtual bool load(drwnXMLNode& xml) = 0;

    //! print object's current state to standard output (for debugging)
    void dump() const;
};

// drwnStdObjIface ----------------------------------------------------------

//! standard Darwin object interface (cloneable and writeable)
//!
//! \sa drwnWriteable
//! \sa drwnCloneable
class drwnStdObjIface : public drwnWriteable, public drwnCloneable {
 public:
    virtual ~drwnStdObjIface() { /* do nothing */ }
};

// helper macros ------------------------------------------------------------

//! helper for defining type and clone functions
#define DRWN_TYPE_AND_CLONE_IMPL(C) \
    virtual const char *type() const { return #C; } \
    virtual C* clone() const { return new C(*this); }


