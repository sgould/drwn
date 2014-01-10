/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFactory.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <map>
#include <list>
#include <typeinfo>

#include "drwnXMLParser.h"
#include "drwnInterfaces.h"
#include "drwnLogger.h"

using namespace std;

// drwnFactoryTraits -------------------------------------------------------
//! Some classes may provide default factory registration (e.g., built-in
//! classes such as drwnClassifier and drwnFeatureTransform).

template <typename T>
struct drwnFactoryTraits {
    static void staticRegistration() { /* do nothing */ };
};

// drwnFactory --------------------------------------------------------------
//! Templated factory for creating or cloning objects for a particular base
//! class.
//!
//! The base class must be default constructable and implement an XML load
//! function (e.g., be derived from drwnWriteable).
//!
//! Each derived object should register with factory using the register
//! function, for example,
//! \code
//!     drwnClassifier *createMyClassifier() { return new myClassifier(); }
//!     ...
//!     drwnFactory<drwnClassifier>::get().registerClass("myClassifier", createMyClassifier);
//! \endcode
//! The macro DRWN_FACTORY_REGISTER(U, T) provides a short way to
//! register class T in factory drwnFactory<U>.
//!
//! Built-in Darwin classes are (statically) registered on construction using
//! the drwnFactoryTraits class. External (i.e., add-on) factory object should
//! be explicitly registered.
//!

template<typename U>
class drwnFactory
{
 protected:
    bool _initialized;
    typedef U* (*drwnCreatorFcn)(void);
    map<string, drwnCreatorFcn> _registry;

 public:
    ~drwnFactory() { /* do nothing */ }
    //! get the factory
    static drwnFactory<U>& get();

    //! register a class and creator function
    void registerClass(const char *name, drwnCreatorFcn fcn);
    //! unregister a previously registered class
    void unregisterClass(const char *name);
    //! get a list of all registered classes
    list<string> getRegisteredClasses() const;

    //! create a default object
    U *create(const char *name) const;
    //! create an object from an XML node
    U *createFromXML(drwnXMLNode& xml) const;
    //! create an object from file
    U *createFromFile(const char *filename) const;

    //! dump the contents of the registery for debugging
    void dump() const;

 protected:
    //! singleton class so hide constructor
    drwnFactory() : _initialized(false) { /* do nothing */ }
};

// drwnFactoryAutoRegister --------------------------------------------------
//! Helper class for registering classes with a drwnFactory.

template<typename U, typename T>
class drwnFactoryAutoRegister
{
 public:
    drwnFactoryAutoRegister(const char *name) {
        drwnFactory<U>::get().registerClass(name, &drwnFactoryAutoRegister<U, T>::creator);
    }

 private:
    static U *creator() { return new T(); }
};

#define DRWN_FACTORY_REGISTER(baseName, className) \
    drwnFactoryAutoRegister<baseName, className> \
        __ ## baseName ## _ ## className ## _ ## AutoRegister(#className);

// drwnFactory implementation ------------------------------------------------

template<typename U>
drwnFactory<U>& drwnFactory<U>::get()
{
    static drwnFactory<U> factory;
    if (!factory._initialized) {
        DRWN_LOG_DEBUG("initializing drwnFactory for " << typeid(U).name());
        factory._initialized = true;
        drwnFactoryTraits<U>::staticRegistration();
    }
    return factory;
}

template<typename U>
void drwnFactory<U>::registerClass(const char *name, drwnCreatorFcn fcn)
{
    DRWN_ASSERT(name != NULL);

    typename map<string, drwnCreatorFcn>::iterator jt = _registry.find(string(name));
    if (jt != _registry.end()) {
        DRWN_LOG_ERROR("class \"" << name << "\" already registered with drwnFactory");
        jt->second = fcn;
    } else {
        DRWN_LOG_DEBUG("registering class \"" << name << "\" with the drwnFactory");
        _registry.insert(make_pair(string(name), fcn));
    }
}

template<typename U>
void drwnFactory<U>::unregisterClass(const char *name)
{
    DRWN_ASSERT(name != NULL);
    DRWN_LOG_DEBUG("unregistering class \"" << name << "\" from drwnFactory");
    typename map<string, drwnCreatorFcn>::iterator jt = _registry.find(string(name));
    if (jt == _registry.end()) {
        DRWN_LOG_ERROR("class \"" << name << "\" does not exist in drwnFactory");
        return;
    }

    _registry.erase(jt);
}

template<typename U>
list<string> drwnFactory<U>::getRegisteredClasses() const
{
    list<string> names;
    for (typename map<string, drwnCreatorFcn>::const_iterator it = _registry.begin();
         it != _registry.end(); ++it) {
        names.push_back(it->first);
    }
    return names;
}

template<typename U>
U *drwnFactory<U>::create(const char *name) const
{
    DRWN_ASSERT(name != NULL);

    typename map<string, drwnCreatorFcn>::const_iterator jt = _registry.find(string(name));
    if (jt == _registry.end()) {
        DRWN_LOG_ERROR("class \"" << name << "\" does not exist in drwnFactory");
        return NULL;
    }

    return (*jt->second)();
}

template<typename U>
U *drwnFactory<U>::createFromXML(drwnXMLNode& xml) const
{
    // construct node and initialize from XML
    U *object = create(xml.name());
    if (object == NULL) {
        return NULL;
    }

    object->load(xml);
    return object;
}

template<typename U>
U *drwnFactory<U>::createFromFile(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);

    // parse XML file
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, filename);
    DRWN_ASSERT(node != NULL);

    return createFromXML(*node);
}

template<typename U>
void drwnFactory<U>::dump() const
{
    DRWN_LOG_MESSAGE("drwnFactory has the following registered classes:");
    for (typename map<string, drwnCreatorFcn>::const_iterator it = _registry.begin();
         it != _registry.end(); ++it) {
        DRWN_LOG_MESSAGE("  " << it->first);
    }
}
