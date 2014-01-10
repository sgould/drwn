/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfigManager.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <string>
#include <map>

#include "drwnConstants.h"
#include "drwnCompatibility.h"
#include "drwnLogger.h"
#include "drwnXMLUtils.h"

#include "drwnConfigManager.h"

using namespace std;

// drwnConfigurableModule -----------------------------------------------------

drwnConfigurableModule::drwnConfigurableModule(const char *module) :
    _moduleName(module)
{
    DRWN_ASSERT(module != NULL);

    // register the module
    drwnConfigurationManager::get().registerModule(this);
}

drwnConfigurableModule::~drwnConfigurableModule()
{
    // unregister the module
    drwnConfigurationManager::get().unregisterModule(this);
}

void drwnConfigurableModule::usage(ostream &os) const {
    // do nothing (derived classes should override)
}

void drwnConfigurableModule::readConfiguration(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc root;
    drwnParseXMLFile(root, filename);
    DRWN_ASSERT(!drwnIsXMLEmpty(root));

    drwnXMLNode *node = root.first_node(_moduleName.c_str());
    if (node == NULL) {
        DRWN_LOG_ERROR("couldn't find configuration for " << _moduleName
            << " in " << filename);
    } else {
        readConfiguration(*node);
    }
}

void drwnConfigurableModule::readConfiguration(drwnXMLNode& node)
{
    // parse attributes
    for (drwnXMLAttr *it = node.first_attribute(); it != NULL; it = it->next_attribute()) {
        setConfiguration(it->name(), it->value());
    }

    // parse <option name="" value=""/>
    for (drwnXMLNode *it = node.first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        setConfiguration(name->value(), value->value());
    }
}

// drwnConfigurationManager ---------------------------------------------------

drwnConfigurationManager::drwnConfigurationManager()
{
    // do nothing
}

drwnConfigurationManager::~drwnConfigurationManager()
{
    // do nothing
}

drwnConfigurationManager& drwnConfigurationManager::get()
{
    static drwnConfigurationManager mgr;
    return mgr;
}

void drwnConfigurationManager::configure(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc root;
    drwnXMLNode *node = drwnParseXMLFile(root, filename);
    if (drwnIsXMLEmpty(root)) {
        DRWN_LOG_ERROR("couldn't find configuration in " << filename);
        return;
    }

    configure(*node);
}

void drwnConfigurationManager::configure(drwnXMLNode& root)
{
    for (drwnXMLNode *nd = root.first_node(); nd != NULL; nd = nd->next_sibling()) {
        drwnConfigRegistry::iterator it = _registry.find(string(nd->name()));
        if (it == _registry.end()) {
            DRWN_LOG_DEBUG("no module with name " << nd->name() << " has been registered");
        } else {
            DRWN_LOG_DEBUG("configuring module " << it->first);
            it->second->readConfiguration(*nd);
        }
    }
}

void drwnConfigurationManager::configure(const char *module, const char *name, const char *value)
{
    DRWN_ASSERT((module != NULL) && (name != NULL) && (value != NULL));
    drwnConfigRegistry::iterator it = _registry.find(string(module));
    if (it == _registry.end()) {
        DRWN_LOG_FATAL("no module with name \"" << module << "\" has been registered");
    } else {
        DRWN_LOG_DEBUG("setting " << module << "::" << name << " to " << value);
        it->second->setConfiguration(name, value);
    }
}

void drwnConfigurationManager::showModuleUsage(const char *module) const
{
    DRWN_ASSERT(module != NULL);
    drwnConfigRegistry::const_iterator it = _registry.find(string(module));
    if (it == _registry.end()) {
        DRWN_LOG_FATAL("no module with name \"" << module << "\" has been registered");
    } else {
        cout << "--- --------------------------------- ---\n";
        cout << "  * " << it->first << "\n";
        it->second->usage(cout);
        cout << "--- --------------------------------- ---\n";
    }
}

void drwnConfigurationManager::showRegistry(bool bIncludeUsage) const
{
    cout << "--- drwnConfigurationManager registry ---\n";
    for (drwnConfigRegistry::const_iterator it = _registry.begin();
         it != _registry.end(); ++it) {
        cout << "  * " << it->first << "\n";
        if (bIncludeUsage) {
            it->second->usage(cout);
        }
    }
    cout << "--- --------------------------------- ---\n";
}

void drwnConfigurationManager::registerModule(drwnConfigurableModule *m)
{
    DRWN_ASSERT(m != NULL);

    DRWN_LOG_DEBUG("registering module " << m->name());
    _registry[m->name()] = m;
}

void drwnConfigurationManager::unregisterModule(drwnConfigurableModule *m)
{
    DRWN_LOG_DEBUG("unregistering module " << m->name());
    drwnConfigRegistry::iterator it = _registry.find(m->name());
    if (it == _registry.end()) {
        DRWN_LOG_WARNING("module with name \"" << m->name()
            << "\" has already been unregistered");
    } else {
        _registry.erase(it);
    }
}
