/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfigManager.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <string>
#include <string.h>
#include <iostream>
#include <map>

#include "drwnXMLParser.h"

using namespace std;

// drwnConfigurableModule -----------------------------------------------------

/*!
** \brief Interface for a configurable module.
**
** Client code should inherit from drwnConfigurableModule and override the
** \ref usage and \ref setConfiguration member functions. 
**
** \sa drwnConfigurationManager
** \sa \ref drwnConfigManagerDoc
*/
class drwnConfigurableModule {
 private:
    string _moduleName; //!< the name of the module

 public:
    //! create a configurable module and register it with the configuration manager
    drwnConfigurableModule(const char *module);
    //! destroy the configurable module and unregister
    //! it from the configuration manager
    virtual ~drwnConfigurableModule();

    //! return the name of the module
    inline const string& name() const { return _moduleName; }
    //! display configuration usage
    virtual void usage(ostream& os) const;

    //! read configuration from an XML file
    void readConfiguration(const char *filename);
    //! read configuration from an XML node
    virtual void readConfiguration(drwnXMLNode& node);
    //! set individual configurable parameter
    virtual void setConfiguration(const char* name, const char *value) = 0;
};

// drwnConfigurationManager ---------------------------------------------------

/*!
** \brief Configuration manager.
**
** Handles configuration of static members in \b Darwin libraries and
** applications from XML configuration files or command line arguments. Also
** allows projects to register their own configuration code.
**
** \sa drwnConfigurableModule
** \sa \ref drwnConfigManagerDoc
*/
class drwnConfigurationManager {
 friend class drwnConfigurableModule;

 private:
    typedef map<string, drwnConfigurableModule *> drwnConfigRegistry;
    drwnConfigRegistry _registry;

 public:
    ~drwnConfigurationManager();

    //! get the configuration manager (singleton object)
    static drwnConfigurationManager& get();

    //! configure all registered modules from XML file
    void configure(const char *filename);
    //! configure all registered modules from XML node
    void configure(drwnXMLNode& root);
    //! set a parameter in a specific module
    void configure(const char *module, const char *name, const char *value);

    //! show usage for a specific module
    void showModuleUsage(const char *module) const;
    //! show all registered modules and their usage
    void showRegistry(bool bIncludeUsage = true) const;

 protected:
    drwnConfigurationManager(); // singleton class so hide constructor

    //! register a module with the configuration manager
    void registerModule(drwnConfigurableModule *m);
    //! unregister a module from the configuration manager
    void unregisterModule(drwnConfigurableModule *m);
};
