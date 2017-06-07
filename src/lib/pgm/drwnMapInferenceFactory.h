/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMapInferenceFactory.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <set>
#include <map>
#include <list>

#include "drwnBase.h"
#include "drwnMapInference.h"

// drwnMAPInferenceFactory -------------------------------------------------
//! Factory for creating drwnMAPInference objects.
//!
//! \todo use drwnFactory templated class (but then drwnMAPInference
//! needs to be default constructable)

class drwnMAPInferenceFactory
{
 public:
    ~drwnMAPInferenceFactory() { /* do nothing */ }
    //! get the factory
    static drwnMAPInferenceFactory& get();

    //! get a list of all registered classes
    list<string> getRegisteredClasses() const;

    //! create a MAP inference object
    drwnMAPInference *create(const char *name, const drwnFactorGraph& graph) const;

 protected:
    //! singleton class so hide constructor
    drwnMAPInferenceFactory() { /* do nothing */ }
};
