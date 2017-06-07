/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVarUniverse.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>

#include "drwnBase.h"
#include "drwnGraphUtils.h"

using namespace std;

// drwnVarUniverse ---------------------------------------------------------
//! Data structure for definining the random variables (name and cardinality)
//! for a given problem instance.

class drwnVarUniverse : public drwnStdObjIface {
 protected:
    int _numVariables;              //!< number of variables in the model (universe)
    int _uniformCards;              //!< cardinality of variables if all the same
    vector<int> _varCards;          //!< cardinality of variables if some are different
    vector<string> _varNames;       //!< optional string name for each variable

 public:
    drwnVarUniverse();
    drwnVarUniverse(int numVars, int varCards = 2);
    drwnVarUniverse(const vector<int>& varCards);
    ~drwnVarUniverse();

    // access functions
    virtual const char *type() const { return "drwnVarUniverse"; }
    virtual drwnVarUniverse *clone() const { return new drwnVarUniverse(*this); }

    //! returns the number of variables in the model
    int numVariables() const { return _numVariables; }
    //! returns true if all variables have the same cardinality
    bool hasUniformCardinality() const { return _varCards.empty(); }
    //! returns the cardinality of variable \b v (between 0 and \p _numVariables - 1)
    int varCardinality(int v) const { return _varCards.empty() ? _uniformCards : _varCards[v]; }
    //! returns the greatest cardinality in the universe or in clique \b c
    int maxCardinality() const { return _varCards.empty() ? _uniformCards : *max_element(_varCards.begin(), _varCards.end()); }
    int maxCardinality(const drwnClique& c) const;
    //! returns the size of the entire state space (in the log domain)
    double logStateSpace() const;
    //! returns the size of the state space of clique \b c (in the log domain)
    double logStateSpace(const drwnClique& c) const;
    //! returns the name of variable \b v (between 0 and \p _numVariables - 1)
    string varName(int v) const { return _varNames.empty() ? (string("X") + toString(v)) : _varNames[v]; }
    //! returns the index of variable with name \b name
    int findVariable(const char* name) const;

    //! add variable with default cadinality (\p name is optional)
    int addVariable(const char *name = NULL);
    //! add a variable with cardinality \p varCard (\p name is optional)
    int addVariable(int varCard, const char *name = NULL);

    //! create a subset of the universe (re-ordered by variables \b vars)
    drwnVarUniverse slice(const vector<int>& vars) const;
    //! create a subset of the universe from variables \b vars
    drwnVarUniverse slice(const drwnClique& vars) const;

    // i/o
    void clear();
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    void print() const;
};

// drwnVarUniversePtr ------------------------------------------------------
//! Shared memory pointer to a drwnVarUniverse object. Construct as
//! drwnVarUniversePtr(new drwnVarUniverse);

typedef drwnSmartPointer<drwnVarUniverse> drwnVarUniversePtr;

