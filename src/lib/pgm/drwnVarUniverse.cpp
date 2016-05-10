/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVarUniverse.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <list>
#include <algorithm>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVarUniverse.h"

using namespace std;
using namespace Eigen;

// drwnVarUniverse ---------------------------------------------------------

drwnVarUniverse::drwnVarUniverse() :
    _numVariables(0), _uniformCards(0)
{
    // do nothing
}

drwnVarUniverse::drwnVarUniverse(int numVars, int varCards) :
    _numVariables(numVars), _uniformCards(varCards)
{
    DRWN_ASSERT((_numVariables > 0) && (_uniformCards > 0));
}

drwnVarUniverse::drwnVarUniverse(const vector<int>& varCards) :
    _uniformCards(0), _varCards(varCards)
{
    _numVariables = (int)varCards.size();
}

drwnVarUniverse::~drwnVarUniverse()
{
    // do nothing
}

int drwnVarUniverse::maxCardinality(const drwnClique& c) const
{
    if (_varCards.empty() && !c.empty()) return _uniformCards;
    int maxCard = 0;
    for (drwnClique::const_iterator it = c.begin(); it != c.end(); ++it) {
        DRWN_ASSERT((*it >= 0) && (*it < _numVariables));
        maxCard = std::max(maxCard, _varCards[*it]);
    }
    return maxCard;
}

double drwnVarUniverse::logStateSpace() const
{
    if (!_varCards.empty()) {
        double d = 0.0;
        for (vector<int>::const_iterator it = _varCards.begin(); it != _varCards.end(); ++it) {
            d += log((double)*it);
        }
        return d;
    }

    return (double)_numVariables * log((double)_uniformCards);
}

double drwnVarUniverse::logStateSpace(const drwnClique& c) const
{
    if (!_varCards.empty()) {
        double d = 0.0;
        for (drwnClique::const_iterator it = c.begin(); it != c.end(); ++it) {
            DRWN_ASSERT((*it >= 0) && (*it < _numVariables));
            d += log((double)_varCards[*it]);
        }
        return d;
    }

   return (double)(c.size()) * log((double)_uniformCards);
}

int drwnVarUniverse::findVariable(const char* name) const
{
    DRWN_ASSERT(name != NULL);

    vector<string>::const_iterator it = std::find(_varNames.begin(), _varNames.end(), string(name));
    return (it == _varNames.end()) ? -1 : (int)(it - _varNames.begin());
}

int drwnVarUniverse::addVariable(const char *name)
{
    DRWN_ASSERT((_varCards.empty()) && (_uniformCards > 0));

    // add variable name
    if (name != NULL) {
        _varNames.reserve(_numVariables + 1);
        while (_varNames.size() < (size_t)_numVariables) {
            _varNames.push_back(string("X") + toString(_varNames.size()));
        }
        _varNames.push_back(toString(name));
    } else {
        if (!_varNames.empty()) {
            _varNames.push_back(string("X") + toString(_numVariables - 1));
        }
    }

    // increment number of variables
    _numVariables += 1;
    return (_numVariables - 1);
}

int drwnVarUniverse::addVariable(int varCard, const char *name)
{
    DRWN_ASSERT(varCard > 0);

    // add variable cardinality
    if ((_numVariables == 0) ||
        (_varCards.empty() && (varCard == _uniformCards))) {
        _uniformCards = varCard;
    } else {
        _varCards.resize(_numVariables + 1, _uniformCards);
        _varCards.back() = varCard;
    }

    // add variable name
    if (name != NULL) {
        _varNames.reserve(_numVariables + 1);
        while (_varNames.size() < (size_t)_numVariables) {
            _varNames.push_back(string("X") + toString(_varNames.size()));
        }
        _varNames.push_back(toString(name));
    } else {
        if (!_varNames.empty()) {
            _varNames.push_back(string("X") + toString(_numVariables - 1));
        }
    }

    // increment number of variables
    _numVariables += 1;
    return (_numVariables - 1);
}

drwnVarUniverse drwnVarUniverse::slice(const vector<int>& vars) const
{
    drwnVarUniverse universe(vars.size(), _uniformCards);

    // copy variable cardinalities
    if (!_varCards.empty()) {
        universe._varCards.resize(vars.size());
        for (unsigned i = 0; i < vars.size(); i++) {
            universe._varCards[i] = _varCards[vars[i]];
        }
    }

    // copy variable names
    if (!_varNames.empty()) {
        universe._varNames.resize(vars.size());
        for (unsigned i = 0; i < vars.size(); i++) {
            universe._varNames[i] = _varNames[vars[i]];
        }
    }

    return universe;
}

drwnVarUniverse drwnVarUniverse::slice(const drwnClique& vars) const
{
    drwnVarUniverse universe(vars.size(), _uniformCards);

    // copy variable cardinalities
    if (!_varCards.empty()) {
        universe._varCards.reserve(vars.size());
        for (drwnClique::const_iterator it = vars.begin(); it != vars.end(); ++it) {
            universe._varCards.push_back(_varCards[*it]);
        }
    }

    // copy variable names
    if (!_varNames.empty()) {
        universe._varNames.reserve(vars.size());
        for (drwnClique::const_iterator it = vars.begin(); it != vars.end(); ++it) {
            universe._varNames.push_back(_varNames[*it]);
        }
    }

    return universe;
}

// i/o
void drwnVarUniverse::clear()
{
    _numVariables = 0;
    _uniformCards = 0;
    _varCards.clear();
    _varNames.clear();
}

bool drwnVarUniverse::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "nVariables", toString(_numVariables).c_str(), false);
    drwnAddXMLAttribute(xml, "uniformCards", toString(_uniformCards).c_str(), false);

    if (!_varCards.empty()) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "varCards", NULL, false);
        drwnAddXMLText(*node, toString(_varCards).c_str());
    }

    return true;
}

bool drwnVarUniverse::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "nVariables") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "uniformCards") != NULL);

    clear();
    _numVariables = atoi(drwnGetXMLAttribute(xml, "nVariables"));
    _uniformCards = atoi(drwnGetXMLAttribute(xml, "uniformCards"));
    DRWN_ASSERT(_numVariables >= 0);

    drwnXMLNode *node = xml.first_node("varCards");
    if (node != NULL) {
        drwn::parseString<int>(string(drwnGetXMLText(*node)), _varCards);
        DRWN_ASSERT(_varCards.size() == (size_t)_numVariables);
    }

    node = xml.first_node("varNames");
    if (node != NULL) {
        drwn::parseString<string>(string(drwnGetXMLText(*node)), _varNames);
        DRWN_ASSERT(_varNames.size() == (size_t)_numVariables);
    }

    return false;
}

void drwnVarUniverse::print() const
{
    for (int i = 0; i < _numVariables; i++) {
        cout << "  [" << setw(3) << i << "]  "
             << setw(3) << varCardinality(i) << "  " << varName(i) << "\n";
    }
}
