/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVarAssignment.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Data structures for encoding assignments to variables.
**
*****************************************************************************/

/*!
** \file drwnVarAssignment.h
** \anchor drwnVarAssignment
** \brief Data structures and utilities for encoding assignments to variables.
*/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>

#include "drwnGraphUtils.h"
#include "drwnVarUniverse.h"

using namespace std;

// drwnFullAssignment ------------------------------------------------------
//! defines a complete assignment to all variables in the universe

typedef std::vector<int> drwnFullAssignment;

// drwnPartialAssignment ---------------------------------------------------
//! defines an assignment to a subset of the variables

class drwnPartialAssignment : public std::map<int, int> {
 public:
    //! construct an empty partial assignment
    drwnPartialAssignment() { /* do nothing */ }
    //! construct a partial assignment from a full assignment (values of -1
    //! in the full assignment are ignored)
    explicit drwnPartialAssignment(const drwnFullAssignment& a) {
        for (int i = 0; i < (int)a.size(); i++) {
            if (a[i] >= 0) {
                this->insert(make_pair(i, a[i]));
            }
        }
    }

    //! construct a partial assignment from a full assignment over a subset
    //! of the variables
    drwnPartialAssignment(const drwnFullAssignment& a, const drwnClique& c) {
        for (drwnClique::const_iterator it = c.begin(); it != c.end(); ++it) {
            if (a[*it] >= 0) {
                this->insert(make_pair(*it, a[*it]));
            }
        }
    }

    //! construct a partial assignment from a partial assignment expressed
    //! using a vector of variables and a vector of values
    drwnPartialAssignment(const vector<int>& vars, const vector<int>& vals) {
        for (int i = 0; i < (int)vars.size(); i++) {
            this->insert(make_pair(vars[i], vals[i]));
        }
    }

    //! return the clique of variables over which the partial
    //! assignment is defined
    drwnClique getClique() const {
        drwnClique c;
        for (const_iterator it = begin(); it != end(); ++it) {
            c.insert(it->first);
        }
        return c;
    }

    //! typecast a partial assignment to a full assignment (missing values
    //! are assigned -1)
    operator drwnFullAssignment() const {
        drwnFullAssignment a(this->rbegin()->first + 1, -1);
        for (const_iterator it = begin(); it != end(); ++it) {
            a[it->first] = it->second;
        }
        return a;
    }
};

// utility functions -------------------------------------------------------

//! next full assignment
void successor(drwnFullAssignment& assignment, const drwnVarUniverse& universe);
//! previous full assignment
void predecessor(drwnFullAssignment& assignment, const drwnVarUniverse& universe);
//! next partial assignment
void successor(drwnPartialAssignment& assignment, const drwnVarUniverse& universe);
//! previous partial assignment
void predecessor(drwnPartialAssignment& assignment, const drwnVarUniverse& universe);
