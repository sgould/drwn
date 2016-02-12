/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSparseFactor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Albert Chen <chenay@student.unimelb.edu.au>
**
*****************************************************************************/

#pragma once

#include "drwnTableFactor.h"
#include "drwnTableFactorMapping.h"

using namespace std;

// drwnSparseFactor --------------------------------------------------------
//! Encapsulates variables that the represented function is a function of.
//! Allows user to add variables, look up variable cardinality and retrieve all
//! variables. Implementation uses coordinate format (Bader and Kolda, SIAM
//! Journal on Scientific Computing, 2007).
//! \todo Replace getAssignments with overloading the [] operator.
class drwnSparseFactor : public drwnFactor
{
    map<vector<int>, double> assignments;	//!< maps variable values to costs
    
 public:
    //! create an empty sparse factor
    drwnSparseFactor(const drwnVarUniversePtr& ptr) : drwnFactor(ptr) { }
    ~drwnSparseFactor() { }

    // access functions
    const char *type() const { return "drwnSparseFactor"; }
    drwnSparseFactor* clone(void) const { DRWN_TODO; return NULL; }

    //! add variable by name
    using drwnFactor::addVariable;

    int entries() const { return assignments.size(); }
    map<vector<int>, double> getAssignments() const { return assignments; }

    //! Returns the value of the factor for a given (full) assignment
    double getValueOf(const drwnFullAssignment& y) const;
    //! Returns the value of the factor for a given partial assignment.
    double getValueOf(const drwnPartialAssignment& y) const;
    //! Sets the value of the factor for a given (full) assignment
    void setValueOf(const drwnFullAssignment& y, double val);
    //! Sets the value of the factor for a given partial assignment.
    void setValueOf(const drwnPartialAssignment& y, double val);

	typedef map<vector<int>, double>::iterator iterator;
	typedef map<vector<int>, double>::const_iterator const_iterator;
	iterator begin() { return assignments.begin(); }
	iterator end() { return assignments.end(); }

 private:
    //! helper function for getValueOf and setValueOf
    void dfaToVals(const drwnFullAssignment& y, vector<int>& vals) const;
};
