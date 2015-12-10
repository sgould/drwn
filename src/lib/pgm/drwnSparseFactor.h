/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSparseFactor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>, Albert Chen
				<chenay@student.unimelb.edu.au>
**
*****************************************************************************/

#pragma once

#ifndef FACTOR_H
#define FACTOR_H
#include "../../src/lib/pgm/drwnTableFactor.h"
#include "../../src/lib/pgm/drwnTableFactorMapping.h"

using namespace std;

typedef pair<vector<int>, double> Assignment;

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
    ~drwnSparseFactor(void) { }

	// access functions
    const char *type() const { return "drwnSparseFactor"; }
    drwnSparseFactor* clone(void) const { return NULL; }

    //! add variable by name
    void addVariable(const char *name) { drwnFactor::addVariable(name); }

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

  private:
	//! helper function for getValueOf and setValueOf
	void dfaToVals(const drwnFullAssignment& y, vector<int>& vals) const;
};
#endif
