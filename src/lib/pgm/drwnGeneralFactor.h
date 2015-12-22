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
#include "drwnSparseFactor.h"

enum drwnFactorStorageType 
{
	DENSE, SPARSE, EVAL
};

//! Either a dense factor, sparse factor, or the sum of two drwnGeneralFactors.
class drwnGeneralFactor : public drwnFactor
{
  public:
	const double THRESHOLD;	//!< if table factor passed in is denser
						//!< than THRESHOLD, then convert it to a sparse factor

  private:
	drwnFactorStorageType _storageType;	//!< whether factor is a dense/sparse/evaluation factor
    drwnTableFactor *_tblFac;		//!< not NULL if this is factor is dense
    drwnSparseFactor *_sparseFac;	//!< not NULL if this factor is sparse
	vector<int> _variables;         //!< list of variables in factor (by index in factor)
	const drwnGeneralFactor *_op1, *_op2;	//!< not NULL if this factor is evaluation

  public:
    //! create an empty factor
	drwnGeneralFactor(const drwnVarUniversePtr& ptr, const double threshold) :
			drwnFactor(ptr), THRESHOLD(threshold) { }
    ~drwnGeneralFactor() { }

	// access functions
    const char *type() const { return "drwnGeneralFactor"; }
    drwnGeneralFactor* clone(void) const { return NULL; }

    //! add variable by id
	void addVariable(int var);
	//! return what type of factor this is
	const drwnFactorStorageType getStorageType() const { return _storageType; }
	//! make factor represent sum of two general factors
	void setOps(const drwnGeneralFactor *f1, const drwnGeneralFactor *f2);

    //! Returns the value of the factor for a given (full) assignment
	double getValueOf(const drwnFullAssignment& y) const;
    //! Returns the value of the factor for a given partial assignment.
    double getValueOf(const drwnPartialAssignment& y) const;
    //! Sets the value of the factor for a given (full) assignment
    void setValueOf(const drwnFullAssignment& y, double val);
    //! Sets the value of the factor for a given partial assignment.
    void setValueOf(const drwnPartialAssignment& y, double val);
    drwnTableFactor* getTableFactor() const;
    drwnSparseFactor* getSparseFactor() const;
	const drwnGeneralFactor* getOp1() const { return _op1; }
	const drwnGeneralFactor* getOp2() const { return _op2; }

  private:
	//! helper function to create sparse factor
    void subtractMostCommonVal();
	//! converts table factor to a sparse factor if density is below THRESHOLD
    void convertIfSparse();
};
