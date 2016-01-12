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

#include <vector>
#include "factor.h"
#include "drwnGeneralFactor.h"
#include "drwnTableFactorOps.h"

using namespace std;

class FactorAdditionOp : public drwnFactorNAryOp {
  protected:
	vector<int> _targetToA, _targetToB;
	//!< storage type to which target will be set when (or if) execute is called.
	drwnFactorStorageType _intendedStorageType;

  public:
	FactorAdditionOp(drwnGeneralFactor *target, const drwnGeneralFactor *A,
	const drwnGeneralFactor *B);
	FactorAdditionOp(drwnGeneralFactor *target,
	const vector<const drwnGeneralFactor *>& A);
	void execute();

    ~FactorAdditionOp() { }
};
