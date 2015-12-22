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
#include "drwnSparseFactor.h"
#include "drwnGeneralFactor.h"
#include "drwnTableFactorOps.h"

using namespace std;

class drwnGeneralFactor;

class FactorAdditionOp : public drwnFactorNAryOp {
  protected:
	drwnGeneralFactor *_target;
	const drwnGeneralFactor *_A, *_B;	//!< for now, this looks like a drwnFactorBinaryOp
	vector<int> _targetToA, _targetToB;	//!< TODO: replace with drwnTableFactorMapping

  public:
	FactorAdditionOp(drwnGeneralFactor *target, const drwnGeneralFactor *A,
		const drwnGeneralFactor *B);
	FactorAdditionOp();
	void lazyExecute();	//!< doesn't do anything unless both operands are dense

    ~FactorAdditionOp() { }
};
