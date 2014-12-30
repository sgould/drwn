/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTableFactorOps.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <set>
#include <map>

#include "drwnBase.h"
#include "drwnTableFactor.h"
#include "drwnTableFactorMapping.h"

// drwnFactorOperation class -----------------------------------------------
//! Base class for implementing various operations on table factors. The
//! derived classes store mappings between factor entries making them very
//! fast for iterative algorithms.

class drwnFactorOperation {
 protected:
    drwnTableFactor *_target;

 public:
    drwnFactorOperation(drwnTableFactor *target);
    drwnFactorOperation(const drwnFactorOperation& op);
    virtual ~drwnFactorOperation();

    //! peform the factor operation on the target
    virtual void execute() = 0;

    //! return a pointer to the target factor
    inline drwnTableFactor *target() const { return _target; }

 protected:
    //! initialize mappings and target factor (with right set of
    //! variables, unless given, e.g. for marginalization)
    virtual void initialize() { /* default: do nothing */ };
    //! check that the target is not empty and has the right set of
    //! variables for this operation
    virtual bool checkTarget() { /* default: true */ return true; }
};

// drwnFactorUnaryOp class -------------------------------------------------
//! Base class for implementing unary factor operations.

class drwnFactorUnaryOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnTableFactorMapping _mappingA;

 public:
    drwnFactorUnaryOp(drwnTableFactor *target, const drwnTableFactor *A);
    drwnFactorUnaryOp(const drwnFactorUnaryOp &op);
    ~drwnFactorUnaryOp();

 protected:
    void initialize();
    bool checkTarget();
};

// drwnFactorBinaryOp class ------------------------------------------------
//! Base class for implementing binary factor operations.

class drwnFactorBinaryOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    const drwnTableFactor * const _operandB;

    drwnTableFactorMapping _mappingA;
    drwnTableFactorMapping _mappingB;

 public:
    drwnFactorBinaryOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorBinaryOp(const drwnFactorBinaryOp &op);
    ~drwnFactorBinaryOp();

 protected:
    void initialize();
    bool checkTarget();
};

// drwnFactorNAryOp class --------------------------------------------------
//! Base class for implementing n-ary factor operations.

class drwnFactorNAryOp : public drwnFactorOperation {
 protected:
    std::vector<const drwnTableFactor *> _factors;
    std::vector<drwnTableFactorMapping> _mappings;

 public:
    drwnFactorNAryOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorNAryOp(drwnTableFactor *target,
	const std::vector<const drwnTableFactor *>& A);
    drwnFactorNAryOp(const drwnFactorNAryOp &phi);
    ~drwnFactorNAryOp();

 protected:
    void initialize();
    bool checkTarget();
};

// drwnFactorAtomicOp class ------------------------------------------------
//! Executes an atomic operation by executing a sequence of factor operations.

class drwnFactorAtomicOp : public drwnFactorOperation {
 protected:
    //! factor operations are owned by object (i.e., they will be deleted when
    //! the object is destroyed)
    std::vector<drwnFactorOperation *> _computations;

 public:
    drwnFactorAtomicOp(drwnFactorOperation *op);
    drwnFactorAtomicOp(const std::vector<drwnFactorOperation *>& ops);
    ~drwnFactorAtomicOp();

    void execute();

    void addOperation(drwnFactorOperation *op);
};

// drwnFactorCopyOp class --------------------------------------------------
//! Copy one factor onto another.

class drwnFactorCopyOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnTableFactorMapping _mappingA;

 public:
    drwnFactorCopyOp(drwnTableFactor *target, const drwnTableFactor *A);
    drwnFactorCopyOp(const drwnFactorCopyOp& op);
    ~drwnFactorCopyOp();

    void execute();

 protected:
    void initialize();
    bool checkTarget();
};

// drwnFactorProductOp class -----------------------------------------------
//! Multiply two or more factors together.

class drwnFactorProductOp : public drwnFactorNAryOp {
 public:
    drwnFactorProductOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorProductOp(drwnTableFactor *target,
	const std::vector<const drwnTableFactor *>& A);
    drwnFactorProductOp(const drwnFactorProductOp &op);
    ~drwnFactorProductOp();

    void execute();
};

// drwnFactorDivideOp class ------------------------------------------------
//! Divide one factor by another.

class drwnFactorDivideOp : public drwnFactorBinaryOp {
 public:
    drwnFactorDivideOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorDivideOp(const drwnFactorDivideOp &op);
    ~drwnFactorDivideOp();

    void execute();
};

// drwnFactorAdditionOp class ----------------------------------------------
//! Add two or more factors together.

class drwnFactorAdditionOp : public drwnFactorNAryOp {
 public:
    drwnFactorAdditionOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorAdditionOp(drwnTableFactor *target,
	const std::vector<const drwnTableFactor *>& A);
    drwnFactorAdditionOp(const drwnFactorAdditionOp &op);
    ~drwnFactorAdditionOp();

    void execute();
};

// drwnFactorSubtractOp class -----------------------------------------------
//! Subtract one factor from another.

class drwnFactorSubtractOp : public drwnFactorBinaryOp {
 public:
    drwnFactorSubtractOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnTableFactor *B);
    drwnFactorSubtractOp(const drwnFactorSubtractOp &op);
    ~drwnFactorSubtractOp();

    void execute();
};

// drwnFactorWeightedSumOp class --------------------------------------------
//! Add a weighted combination of factors.

class drwnFactorWeightedSumOp : public drwnFactorBinaryOp {
 protected:
    double _weightA;
    double _weightB;

 public:
    drwnFactorWeightedSumOp(drwnTableFactor *target, const drwnTableFactor *A,
        const drwnTableFactor *B, double wA = 1.0, double wB = 1.0);
    drwnFactorWeightedSumOp(const drwnFactorWeightedSumOp &op);
    ~drwnFactorWeightedSumOp();

    void execute();
};

// drwnFactorPlusEqualsOp class ---------------------------------------------
//! Add one (weighted) factor to another inline.

class drwnFactorPlusEqualsOp : public drwnFactorUnaryOp {
 protected:
    double _weightA;

 public:
    drwnFactorPlusEqualsOp(drwnTableFactor *target, const drwnTableFactor *A,
        double wA = 1.0);
    drwnFactorPlusEqualsOp(const drwnFactorPlusEqualsOp &op);
    ~drwnFactorPlusEqualsOp();

    void execute();
};

// drwnFactorMinusEqualsOp class ---------------------------------------------
//! Subtract one (weighted) factor from another inline.

class drwnFactorMinusEqualsOp : public drwnFactorUnaryOp {
 protected:
    double _weightA;

 public:
    drwnFactorMinusEqualsOp(drwnTableFactor *target, const drwnTableFactor *A,
        double wA = 1.0);
    drwnFactorMinusEqualsOp(const drwnFactorMinusEqualsOp &op);
    ~drwnFactorMinusEqualsOp();

    void execute();
};

// drwnFactorTimesEqualsOp class --------------------------------------------
//! Multiply one factor by another inline.

class drwnFactorTimesEqualsOp : public drwnFactorUnaryOp {
 public:
    drwnFactorTimesEqualsOp(drwnTableFactor *target, const drwnTableFactor *A);
    drwnFactorTimesEqualsOp(const drwnFactorTimesEqualsOp &op);
    ~drwnFactorTimesEqualsOp();

    void execute();
};

// drwnFactorMarginalizeOp class --------------------------------------------
//! Marginalize out one or more variables in a factor.

class drwnFactorMarginalizeOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnTableFactorMapping _mappingA;

 public:
    drwnFactorMarginalizeOp(drwnTableFactor *target,
	const drwnTableFactor *A);
    drwnFactorMarginalizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, int v);
    drwnFactorMarginalizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, const std::set<int>& v);
    drwnFactorMarginalizeOp(const drwnFactorMarginalizeOp &op);
    ~drwnFactorMarginalizeOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorMaximizeOp class -----------------------------------------------
//! Maximize over one or more variables in a factor.

class drwnFactorMaximizeOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnTableFactorMapping _mappingA;

 public:
    drwnFactorMaximizeOp(drwnTableFactor *target,
	const drwnTableFactor *A);
    drwnFactorMaximizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, int v);
    drwnFactorMaximizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, const std::set<int>& v);
    drwnFactorMaximizeOp(const drwnFactorMaximizeOp &op);
    ~drwnFactorMaximizeOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorMinimizeOp class ----------------------------------------------
//! Minimize over one or more variables in a factor.

class drwnFactorMinimizeOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnTableFactorMapping _mappingA;

 public:
    drwnFactorMinimizeOp(drwnTableFactor *target,
	const drwnTableFactor *A);
    drwnFactorMinimizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, int v);
    drwnFactorMinimizeOp(drwnTableFactor *target,
	const drwnTableFactor *A, const std::set<int>& v);
    drwnFactorMinimizeOp(const drwnFactorMinimizeOp &op);
    ~drwnFactorMinimizeOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorReduceOp class ------------------------------------------------
//! Reduce factor by oberving the value of one or more variables.

class drwnFactorReduceOp : public drwnFactorOperation {
 protected:
    const drwnTableFactor * const _operandA;
    drwnPartialAssignment _assignment;

 public:
    drwnFactorReduceOp(drwnTableFactor *target,
	const drwnTableFactor *A, int var, int val);
    drwnFactorReduceOp(drwnTableFactor *target,
	const drwnTableFactor *A, const drwnPartialAssignment& assignment);
    drwnFactorReduceOp(const drwnFactorReduceOp &op);
    ~drwnFactorReduceOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorNormalizeOp class ---------------------------------------------
//! Normalize all the entries in a factor to sum to one. Assumes non-negative
//! entries.

class drwnFactorNormalizeOp : public drwnFactorOperation {
 public:
    drwnFactorNormalizeOp(drwnTableFactor *target);
    ~drwnFactorNormalizeOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorExpAndNormalizeOp class ---------------------------------------
//! Exponentiate and normalize all the entries in a factor to sum to one.

class drwnFactorExpAndNormalizeOp : public drwnFactorOperation {
 public:
    drwnFactorExpAndNormalizeOp(drwnTableFactor *target);
    ~drwnFactorExpAndNormalizeOp();

    void execute();

 protected:
    bool checkTarget();
};

// drwnFactorLogNormalizeOp class -------------------------------------------
//! Shift all the entries in a factor so that the maximum is zero.

class drwnFactorLogNormalizeOp : public drwnFactorOperation {
 public:
    drwnFactorLogNormalizeOp(drwnTableFactor *target);
    ~drwnFactorLogNormalizeOp();

    void execute();

 protected:
    bool checkTarget();
};

