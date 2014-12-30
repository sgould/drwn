/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTableFactorOps.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

#include "drwnBase.h"
#include "drwnPGM.h"

using namespace std;

// drwnFactorOperation class -----------------------------------------------

drwnFactorOperation::drwnFactorOperation(drwnTableFactor *target) :
    _target(target)
{
    DRWN_ASSERT(target != NULL);
    // do nothing
}

drwnFactorOperation::drwnFactorOperation(const drwnFactorOperation& op) :
    _target(op._target)
{
    // do nothing
}

drwnFactorOperation::~drwnFactorOperation()
{
    // do nothing
}

// drwnFactorUnaryOp -------------------------------------------------------

drwnFactorUnaryOp::drwnFactorUnaryOp(drwnTableFactor *target,
    const drwnTableFactor *A) : drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    initialize();
}

drwnFactorUnaryOp::drwnFactorUnaryOp(const drwnFactorUnaryOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _mappingA(op._mappingA)
{
    // do nothing
}

drwnFactorUnaryOp::~drwnFactorUnaryOp()
{
    // do nothing
}

void drwnFactorUnaryOp::initialize()
{
    // add variables and check domains match
    if (_target->empty()) {
	_target->addVariables(*_operandA);
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create index mappings
    _mappingA = drwnTableFactorMapping(_target->getOrderedVars(),
        _operandA->getOrderedVars(), _target->getUniverse());
}

bool drwnFactorUnaryOp::checkTarget()
{
    for (unsigned i = 0; i < _operandA->size(); i++) {
	if (!_target->hasVariable(_operandA->getOrderedVars()[i])) {
            DRWN_LOG_ERROR("target vars:" <<
                toString(_target->getOrderedVars()) << "; A vars:" <<
                toString(_operandA->getOrderedVars()));
	    return false;
	}
    }

    // TO DO: check reverse direction and domain sizes

    return true;
}

// drwnFactorBinaryOp ------------------------------------------------------

drwnFactorBinaryOp::drwnFactorBinaryOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorOperation(target), _operandA(A), _operandB(B)
{
    DRWN_ASSERT((A != NULL) && (B != NULL));
    initialize();
}

drwnFactorBinaryOp::drwnFactorBinaryOp(const drwnFactorBinaryOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _operandB(op._operandB),
    _mappingA(op._mappingA), _mappingB(op._mappingB)
{
    // do nothing
}

drwnFactorBinaryOp::~drwnFactorBinaryOp()
{
    // do nothing
}

void drwnFactorBinaryOp::initialize()
{
    // add variables and check domains match
    if (_target->empty()) {
	_target->addVariables(*_operandA);
	_target->addVariables(*_operandB);
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create index mappings
    _mappingA = drwnTableFactorMapping(_target->getOrderedVars(),
        _operandA->getOrderedVars(), _target->getUniverse());
    _mappingB = drwnTableFactorMapping(_target->getOrderedVars(),
        _operandB->getOrderedVars(), _target->getUniverse());
}

bool drwnFactorBinaryOp::checkTarget()
{
    for (unsigned i = 0; i < _operandA->size(); i++) {
	if (!_target->hasVariable(_operandA->getOrderedVars()[i])) {
            DRWN_LOG_ERROR("target vars:" <<
                toString(_target->getOrderedVars()) << "; A vars:" <<
                toString(_operandA->getOrderedVars()));
	    return false;
	}
    }

    for (unsigned i = 0; i < _operandB->size(); i++) {
	if (!_target->hasVariable(_operandB->getOrderedVars()[i])) {
            DRWN_LOG_ERROR("target vars:" <<
                toString(_target->getOrderedVars()) << "; B vars:" <<
                toString(_operandB->getOrderedVars()));
	    return false;
	}
    }

    // TO DO: check reverse direction and domain sizes

    return true;
}

// drwnFactorNAryOp -------------------------------------------------------

drwnFactorNAryOp::drwnFactorNAryOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorOperation(target)
{
    DRWN_ASSERT((A != NULL) && (B != NULL));

    // add factors to list
    _factors.push_back(A);
    _factors.push_back(B);

    initialize();
}

drwnFactorNAryOp::drwnFactorNAryOp(drwnTableFactor *target,
    const std::vector<const drwnTableFactor *>& A) :
    drwnFactorOperation(target)
{
    DRWN_ASSERT(!A.empty());

    // add factors to list
    for (unsigned i = 0; i < _factors.size(); i++) {
	DRWN_ASSERT(_factors[i] != NULL);
    }
    _factors.insert(_factors.end(), A.begin(), A.end());

    initialize();
}

drwnFactorNAryOp::drwnFactorNAryOp(const drwnFactorNAryOp &op) :
    drwnFactorOperation(op)
{
    _factors = op._factors;
    _mappings = op._mappings;
}

drwnFactorNAryOp::~drwnFactorNAryOp()
{
    // do nothing
}

void drwnFactorNAryOp::initialize()
{
    // add variables and check domains match
    if (_target->empty()) {
	for (unsigned i = 0; i < _factors.size(); i++) {
	    _target->addVariables(*_factors[i]);
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mappings
    _mappings.resize(_factors.size());
    for (unsigned i = 0; i < _factors.size(); i++) {
        _mappings[i] = drwnTableFactorMapping(_target->getOrderedVars(),
            _factors[i]->getOrderedVars(), _target->getUniverse());
    }
}

bool drwnFactorNAryOp::checkTarget()
{
    for (unsigned i = 0; i < _factors.size(); i++) {
	for (unsigned j = 0; j < _factors[i]->size(); j++) {
	    if (!_target->hasVariable(_factors[i]->getOrderedVars()[j])) {
		return false;
	    }
	}
    }

    // TO DO: check reverse direction and domain sizes

    return true;
}

// drwnFactorAtomicOp class ------------------------------------------------

drwnFactorAtomicOp::drwnFactorAtomicOp(drwnFactorOperation *op) :
    drwnFactorOperation(op->target())
{
    _computations.push_back(op);
}

drwnFactorAtomicOp::drwnFactorAtomicOp(const vector<drwnFactorOperation *>& ops) :
    drwnFactorOperation(ops.back()->target()), _computations(ops)
{
    DRWN_ASSERT(!_computations.empty());
    for (unsigned i = 0; i < _computations.size(); i++) {
	DRWN_ASSERT(_computations[i] != NULL);
    }
    _target = _computations.back()->target();
}

drwnFactorAtomicOp::~drwnFactorAtomicOp()
{
    // delete computation tree
    for (unsigned i = 0; i < _computations.size(); i++) {
	delete _computations[i];
    }
    _computations.clear();
}

void drwnFactorAtomicOp::execute()
{
    for (unsigned i = 0; i < _computations.size(); i++) {
	_computations[i]->execute();
    }
}

void drwnFactorAtomicOp::addOperation(drwnFactorOperation *op)
{
    DRWN_ASSERT(op != NULL);
    _computations.push_back(op);
    _target = _computations.back()->target();
}

// drwnFactorCopyOp class --------------------------------------------------

drwnFactorCopyOp::drwnFactorCopyOp(drwnTableFactor *target, const drwnTableFactor *A) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    initialize();
}

drwnFactorCopyOp::drwnFactorCopyOp(const drwnFactorCopyOp& op) :
    drwnFactorOperation(op), _operandA(op._operandA), _mappingA(op._mappingA)
{
    // do nothing
}

drwnFactorCopyOp::~drwnFactorCopyOp()
{
    // do nothing
}

void drwnFactorCopyOp::execute()
{
    if (_mappingA.empty()) {
        memcpy(&_target[0], &_operandA[0], _target->size() * sizeof(double));
    } else {
        drwnTableFactorMapping::iterator ia(_mappingA.begin());
        for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
            (*_target)[i] = (*_operandA)[*ia];
        }
    }
}

void drwnFactorCopyOp::initialize()
{
    if (_target->empty()) {
        // direct copy of tables
	_target->addVariables(*_operandA);
    } else if (_operandA->getOrderedVars() != _target->getOrderedVars()) {
        // create mapping for copying table entries
        _mappingA = drwnTableFactorMapping(_target->getOrderedVars(),
            _operandA->getOrderedVars(), _target->getUniverse());
    }
}

bool drwnFactorCopyOp::checkTarget()
{
    if (_mappingA.empty()) {
        if (_operandA->getOrderedVars() != _target->getOrderedVars()) {
            return false;
        }
    }

    return true;
}

// drwnFactorProductOp -----------------------------------------------------

drwnFactorProductOp::drwnFactorProductOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorNAryOp(target, A, B)
{
    // do nothing
}

drwnFactorProductOp::drwnFactorProductOp(drwnTableFactor *target,
    const std::vector<const drwnTableFactor *>& A) :
    drwnFactorNAryOp(target, A)
{
    // do nothing
}

drwnFactorProductOp::drwnFactorProductOp(const drwnFactorProductOp &phi) :
    drwnFactorNAryOp(phi)
{
    // do nothing
}

drwnFactorProductOp::~drwnFactorProductOp()
{
    // do nothing
}

void drwnFactorProductOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsProductCount += (int)_factors.size() - 1;
#endif

    // copy first factor
    if (_factors[0]->empty()) {
        _target->fill(1.0);
    } else if (_target != _factors[0]) {
        drwnTableFactorMapping::iterator it = _mappings[0].begin();
        for (unsigned i = 0; i < _target->entries(); ++i, ++it) {
            (*_target)[i] = (*_factors[0])[*it];
        }
    }

    // multiply remaining factors
    for (unsigned k = 1; k < _factors.size(); k++) {
        if (_factors[k]->empty())
            continue;
        drwnTableFactorMapping::iterator it = _mappings[k].begin();
        for (unsigned i = 0; i < _target->entries(); ++i, ++it) {
            (*_target)[i] *= (*_factors[k])[*it];
        }
    }
}

// drwnFactorDivideOp ------------------------------------------------------

drwnFactorDivideOp::drwnFactorDivideOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorBinaryOp(target, A, B)
{
    // do nothing
}

drwnFactorDivideOp::drwnFactorDivideOp(const drwnFactorDivideOp &phi) :
    drwnFactorBinaryOp(phi)
{
    // do nothing
}

drwnFactorDivideOp::~drwnFactorDivideOp()
{
    // do nothing
}

void drwnFactorDivideOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsDivideCount += 1;
#endif

    drwnTableFactorMapping::iterator ia(_mappingA.begin());
    drwnTableFactorMapping::iterator ib(_mappingB.begin());
    for (unsigned i = 0; i < _target->entries(); ++i, ++ia, ++ib) {
        // check for zero (and prevent zero/zero)
        if ((*_operandA)[*ia] == 0.0) {
            (*_target)[i] = 0.0;
            continue;
        }
        (*_target)[i] = (*_operandA)[*ia] / (*_operandB)[*ib];
    }
}

// drwnFactorAdditionOp ----------------------------------------------------

drwnFactorAdditionOp::drwnFactorAdditionOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorNAryOp(target, A, B)
{
    // do nothing
}

drwnFactorAdditionOp::drwnFactorAdditionOp(drwnTableFactor *target,
    const std::vector<const drwnTableFactor *>& A) :
    drwnFactorNAryOp(target, A)
{
    // do nothing
}

drwnFactorAdditionOp::drwnFactorAdditionOp(const drwnFactorAdditionOp &phi) :
    drwnFactorNAryOp(phi)
{
    // do nothing
}

drwnFactorAdditionOp::~drwnFactorAdditionOp()
{
    // do nothing
}

void drwnFactorAdditionOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsAdditionCount += (int)_factors.size() - 1;
#endif

    // copy first factor
    if (_factors[0]->empty()) {
        _target->fill(0.0);
    } else if (_target != _factors[0]) {
        drwnTableFactorMapping::iterator it = _mappings[0].begin();
#if 0
        for (unsigned i = 0; i < _target->entries(); ++i, ++it) {
            (*_target)[i] = (*_factors[0])[*it];
        }
#else
        double *p = &((*_target)[0]);
        const double *q = &(*_factors[0])[0];
        for (unsigned i = _target->entries(); i > 0; --i, ++it) {
            (*p++) = q[*it];
        }
#endif
    }

    // add remaining factors
    for (unsigned k = 1; k < _factors.size(); k++) {
        if (_factors[k]->empty())
            continue;
        drwnTableFactorMapping::iterator it = _mappings[k].begin();
#if 0
        for (unsigned i = 0; i < _target->entries(); ++i, ++it) {
            (*_target)[i] += (*_factors[k])[*it];
        }
#else
        double *p = &((*_target)[0]);
        const double *q = &(*_factors[k])[0];
        for (unsigned i = _target->entries(); i > 0; --i, ++it) {
            (*p++) += q[*it];
        }
#endif
    }
}

// drwnFactorSubtractOp ------------------------------------------------------

drwnFactorSubtractOp::drwnFactorSubtractOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B) :
    drwnFactorBinaryOp(target, A, B)
{
    // do nothing
}

drwnFactorSubtractOp::drwnFactorSubtractOp(const drwnFactorSubtractOp &phi) :
    drwnFactorBinaryOp(phi)
{
    // do nothing
}

drwnFactorSubtractOp::~drwnFactorSubtractOp()
{
    // do nothing
}

void drwnFactorSubtractOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsSubtractionCount += 1;
#endif

    drwnTableFactorMapping::iterator ia(_mappingA.begin());
    drwnTableFactorMapping::iterator ib(_mappingB.begin());
    if (_operandA->empty()) {
        for (unsigned i = 0; i < _target->entries(); ++i, ++ib) {
            (*_target)[i] = - (*_operandB)[*ib];
        }
    } else if (_operandB->empty()) {
        for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
            (*_target)[i] = (*_operandA)[*ia];
        }
    } else {
        for (unsigned i = 0; i < _target->entries(); ++i, ++ia, ++ib) {
            (*_target)[i] = (*_operandA)[*ia] - (*_operandB)[*ib];
        }
    }
}

// drwnFactorWeightedSumOp ---------------------------------------------------

drwnFactorWeightedSumOp::drwnFactorWeightedSumOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnTableFactor *B, double wA, double wB) :
    drwnFactorBinaryOp(target, A, B), _weightA(wA), _weightB(wB)
{
    // do nothing
}

drwnFactorWeightedSumOp::drwnFactorWeightedSumOp(const drwnFactorWeightedSumOp &phi) :
    drwnFactorBinaryOp(phi), _weightA(phi._weightA), _weightB(phi._weightB)
{
    // do nothing
}

drwnFactorWeightedSumOp::~drwnFactorWeightedSumOp()
{
    // do nothing
}

void drwnFactorWeightedSumOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsAdditionCount += 1;
#endif

    drwnTableFactorMapping::iterator ia = _mappingA.begin();
    drwnTableFactorMapping::iterator ib = _mappingB.begin();
    for (unsigned indx = 0; indx < _target->entries(); ++indx, ++ia, ++ib) {
        (*_target)[indx] = _weightA * (*_operandA)[*ia] + _weightB * (*_operandB)[*ib];
    }
}

// drwnFactorPlusEqualsOp class ---------------------------------------------

drwnFactorPlusEqualsOp::drwnFactorPlusEqualsOp(drwnTableFactor *target,
    const drwnTableFactor *A, double wA) : drwnFactorUnaryOp(target, A), _weightA(wA)
{
    // do nothing
}

drwnFactorPlusEqualsOp::drwnFactorPlusEqualsOp(const drwnFactorPlusEqualsOp &phi) :
    drwnFactorUnaryOp(phi), _weightA(phi._weightA)
{
    // do nothing
}

drwnFactorPlusEqualsOp::~drwnFactorPlusEqualsOp()
{
    // do nothing
}

void drwnFactorPlusEqualsOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsAdditionCount += 1;
#endif

    if (!_operandA->empty()) {
        drwnTableFactorMapping::iterator ia(_mappingA.begin());
#if 0
        if (_weightA == 1.0) {
            for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
                (*_target)[i] += (*_operandA)[*ia];
            }
        } else {
            for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
                (*_target)[i] += _weightA * (*_operandA)[*ia];
            }
        }
#else
        double *p = &(*_target)[0];
        const double *q = &(*_operandA)[0];
        if (_weightA == 1.0) {
            for (unsigned i = _target->entries(); i > 0; --i, ++ia) {
                (*p++) += q[*ia];
            }
        } else {
            for (unsigned i = _target->entries(); i > 0; --i, ++ia) {
                (*p++) += _weightA * q[*ia];
            }
        }
#endif
    }
}

// drwnFactorMinusEqualsOp class --------------------------------------------

drwnFactorMinusEqualsOp::drwnFactorMinusEqualsOp(drwnTableFactor *target,
    const drwnTableFactor *A, double wA) : drwnFactorUnaryOp(target, A), _weightA(wA)
{
    // do nothing
}

drwnFactorMinusEqualsOp::drwnFactorMinusEqualsOp(const drwnFactorMinusEqualsOp &phi) :
    drwnFactorUnaryOp(phi), _weightA(phi._weightA)
{
    // do nothing
}

drwnFactorMinusEqualsOp::~drwnFactorMinusEqualsOp()
{
    // do nothing
}

void drwnFactorMinusEqualsOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsSubtractionCount += 1;
#endif

    if (!_operandA->empty()) {
        drwnTableFactorMapping::iterator ia(_mappingA.begin());
        if (_weightA == 1.0) {
            for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
                (*_target)[i] -= (*_operandA)[*ia];
            }
        } else {
            for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
                (*_target)[i] -= _weightA * (*_operandA)[*ia];
            }
        }
    }
}

// drwnFactorTimesEqualsOp class --------------------------------------------

drwnFactorTimesEqualsOp::drwnFactorTimesEqualsOp(drwnTableFactor *target,
    const drwnTableFactor *A) : drwnFactorUnaryOp(target, A)
{
    // do nothing
}

drwnFactorTimesEqualsOp::drwnFactorTimesEqualsOp(const drwnFactorTimesEqualsOp &phi) :
    drwnFactorUnaryOp(phi)
{
    // do nothing
}

drwnFactorTimesEqualsOp::~drwnFactorTimesEqualsOp()
{
    // do nothing
}

void drwnFactorTimesEqualsOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsProductCount += 1;
#endif

    if (!_operandA->empty()) {
        drwnTableFactorMapping::iterator ia(_mappingA.begin());
        for (unsigned i = 0; i < _target->entries(); ++i, ++ia) {
            (*_target)[i] *= (*_operandA)[*ia];
        }
    }
}

// drwnFactorMarginalizeOp class --------------------------------------------

drwnFactorMarginalizeOp::drwnFactorMarginalizeOp(drwnTableFactor *target,
    const drwnTableFactor *A) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(checkTarget());

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}


drwnFactorMarginalizeOp::drwnFactorMarginalizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, int v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(_operandA->hasVariable(v));

    // initialize target
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (_operandA->varId(i) == v)
		continue;
	    _target->addVariable(_operandA->varId(i));
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMarginalizeOp::drwnFactorMarginalizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, const std::set<int>& v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);

    // add variables
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (v.find(_operandA->varId(i)) != v.end())
		continue;
	    _target->addVariable(_operandA->varId(i));
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMarginalizeOp::drwnFactorMarginalizeOp(const drwnFactorMarginalizeOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _mappingA(op._mappingA)
{
    // do nothing
}

drwnFactorMarginalizeOp::~drwnFactorMarginalizeOp()
{
    // do nothing
}

void drwnFactorMarginalizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsMarginalCount += 1;
#endif
    _target->fill(0.0);

    drwnTableFactorMapping::iterator it(_mappingA.begin());
    for (unsigned i = 0; i < _operandA->entries(); ++i, ++it) {
        (*_target)[*it] += (*_operandA)[i];
    }
}

bool drwnFactorMarginalizeOp::checkTarget()
{
    // TO DO
    return true;
}

// drwnFactorMaximizeOp class ------------------------------------------------

drwnFactorMaximizeOp::drwnFactorMaximizeOp(drwnTableFactor *target,
    const drwnTableFactor *A) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(checkTarget());

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMaximizeOp::drwnFactorMaximizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, int v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(_operandA->hasVariable(v));

    // initialize target
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (_operandA->varId(i) != v) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMaximizeOp::drwnFactorMaximizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, const std::set<int>& v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);

    // add variables
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (v.find(_operandA->varId(i)) == v.end()) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMaximizeOp::drwnFactorMaximizeOp(const drwnFactorMaximizeOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _mappingA(op._mappingA)
{
    // do nothing
}

drwnFactorMaximizeOp::~drwnFactorMaximizeOp()
{
    // do nothing
}

void drwnFactorMaximizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsMaxMinCount += 1;
#endif
    if (_target->empty())
        return;

    _target->fill(-numeric_limits<double>::max());

    drwnTableFactorMapping::iterator it(_mappingA.begin());
    for (unsigned i = 0; i < _operandA->entries(); ++i, ++it) {
        (*_target)[*it] = std::max((*_target)[*it], (*_operandA)[i]);
    }
}

bool drwnFactorMaximizeOp::checkTarget()
{
    // TO DO
    DRWN_ASSERT(!_operandA->empty());

    return true;
}

// drwnFactorMinimizeOp class -----------------------------------------------

drwnFactorMinimizeOp::drwnFactorMinimizeOp(drwnTableFactor *target,
    const drwnTableFactor *A) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(checkTarget());

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMinimizeOp::drwnFactorMinimizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, int v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(_operandA->hasVariable(v));

    // initialize target
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (_operandA->varId(i) != v) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMinimizeOp::drwnFactorMinimizeOp(drwnTableFactor *target,
    const drwnTableFactor *A, const std::set<int>& v) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);

    // add variables
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (v.find(_operandA->varId(i)) == v.end()) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }

    // create mapping
    _mappingA = drwnTableFactorMapping(_operandA->getOrderedVars(),
        _target->getOrderedVars(), _target->getUniverse());
}

drwnFactorMinimizeOp::drwnFactorMinimizeOp(const drwnFactorMinimizeOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _mappingA(op._mappingA)
{
    // do nothing
}

drwnFactorMinimizeOp::~drwnFactorMinimizeOp()
{
    // do nothing
}

void drwnFactorMinimizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsMaxMinCount += 1;
#endif
    if (_target->empty())
        return;

    _target->fill(numeric_limits<double>::max());

    drwnTableFactorMapping::iterator it(_mappingA.begin());
#if 0
    for (unsigned i = 0; i < _operandA->entries(); ++i, ++it) {
        (*_target)[*it] = std::min((*_target)[*it], (*_operandA)[i]);
    }
#else
    double *p = &(*_target)[0];
    const double *q = &(*_operandA)[0];
    for (unsigned i = _operandA->entries(); i > 0; --i, ++it) {
        p[*it] = std::min(p[*it], *q++);
    }
#endif
}

bool drwnFactorMinimizeOp::checkTarget()
{
    // TO DO
    DRWN_ASSERT(!_operandA->empty());

    return true;
}

// drwnFactorReduceOp class -------------------------------------------------

drwnFactorReduceOp::drwnFactorReduceOp(drwnTableFactor *target,
    const drwnTableFactor *A, int var, int val) :
    drwnFactorOperation(target), _operandA(A)
{
    DRWN_ASSERT(A != NULL);
    DRWN_ASSERT(_operandA->hasVariable(var));
    DRWN_ASSERT((val >= 0) && (val < _operandA->getUniverse()->varCardinality(var)));

    _assignment[var] = val;

    // initialize target
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (_operandA->varId(i) != var) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }
}

drwnFactorReduceOp::drwnFactorReduceOp(drwnTableFactor *target,
    const drwnTableFactor *A, const drwnPartialAssignment& assignment) :
    drwnFactorOperation(target), _operandA(A), _assignment(assignment)
{
    DRWN_ASSERT(A != NULL);

    // add variables
    if (_target->empty()) {
	for (unsigned i = 0; i < _operandA->size(); i++) {
	    if (assignment.find(_operandA->varId(i)) == assignment.end()) {
                _target->addVariable(_operandA->varId(i));
            }
	}
    } else {
	DRWN_ASSERT(checkTarget());
    }
}

drwnFactorReduceOp::drwnFactorReduceOp(const drwnFactorReduceOp &op) :
    drwnFactorOperation(op), _operandA(op._operandA), _assignment(op._assignment)
{
    // do nothing
}

drwnFactorReduceOp::~drwnFactorReduceOp()
{
    // do nothing
}

void drwnFactorReduceOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsReductionCount += 1;
#endif
    if (_target->empty())
        return;

    //! \todo speed up
    drwnPartialAssignment assignment(_assignment);
    for (unsigned indx = 0; indx < _target->entries(); indx++) {
        _target->assignmentOf(indx, assignment);
        (*_target)[indx] = (*_operandA)[_operandA->indexOf(assignment)];
    }
}

bool drwnFactorReduceOp::checkTarget()
{
    // TO DO
    DRWN_ASSERT(!_operandA->empty());

    return true;
}

// drwnFactorNormalizeOp class ---------------------------------------------

drwnFactorNormalizeOp::drwnFactorNormalizeOp(drwnTableFactor *target) :
    drwnFactorOperation(target)
{
    // do nothing
}

drwnFactorNormalizeOp::~drwnFactorNormalizeOp()
{
    // do nothing
}

void drwnFactorNormalizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsNormalizeCount += 1;
#endif

    if (_target->empty())
	return;

    double total = 0.0;
    for (unsigned i = 0; i < _target->entries(); i++) {
        total += (*_target)[i];
    }
    if (total > 0.0) {
        if (total != 1.0) {
            const double invTotal = 1.0 / total;
            for (unsigned i = 0; i < _target->entries(); i++) {
                (*_target)[i] *= invTotal;
            }
        }
    } else {
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
	drwnFactor::_dbStatsNormalizeErrors += 1;
#endif
        _target->fill(1.0 / (double)_target->entries());
    }
}

bool drwnFactorNormalizeOp::checkTarget()
{
    // normalizing into yourself, so always true
    return true;
}

// drwnFactorExpAndNormalizeOp class ---------------------------------------

drwnFactorExpAndNormalizeOp::drwnFactorExpAndNormalizeOp(drwnTableFactor *target) :
    drwnFactorOperation(target)
{
    // do nothing
}

drwnFactorExpAndNormalizeOp::~drwnFactorExpAndNormalizeOp()
{
    // do nothing
}

void drwnFactorExpAndNormalizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsNormalizeCount += 1;
#endif

    if (_target->empty())
	return;

    const double maxValue = (*_target)[_target->indexOfMax()];
    double total = 0.0;
    for (unsigned i = 0; i < _target->entries(); i++) {
        (*_target)[i] = exp((*_target)[i] - maxValue);
        total += (*_target)[i];
    }

    if (total != 1.0) {
        const double invTotal = 1.0 / total;
        for (unsigned i = 0; i < _target->entries(); i++) {
            (*_target)[i] *= invTotal;
        }
    }
}

bool drwnFactorExpAndNormalizeOp::checkTarget()
{
    // normalizing into yourself, so always true
    return true;
}

// drwnLogFactorNormalizeOp class ------------------------------------------

drwnFactorLogNormalizeOp::drwnFactorLogNormalizeOp(drwnTableFactor *target) :
    drwnFactorOperation(target)
{
    // do nothing
}

drwnFactorLogNormalizeOp::~drwnFactorLogNormalizeOp()
{
    // do nothing
}

void drwnFactorLogNormalizeOp::execute()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsNormalizeCount += 1;
#endif

    if (_target->empty())
	return;

    double maxVal = -numeric_limits<double>::max();
    for (unsigned i = 0; i < _target->entries(); i++) {
        maxVal = std::max(maxVal, (*_target)[i]);
    }
    for (unsigned i = 0; i < _target->entries(); i++) {
        (*_target)[i] -= maxVal;
    }
}

bool drwnFactorLogNormalizeOp::checkTarget()
{
    // normalizing into yourself, so always true
    return true;
}
