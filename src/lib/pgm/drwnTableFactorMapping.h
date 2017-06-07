/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTableFactorMapping.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>

#include "drwnBase.h"
#include "drwnIO.h"

#include "drwnVarUniverse.h"

using namespace std;

//! \todo make this a configurable option
//! \todo cache mappings
#define DRWN_FACTOR_MAPPING_FULL
#undef DRWN_FACTOR_MAPPING_FULL

// drwnTableFactorMapping --------------------------------------------------
//! \brief Creates a mapping between entries in two tables.
//!
//! The target table must contain a superset of the variables of the
//! source table.
//! \todo create a multi-table mapping (will reduce amount of repeated computation in iterators)
//! \todo specialize for unary and pairwise cases

class drwnTableFactorMapping {
 protected:
    int _targetSize;            //!< number of entries in the target table

    // the following are not used for identity mappings
    vector<int> _targetCards;   //!< cardinality of each variable in the target
    vector<int> _strideMapping; //!< stride mapping from source to target
#ifdef DRWN_FACTOR_MAPPING_FULL
    vector<int> _fullMapping;   //!< full mapping of size _targetSize
#endif

 public:
    //! iterator for indexing entries in the tables
    class iterator {
    protected:
        friend class drwnTableFactorMapping;

        const drwnTableFactorMapping *_owner; //!< the factor mapping that owns this iterator
        int _dstIndx;             //!< current index into the target table
        int _srcIndx;             //!< current index into the source table
        vector<int> _assignment;  //!< current assignement to the variables in the target table

        inline iterator(const drwnTableFactorMapping& m) :
            _owner(&m), _dstIndx(0), _srcIndx(0) {

            if (!_owner->identity()) {
#ifdef DRWN_FACTOR_MAPPING_FULL
                if (_owner->_fullMapping.empty())
#endif
                _assignment.resize(_owner->_targetCards.size(), 0);
            }
        }

        inline iterator(const drwnTableFactorMapping& m, size_t indx) :
            _owner(&m), _dstIndx(indx), _srcIndx(0) {

            if (_owner->identity()) {
                _srcIndx = _dstIndx;
                return;
            }

#ifdef DRWN_FACTOR_MAPPING_FULL
            // use full mapping if available, otherwise compute srcIndx
            if (!_owner->_fullMapping.empty()) {
                _srcIndx = (_dstIndx < (int)_owner->_fullMapping.size()) ?
                    _owner->_fullMapping[_dstIndx] : 0;
            } else {
#else
            {
#endif
                _assignment.resize(_owner->_targetCards.size(), 0);
                if (_dstIndx != 0) {
                    int stride = 1;
                    for (unsigned i = 0; i < _assignment.size(); i++) {
                        int r = (int)(_dstIndx / stride);
                        if (r == 0) break;
                        _assignment[i] = r % _owner->_targetCards[i];
                        _srcIndx += r * _owner->_strideMapping[i];
                        stride *= _owner->_targetCards[i];
                    }
                }
            }
        }

    public:
        inline iterator() : _owner(NULL), _dstIndx(0), _srcIndx(0) { /* do nothing */ }
        inline ~iterator() { /* do nothing */ }

        inline iterator& operator=(const iterator& it) {
            _owner = it._owner; _dstIndx = it._dstIndx; _srcIndx = it._srcIndx;
            _assignment = it._assignment;
            return *this;
        }
        inline bool operator==(const iterator& it) const {
            return ((_owner == it._owner) && (_dstIndx == it._dstIndx));
        }
        inline bool operator!=(const iterator& it) const {
            return (!operator==(it));
        }
        inline bool operator<(const iterator& it) const {
            return (_dstIndx < it._dstIndx);
        }
        inline bool operator>(const iterator& it) const {
            return (_dstIndx > it._dstIndx);
        }
        inline bool operator<=(const iterator& it) const {
            return (_dstIndx <= it._dstIndx);
        }
        inline bool operator>=(const iterator& it) const {
            return (_dstIndx >= it._dstIndx);
        }
        inline const int& operator*() const {
            return _srcIndx;
        }
        inline iterator& operator++() {
            _dstIndx += 1;

            if (_owner->identity()) {
                _srcIndx = _dstIndx;
                return *this;
            }

#ifdef DRWN_FACTOR_MAPPING_FULL
            if (!_owner->_fullMapping.empty()) {
                _srcIndx = (_dstIndx < (int)_owner->_fullMapping.size()) ?
                    _owner->_fullMapping[_dstIndx] : 0;
                return *this;
            }
#endif
            _assignment[0] += 1;
            _srcIndx += _owner->_strideMapping[0];
            for (unsigned i = 1; i < _assignment.size(); i++) {
                if (_assignment[i - 1] < _owner->_targetCards[i - 1])
                    break;
                _assignment[i - 1] = 0;
                _assignment[i] += 1;
                _srcIndx += _owner->_strideMapping[i];
            }

            return *this;
        }

        inline iterator& operator--() {
            _dstIndx -= 1;

            if (_owner->identity()) {
                _srcIndx = _dstIndx;
                return *this;
            }

#ifdef DRWN_FACTOR_MAPPING_FULL
            if (!_owner->_fullMapping.empty()) {
                _srcIndx = (_dstIndx >= 0) ? _owner->_fullMapping[_dstIndx] : 0;
                return *this;
            }
#endif
            _assignment[0] -= 1;
            _srcIndx -= _owner->_strideMapping[0];
            for (unsigned i = 1; i < _assignment.size(); i++) {
                if (_assignment[i - 1] >= 0)
                    break;
                _assignment[i - 1] = _owner->_targetCards[i - 1] - 1;
                _assignment[i] -= 1;
                _srcIndx -= _owner->_strideMapping[i];
            }
            return *this;
        }

        inline iterator operator++(int) {
            iterator copy(*this);
            ++(*this);
            return copy;
        }
        inline iterator operator--(int) {
            iterator copy(*this);
            --(*this);
            return copy;
        }
        inline const int& operator[](int n) const {
            iterator copy(*_owner, _dstIndx + n);
            return *copy;
        }
    };
    friend class iterator;

 public:
    //! default constructor
    drwnTableFactorMapping() : _targetSize(0) { /* do nothing */ }

    //! dstVar is a superset of srcVar (and variables don't repeat)
    drwnTableFactorMapping(const vector<int>& dstVars, const vector<int>& srcVars,
        const drwnVarUniversePtr& pUniverse) : _targetSize(1) {

#if 1
        // special case: unary source
        if (srcVars.size() == 1) {

            // check for identity mapping
            if (dstVars.size() == 1) {
                DRWN_ASSERT(dstVars[0] == srcVars[0]);
                _targetSize = pUniverse->varCardinality(dstVars[0]);
                return;
            }

            // find location of srcVars in dstVars
            unsigned k = dstVars.size();
            for (unsigned i = 0; i < dstVars.size(); i++) {
                if (dstVars[i] == srcVars[0]) k = i;
                _targetSize *= pUniverse->varCardinality(dstVars[i]);
            }
            DRWN_ASSERT(k != dstVars.size());

            // collapse first and last dstVars into a single variable
            if (k == 0) {
                _targetCards.resize(2);
                _strideMapping.resize(2);
                _targetCards[0] = pUniverse->varCardinality(srcVars[0]);                
                _targetCards[1] = _targetSize / _targetCards[0];
                _strideMapping[0] = 1;
                _strideMapping[1] = -1 * _targetCards[0];
            } else if (k + 1 == dstVars.size()) {
                _targetCards.resize(2);
                _strideMapping.resize(2);
                _targetCards[1] = pUniverse->varCardinality(srcVars[0]);                
                _targetCards[0] = _targetSize / _targetCards[1];
                _strideMapping[0] = 0;
                _strideMapping[1] = 1;
            } else {
                _targetCards.resize(3);
                _strideMapping.resize(3);
                _targetCards[0] = 1;
                for (unsigned i = 0; i < k; i++) {
                    _targetCards[0] *= pUniverse->varCardinality(dstVars[i]);
                }
                _targetCards[1] = pUniverse->varCardinality(srcVars[0]);                
                _targetCards[2] = _targetSize / (_targetCards[0] * _targetCards[1]);
                _strideMapping[0] = 0;
                _strideMapping[1] = 1;
                _strideMapping[2] = -1 * _targetCards[1];
            }

            return;
        }
#endif

        // check for identity or near identity mappings (first n variables
        // of dstVars match order of all variables in srcVars)
        if (std::equal(srcVars.begin(), srcVars.end(), dstVars.begin())) {
            for (unsigned i = 0; i < dstVars.size(); i++) {
                _targetSize *= pUniverse->varCardinality(dstVars[i]);
            }

            // collapse first srcVars variables into a single variable
            if (dstVars.size() > srcVars.size()) {
                _targetCards.resize(2, 1);
                _strideMapping.resize(2, 1);
                for (unsigned i = 0; i < srcVars.size(); i++) {
                    _targetCards[0] *= pUniverse->varCardinality(srcVars[i]);
                }
                _targetCards[1] = _targetSize / _targetCards[0];
                _strideMapping[1] = -1 * _targetCards[0];
            }

            return;
        }

#if 1
        // check for near identity mappings on last n variables
        if ((dstVars.size() > srcVars.size()) &&
            (std::equal(srcVars.begin(), srcVars.end(), dstVars.end() - srcVars.size()))) {
            for (unsigned i = 0; i < dstVars.size(); i++) {
                _targetSize *= pUniverse->varCardinality(dstVars[i]);
            }

            // collapse last srcVars variables into a single variable
            _targetCards.resize(2, 1);
            _strideMapping.resize(2, 1);
            for (unsigned i = 0; i < srcVars.size(); i++) {
                _targetCards[1] *= pUniverse->varCardinality(srcVars[i]);
            }
            _targetCards[0] = _targetSize / _targetCards[1];
            _strideMapping[0] = 0;

            return;
        }
#endif

        // build stride index for source variables
        //! \todo group variables as above if appear contiguously in src and dst
        map<int, unsigned> varIndex;
        unsigned stride = 1;
        for (unsigned i = 0; i < srcVars.size(); i++) {
            varIndex[srcVars[i]] = stride;
            stride *= pUniverse->varCardinality(srcVars[i]);
        }

        _targetCards.resize(dstVars.size(), 0);
        _strideMapping.resize(dstVars.size(), 0);
        for (unsigned i = 0; i < dstVars.size(); i++) {
            _targetCards[i] = pUniverse->varCardinality(dstVars[i]);
            _targetSize *= _targetCards[i];

            map<int, unsigned>::const_iterator v = varIndex.find(dstVars[i]);
            if (v != varIndex.end()) {
                _strideMapping[i] += v->second;
                if (i + 1 < dstVars.size()) {
                    _strideMapping[i + 1] = -1 * _targetCards[i] * v->second;
                }
            }
        }

#ifdef DRWN_FACTOR_MAPPING_FULL
        _fullMapping = mapping();
#endif
    }
    ~drwnTableFactorMapping() { /* do nothing */ }

    //! returns true if the mapping has size zero
    inline bool empty() const { return (_targetSize == 0); }

    //! returns true if the mapping is identity (destiation variables and source
    //! variables are in the same order)
    //inline bool identity() const { return (_targetSize != 0) && (_targetCards.empty()); }
    inline bool identity() const { return _targetCards.empty(); }

    //! iterators
    iterator begin() const { return iterator(*this); }
    iterator end() const { return iterator(*this, _targetSize); }

    //! returns the mapping of indexes from the source to the destiation
    vector<int> mapping() const {
        vector<int> m;
        m.reserve(_targetSize);
        for (iterator it = begin(); it != end(); ++it) {
            m.push_back(*it);
        }
        return m;
    }
};
