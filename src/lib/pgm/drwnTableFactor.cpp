/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTableFactor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <cstdarg>
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

#include "drwnTableFactor.h"

using namespace std;
using namespace Eigen;

// drwnFactor globals and constants ----------------------------------------

#ifdef DRWN_FACTOR_DEBUG_STATISTICS
unsigned drwnFactor::_dbStatsRefCount = 0;
unsigned drwnFactor::_dbStatsProductCount = 0;
unsigned drwnFactor::_dbStatsAdditionCount = 0;
unsigned drwnFactor::_dbStatsSubtractionCount = 0;
unsigned drwnFactor::_dbStatsDivideCount = 0;
unsigned drwnFactor::_dbStatsMarginalCount = 0;
unsigned drwnFactor::_dbStatsMaxMinCount = 0;
unsigned drwnFactor::_dbStatsReductionCount = 0;
unsigned drwnFactor::_dbStatsNormalizeCount = 0;
unsigned drwnFactor::_dbStatsNormalizeErrors = 0;
unsigned drwnFactor::_dbStatsLargestFactorSize = 0;
unsigned drwnFactor::_dbStatsCurrentMem = 0;
unsigned drwnFactor::_dbStatsMaxMem = 0;
unsigned drwnFactor::_dbStatsMaxTable = 0;
unsigned drwnFactor::_dbStatsTotalMem = 0;
static drwnVarUniversePtr dbDummyUniverse(NULL);
static drwnTableFactor dbDummyFactor(dbDummyUniverse);
#endif

// drwnFactor --------------------------------------------------------------

drwnFactor::drwnFactor(const drwnVarUniversePtr& ptr) :
    _pUniverse(ptr)
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    _dbStatsRefCount += 1;
#endif
}

drwnFactor::drwnFactor(const drwnFactor& psi) :
    _pUniverse(psi._pUniverse), _variables(psi._variables), _varIndex(psi._varIndex)
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    _dbStatsRefCount += 1;
#endif
}

drwnFactor::~drwnFactor()
{
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    _dbStatsRefCount -= 1;
    if ((_dbStatsRefCount == 0) && (_dbStatsLargestFactorSize > 0)) {
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsProductCount << " products");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsDivideCount << " divisions");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsAdditionCount << " additions");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsSubtractionCount << " subtractions");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsMarginalCount << " marginals");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsMaxMinCount << " maximizations/minimizations");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsReductionCount << " reductions");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsNormalizeCount << " normalizations");
	DRWN_LOG_MESSAGE("drwnFactor class computed " << _dbStatsNormalizeErrors << " normalization errors");
        DRWN_LOG_MESSAGE("drwnFactor class largest factor had " << _dbStatsLargestFactorSize << " variables ");
        DRWN_LOG_MESSAGE("drwnFactor class allocated " << _dbStatsTotalMem << " total entries");
        DRWN_LOG_MESSAGE("drwnFactor class allocated " << _dbStatsMaxMem << " max. concurrent entries");
        DRWN_LOG_MESSAGE("drwnFactor class allocated " << _dbStatsCurrentMem << " current entries");
        DRWN_LOG_MESSAGE("drwnFactor class allocated " << _dbStatsMaxTable << " entries in largest factor");
    }
#endif
}

// add variables
void drwnFactor::addVariable(int var)
{
    DRWN_ASSERT((var >= 0) && (var < _pUniverse->numVariables()));
    map<int, int>::iterator it = _varIndex.find(var);
    DRWN_ASSERT(it == _varIndex.end());
    _varIndex.insert(it, make_pair(var, (int)_varIndex.size()));
    _variables.push_back(var);

#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    if (_variables.size() > _dbStatsLargestFactorSize) {
        _dbStatsLargestFactorSize = _variables.size();
    }
#endif
}

void drwnFactor::addVariable(const char *name)
{
    DRWN_ASSERT(name != NULL);
    int var = _pUniverse->findVariable(name);
    DRWN_ASSERT(var != -1);
    addVariable(var);
}

void drwnFactor::addVariables(const char *name, ...)
{
    va_list ap;
    va_start(ap, name);
    while (name != NULL) {
        addVariable(_pUniverse->findVariable(name));
        name = va_arg(ap, const char *);
    }
    va_end(ap);
}

void drwnFactor::addVariables(const vector<int>& c)
{
    for (vector<int>::const_iterator it = c.begin(); it != c.end(); ++it) {
        if (_varIndex.find(*it) == _varIndex.end()) {
            addVariable(*it);
        }
    }
}

void drwnFactor::addVariables(const drwnClique& c)
{
    for (drwnClique::const_iterator it = c.begin(); it != c.end(); ++it) {
        if (_varIndex.find(*it) == _varIndex.end()) {
            addVariable(*it);
        }
    }
}

void drwnFactor::addVariables(const drwnFactor& psi)
{
    addVariables(psi._variables);
}

// i/o
bool drwnFactor::save(drwnXMLNode& xml) const
{
    drwnAddXMLChildNode(xml, "vars", toString(_variables).c_str(), false);
    return true;
}

bool drwnFactor::load(drwnXMLNode& xml)
{
    // clear existing variables
    _variables.clear();
    _varIndex.clear();

    // read variables
    vector<int> vars;
    drwnXMLNode *node = xml.first_node("vars");
    DRWN_ASSERT(node != NULL);
    parseString<int>(string(drwnGetXMLText(*node)), vars);

    addVariables(vars);

    return true;
}

#ifdef DRWN_FACTOR_DEBUG_STATISTICS
void drwnFactor::updateMemoryStats(int nSize, int previousSize)
{
    _dbStatsCurrentMem += (nSize - previousSize);
    if (_dbStatsCurrentMem > _dbStatsMaxMem)
        _dbStatsMaxMem = _dbStatsCurrentMem;
    if (nSize > (int)_dbStatsMaxTable)
        _dbStatsMaxTable = nSize;
    _dbStatsTotalMem += (nSize - previousSize);
    // check for overflow
    if ((int)_dbStatsTotalMem - (nSize - previousSize) < 0) {
        DRWN_LOG_WARNING_ONCE("_dbStatsTotalMem overflowed");
        _dbStatsTotalMem = (unsigned)-1;
    }
}
#endif

// drwnTableFactor --------------------------------------------------------

drwnTableFactor::drwnTableFactor(const drwnVarUniversePtr& ptr, drwnTableFactorStorage *storage) :
    drwnFactor(ptr), _nSize(0), _data(NULL), _storage(storage)
{
    // register shared storage
    if (_storage != NULL) {
        _storage->registerFactor(this);
    }
}

drwnTableFactor::drwnTableFactor(const drwnTableFactor& psi) :
    drwnFactor(psi), _stride(psi._stride), _nSize(psi._nSize), _data(NULL), _storage(psi._storage)
{
    // register shared storage
    if (_storage != NULL) {
        _storage->registerFactor(this);
    } else if (psi._data != NULL) {
        _data = new double[_nSize];
        memcpy(_data, psi._data, _nSize * sizeof(double));
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
        drwnFactor::updateMemoryStats(_nSize);
#endif
    }
}

drwnTableFactor::~drwnTableFactor()
{
    if (_storage != NULL) {
        _storage->unregisterFactor(this);
    } else if (_data != NULL) {
        delete[] _data;
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
        drwnFactor::_dbStatsCurrentMem -= _nSize;
#endif
    }
}

// add variables
// TODO: addVariables variant for multiple variables
void drwnTableFactor::addVariable(int var)
{
    drwnFactor::addVariable(var);

    int oldSize = _nSize;
    _stride.push_back(_nSize == 0 ? 1 : _nSize);
    _nSize = _stride.back() * _pUniverse->varCardinality(var);

    if (oldSize == _nSize)
        return;

    if (_storage == NULL) {
        if (_data == NULL) {
            _data = new double[_nSize];
            initialize();
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
            drwnFactor::updateMemoryStats(_nSize);
#endif
        } else {
            double *newData = new double[_nSize];

            // replicate table
            DRWN_ASSERT(oldSize != 0);
            for (int i = 0; i < _nSize; i += oldSize) {
                memcpy(&newData[i], &_data[0], oldSize * sizeof(double));
            }

            // delete old data table
            delete[] _data;
            _data = newData;
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
            drwnFactor::updateMemoryStats(_nSize, oldSize);
#endif
        }
    } else {
        _storage->reserve(_nSize);
        initialize();
    }
}

drwnTableFactor& drwnTableFactor::initialize()
{
    if (_nSize != 0) {
        memset((void *)_data, (int)0.0, _nSize * sizeof(double));
    }
    return *this;
}

drwnTableFactor& drwnTableFactor::fill(double alpha)
{
    if (_nSize != 0) {
        double *p = &_data[0];
        for (int i = _nSize / 4; i != 0; i--) {
            p[0] = p[1] = p[2] = p[3] = alpha;
            p += 4;
        }
        for (int i = 0; i < _nSize % 4; i++) {
            (*p++) = alpha;
        }
    }

    return *this;
}

drwnTableFactor& drwnTableFactor::copy(const double *data)
{
    memcpy(_data, data, _nSize * sizeof(double));
    return *this;
}

drwnTableFactor& drwnTableFactor::scale(double alpha)
{
    for (int i = 0; i < _nSize; i++) {
        _data[i] *= alpha;
    }

    return *this;
}

drwnTableFactor& drwnTableFactor::offset(double alpha)
{
    for (int i = 0; i < _nSize; i++) {
        _data[i] += alpha;
    }

    return *this;
}

drwnTableFactor& drwnTableFactor::normalize()
{
    if (_data == NULL) return *this;

#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    _dbStatsNormalizeCount += 1;
#endif
    double total = 0.0;
    for (int i = 0; i < _nSize; i++) {
        total += _data[i];
    }
    if (total > 0.0) {
        if (total != 1.0) {
            double invTotal = 1.0 / total;
            for (int i = 0; i < _nSize; i++) {
                _data[i] *= invTotal;
            }
        }
    } else {
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
	_dbStatsNormalizeErrors += 1;
#endif
        fill(1.0 / (double)_nSize);
    }

    return *this;
}

int drwnTableFactor::indexOf(int var, int val, int indx) const {
    //DRWN_ASSERT(hasVariable(var));
    int vi = _varIndex.find(var)->second;
    //DRWN_ASSERT((val >= 0) && (val < _pUniverse->varCardinality(var)));

    val -= ((int)(indx / _stride[vi])) % _pUniverse->varCardinality(var);
    indx += val * _stride[vi];

    return indx;
}

int drwnTableFactor::indexOf(const drwnPartialAssignment& assignment) const
{
    int indx = 0;
    for (int i = 0; i < (int)_variables.size(); i++) {
	drwnPartialAssignment::const_iterator it = assignment.find(_variables[i]);
	//DRWN_ASSERT(it != assignment.end());
	//DRWN_ASSERT((it->second >= 0) && (it->second < _cards[i]));
	indx += it->second * _stride[i];
    }

    return indx;
}

int drwnTableFactor::indexOf(const drwnFullAssignment& assignment) const
{
    //DRWN_ASSERT(assignment.size() == _pUniverse.numVariables());
    int indx = 0;
    for (int i = 0; i < (int)_variables.size(); i++) {
	//DRWN_ASSERT((assignment[_variables[i]] >= 0) &&
        //    (assignment[_variables[i]] < _pUniverse->varCardinality(_variables[i])));
	indx += assignment[_variables[i]] * _stride[i];
    }
    return indx;
}

int drwnTableFactor::valueOf(int var, int indx) const {
    int vi = _varIndex.find(var)->second;
    return ((int)(indx / _stride[vi])) % _pUniverse->varCardinality(var);
}

void drwnTableFactor::assignmentOf(int indx, drwnPartialAssignment& assignment) const
{
    for (int i = 0; i < (int)_variables.size(); i++) {
	assignment[_variables[i]] = ((int)(indx / _stride[i])) % _pUniverse->varCardinality(_variables[i]);
    }
}

void drwnTableFactor::assignmentOf(int indx, drwnFullAssignment& assignment) const
{
    assignment.resize(_pUniverse->numVariables(), -1);
    for (int i = 0; i < (int)_variables.size(); i++) {
	assignment[_variables[i]] = ((int)(indx / _stride[i])) % _pUniverse->varCardinality(_variables[i]);
    }
}

int drwnTableFactor::indexOfMin() const
{
    int indx = 0;
    for (int i = 1; i < _nSize; i++) {
        if (_data[indx] > _data[i]) {
            indx = i;
        }
    }

    return indx;
}

int drwnTableFactor::indexOfMax() const
{
    int indx = 0;
    for (int i = 1; i < _nSize; i++) {
        if (_data[indx] < _data[i]) {
            indx = i;
        }
    }

    return indx;
}

pair<int, int> drwnTableFactor::indexOfMinAndMax() const
{
    pair<int, int> indx(0, 0);
    for (int i = 1; i < _nSize; i++) {
        if (_data[indx.first] > _data[i]) {
            indx.first = i;
        }
        if (_data[indx.second] < _data[i]) {
            indx.second = i;
        }
    }

    return indx;
}

// i/o
bool drwnTableFactor::save(drwnXMLNode& xml) const
{
    drwnFactor::save(xml);

    if (_data != NULL) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "data", NULL, false);
        const VectorXd& v = Eigen::Map<VectorXd>(_data, _nSize);
        drwnXMLUtils::serialize(*node, v);
    }

    return true;
}

bool drwnTableFactor::load(drwnXMLNode& xml)
{
    // free existing storage
    if ((_storage == NULL) && (_data != NULL)) {
        delete[] _data;
        _data = NULL;
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
        drwnFactor::_dbStatsCurrentMem -= _nSize;
#endif
    }

    // load and add variables
    drwnFactor::load(xml);

    // read table data
    drwnXMLNode *node = xml.first_node("data");
    if ((node == NULL) || drwnIsXMLEmpty(*node)) {
        initialize();
    } else {
        VectorXd v;
        drwnXMLUtils::deserialize(*node, v);
        DRWN_ASSERT_MSG(v.size() == _nSize, v.size() << " != " << _nSize);

        Eigen::Map<VectorXd>(_data, _nSize) = v;
    }

    return true;
}

bool drwnTableFactor::dataCompare(const drwnTableFactor& psi) const
{
    if (psi._nSize != _nSize) return false;

    // unroll loop
    const double *p = &_data[0];
    const double *q = &psi._data[0];
    for (int i = _nSize / 2; i != 0; i--, p += 2, q += 2) {
        if ((fabs(p[0] - q[0]) > DRWN_EPSILON) || (fabs(p[1] - q[1]) > DRWN_EPSILON)) {
            return false;
        }
    }

    return ((_nSize % 2 == 0) || (fabs(p[0] - q[0]) <= DRWN_EPSILON));
}

bool drwnTableFactor::dataCompareAndCopy(const drwnTableFactor& psi)
{
    if (psi._nSize != _nSize) {
        *this = psi;
        return false;
    }

    for (int i = 0; i < _nSize; i++) {
        if (fabs(_data[i] - psi._data[i]) > DRWN_EPSILON) {
            memcpy(&_data[i], &psi._data[i], (_nSize - i) * sizeof(double));
            return false;
        }
        _data[i] = psi._data[i];
    }

    return true;
}

drwnTableFactor& drwnTableFactor::operator=(const drwnTableFactor& psi)
{
    if (this->_data == psi._data) {
        // Really want to check (*this == psi), but check on
        // data is much quicker. Also works for _data == NULL
        return *this;
    }

    if ((_storage == NULL) && (_data != NULL) && (_nSize != psi._nSize)) {
        delete[] _data;
        _data = NULL;
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
        drwnFactor::_dbStatsCurrentMem -= _nSize;
#endif
    }

    // drwnFactor data members
    _pUniverse = psi._pUniverse;
    _variables = psi._variables;
    _varIndex = psi._varIndex;

    // drwnTableFactor data members
    _nSize = psi._nSize;
    _stride = psi._stride;

    if (psi._data != NULL) {
        if (_storage == NULL) {
            if (_data == NULL) {
                _data = new double[_nSize];
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
                drwnFactor::updateMemoryStats(_nSize);
#endif
            }
            memcpy(_data, psi._data, _nSize * sizeof(double));
        } else {
            if (_storage != psi._storage) {
                _storage->copy(psi._data, _nSize);
            }
        }
    }

    return *this;
}


// drwnTableFactorStorage class -------------------------------------------

drwnTableFactorStorage::drwnTableFactorStorage(int nSize) :
    _dataSize(0), _data(NULL)
{
    reserve(nSize);
}

drwnTableFactorStorage::~drwnTableFactorStorage()
{
    // all tables must be unregistered
    DRWN_ASSERT(_tables.empty());

    // delete memory
    if (_data != NULL)
        delete[] _data;
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    drwnFactor::_dbStatsCurrentMem -= _dataSize;
#endif
}

void drwnTableFactorStorage::reserve(int nSize)
{
    if (_dataSize < nSize) {
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
        drwnTableFactor::updateMemoryStats(nSize, _dataSize);
#endif
        // allocate new data
        double *newData = new double[nSize];
        DRWN_ASSERT(newData != NULL);
        if (_data != NULL) {
            memcpy(newData, _data, _dataSize * sizeof(double));
            delete[] _data;
        }
        _data = newData;
        _dataSize = nSize;

        // update tables internal pointers
        for (set<drwnTableFactor *>::iterator it = _tables.begin(); it != _tables.end(); ++it) {
            (*it)->_data = _data;
        }
    }
}

void drwnTableFactorStorage::fill(double v, int nSize)
{
    if (nSize < 0) nSize = _dataSize;
    reserve(nSize);

    double *p = &_data[0];
    for (int i = nSize / 4; i != 0; i--) {
        p[0] = p[1] = p[2] = p[3] = v;
        p += 4;
    }
    for (int i = 0; i < nSize % 4; i++) {
        (*p++) = v;
    }
}

void drwnTableFactorStorage::copy(const double *p, int nSize)
{
    DRWN_ASSERT(p != NULL);
    if (nSize < 0) nSize = _dataSize;
    reserve(nSize);
    memcpy(_data, p, nSize * sizeof(double));
}

void drwnTableFactorStorage::copy(const drwnTableFactorStorage *p, int nSize)
{
    DRWN_ASSERT((p != NULL) && (p->_dataSize >= nSize));
    if (nSize < 0) nSize = _dataSize;
    reserve(nSize);
    memcpy(_data, p->_data, nSize * sizeof(double));
}

void drwnTableFactorStorage::registerFactor(drwnTableFactor *factor)
{
    //DRWN_ASSERT(_bShared || (_tables.empty()));
    _tables.insert(factor);
    factor->_data = _data;
}

void drwnTableFactorStorage::unregisterFactor(drwnTableFactor *factor)
{
    set<drwnTableFactor *>::iterator it = _tables.find(factor);
    DRWN_ASSERT(it != _tables.end());
    _tables.erase(it);
    factor->_data = NULL;
}


