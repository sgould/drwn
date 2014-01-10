/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTableFactor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <list>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVarUniverse.h"
#include "drwnVarAssignment.h"

using namespace std;
using namespace Eigen;

// drwnFactor --------------------------------------------------------------
//! Generic interface for a factor. Currently only inherited by drwnTableFactor.
//! \todo move into own file.
//! \todo allow for more efficient storage of higher-order factors
//! \todo move to a templated design

class drwnFactor : public drwnStdObjIface {
 public:
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    static unsigned _dbStatsRefCount;
    static unsigned _dbStatsProductCount;
    static unsigned _dbStatsDivideCount;
    static unsigned _dbStatsAdditionCount;
    static unsigned _dbStatsSubtractionCount;
    static unsigned _dbStatsMarginalCount;
    static unsigned _dbStatsMaxMinCount;
    static unsigned _dbStatsReductionCount;
    static unsigned _dbStatsNormalizeCount;
    static unsigned _dbStatsNormalizeErrors;
    static unsigned _dbStatsLargestFactorSize;
    static unsigned _dbStatsCurrentMem;
    static unsigned _dbStatsMaxMem;
    static unsigned _dbStatsMaxTable;
    static unsigned _dbStatsTotalMem;
#endif

 protected:
    drwnVarUniversePtr _pUniverse;  //!< all variables in the universe
    vector<int> _variables;         //!< list of variables in factor (by index in factor)
    map<int, int> _varIndex;        //!< index of variable in factor (by variable id)

 public:
    //! create an empty factor
    drwnFactor(const drwnVarUniversePtr& ptr);
    //! copy constructor
    drwnFactor(const drwnFactor& psi);
    virtual ~drwnFactor();

    // access functions
    virtual drwnFactor *clone() const = 0;

    //! true if the factor contains no variables
    bool empty() const { return _variables.empty(); }
    //! number of variables in this factor
    size_t size() const { return _variables.size(); }
    //! number of bytes comsumed by the factor data (excludes variables lists, etc)
    virtual size_t memory() const { return 0; }

    //! returns the variable universe
    const drwnVarUniversePtr& getUniverse() const { return _pUniverse; }
    //! returns the subset of the universe for this factor
    drwnVarUniverse getSubUniverse() const;
    //! returns true if the variable is included in this factor
    bool hasVariable(int var) const { return (_varIndex.find(var) != _varIndex.end()); }
    //! returns variable at given index
    int varId(int indx) const { return _variables[indx]; }
    //! return cardinality of variable at given index
    int varCard(int indx) const { return _pUniverse->varCardinality(_variables[indx]); }

    //! add variable by id
    virtual void addVariable(int var);
    //! add variable by name
    void addVariable(const char *name);
    //! add multiple variables
    void addVariables(const char *name, ...);
    void addVariables(const vector<int>& c);
    void addVariables(const drwnClique& c);
    void addVariables(const drwnFactor& psi);

    //! returns the ordered set of variables over which this factor is defined
    const vector<int>& getOrderedVars() const { return _variables; }
    //! returns the set of variables over which this factor is defined
    drwnClique getClique() const { return drwnClique(_variables.begin(), _variables.end()); }

    // i/o
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    //! Returns the value of the factor for a given (full) assignment
    virtual double value(const drwnFullAssignment& y) const = 0;
    //! Returns the value of the factor for a given partial assignment.
    //! The variables in the scope of this factor must be defined.
    double value(const drwnPartialAssignment& y) const {
        return value((drwnFullAssignment)y);
    }

 protected:
#ifdef DRWN_FACTOR_DEBUG_STATISTICS
    static void updateMemoryStats(int nSize, int previousSize = 0);
#endif
};

// drwnTableFactor --------------------------------------------------------

class drwnTableFactorStorage;

//! Factor which stores the value of each assignment explicitly in table
//! form.
//!
//! Storage can be shared by using a drwnTableFactorStorage object. Operations
//! on table factors should use the drwnFactorOperations objects.
//! \todo drwnSparseTableFactor class and drwnPottsFactor class

class drwnTableFactor : public drwnFactor {
 protected:
    friend class drwnTableFactorStorage;

    std::vector<int> _stride;           //!< stride of variable in table (by index)

    int _nSize;                         //!< total size of factor (or 1 for singular factors)
    double *_data;                      //!< data is stored as (a0, b0, ...), (a1, b0, ...), ...
    drwnTableFactorStorage *_storage;   //!< pointer to shared storage (updates _data member)

 public:
    drwnTableFactor(const drwnVarUniversePtr& ptr, drwnTableFactorStorage *storage = NULL);
    drwnTableFactor(const drwnTableFactor& psi);
    virtual ~drwnTableFactor();

    // access functions
    const char *type() const { return "drwnTableFactor"; }
    drwnTableFactor *clone() const { return new drwnTableFactor(*this); }

    size_t memory() const { return _nSize * sizeof(double); }
    size_t entries() const { return _nSize; }
    bool isShared() const { return (_storage != NULL); }

    // add variables
    using drwnFactor::addVariable;
    void addVariable(int var);

    //! Returns the index into the data table with variable \p var taking
    //! assignment \p val. Can be relative to another index (calculated
    //! for a different variable-value pair).
    int indexOf(int var, int val, int indx = 0) const;
    //! Returns the index into the data table for a given full assignment.
    int indexOf(const drwnFullAssignment& assignment) const;
    //! Returns the index into the data table for a given partial assignment.
    //! All variables in the scope of the factor must be defined.
    int indexOf(const drwnPartialAssignment& assignment) const;
    //! Returns the value of variable \p var for the given index.
    int valueOf(int var, int indx) const;
    //! Returns the assignment for all variables in the scope of the
    //! factor for a given index into the data table.
    void assignmentOf(int indx, drwnPartialAssignment& assignment) const;
    //! Returns the assignment for all variables in the scope of the
    //! factor for a given index into the data table.
    void assignmentOf(int indx, drwnFullAssignment& assignment) const;

    int indexOfMin() const;
    int indexOfMax() const;
    pair<int, int> indexOfMinAndMax() const;

    // inline modifications
    virtual drwnTableFactor& initialize();
    drwnTableFactor& fill(double alpha);
    drwnTableFactor& copy(const double *data);
    drwnTableFactor& scale(double alpha);
    drwnTableFactor& offset(double alpha);
    virtual drwnTableFactor& normalize();

    // i/o
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    double value(const drwnFullAssignment& y) const {
        return _data[indexOf(y)];
    }

    // epsilon-tolerant data comparison
    bool dataCompare(const drwnTableFactor& psi) const;
    bool dataCompareAndCopy(const drwnTableFactor& psi);

    // operators
    inline double& operator[](unsigned index) { return _data[index]; }
    inline const double& operator[](unsigned index) const { return _data[index]; }

    drwnTableFactor& operator=(const drwnTableFactor& psi);
};

// drwnTableFactorStorage class -------------------------------------------
//! Shared memory for table factors.
//!
//! \warning Shared storage cannot be deleted until all factors using the
//! storage have been deleted (i.e., unregistered). This means that if
//! factors in a graph are using shared storage then the graph must be
//! destroyed before the storage can be deleted.

class drwnTableFactorStorage {
    friend class drwnTableFactor;

 protected:
    int _dataSize;                   //!< amount of memory allocated
    double *_data;                   //!< memory allocation

    set<drwnTableFactor *> _tables;  //!< tables using this storage

 public:
    drwnTableFactorStorage(int nSize = 0);
    ~drwnTableFactorStorage();

    //! return the total capacity of the storage
    inline int capacity() const { return _dataSize; }
    //! reserve capacity for \p nSize entries
    void reserve(int nSize);

    //! set \p nSize (or all) entries to zero
    inline void zero(int nSize = -1) { fill(0.0, nSize); }
    //! set \p nSize (or all) entries to the value \p v
    void fill(double v, int nSize = -1);
    //! copy \p nSize (or all) entries from array \p p
    void copy(const double *p, int nSize = -1);
    //! copy \p nSize (or all) entries from storage object \p p
    void copy(const drwnTableFactorStorage *p, int nSize = -1);

    // data access
    inline double& operator[](unsigned index) { return _data[index]; }
    inline const double& operator[](unsigned index) const { return _data[index]; }

 protected:
    //! registers the factor with the storage so that data pointer 
    //! can be updated when storage location changes
    void registerFactor(drwnTableFactor *factor);
    //! unregisters a factor with the storage
    void unregisterFactor(drwnTableFactor *factor);
};
