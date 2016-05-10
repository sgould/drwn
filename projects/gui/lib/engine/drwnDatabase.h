/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDatabase.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Defines data storage and management classes. Data is stored in records
**  within tables. Each record contains both forward and backward information
**  if any. Storage (databases and tables) can be in-memory or persistent.
**  Persistent tables are stored in three separate files on disk:
**    1. <tblName>.index has a list of keys and indices into the data file
**    2. <tblName>.data contains the records in binary format
**    3. <tblName>.attributes contains the field/attribute names (if any)
**    4. <tblName>.modified contains last modification date
**
**  Persistent databases are stored in files and directories on disk:
**    1. <dbName>/tables has a list of tables in the database
**    2. <dbName>/data is a directory containing the actual tables
**    3. <dbName>/colours has a list of keys and their colours (fold)
**    4. <dbName>/locked if present the database is in use by another application
**
**  The database manager handles reading and writing of databases (since
**  multiple databases can be active at any one time). The database cache
**  owns all (active) records and handles reading and writing inactive ones
**  to disk (via the corresponding tables). In-memory database records cannot
**  be flushed from the database cache. The only way to remove these records
**  is to close the database (at which point they are deleted and lost).
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <map>
#include <list>
#include <vector>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnDataRecord.h"

using namespace std;

// forward declarations ------------------------------------------------------

class drwnDatabase;

// drwnDataTable -------------------------------------------------------------

class drwnDataTable {
 protected:
    drwnDatabase *_owner;
    string _tableName;                          // table name
    vector<string> _fieldNames;                 // attributes
    set<string> _keys;                          // keys
    string _lastModified;                       // last modification date

    drwnPersistentStorage _storage;             // record storage

 public:
    drwnDataTable(drwnDatabase *o = NULL, const char *name = NULL);
    virtual ~drwnDataTable();

    drwnDatabase *getOwner() const { return _owner; }
    void setOwner(drwnDatabase *o) { flush(); _owner = o; }

    const string& name() const { return _tableName; }
    bool rename(const char *name);

    // i/o
    void clear();
    void read();
    void write();
    void flush();
    void defragment() { if (!_tableName.empty()) _storage.defragment(); }
    int size() const { return _storage.numTotalBytes(); }

    string getLastModified() const { return _lastModified; }
    void setLastModified(const string& s) { _lastModified = s; }

    // record manipulation
    int numRecords() const { return (int)_keys.size(); }
    // TODO: change this to set<string>
    vector<string> getKeys() const;
    bool hasKey(const string& key) const { return (_keys.find(key) != _keys.end()); }
    // TODO: provide record iterators

    void addRecord(const string& key, drwnDataRecord *record); // adds a record and unlocks it (replaces any existing record with the same key)
    drwnDataRecord *lockRecord(const string& key);  // the only way to get access to (or create) a record is to lock it
    void unlockRecord(const string& key);
    void writeRecord(const string& key, drwnDataRecord *record); // commits a record to disk
    drwnDataRecord *readRecord(const string& key); // reads a record from disk
    void deleteRecord(const string& key);
};


// drwnDatabase --------------------------------------------------------------

class drwnDatabase {
 protected:
    static const int DBVERSION;
    string _dbName;
    bool _bLocked;
    map<string, drwnDataTable *> _dataTables;
    map<string, int> _colourTable;

 public:
    drwnDatabase();                 // creates an in-memory database
    drwnDatabase(const char *name); // creates a persistent database
    virtual ~drwnDatabase();

    inline bool isPersistent() const { return !_dbName.empty(); }

    // i/o
    const string& name() const { return _dbName; }
    bool create();                  // creates a new database
    bool read();                    // reads tables and colours
    bool write();                   // writes tables and colours

    // data tables
    void clearTables();
    vector<string> getTableNames() const;
    drwnDataTable *getTable(const string& tblName);
    bool deleteTable(const string& tblName);
    bool renameTable(const string& srcName, const string& dstName);

    // data colours (partitions)
    void clearColours();
    int getColour(const string& key) const;
    void setColour(const string& key, int c);
    void clearColour(const string& key);
    bool matchColour(const string& key, int c) const;

    vector<string> getAllKeys() const;
    vector<string> getColourKeys(int c, bool bSkipWildCard = true) const;

 protected:
    drwnDatabase(const drwnDatabase& db); // databases cannot be copied

    bool lock();
    void unlock();

    void readColourTable();
    void writeColourTable() const;
    void readDataTables();
    void writeDataTables() const;
};

// drwnDataCache -------------------------------------------------------------
// Keeps track of amount of memory used by "unlocked" records (since locked
// record sizes may change). When memory exceeds pre-set limit, records are
// flushed to disk.

class drwnDataCacheEntry;

typedef map<string, drwnDataCacheEntry *> drwnDataCacheKeyMap;

typedef map<drwnDataTable *, drwnDataCacheKeyMap> drwnDataCacheTableMap;

class drwnDataCacheEntry {
 public:
    // data
    drwnDataRecord *record;
    bool bLocked;

    // reverse lookup
    drwnDataCacheKeyMap::iterator keyMapIterator;
    drwnDataCacheTableMap::iterator tblMapIterator;

 public:
    drwnDataCacheEntry(drwnDataRecord *r = NULL) : record(r), bLocked(false) { };
    virtual ~drwnDataCacheEntry() { if (record != NULL) delete record; };
};

class drwnDataCache {
 public:
    static int DEFAULT_SIZE_LIMIT;
    static int DEFAULT_MEMORY_LIMIT;

 protected:
    drwnDataCacheTableMap _lookup;        // fast lookup of (table, key) pairs
    list<drwnDataCacheEntry *> _entries;  // entry storage
    int _entriesUsed;                     // size of _entries (since list::size is O(N))

    int _sizeLimit;                       // maximum number of concurrent entries
    int _memoryLimit;                     // maximum memory allocated to cache
    int _memoryUsed;                      // total memory consumed by records

 public:
    ~drwnDataCache();
    static drwnDataCache& get();          // singleton class (controls initialization)

    int getMemoryUsed() const { return _memoryUsed; }
    int getMemoryLimit() const { return _memoryLimit; }
    void setMemoryLimit(int ml);
    int getSize() const { return _entriesUsed; }
    int getSizeLimit() const { return _sizeLimit; }
    void setSizeLimit(int sl);

    void idleFlush();                     // flushes a single record when idle
    void flush();                         // flush all databases
    void flush(const drwnDatabase *db);   // flush data from a single database
    void flush(const drwnDataTable *tbl); // flush data from a single table
    void clear();                         // clears all data (no saving)
    void clear(const drwnDatabase *db);   // clears data from a database (no saving)
    void clear(const drwnDataTable *tbl); // clears fata from a table (no saving)

    void dump() const;                    // dumps content for debugging

    // access to records must be handled through the data cache
    void insertRecord(drwnDataTable *tbl, const string& key, drwnDataRecord *record);
    drwnDataRecord *lockRecord(drwnDataTable *tbl, const string& key);
    void unlockRecord(drwnDataTable *tbl, const string& key);

    bool isInMemory(drwnDataTable *tbl, const string& key);
    bool isLocked(drwnDataTable *tbl, const string& key);

 protected:
    drwnDataCache(); // singleton class (so hide constructor)

    void pack(); // ensure that memory/size limit has is not exceeded
};

// drwnDbManager -------------------------------------------------------------

class drwnDbManager {
 protected:
    typedef map<string, pair<drwnDatabase *, int> > drwnPersistentDbList;
    drwnPersistentDbList _persistentDatabases;
    set<drwnDatabase *> _inMemoryDatabases;

 public:
    ~drwnDbManager();
    static drwnDbManager& get();

    // open/close persistent and in-memory databases
    // (persistent databases are reference counted)
    drwnDatabase *openDatabase(const char *dbName);
    drwnDatabase *openMemoryDatabase();
    bool closeDatabase(drwnDatabase *db);

 protected:
    drwnDbManager(); // singleton class (so hide constructor)
};
