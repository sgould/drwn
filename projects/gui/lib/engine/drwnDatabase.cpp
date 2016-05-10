/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDatabase.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <map>
#include <vector>
#include <iomanip>

#include "drwnBase.h"
#include "drwnDatabase.h"

using namespace std;

// drwnDataTable -------------------------------------------------------------

drwnDataTable::drwnDataTable(drwnDatabase *o, const char *name) :
    _owner(o) {
    if (name != NULL) _tableName = string(name);
    DRWN_ASSERT(_tableName.size() > 0);
}

drwnDataTable::~drwnDataTable()
{
    // do nothing
}

bool drwnDataTable::rename(const char *name)
{
    DRWN_ASSERT(name != NULL);
    DRWN_LOG_DEBUG("renaming table " << _tableName << " to " << name << "...");

    // nothing to do for in-memory tables
    if (!_owner->isPersistent()) {
        _tableName = string(name);
        return true;
    }

    // persistent storage
    if (_storage.isOpen()) {
        _storage.close();
    }

    // old filenames
    string dir = _owner->name() + DRWN_DIRSEP + string("data") + DRWN_DIRSEP;
    DRWN_ASSERT(drwnDirExists(dir.c_str()));

    string oldIndxName = dir + _tableName + string(".index");
    string oldDataName = dir + _tableName + string(".data");
    string oldAttrName = dir + _tableName + string(".attributes");

    _tableName = string(name);

    // new filenames
    string indxName = dir + _tableName + string(".index");
    string dataName = dir + _tableName + string(".data");
    string attrName = dir + _tableName + string(".attributes");

    // TODO: check existing filename doesn't already exist
    if (drwnFileExists(indxName.c_str()) || drwnFileExists(dataName.c_str())) {
        DRWN_LOG_WARNING("table with name \"" << _tableName << "\" already exists in database");
    }

    // rename files
    ::rename(oldIndxName.c_str(), indxName.c_str());
    ::rename(oldDataName.c_str(), dataName.c_str());
    //::rename(oldAttrName.c_str(), attrName.c_str());

    // open storage with new name
    _storage.open(indxName.c_str(), dataName.c_str());

    return true;
}

void drwnDataTable::clear()
{
    // clear storage and in-memory cache
    _keys.clear();
    _storage.clear();
    drwnDataCache::get().clear(this);
}

void drwnDataTable::read()
{
    DRWN_ASSERT(_owner != NULL);
    DRWN_LOG_DEBUG("reading table " << _tableName << "...");

    // nothing to read for in-memory tables
    if (!_owner->isPersistent()) {
        return;
    }

    // filenames
    string dir = _owner->name() + DRWN_DIRSEP + string("data") + DRWN_DIRSEP;
    DRWN_ASSERT(drwnDirExists(dir.c_str()));

    string indxName = dir + _tableName + string(".index");
    string dataName = dir + _tableName + string(".data");
    string attrName = dir + _tableName + string(".attributes");

    // read keys
    if (!_storage.isOpen()) {
        if (_storage.canReopen()) {
            _storage.reopen();
        } else {
            _storage.open(indxName.c_str(), dataName.c_str());
        }
    }
    _keys = _storage.getKeys();

    DRWN_LOG_DEBUG("...table has " << _keys.size() << " records");

    // read attributes
    //DRWN_NOT_IMPLEMENTED_YET;
}

void drwnDataTable::write()
{
    DRWN_ASSERT(_owner != NULL);

    // nothing to write for in-memory tables
    if (!_owner->isPersistent()) {
        return;
    }

    // filenames
    string dir = _owner->name() + DRWN_DIRSEP + string("data") + DRWN_DIRSEP;
    DRWN_ASSERT(drwnDirExists(dir.c_str()));

    string indxName = dir + _tableName + string(".index");
    string dataName = dir + _tableName + string(".data");
    string attrName = dir + _tableName + string(".attributes");

    // TODO: write last modified date

    // write storage
    if (_storage.isOpen()) {
        _storage.close();
    } else if (!_storage.canReopen()) {
        _storage.open(indxName.c_str(), dataName.c_str());
        _storage.close();
    }

    // write attributes
    //DRWN_NOT_IMPLEMENTED_YET;

}

void drwnDataTable::flush()
{
    drwnDataCache::get().flush(this);
}

vector<string> drwnDataTable::getKeys() const
{
    vector<string> keys;
    keys.reserve(_keys.size());
    for (set<string>::const_iterator it = _keys.begin(); it != _keys.end(); it++) {
        keys.push_back(*it);
    }

    return keys;
}

void drwnDataTable::addRecord(const string& key, drwnDataRecord *record)
{
    DRWN_ASSERT(record != NULL);
    if (!hasKey(key)) {
        _keys.insert(key);
    }
    drwnDataCache::get().insertRecord(this, key, record);
}

drwnDataRecord *drwnDataTable::lockRecord(const string& key)
{
    // TODO: change this around---check cache for record, if not
    // present load and add to cache. Then get rid of readRecord
    // function.
    if (!hasKey(key)) {
        _keys.insert(key);
    }
    return drwnDataCache::get().lockRecord(this, key);
}

void drwnDataTable::unlockRecord(const string& key)
{
    //DRWN_NOT_IMPLEMENTED_YET;
    return drwnDataCache::get().unlockRecord(this, key);
}

void drwnDataTable::writeRecord(const string& key, drwnDataRecord *record)
{
    DRWN_ASSERT_MSG(hasKey(key), "table: " << _tableName << ", key: " << key);

    // delete from table if NULL
    if (record == NULL) {
        if (this->hasKey(key)) {
            deleteRecord(key);
        }
        return;
    }

    // nothing to write for in-memory tables
    if (!_owner->isPersistent()) {
        return;
    }

    string dir = _owner->name() + DRWN_DIRSEP + string("data") + DRWN_DIRSEP;
    string indxName = dir + _tableName + string(".index");
    string dataName = dir + _tableName + string(".data");
    if (!_storage.isOpen()) {
        if (_storage.canReopen()) {
            _storage.reopen();
        } else {
            _storage.open(indxName.c_str(), dataName.c_str());
        }
    }
    _storage.write(key.c_str(), record);
}

drwnDataRecord * drwnDataTable::readRecord(const string& key)
{
    DRWN_ASSERT_MSG(hasKey(key), "table: " << _tableName << ", key: " << key);

    // always return new record for in-memory tables
    if (!_owner->isPersistent()) {
        return new drwnDataRecord(this);
    }

    string dir = _owner->name() + DRWN_DIRSEP + string("data") + DRWN_DIRSEP;
    string indxName = dir + _tableName + string(".index");
    string dataName = dir + _tableName + string(".data");
    if (!_storage.isOpen()) {
        if (_storage.canReopen()) {
            _storage.reopen();
        } else {
            _storage.open(indxName.c_str(), dataName.c_str());
        }
    }
    drwnDataRecord *record = new drwnDataRecord(this);
    if (_storage.hasKey(key.c_str())) {
        bool success = _storage.read(key.c_str(), record);
        DRWN_ASSERT(success);
    }

    return record;
}

void drwnDataTable::deleteRecord(const string& key)
{
    set<string>::iterator it = _keys.find(key);
    DRWN_ASSERT_MSG(it != _keys.end(), "table: " << _tableName << ", key: " << key);

    if (_owner->isPersistent()) {
        _storage.erase(key.c_str());
    }
    _keys.erase(it);
}


// drwnDatabase --------------------------------------------------------------

const int drwnDatabase::DBVERSION = 100;

drwnDatabase::drwnDatabase() : _bLocked(false)
{
    // do nothing (in-memory database)
}

drwnDatabase::drwnDatabase(const char *name) : _dbName(name), _bLocked(false)
{
    // do nothing (persistent database)
}

drwnDatabase::drwnDatabase(const drwnDatabase& db)
{
    DRWN_LOG_FATAL("drwnDatabase objects cannot be copied");
}

drwnDatabase::~drwnDatabase()
{
    for (map<string, drwnDataTable *>::iterator it = _dataTables.begin();
         it != _dataTables.end(); it++) {
        delete it->second;
    }

    if (isPersistent()) {
        unlock();
    }
}

// i/o
bool drwnDatabase::create()
{
    // in-memory database are created by default
    if (!isPersistent()) return true;

    // check that the database does not already exist
    if (drwnPathExists(_dbName.c_str()) && (drwnDirSize(_dbName.c_str()) != 0)) {
        DRWN_LOG_ERROR("database " << _dbName << " already exists");
        return false;
    }

    // create and lock the database directory
    if (!drwnDirExists(_dbName.c_str()) && !drwnCreateDirectory(_dbName.c_str())) {
        DRWN_LOG_ERROR("failed to create database " << _dbName);
        return false;
    }
    lock();

    // create directories and files
    drwnCreateDirectory((_dbName + DRWN_DIRSEP + "data").c_str());
    write(); // creates directories

    return true;
}

bool drwnDatabase::read()
{
    // nothing to read from in-memory
    if (!isPersistent()) return true;

    DRWN_LOG_DEBUG("reading database " << _dbName << "...");

    // check database directory exists
    if (!drwnDirExists(_dbName.c_str()) ||
        !drwnFileExists((_dbName + DRWN_DIRSEP + "tables").c_str())) {
        DRWN_LOG_MESSAGE("database " << _dbName << " does not exist");
        return false;
    }

    // acquire lock on database
    if (!lock()) {
        DRWN_LOG_MESSAGE("database " << _dbName << " locked by another application");
        return false;
    }

    readColourTable();
    readDataTables();

    DRWN_LOG_DEBUG("...database has " << _dataTables.size() << " tables");
    return true;
}

bool drwnDatabase::write()
{
    // in-memory databases cannot be written
    if (!isPersistent()) return false;

    DRWN_LOG_DEBUG("writing database " << _dbName << "...");
    writeColourTable();
    writeDataTables();
    return true;
}

// data tables
void drwnDatabase::clearTables()
{
    for (map<string, drwnDataTable *>::iterator it = _dataTables.begin();
         it != _dataTables.end(); it++) {
        it->second->clear();
    }
}

vector<string> drwnDatabase::getTableNames() const
{
    vector<string> tblNames;
    tblNames.reserve(_dataTables.size());
    for (map<string, drwnDataTable *>::const_iterator it = _dataTables.begin();
         it != _dataTables.end(); it++) {
        tblNames.push_back(it->first);
    }

    return tblNames;
}

drwnDataTable *drwnDatabase::getTable(const string& tblName)
{
    map<string, drwnDataTable *>::iterator it = _dataTables.find(tblName);
    if (it == _dataTables.end()) {
        DRWN_LOG_VERBOSE("creating table " << tblName << " in database " << _dbName);
        drwnDataTable *tbl = new drwnDataTable(this, tblName.c_str());
        it = _dataTables.insert(it, make_pair(tblName, tbl));
    }

    return it->second;
}

bool drwnDatabase::deleteTable(const string& tblName)
{
    map<string, drwnDataTable *>::iterator it = _dataTables.find(tblName);
    if (it != _dataTables.end()) {
        drwnDataCache::get().clear(it->second);
        delete it->second;
        _dataTables.erase(it);
        return true;
    }

    if (isPersistent()) {
        DRWN_LOG_WARNING("deleting non-existent table " << tblName << " from database " << _dbName);
    } else {
        DRWN_LOG_WARNING("deleting non-existent table " << tblName << " from in-memory database");
    }

    return false;
}

bool drwnDatabase::renameTable(const string& srcName, const string& dstName)
{
    map<string, drwnDataTable *>::iterator it = _dataTables.find(dstName);
    if (it != _dataTables.end()) {
        DRWN_LOG_ERROR("table \"" << dstName << "\" already exists in database");
        return false;
    }

    it = _dataTables.find(srcName);
    if (it == _dataTables.end()) {
        DRWN_LOG_FATAL("table \"" << srcName << "\" does not exist in database");
    }

    it->second->rename(dstName.c_str());
    _dataTables.insert(make_pair(dstName, it->second));
    _dataTables.erase(it);

    return true;
}

// data colours (partitions)
void drwnDatabase::clearColours()
{
    _colourTable.clear();
}

int drwnDatabase::getColour(const string& key) const
{
    map<string, int>::const_iterator it = _colourTable.find(key);
    return (it == _colourTable.end()) ? -1 : it->second;
}

void drwnDatabase::setColour(const string& key, int c)
{
    _colourTable[key] = c;
}

void drwnDatabase::clearColour(const string& key)
{
    map<string, int>::iterator it = _colourTable.find(key);
    if (it != _colourTable.end()) {
        _colourTable.erase(it);
    }
}

bool drwnDatabase::matchColour(const string& key, int c) const
{
    if (c < 0) return true;
    map<string, int>::const_iterator it = _colourTable.find(key);
    return ((it == _colourTable.end()) || (it->second < 0) || (it->second == c));
}

vector<string> drwnDatabase::getAllKeys() const
{
    set<string> allKeys;
    for (map<string, drwnDataTable *>::const_iterator it = _dataTables.begin();
         it != _dataTables.end(); it++) {
        vector<string> tblKeys = it->second->getKeys();
        allKeys.insert(tblKeys.begin(), tblKeys.end());
    }

    return vector<string>(allKeys.begin(), allKeys.end());
}

vector<string> drwnDatabase::getColourKeys(int c, bool bSkipWildCard) const
{
    vector<string> matchingKeys;
    for (map<string, int>::const_iterator it = _colourTable.begin();
         it != _colourTable.end(); it++) {
        if ((it->second == c) || (!bSkipWildCard && (it->second < 0))) {
            matchingKeys.push_back(it->first);
        }
    }

    return matchingKeys;
}

bool drwnDatabase::lock()
{
    if (_bLocked) return true;

    string lockFile(_dbName + DRWN_DIRSEP + string("locked"));
    if (drwnFileExists(lockFile.c_str())) {
        return false;
    }

    ofstream ofs(lockFile.c_str());

    time_t rawtime;
    time(&rawtime);
    ofs << ctime(&rawtime) << "\n";
    ofs.close();

    _bLocked = true;
    return true;
}

void drwnDatabase::unlock()
{
    // check if database has been locked by this object
    if (!_bLocked) return;

    // unlock database
    string lockFile(_dbName + DRWN_DIRSEP + string("locked"));
    if (!drwnFileExists(lockFile.c_str())) {
        DRWN_LOG_ERROR("database lock broken for " << _dbName << "; data may be corrupt");
    }
    remove(lockFile.c_str());
    DRWN_ASSERT(!drwnFileExists(lockFile.c_str()));
    _bLocked = false;
}

void drwnDatabase::readColourTable()
{
    _colourTable.clear();

    string colourTableFilename(_dbName + DRWN_DIRSEP + "colours");
    ifstream ifs(colourTableFilename.c_str());

    if (ifs.fail()) {
        DRWN_LOG_WARNING("no colour table for database " << _dbName);
        ifs.close();
        return;
    }

    int colour;
    string key;
    while (!ifs.fail()) {
        ifs >> colour >> key;
        if (ifs.fail()) break;
        pair<map<string, int>::iterator, bool> success =
            _colourTable.insert(make_pair(key, colour));
        if (!success.second) {
            DRWN_LOG_ERROR("duplicate keys in colour table for database " << _dbName);
        }
    }

    ifs.close();
}

void drwnDatabase::writeColourTable() const
{
    string colourTableFilename(_dbName + DRWN_DIRSEP + "colours");
    ofstream ofs(colourTableFilename.c_str());
    DRWN_ASSERT(!ofs.fail());

    for (map<string, int>::const_iterator it = _colourTable.begin();
         it != _colourTable.end(); it++) {
        ofs << it->second << " " << it->first << "\n";
    }

    DRWN_ASSERT(!ofs.fail());
    ofs.close();
}

void drwnDatabase::readDataTables()
{
    DRWN_ASSERT(_dataTables.empty()); // TODO: or delete them?

    string tablesFilename(_dbName + DRWN_DIRSEP + "tables");
    ifstream ifs(tablesFilename.c_str());

    if (ifs.fail()) {
        DRWN_LOG_WARNING("no data tables for database " << _dbName);
        ifs.close();
        return;
    }

    // open tables
    string tableName;
    while (1) {
        getline(ifs, tableName);
        if (ifs.fail()) break;
        if (tableName.empty()) continue;

        drwnDataTable *table = new drwnDataTable(this, tableName.c_str());
        _dataTables.insert(make_pair(tableName, table));
        table->read();
    }

    ifs.close();
}

void drwnDatabase::writeDataTables() const
{
    string tablesFilename(_dbName + DRWN_DIRSEP + "tables");
    ofstream ofs(tablesFilename.c_str());
    DRWN_ASSERT(!ofs.fail());

    for (map<string, drwnDataTable *>::const_iterator it = _dataTables.begin();
         it != _dataTables.end(); it++) {
        ofs << it->first << "\n";
    }

    DRWN_ASSERT(!ofs.fail());
    ofs.close();
}

// drwnDataCache -------------------------------------------------------------

//int drwnDataCache::DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024; // 512 MB
int drwnDataCache::DEFAULT_MEMORY_LIMIT = 1536 * 1024 * 1024; // 1.5 GB
int drwnDataCache::DEFAULT_SIZE_LIMIT = 65536; // 64k

drwnDataCache::drwnDataCache() : _entriesUsed(0), _sizeLimit(DEFAULT_SIZE_LIMIT),
    _memoryLimit(DEFAULT_MEMORY_LIMIT), _memoryUsed(0)
{
    // do nothing
}

drwnDataCache::~drwnDataCache()
{
    // write all unsaved data
    flush();
    for (list<drwnDataCacheEntry *>::iterator it = _entries.begin();
         it != _entries.end(); it++) {
        delete *it;
    }
}

drwnDataCache& drwnDataCache::get()
{
    static drwnDataCache cache;
    return cache;
}

void drwnDataCache::idleFlush()
{
    list<drwnDataCacheEntry *>::iterator it = _entries.begin();
    while (it != _entries.end()) {
        if (!(*it)->bLocked && (*it)->record->isDirty()) {
            break;
        }
        it++;
    }

    if (it != _entries.end()) {
        string key = (*it)->keyMapIterator->first;
        drwnDataTable *table = (*it)->tblMapIterator->first;
        table->writeRecord(key, (*it)->record);
    }
}

void drwnDataCache::flush()
{
    for (list<drwnDataCacheEntry *>::iterator it = _entries.begin();
         it != _entries.end(); it++) {
        const string& key = (*it)->keyMapIterator->first;
        if ((*it)->bLocked) {
            DRWN_LOG_VERBOSE("can't flush locked record " << key);
            continue;
        }
        drwnDataTable *table = (*it)->tblMapIterator->first;
        DRWN_ASSERT(table != NULL);
        drwnDataRecord *record = (*it)->record;
        if (record->isDirty()) {
            table->writeRecord(key, record);
        }
    }
}

void drwnDataCache::flush(const drwnDatabase *db)
{
    DRWN_ASSERT(db != NULL);

    // prepare list of tables that belong to this database
    list<drwnDataTable *> tableList;
    for (drwnDataCacheTableMap::const_iterator it = _lookup.begin(); it != _lookup.end(); it++) {
        if (it->first->getOwner() == db) {
            tableList.push_back(it->first);
        }
    }

    // flush the tables
    for (list<drwnDataTable *>::const_iterator it = tableList.begin(); it != tableList.end(); it++) {
        flush(*it);
    }
}

void drwnDataCache::flush(const drwnDataTable *tbl)
{
    DRWN_ASSERT(tbl != NULL);
    drwnDataCacheTableMap::const_iterator it = _lookup.find((drwnDataTable *)tbl);
    if (it != _lookup.end()) {
        for (drwnDataCacheKeyMap::const_iterator jt = it->second.begin(); jt != it->second.end(); jt++) {
            const string& key = jt->first;
            if (jt->second->bLocked) {
                DRWN_LOG_VERBOSE("can't flush locked record " << key);
                continue;
            }
            drwnDataRecord *record = jt->second->record;
            if (record->isDirty()) {
                it->first->writeRecord(key, record);
            }
        }
    }
}

void drwnDataCache::clear()
{
    _lookup.clear();
    int nDirtyRecords = 0;
    int nLockedRecords = 0;
    for (list<drwnDataCacheEntry *>::iterator it = _entries.begin(); it != _entries.end(); it++) {
        nDirtyRecords += (*it)->record->isDirty() ? 1 : 0;
        nLockedRecords += (*it)->bLocked ? 1 : 0;
        delete (*it);
    }
    _entries.clear();
    _entriesUsed = 0;
    _memoryUsed = 0;

    if (nDirtyRecords > 0) {
        DRWN_LOG_WARNING(nDirtyRecords << " records lost when clearing the data cache");
    }
    if (nLockedRecords > 0) {
        DRWN_LOG_ERROR(nLockedRecords << " locked records lost when clearing the data cache");
    }
}

void drwnDataCache::clear(const drwnDatabase *db)
{
    DRWN_ASSERT(db != NULL);

    // prepare list of tables that belong to this database
    list<drwnDataTable *> tableList;
    for (drwnDataCacheTableMap::const_iterator it = _lookup.begin(); it != _lookup.end(); it++) {
        if (it->first->getOwner() == db) {
            tableList.push_back(it->first);
        }
    }

    // clear the tables
    for (list<drwnDataTable *>::const_iterator it = tableList.begin(); it != tableList.end(); it++) {
        clear(*it);
    }
}

void drwnDataCache::clear(const drwnDataTable *tbl)
{
    DRWN_ASSERT(tbl != NULL);

    drwnDataCacheTableMap::iterator it = _lookup.find((drwnDataTable *)tbl);
    if (it != _lookup.end()) {
        // delete all records and remove mapping
        for (drwnDataCacheKeyMap::iterator jt = it->second.begin(); jt != it->second.end(); jt++) {
            DRWN_ASSERT(!jt->second->bLocked);
            _memoryUsed -= jt->second->record->numBytes();
            // TODO: replace drwnDataCacheKeyMap with an iterator so that the
            // following can be done without a find
            _entries.erase(find(_entries.begin(), _entries.end(), jt->second));
            _entriesUsed -= 1;
            delete jt->second;
        }
        _lookup.erase(it);
    }

    //DRWN_LOG_DEBUG("cache using " << _memoryUsed << " bytes after clearing table " << tbl->name());
}

void drwnDataCache::dump() const
{
    for (list<drwnDataCacheEntry *>::const_iterator it = _entries.begin(); it != _entries.end(); it++) {
        cout << " " << ((*it)->bLocked ? "X" : "_")
             << ((*it)->record->isDirty() ? "D" : "_")
             << ((*it)->tblMapIterator->first->getOwner()->isPersistent() ? "P" : "M")
             << " " << setw(7) << drwn::bytesToString((*it)->record->numBytes())
             << "  " << (*it)->tblMapIterator->first->name()
             << "::" << (*it)->keyMapIterator->first << "\n";
    }
}

// access to records must be handled through the data cache
void drwnDataCache::insertRecord(drwnDataTable *tbl, const string& key, drwnDataRecord *record)
{
    DRWN_ASSERT((tbl != NULL) && (record != NULL));

    // first check if record already exists in memory
    drwnDataCacheTableMap::iterator it = _lookup.find(tbl);
    if (it == _lookup.end()) {
        // table does not exist in the cache so add it
        it = _lookup.insert(it, make_pair(tbl, drwnDataCacheKeyMap()));
    }

    drwnDataCacheKeyMap::iterator jt = it->second.find(key);
    if (jt != it->second.end()) {
        // update existing cache entry
        DRWN_ASSERT(!jt->second->bLocked);
        _memoryUsed -= record->numBytes();
        delete jt->second->record;
        jt->second->record = record;
    } else {
        // add new cache entry
        drwnDataCacheEntry *e = new drwnDataCacheEntry(record);
        e->keyMapIterator = it->second.insert(it->second.end(), make_pair(key, e));
        e->tblMapIterator = it;
        _entries.push_back(e);
        _entriesUsed += 1;
    }

    _memoryUsed += record->numBytes();
    //DRWN_LOG_DEBUG("cache using " << _memoryUsed << " bytes after inserting record \"" << key << "\"");

    // flush old records if memory/space is exceeded
    pack();
}

drwnDataRecord *drwnDataCache::lockRecord(drwnDataTable *tbl, const string& key)
{
    DRWN_ASSERT(tbl != NULL);

    // first check if record already exists in memory
    drwnDataCacheTableMap::iterator it = _lookup.find(tbl);
    if (it == _lookup.end()) {
        // table does not exist in the cache so add it
        it = _lookup.insert(it, make_pair(tbl, drwnDataCacheKeyMap()));
    }

    drwnDataCacheEntry *cacheEntry;
    drwnDataCacheKeyMap::iterator jt = it->second.find(key);
    if (jt != it->second.end()) {
        // lock existing cache entry
        // TODO: move to back of list (least recently used cache rule?)
        cacheEntry = jt->second;
        if (!cacheEntry->bLocked) {
            // subtract memory used from cache memory used (to be added back when unlocked)
            _memoryUsed -= cacheEntry->record->numBytes();
        }
    } else {
        // load cache entry from table (don't update memory; this will be added when unlocked)
        drwnDataRecord *record = tbl->readRecord(key);

        cacheEntry = new drwnDataCacheEntry(record);
        cacheEntry->keyMapIterator = it->second.insert(it->second.end(), make_pair(key, cacheEntry));
        cacheEntry->tblMapIterator = it;
        _entries.push_back(cacheEntry);
        _entriesUsed += 1;
    }

    // lock the record
    cacheEntry->bLocked = true;

    // flush old records if memory/space is exceeded and new record added
    if (jt == it->second.end()) {
        pack();
    }

    return cacheEntry->record;
}

void drwnDataCache::unlockRecord(drwnDataTable *tbl, const string& key)
{
    DRWN_ASSERT(tbl != NULL);

    // check that record exists and is indeed locked
    drwnDataCacheTableMap::iterator it = _lookup.find(tbl);
    DRWN_ASSERT(it != _lookup.end());
    drwnDataCacheKeyMap::iterator jt = it->second.find(key);
    DRWN_ASSERT(jt != it->second.end());

    DRWN_ASSERT(jt->second->bLocked);

    // unlock and update memory used
    jt->second->bLocked = false;
    _memoryUsed += jt->second->record->numBytes();
    //DRWN_LOG_DEBUG("cache using " << _memoryUsed << " bytes after unlocking record \"" << key << "\"");

    // flush old records if memory/space is exceeded and record changed
    if (jt->second->record->isDirty()) {
        pack();
    }
}

bool drwnDataCache::isInMemory(drwnDataTable *tbl, const string& key)
{
    DRWN_ASSERT(tbl != NULL);
    drwnDataCacheTableMap::const_iterator it = _lookup.find(tbl);
    if (it == _lookup.end()) return false;

    return (it->second.find(key) != it->second.end());
}

bool drwnDataCache::isLocked(drwnDataTable *tbl, const string& key)
{
    DRWN_ASSERT(tbl != NULL);
    drwnDataCacheTableMap::const_iterator it = _lookup.find(tbl);
    if (it == _lookup.end()) return false;

    drwnDataCacheKeyMap::const_iterator jt = it->second.find(key);
    if (jt == it->second.end()) return false;

    return jt->second->bLocked;
}

void drwnDataCache::pack()
{
    // remove earliest records from cache until memory/size limit satisfied
    list<drwnDataCacheEntry *>::iterator it = _entries.begin();

    while ((_entriesUsed > _sizeLimit) || (_memoryUsed > _memoryLimit)) {
        if (it == _entries.end()) {
            DRWN_LOG_WARNING("data cache exceeds memory/size limits, but cannot remove records");
            break;
        }

        bool bInMemoryRecord = !(*it)->tblMapIterator->first->getOwner()->isPersistent();
        if ((*it)->bLocked || bInMemoryRecord) {
            it++;
            continue;
        }

        drwnDataRecord *record = (*it)->record;
        if (record->isDirty()) {
            string key = (*it)->keyMapIterator->first;
            drwnDataTable *table = (*it)->tblMapIterator->first;
            DRWN_ASSERT(table != NULL);
            table->writeRecord(key, record);
        }

        // erase record
        _memoryUsed -= record->numBytes();
        (*it)->tblMapIterator->second.erase((*it)->keyMapIterator);
        if ((*it)->tblMapIterator->second.empty()) {
            _lookup.erase((*it)->tblMapIterator);
        }
        delete *it;
        list<drwnDataCacheEntry *>::iterator jt = it++;
        _entries.erase(jt);
        _entriesUsed -= 1;
    }

    DRWN_ASSERT((_entriesUsed != 0) || (_entries.empty()));
}

// drwnDbManager -------------------------------------------------------------

drwnDbManager::drwnDbManager()
{
    // do nothing
}

drwnDbManager::~drwnDbManager()
{
    // write and delete persistent databases
    for (drwnPersistentDbList::iterator it = _persistentDatabases.begin();
         it != _persistentDatabases.end(); it++) {
        if (it->second.second > 0) {
            it->second.first->write();
        }
        delete it->second.first;
    }

    // delete in-memory databases
    for (set<drwnDatabase *>::iterator it = _inMemoryDatabases.begin();
         it != _inMemoryDatabases.end(); it++) {
        delete (*it);
    }
}

drwnDbManager& drwnDbManager::get()
{
    static drwnDbManager manager;
    return manager;
}

drwnDatabase *drwnDbManager::openDatabase(const char *dbName)
{
    DRWN_ASSERT(dbName != NULL);

    drwnPersistentDbList::iterator it = _persistentDatabases.find(string(dbName));
    if (it == _persistentDatabases.end()) {
        drwnDatabase *db = new drwnDatabase(dbName);

        bool success = false;
        if (drwnDirExists(dbName) && (drwnDirSize(dbName) > 0)) {
            DRWN_LOG_DEBUG("reading existing database " << dbName);
            success = db->read();
        } else {
            DRWN_LOG_DEBUG("creating new database " << dbName);
            success = db->create();
        }

        if (!success) {
            DRWN_LOG_ERROR("failed to open or create database " << dbName);
            delete db;
            return NULL;
        }

        it = _persistentDatabases.insert(it, make_pair(string(dbName), make_pair(db, 0)));
    }
    it->second.second += 1;

    DRWN_LOG_VERBOSE("Opening database " << dbName
        << " (ref. count: " << it->second.second << ")");

    return it->second.first;
}

drwnDatabase *drwnDbManager::openMemoryDatabase()
{
    drwnDatabase *db = new drwnDatabase();
    _inMemoryDatabases.insert(db);

    DRWN_LOG_VERBOSE("Opening in-memory database");
    return db;
}

bool drwnDbManager::closeDatabase(drwnDatabase *db)
{
    DRWN_ASSERT(db != NULL);

    // try in-memory databases first
    set<drwnDatabase *>::iterator jt = _inMemoryDatabases.find(db);
    if (jt != _inMemoryDatabases.end()) {
        drwnDataCache::get().clear(*jt);
        delete *jt;
        _inMemoryDatabases.erase(jt);
        return true;
    }

    // must be a persistent database
    drwnPersistentDbList::iterator it = _persistentDatabases.find(db->name());
    DRWN_ASSERT(it != _persistentDatabases.end());
    DRWN_ASSERT(it->second.second > 0);

    // decrement reference counter
    it->second.second -= 1;
    DRWN_LOG_VERBOSE("Closing database " << db->name()
        << " (ref. count: " << it->second.second << ")");

    // write database (can't delete yet since records may still be in the cache)
    if (it->second.second == 0) {
        it->second.first->write();
    }

    return true;
}

// drwnDataCacheConfig ------------------------------------------------------

class drwnDataCacheConfig : public drwnConfigurableModule {
public:
    drwnDataCacheConfig() : drwnConfigurableModule("drwnDataCache") { }
    ~drwnDataCacheConfig() { }

    void usage(ostream &os) const {
        os << "      maxSize       :: maximum number of records to keep in memory (default: "
           << drwnDataCache::DEFAULT_SIZE_LIMIT << ")\n";
        os << "      maxMemory     :: maximum size (in bytes) of records to keep in memory  (default: "
           << drwnDataCache::DEFAULT_MEMORY_LIMIT << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxSize")) {
            drwnDataCache::DEFAULT_SIZE_LIMIT = atoi(value);
            if (drwnDataCache::DEFAULT_SIZE_LIMIT < 0) {
                drwnDataCache::DEFAULT_SIZE_LIMIT = DRWN_INT_MAX;
            }
        } else if (!strcmp(name, "maxMemory")) {
            drwnDataCache::DEFAULT_MEMORY_LIMIT = atoi(value);
            if (drwnDataCache::DEFAULT_MEMORY_LIMIT < 0) {
                drwnDataCache::DEFAULT_MEMORY_LIMIT = DRWN_INT_MAX;
            }
        } else {
            DRWN_LOG_FATAL("unrecognized configuration property " << name << " for " << this->name());
        }
    }
};

static drwnDataCacheConfig gDataCacheConfig;
