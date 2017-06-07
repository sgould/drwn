/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPersistentStorageBuffer.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <map>
#include <list>
#include <string>

#include "drwnBase.h"
#include "drwnPersistentStorage.h"

using namespace std;

// drwnPersistentStorageBuffer -----------------------------------------------
//! \brief Provides buffered storage (delayed write-back) of objects with a
//! drwnPersistentRecord interface.
//!
//! Records are removed in least-recently-used order.
//!
//! \warning The API for this class may change in future versions.
//! \warning The class is not thread-safe.
//!
//! \todo rename to drwnPersistentRecordBuffer
//!

template <class T>
class drwnPersistentStorageBuffer {
 public:
    static size_t MAX_MEMORY;  //!< maximum number of bytes stored
    static size_t MAX_RECORDS; //!< maximum number of records stored

 protected:
    class _drwnRecordEntry {
    public:
        string key; //!< the record's key
        drwnSmartPointer<T> record; //!< the record's data
        bool dirty; //!< dirty flag (requires write-back)

    public:
        inline _drwnRecordEntry() : dirty(false) { /* do nothing */ }
        inline _drwnRecordEntry(const string& k, drwnSmartPointer<T>& r) :
            key(k), record(r), dirty(false) { /* do nothing */ }
        inline ~_drwnRecordEntry() { /* do nothing */ }
    };

    typedef list<_drwnRecordEntry> _drwnRecordList;
    typedef map<string, typename _drwnRecordList::iterator> _drwnRecordRef;

    drwnPersistentStorage _storage; //!< storage object
    _drwnRecordRef _recref; //!< reference to records
    _drwnRecordList _records; //!< cached records (in most-recently-used order)
    size_t _memoryUsed; //!< current memory usage

 public:
    drwnPersistentStorageBuffer(const char *fileStem, bool bCompressed = false);
    ~drwnPersistentStorageBuffer();

    //! returns number of records in the buffer
    size_t size() const { return _records.size(); }
    //! returns memory used by in-memory records
    size_t memory() const { return _memoryUsed; }
    //! returns a reference to the storage object
    const drwnPersistentStorage& storage() const { return _storage; }

    //! returns a record from the in-memory buffer or disk storage
    drwnSmartPointer<T> read(const string& key);

    //! writes a record to storage (the write is delayed until the
    //! record is removed from memory or flush is called)
    void write(const string& key, drwnSmartPointer<T>& p);

    //! erases a record from storage (and memory)
    void erase(const string& key);

    //! write all pending records to disk
    void flush();

protected:
    //! checks memory and count limits, writes and frees pending records
    //! if exceeded
    void enforceLimits();
};

// drwnPersistentStorageBuffer Implementation --------------------------------

template <class T>
size_t drwnPersistentStorageBuffer<T>::MAX_MEMORY = 1.0e9;

template <class T>
size_t drwnPersistentStorageBuffer<T>::MAX_RECORDS = 1000;

template <class T>
drwnPersistentStorageBuffer<T>::drwnPersistentStorageBuffer(const char *fileStem, bool bCompressed) :
    _storage(bCompressed), _memoryUsed(0)
{
    _storage.open(fileStem);
    DRWN_ASSERT(_storage.isOpen());
}

template <class T>
drwnPersistentStorageBuffer<T>::~drwnPersistentStorageBuffer()
{
    flush(); // write pending records
    _storage.close();
}

template <class T>
drwnSmartPointer<T> drwnPersistentStorageBuffer<T>::read(const string& key)
{
    typename _drwnRecordRef::iterator it = _recref.find(key);
    if (it != _recref.end()) {
        // move record to front of record list
        _records.push_front(*it->second);
        _records.erase(it->second);
        it->second = _records.begin();
        return it->second->record;
    }

    // read record from storage and push to front of record list
    drwnSmartPointer<T> p(new T());
    _storage.read(key.c_str(), p);
    _records.push_front(_drwnRecordEntry(key, p));
    _recref.insert(make_pair(key, _records.begin()));
    _memoryUsed += p->numBytesOnDisk();
    enforceLimits();

    return p;
}

template <class T>
void drwnPersistentStorageBuffer<T>::write(const string& key, drwnSmartPointer<T>& p)
{
    // delete existing record with same key
    typename _drwnRecordRef::iterator it = _recref.find(key);
    if (it != _recref.end()) {
        _memoryUsed -= it->second->record->numBytesOnDisk();
        _records.erase(it->second);
        _recref.erase(it);
    }

    // add record to front of record list
    _records.push_front(_drwnRecordEntry(key, p));
    _records.front().dirty = true;
    _recref.insert(make_pair(key, _records.begin()));

    _memoryUsed += p->numBytesOnDisk();
    enforceLimits();
}

template <class T>
void drwnPersistentStorageBuffer<T>::erase(const string& key)
{
    typename _drwnRecordRef::iterator it = _recref.find(key);
    if (it != _recref.end()) {
        _memoryUsed -= it->second->record->numBytesOnDisk();
        _records.erase(it->second);
        _recref.erase(it);
    }
    _storage.erase(key.c_str());
}

template <class T>
void drwnPersistentStorageBuffer<T>::flush()
{
    for (typename _drwnRecordList::const_iterator it = _records.begin(); it != _records.end(); it++) {
        if (it->dirty) {
            _storage.write(it->key.c_str(), it->record);
        }
    }

    _recref.clear();
    _records.clear();
    _memoryUsed = 0;
}

template <class T>
void drwnPersistentStorageBuffer<T>::enforceLimits()
{
    while (!_records.empty()) {
        if ((_records.size() <= MAX_RECORDS) && (_memoryUsed <= MAX_MEMORY)) break;

        const string key = _records.back().key;
        DRWN_LOG_DEBUG("removing " << (_records.back().dirty ? "dirty" : "clean") <<
            " record \"" << key << "\" from drwnPersistentStorageBuffer");
        _memoryUsed -= _records.back().record->numBytesOnDisk();
        if (_records.back().dirty) {
            _storage.write(key.c_str(), _records.back().record);
        }

        typename _drwnRecordRef::iterator it = _recref.find(key);
        DRWN_ASSERT_MSG(it != _recref.end(), key);
        _recref.erase(it);

        _records.pop_back();
    }
}
