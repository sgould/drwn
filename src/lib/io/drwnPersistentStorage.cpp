/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPersistentStorage.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <map>
#include <vector>

#include "drwnBase.h"
#include "drwnCompressionBuffer.h"
#include "drwnPersistentStorage.h"

using namespace std;

// drwnPersistentStorage -----------------------------------------------------

int drwnPersistentStorage::MAX_OPEN = 16;
string drwnPersistentStorage::DEFAULT_INDEX_EXT = string(".index");
string drwnPersistentStorage::DEFAULT_DATA_EXT = string(".data");

list<drwnPersistentStorage *> drwnPersistentStorage::_openList;
#ifdef DRWN_USE_PTHREADS
pthread_mutex_t drwnPersistentStorage::_gmutex = PTHREAD_MUTEX_INITIALIZER;
#endif

drwnPersistentStorage::drwnPersistentStorage(bool bCompressed) :
    _bCompressed(bCompressed), _fsdata(NULL), _bSuspended(false), _bDirty(false)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
}

drwnPersistentStorage::drwnPersistentStorage(const drwnPersistentStorage& storage) :
    _indexFilename(storage._indexFilename),
    _dataFilename(storage._dataFilename),
    _recordMapping(storage._recordMapping),
    _freeSpace(storage._freeSpace),
    _bCompressed(storage._bCompressed),
    _fsdata(NULL), _bSuspended(false), _bDirty(false)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_init(&_mutex, NULL);
#endif
}

drwnPersistentStorage::~drwnPersistentStorage()
{
    if ((_fsdata != NULL) || _bSuspended) {
        close();
    }
}

bool drwnPersistentStorage::open(const char *fileStem)
{
    DRWN_ASSERT(fileStem != NULL);
    string indexFile = string(fileStem) + DEFAULT_INDEX_EXT;
    string dataFile = string(fileStem) + DEFAULT_DATA_EXT;
    return open(indexFile.c_str(), dataFile.c_str());
}

bool drwnPersistentStorage::open(const char *indexFile, const char *dataFile)
{
    DRWN_ASSERT((indexFile != NULL) && (dataFile != NULL));
    if (_fsdata != NULL) {
        close();
    }

    _indexFilename = string(indexFile);
    _dataFilename = string(dataFile);
    _recordMapping.clear();
    _freeSpace.clear();

    // read index file
    ifstream ifs(_indexFilename.c_str());
    drwnPersistentBlock location;
    string key;
    while (1) {
        ifs >> location.start >> location.length >> key;
        if (ifs.fail()) break;
        DRWN_ASSERT(_recordMapping.find(key) == _recordMapping.end());
        _recordMapping.insert(make_pair(key, location));
    }
    ifs.close();
    _bDirty = false;

    // compute free space
    map<size_t, size_t> recordBlocks;
    for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        recordBlocks.insert(make_pair(it->second.start, it->second.length));
    }

    size_t lastRecordEnd = 0;
    for (map<size_t, size_t>::const_iterator it = recordBlocks.begin();
         it != recordBlocks.end(); it++) {
        if (it->first != lastRecordEnd) {
            _freeSpace.push_back(drwnPersistentBlock(lastRecordEnd, it->first - lastRecordEnd));
        }
        lastRecordEnd = it->first + it->second;
    }

    // create empty data file (if no entries exist)
    if (_recordMapping.empty()) {
        ofstream ofs(_dataFilename.c_str());
        DRWN_ASSERT(!ofs.fail());
        ofs.close();
        _bDirty = true;
    }

    // open data file
    reopen();

    return true;
}

bool drwnPersistentStorage::reopen()
{
    return atomic_reopen(false);
}

bool drwnPersistentStorage::close()
{
    return atomic_close(false);
}

size_t drwnPersistentStorage::numTotalBytes() const
{
    size_t totalBytes = 0;
    for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        totalBytes = std::max(it->second.start + it->second.length, totalBytes);
    }

    return totalBytes;
}

size_t drwnPersistentStorage::numUsedBytes() const
{
    size_t usedBytes = 0;
    for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        usedBytes += it->second.length;
    }

    return usedBytes;
}

size_t drwnPersistentStorage::numFreeBytes() const
{
    size_t freeBytes = 0;
    for (list<drwnPersistentBlock>::const_iterator it = _freeSpace.begin();
         it != _freeSpace.end(); it++) {
        freeBytes += it->length;
    }

    return freeBytes;
}

set<string> drwnPersistentStorage::getKeys() const
{
    set<string> keys;
    for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        keys.insert(it->first);
    }

    return keys;
}

bool drwnPersistentStorage::erase(const char *key)
{
    DRWN_ASSERT(key != NULL);
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    resume(); // re-open if self-suspended
    DRWN_ASSERT_MSG(_fsdata != NULL, "can't erase from unopened data storage");

    map<string, drwnPersistentBlock>::iterator it = _recordMapping.find(string(key));
    if (it == _recordMapping.end()) {
        DRWN_LOG_ERROR("attempting to erase non-existent record \"" << key << "\"");
        return false;
    }

    _freeSpace.push_back(it->second);
    _recordMapping.erase(it);
    _bDirty = true;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
    return true;
}

bool drwnPersistentStorage::read(const char *key, drwnPersistentRecord *record)
{
    DRWN_ASSERT((key != NULL) && (record != NULL));
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    resume(); // re-open if self-suspended
    DRWN_ASSERT_MSG(_fsdata != NULL, "can't read from unopened data storage");

    map<string, drwnPersistentBlock>::iterator it = _recordMapping.find(string(key));
    if (it == _recordMapping.end()) {
        DRWN_LOG_ERROR("attempting to read non-existent record \"" << key << "\"");
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_unlock(&_mutex);
#endif
        return false;
    }

    DRWN_LOG_DEBUG("reading record \"" << key << "\" of size " << it->second.length
        << " beginning at " << it->second.start << "...");
    _fsdata->seekg(it->second.start, ios::beg);
    if (_bCompressed) {
        drwnCompressionBuffer buffer;
        buffer.read(*_fsdata);
        unsigned char *data = new unsigned char[buffer.originalBytes()];
        buffer.decompress(data);
        stringstream strio;
        strio.write((const char *)data, buffer.originalBytes());
        delete[] data;
        record->read(strio);
    } else {
        record->read(*_fsdata);
    }
    DRWN_ASSERT_MSG((size_t)_fsdata->tellg() == it->second.start + it->second.length,
        "key: " << key << ", start: " << it->second.start << ", size: "
        << it->second.length << ", tellg: " << _fsdata->tellg());

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
    return true;
}

bool drwnPersistentStorage::write(const char *key, const drwnPersistentRecord *record)
{
    DRWN_ASSERT((key != NULL) && (record != NULL));
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif

    resume(); // re-open if self-suspended
    DRWN_ASSERT_MSG(_fsdata != NULL, "can't write to unopened data storage");

    // compute disk space needed
    size_t bytesRequired = record->numBytesOnDisk();
    drwnCompressionBuffer buffer;
    if (_bCompressed) {
        stringstream strio;
        record->write(strio);
        buffer.compress((const unsigned char *)strio.str().c_str(), strio.str().size());
        bytesRequired = buffer.numBytesOnDisk();
    }

    // check if record exists and fits into current location
    map<string, drwnPersistentBlock>::iterator it = _recordMapping.find(string(key));
    if (it != _recordMapping.end()) {
        if (it->second.length >= bytesRequired) {
            _fsdata->seekp(it->second.start, ios::beg);
            if (_bCompressed) {
                buffer.write(*_fsdata);
            } else {
                record->write(*_fsdata);
            }
            _bDirty = true;

            if (it->second.length > bytesRequired) {
                _freeSpace.push_back(drwnPersistentBlock(it->second.start + bytesRequired,
                    it->second.length - bytesRequired));
                it->second.length = bytesRequired;
            }

#ifdef DRWN_USE_PTHREADS
            pthread_mutex_unlock(&_mutex);
#endif
            return true;
        } else {
            _freeSpace.push_back(it->second);
            _recordMapping.erase(it);
        }
    }

    // record either doesn't exist or exceeds current location
    drwnPersistentBlock location((size_t)-1, bytesRequired);

    // search through free space
    for (list<drwnPersistentBlock>::iterator it = _freeSpace.begin();
         it != _freeSpace.end(); it++) {

        if (it->length >= bytesRequired) {
            location.start = it->start;
            _freeSpace.erase(it);

            if (it->length > bytesRequired) {
                _freeSpace.push_back(drwnPersistentBlock(location.start + location.length,
                    it->length - bytesRequired));
            }

            break;
        }
    }

    // if we couldn't find a free slot, then append to end of file
    if (location.start == (size_t)-1) {
        _fsdata->seekp(0, ios::end);
        location.start = _fsdata->tellp();
    } else {
        _fsdata->seekp(location.start, ios::beg);
    }

    DRWN_LOG_DEBUG("writing record \"" << key << "\" of size " << location.length
        << " to " << location.start << "...");
    if (_bCompressed) {
        buffer.write(*_fsdata);
    } else {
        record->write(*_fsdata);
    }
    DRWN_ASSERT_MSG((size_t)_fsdata->tellp() == location.start + location.length,
        "key: " << key << ", start: " << location.start << ", size: "
        << location.length << ", tellp: " << _fsdata->tellp());

    _recordMapping.insert(make_pair(string(key), location));
    _bDirty = true;

#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
    return true;
}

size_t drwnPersistentStorage::bytes(const char *key) const
{
    map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.find(string(key));
    DRWN_ASSERT_MSG(it != _recordMapping.end(), "non-existent record \"" << key << "\"");

    return it->second.length;
}

bool drwnPersistentStorage::clear()
{
    if (_recordMapping.empty())
        return true;

#if 1
    drwnPersistentBlock block(0, 0);

    for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        block.length = std::max(block.length, it->second.start + it->second.length);
    }
    for (list<drwnPersistentBlock>::const_iterator it = _freeSpace.begin(); it != _freeSpace.end(); it++) {
        block.length = std::max(block.length, it->start + it->length);
    }

    _recordMapping.clear();
    _freeSpace.clear();

    if (block.length > 0) {
        _freeSpace.push_back(block);
    }

    _bDirty = true;
    return true;
#else
    _recordMapping.clear();
    _bDirty = true;
    return defragment();
#endif
}

bool drwnPersistentStorage::defragment()
{
    DRWN_ASSERT(!_indexFilename.empty() && !_dataFilename.empty());
    bool bReopened = (_fsdata == NULL);
    if (bReopened) reopen();

    map<size_t, map<string, drwnPersistentBlock>::iterator> recordOrder;
    for (map<string, drwnPersistentBlock>::iterator it = _recordMapping.begin();
         it != _recordMapping.end(); it++) {
        recordOrder.insert(make_pair(it->second.start, it));
    }

    // note, get and put pointers are not independent of each other
    size_t totalBytes = 0;
    _fsdata->seekp(0, ios::beg);
    for (map<size_t, map<string, drwnPersistentBlock>::iterator>::const_iterator
             it = recordOrder.begin(); it != recordOrder.end(); it++) {
        map<string, drwnPersistentBlock>::iterator jt = it->second;
        DRWN_LOG_DEBUG("defragment(): writing record \"" << jt->first
            << "\" to " << totalBytes << " from " << jt->second.start);

        // copy data backwards if not in correct location
        if (jt->second.start != totalBytes) {
            _fsdata->seekg(jt->second.start, ios::beg);
            jt->second.start = totalBytes;
            char *data = new char[jt->second.length];
            _fsdata->read(data, jt->second.length);
            _fsdata->seekp(totalBytes, ios::beg);
            _fsdata->write(data, jt->second.length);
            delete[] data;
        }

        // update total bytes written
        totalBytes += jt->second.length;
    }

    _freeSpace.clear();
    _bDirty = true;

    close();
    drwnFileResize(_dataFilename.c_str(), totalBytes);
    if (!bReopened) reopen();
    return true;
}

void drwnPersistentStorage::suspend()
{
    atomic_suspend(false);
}

void drwnPersistentStorage::resume()
{
    if (_bSuspended) {
        DRWN_LOG_DEBUG("resuming storage " << _dataFilename << "...");
        reopen();
    }
}

bool drwnPersistentStorage::atomic_reopen(bool locked)
{
    DRWN_ASSERT(!_indexFilename.empty() && !_dataFilename.empty());
    DRWN_ASSERT_MSG(_fsdata == NULL, "data storage already open");

    // open the data file
    _fsdata = new fstream(_dataFilename.c_str(), ios::binary | ios::in | ios::out);
    _bSuspended = false;
    DRWN_ASSERT(!_fsdata->fail());

    // add to open list and suspend other storage
#ifdef DRWN_USE_PTHREADS
    if (!locked) pthread_mutex_lock(&_gmutex);
#endif

    _openList.push_back(this);
    while ((int)_openList.size() > std::max(MAX_OPEN, 1)) {
        _openList.front()->atomic_suspend(true);
    }
    DRWN_ASSERT(find(_openList.begin(), _openList.end(), this) != _openList.end());

#ifdef DRWN_USE_PTHREADS
    if (!locked) pthread_mutex_unlock(&_gmutex);
#endif

    return true;
}

bool drwnPersistentStorage::atomic_close(bool locked)
{
    DRWN_ASSERT((_fsdata != NULL) || _bSuspended);

    // close data file and remove from open list
    if (_fsdata != NULL) {
        _fsdata->close();
        delete _fsdata;

#ifdef DRWN_USE_PTHREADS
        if (!locked) pthread_mutex_lock(&_gmutex);
#endif
        list<drwnPersistentStorage *>::iterator it = find(_openList.begin(), _openList.end(), this);
        DRWN_ASSERT_MSG(it != _openList.end(), _dataFilename);
        _openList.erase(it);

#ifdef DRWN_USE_PTHREADS
        if (!locked) pthread_mutex_unlock(&_gmutex);
#endif
    }
    _fsdata = NULL;
    _bSuspended = false;

    // write index file
    if (_bDirty) {
        ofstream ofs(_indexFilename.c_str());
        DRWN_ASSERT(!ofs.fail());

        for (map<string, drwnPersistentBlock>::const_iterator it = _recordMapping.begin();
             it != _recordMapping.end(); it++) {
            ofs << it->second.start << " " << it->second.length << " " << it->first << "\n";
        }

        ofs.close();
        _bDirty = false;
    }

    return true;
}

void drwnPersistentStorage::atomic_suspend(bool locked)
{
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_lock(&_mutex);
#endif
    if (_fsdata != NULL) {
        DRWN_LOG_DEBUG("suspending storage " << _dataFilename << "...");
        atomic_close(locked);
        _bSuspended = true;
    }
#ifdef DRWN_USE_PTHREADS
    pthread_mutex_unlock(&_mutex);
#endif
}
