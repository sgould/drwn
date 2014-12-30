/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPersistentStorage.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <map>
#include <list>
#include <vector>

using namespace std;

// drwnPersistentRecord ------------------------------------------------------
//! Interface class for drwnPersistentStorage.

class drwnPersistentRecord {
 public:
    drwnPersistentRecord() { /* do nothing */ };
    virtual ~drwnPersistentRecord() { /* do nothing */ };

    //! number of bytes required to store object on disk or in a character
    //! stream (without compression)
    virtual size_t numBytesOnDisk() const = 0;
    //! write the object to an output stream
    virtual bool write(ostream& os) const = 0;
    //! read the object from an input stream
    virtual bool read(istream& is) = 0;
};

// drwnPersistentVectorRecord ------------------------------------------------
//! Templated class for storing vector records.

template <class T>
class drwnPersistentVectorRecord : public drwnPersistentRecord {
 public:
    std::vector<T> data; //<! the data

 public:
    //! default constructor
    drwnPersistentVectorRecord() { /* do nothing */ }
    //! destructor
    ~drwnPersistentVectorRecord() { /* do nothing */ }

    // i/o (from drwnPersistentRecord)
    size_t numBytesOnDisk() const {
        return sizeof(size_t) + data.size() * sizeof(T);
    }

    bool write(ostream& os) const {
        size_t n = data.size();
        os.write((char *)&n, sizeof(size_t));
        if (n != 0) {
            os.write((char *)&data[0], n * sizeof(T));
        }
        return true;
    }

    bool read(istream& is) {
        size_t n = 0;
        is.read((char *)&n, sizeof(size_t));
        data.resize(n);
        if (n > 0) {
            is.read((char *)&data[0], n * sizeof(T));
        }
        return true;
    }
};

// drwnPersistentVectorVectorRecord ------------------------------------------
//! Templated class for storing vector-of-vector records.

template <class T>
class drwnPersistentVectorVectorRecord : public drwnPersistentRecord {
 public:
    std::vector<std::vector<T> > data; //<! the data

 public:
    //! default constructor
    drwnPersistentVectorVectorRecord() { /* do nothing */ }
    //! destructor
    ~drwnPersistentVectorVectorRecord() { /* do nothing */ }

    // i/o (from drwnPersistentRecord)
    size_t numBytesOnDisk() const {
        size_t n = sizeof(size_t) * (data.size() + 1);
        for (unsigned i = 0; i < data.size(); i++) {
            n += data[i].size() * sizeof(T);
        }
        return n;
    }

    bool write(ostream& os) const {
        size_t n = data.size();
        os.write((char *)&n, sizeof(size_t));
        for (unsigned i = 0; i < data.size(); i++) {
            n = data[i].size();
            os.write((char *)&n, sizeof(size_t));
            if (n != 0) {
                os.write((char *)&data[i][0], n * sizeof(T));
            }
        }
        return true;
    }

    bool read(istream& is) {
        size_t n = 0;
        is.read((char *)&n, sizeof(size_t));
        data.resize(n);
        for (unsigned i = 0; i < data.size(); i++) {
            is.read((char *)&n, sizeof(size_t));
            data[i].resize(n);
            if (n > 0) {
                is.read((char *)&data[i][0], n * sizeof(T));
            }
        }
        return true;
    }
};

// drwnPersistentBlock -------------------------------------------------------
//! Persistent storage block used internally by drwnPersistentStorage

class drwnPersistentBlock {
 public:
    size_t start;   //!< start of block on disk
    size_t length;  //!< length of block on disk

 public:
    drwnPersistentBlock() { /* do nothing */ };
    drwnPersistentBlock(size_t strt, size_t len) : start(strt), length(len) { /* do nothing */ };
    ~drwnPersistentBlock() { /* do nothing */ };
};

// drwnPersistentStorage -----------------------------------------------------
//! \brief Provides indexed storage for multiple records using two files (a
//! binary data file and a text index file).
//!
//! The class maintains an open stream to the data file. The index file is written
//! on closing. A closed storage object can be reopened without having to re-parse
//! the index file. The class also manages the maximum number of actively open
//! storage objects. If more storage objects are opened, the oldest ones will be
//! suspended (temporarily closes). Any operation to the storage object will
//! automatically re-open (resume) the storage (possibly suspending other storage
//! objects). To use this class, objects must implement the drwnPersistentRecord
//! interface. Automatic compression of records can be set and is
//! managed by through the drwnCompressionBuffer class.
//!
//! The erase(), read() and write() methods are thread-safe allowing the same
//! persistent storage object to be accessed from multiple threads.
//!
//! \sa \ref drwnTutorial for an example.

class drwnPersistentStorage {
 public:
    static int MAX_OPEN;                          //!< max. number of open files
    static string DEFAULT_INDEX_EXT;              //!< default extension for index files
    static string DEFAULT_DATA_EXT;               //!< default extension for data files

 protected:
    static list<drwnPersistentStorage *> _openList; //!< list of open storage

    string _indexFilename;                        //!< name of index file
    string _dataFilename;                         //!< name of data file
    map<string, drwnPersistentBlock> _recordMapping; //!< key, start, length
    list<drwnPersistentBlock> _freeSpace;         //!< start, length

    bool _bCompressed;                            //!< data in storage is compressed
    fstream *_fsdata;                             //!< data file stream (if open)
    bool _bSuspended;                             //!< self-suspended vs. closed (will reopen on any operation)
    bool _bDirty;                                 //!< data has been written to the storage since opened

#ifdef DRWN_USE_PTHREADS
    static pthread_mutex_t _gmutex;               //!< thread safety for all drwnPersistentStorage objects
    pthread_mutex_t _mutex;                       //!< thread safety for this drwnPersistentStorage object
#endif

 public:
    drwnPersistentStorage(bool bCompressed = false);
    drwnPersistentStorage(const drwnPersistentStorage& storage);
    virtual ~drwnPersistentStorage();

    //! open persistent storage index and data files
    bool open(const char *indexFile, const char *dataFile);
    //! open persistent storage using default extension for index and data files
    bool open(const char *fileStem);
    //! re-open previously opened persistent storage files
    bool reopen();
    //! close (and write) persistent storage files
    bool close();

    //! returns true if the persistent storage has been opened and not closed
    bool isOpen() const { return (_fsdata != NULL) || _bSuspended; }
    //! returns true if the persistent storage object has valid filenames but is currently closed
    bool canReopen() const { return (!_indexFilename.empty() && !_dataFilename.empty() && (_fsdata == NULL)); }

    //! number of drwnPersistentRecord records stored on disk
    int numRecords() const { return (int)_recordMapping.size(); }
    //! total number of bytes on disk used by the data file
    size_t numTotalBytes() const;
    //! number of bytes on disk actually needed by the data file
    size_t numUsedBytes() const;
    //! number of free bytes (due to fragmentation) in data file
    size_t numFreeBytes() const;

    //! returns true if a record with given key exists in the persistent storage
    bool hasKey(const char *key) const { return _recordMapping.find(string(key)) != _recordMapping.end(); }
    //! returns all keys stored in the persistent storage object
    set<string> getKeys() const;

    //! erases the record with given key
    bool erase(const char *key);
    //! reads the record with given key
    bool read(const char *key, drwnPersistentRecord *record);
    //! writes the record with given key
    bool write(const char *key, const drwnPersistentRecord *record);
    //! number of bytes on disk used for record with given key
    size_t bytes(const char *key) const;

    //! clears all records from the persistent storage object (takes affect when closed)
    bool clear();
    //! defragments data file
    bool defragment();

 protected:
    void suspend();  //!< self-suspend (close)
    void resume();   //!< reopen if suspended

    bool atomic_reopen(bool locked);  //!< perform an atomic reopen operation
    bool atomic_close(bool locked);   //!< perform an atomic close operation
    void atomic_suspend(bool locked); //!< perform an atomic suspend operation
};
