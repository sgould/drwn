/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinDatabase.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

// eigen matrix library headers
#include "Eigen/Core"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnEngine.h"

using namespace std;
using namespace Eigen;

// prototypes ----------------------------------------------------------------

void printRecord(const drwnDataRecord& r);
void printStorage(const drwnPersistentStorage& s);

void testRecord();
void testStorage();
void testTable();
void testDatabase();
void testCache();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinDatabase [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << "  -testRecord       :: test database record functionality\n"
         << "  -testStorage      :: test database storage functionality\n"
         << "  -testTable        :: test database table functionality\n"
         << "  -testDatabase     :: test database functionality\n"
         << "  -testCache        :: test database cache functionality\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    if (argc == 1) {
        usage();
        return 0;
    }

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("-testRecord")
        testRecord();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("-testStorage")
        testStorage();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("-testTable")
        testTable();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("-testDatabase")
        testDatabase();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("-testCache")
        testCache();
        DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());

    // print profile information
    drwnCodeProfiler::print(cerr);
    return 0;
}

// private functions ---------------------------------------------------------

void printRecord(const drwnDataRecord& r)
{
    cout << "Size: " << r.numObservations() << "-by-" << r.numFeatures() << "\n";
    cout << "Memory: " << r.numBytes() << " bytes (" << r.numBytesOnDisk() << " bytes on disk)\n";
    cout << "Dirty: " << (r.isDirty() ? "yes" : "no") << "\n";
    cout << "Data: ";
    if (!r.hasData()) {
        cout << "none\n";
    } else {
        cout << r.data() << "\n";
    }
    cout << "Objective: " << (r.hasObjective() ? "yes" : "no") << "\n";
    cout << "Gradient: " << (r.hasGradient() ? "yes" : "no") << "\n";
    cout << "\n";
}

void printStorage(const drwnPersistentStorage& s)
{
    cout << "r: " << s.numRecords() << ", t: " << s.numTotalBytes()
         << ", u: " << s.numUsedBytes() << ", f: " << s.numFreeBytes() << endl;
}

void testRecord()
{
    DRWN_FCN_TIC;

    drwnDataRecord record;
    DRWN_ASSERT(!record.isDirty());

    record.clear();
    DRWN_ASSERT(!record.isDirty());

    record.data().resize(1, 10);
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        printRecord(record);
    }
    DRWN_ASSERT(record.isDirty());
    DRWN_ASSERT(record.hasData());
    DRWN_ASSERT(record.numFeatures() == 10);
    DRWN_ASSERT(record.numObservations() == 1);

    record.data() = MatrixXd::Random(record.numObservations(), record.numFeatures());
    DRWN_ASSERT(record.isDirty());

    ofstream ofs("testDataRecord.bin");
    record.write(ofs);
    ofs.close();
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        printRecord(record);
    }
    DRWN_ASSERT(!record.isDirty());

    record.clear();
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        printRecord(record);
    }
    DRWN_ASSERT(!record.hasData());

    ifstream ifs("testDataRecord.bin");
    record.read(ifs);
    ifs.close();
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        printRecord(record);
    }
    DRWN_ASSERT(!record.isDirty());

    remove("testDataRecord.bin");

    DRWN_FCN_TOC;
}

void testStorage()
{
    DRWN_FCN_TIC;

    DRWN_LOG_MESSAGE("testing open...");
    drwnPersistentStorage storage;
    storage.open("testStorage.index", "testStorage.data");

    DRWN_LOG_MESSAGE("testing writes...");
    drwnDataRecord record;
    record.data() = MatrixXd::Zero(1, 10);
    storage.write("key1", &record);
    DRWN_ASSERT(!record.isDirty());

    record.data() = MatrixXd::Ones(1, 10);
    storage.write("key2", &record);
    DRWN_ASSERT(!record.isDirty());

    record.data() = MatrixXd::Random(1, 10);
    storage.write("key3", &record);
    DRWN_ASSERT(!record.isDirty());

    storage.close();
    printStorage(storage);

    DRWN_LOG_MESSAGE("testing reopen and erase...");
    storage.reopen();
    storage.erase("key2");
    printStorage(storage);

    DRWN_LOG_MESSAGE("testing defragment...");
    storage.defragment();

    DRWN_LOG_MESSAGE("testing reads...");
    drwnDataRecord *readRecord = new drwnDataRecord();
    bool success = storage.read("key3", readRecord);
    DRWN_ASSERT(success && (readRecord != NULL));
    DRWN_ASSERT(*readRecord == record);

    success = storage.read("key1", readRecord);
    DRWN_ASSERT(success);
    DRWN_ASSERT(*readRecord != record);

    success = storage.read("key2", readRecord);
    DRWN_ASSERT(!success);

    storage.close();
    printStorage(storage);

    DRWN_LOG_MESSAGE("testing clear...");
    storage.clear();
    printStorage(storage);

    DRWN_LOG_MESSAGE("testing free space...");
    storage.reopen();
    storage.write("key1", &record);
    storage.write("key2", &record);
    storage.write("key3", &record);
    printStorage(storage);
    storage.erase("key2");
    printStorage(storage);
    storage.write("key4", &record);
    printStorage(storage);

    DRWN_LOG_MESSAGE("testing clear...");
    storage.clear();
    printStorage(storage);
    storage.close();

    //remove("testStorage.index");
    //remove("testStorage.data");

    // try opening lots of storages
    DRWN_LOG_MESSAGE("writing and reading multiple storages...");
    vector<drwnPersistentStorage> storageArray(100);
    for (int i = 0; i < (int)storageArray.size(); i++) {
        string baseName = string("storage") + toString(i);
        storageArray[i].open((baseName + string(".index")).c_str(),
            (baseName + string(".data")).c_str());
        record.data() = MatrixXd::Zero(1, 10);
        storageArray[i].write("key", &record);
    }

    for (int i = 0; i < (int)storageArray.size(); i++) {
        drwnDataRecord *readRecord = new drwnDataRecord();
        bool success = storageArray[i].read("key", readRecord);
        DRWN_ASSERT(success);
        delete readRecord;
    }

    for (int i = 0; i < (int)storageArray.size(); i++) {
        storageArray[i].close();

        string baseName = string("storage") + toString(i);
        remove((baseName + string(".index")).c_str());
        remove((baseName + string(".data")).c_str());
    }

    DRWN_FCN_TOC;
}

void testTable()
{
    DRWN_NOT_IMPLEMENTED_YET;
}

void testDatabase()
{
    DRWN_FCN_TIC;

    // create a new database
    drwnDatabase *db = new drwnDatabase("testDatabase");
    DRWN_LOG_VERBOSE("attempting database creation for the first time...");
    bool success = db->create();
    DRWN_ASSERT(success);

    DRWN_LOG_VERBOSE("attempting database creation for the second time...");
    success = db->create();
    DRWN_ASSERT(!success);

    DRWN_LOG_VERBOSE("deleting database...");
    delete db;

    // read the newly created database
    DRWN_LOG_VERBOSE("attempting database read for the first time...");
    db = new drwnDatabase("testDatabase");
    success = db->read();
    DRWN_ASSERT(success);

    DRWN_LOG_VERBOSE("attempting database read for the second time...");
    success = db->read();
    DRWN_ASSERT(!success);

    delete db;

    DRWN_LOG_VERBOSE("removing database...");
    drwnRemoveDirectory("testDatabase");
    DRWN_FCN_TOC;
}

void testCache()
{
    DRWN_NOT_IMPLEMENTED_YET;
}
