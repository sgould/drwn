/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinIO.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// eigen matrix library headers
#include "Eigen/Core"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;
using namespace Eigen;

// prototypes ----------------------------------------------------------------

void testZlibUtils();
void testPersistentStorageBuffer(bool bCompress);
void testThreadedPersistentStorage();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinIO [OPTIONS] (<test>)*\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "TESTS:\n"
         << "  zlib              :: zlib compression utilities\n"
         << "  storagebuffer     :: persistent storage buffer\n"
         << "  threadedstorage   :: threaded persistent storage\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("zlib")
            testZlibUtils();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("storagebuffer")
            testPersistentStorageBuffer(false);
            testPersistentStorageBuffer(true);
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("threadedstorage")
            testThreadedPersistentStorage();
        DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// regression tests ----------------------------------------------------------

void testZlibUtils()
{
    // test random vector
    {
        VectorXd x = VectorXd::Random(1024);
        drwnCompressionBuffer buffer = drwnCompressVector(x);
        DRWN_LOG_MESSAGE("Compressed " << x.rows()
            << "-vector to " << buffer.compressedBytes() << " bytes");
        VectorXd y = drwnDecompressVector(buffer, x.rows());
        DRWN_ASSERT((x - y).norm() == 0.0);
    }

    // test uniform vector
    {
        VectorXd x = VectorXd::Constant(1024, 22);
        drwnCompressionBuffer buffer = drwnCompressVector(x);
        DRWN_LOG_MESSAGE("Compressed " << x.rows()
            << "-vector to " << buffer.compressedBytes() << " bytes");
        VectorXd y = drwnDecompressVector(buffer, x.rows());
        DRWN_ASSERT((x - y).norm() == 0.0);
    }

}

void testPersistentStorageBuffer(bool bCompress)
{
    const char *storageName = "_test";

    // delete existing storage (if it exists)
    {
        drwnPersistentStorage storage;
        storage.open(storageName);
        storage.clear();
        storage.close();
    }

    // create stroage buffer
    {
        typedef drwnPersistentVectorRecord<double> Record;
        drwnPersistentStorageBuffer<Record>::MAX_RECORDS = 10;
        drwnPersistentStorageBuffer<Record> buffer(storageName, bCompress);
        
        drwnSmartPointer<Record> recA(new Record());
        recA->data.resize(100, 0);
        buffer.write(string("A"), recA);
        DRWN_LOG_VERBOSE("wrote record A to buffer: " << buffer.size());
        buffer.flush();
        DRWN_LOG_VERBOSE("flushed buffer: " << buffer.size());
        
        drwnSmartPointer<Record> recB(new Record());
        recB->data.resize(200, 0);
        buffer.write(string("B"), recB);
        DRWN_LOG_VERBOSE("wrote record B to buffer: " << buffer.size());
        
        recA = buffer.read(string("A"));
        DRWN_LOG_VERBOSE("read record A from buffer: " << buffer.size());
        
        // test memory usage
        for (int i = 0; i < 1000; i++) {
            drwnSmartPointer<Record> rec(new Record());
            rec->data.resize(1000, 0);
            string key = string("REC") + toString(i);
            buffer.write(key, rec);
            DRWN_LOG_VERBOSE("wrote record " << key << " to buffer: " << buffer.size());
            
            key = string("REC") + toString((int)(i * drand48()));
            rec = buffer.read(key);
            DRWN_LOG_VERBOSE("read record " << key << " from buffer: " << buffer.size());
        }
    }

    // clear storage
    {
        string filename = string(storageName) + drwnPersistentStorage::DEFAULT_INDEX_EXT;
        drwnRemoveFile(filename.c_str());

        filename = string(storageName) + drwnPersistentStorage::DEFAULT_DATA_EXT;
        drwnRemoveFile(filename.c_str());
    }
}

// testThreadedPersistentStorage -------------------------------------------------------

class PersistentStorageJob : public drwnThreadJob {
protected:    
    drwnPersistentStorage _storage;
    bool _bWritten;

public:
    PersistentStorageJob(const string& name) : _bWritten(false) {
        _storage.open(name.c_str());
    }
    virtual ~PersistentStorageJob() {
        _storage.close();
    }
    
    void operator()() {
        drwnPersistentVectorRecord<double> *rec;
        rec = new drwnPersistentVectorRecord<double>();

        if (!_bWritten) {
            // write a record
            rec->data.resize(100, 0);
            for (int i = 0; i < (int)rec->data.size(); i++) {
                rec->data[i] = drand48();
            }
            _storage.write(string("RECORD").c_str(), rec);
            _bWritten = true;

        } else {
            // read the record
            _storage.read("RECORD", rec);
            DRWN_ASSERT(rec->data.size() == 100);
        }

        delete rec;
    }
};

void testThreadedPersistentStorage()
{
    const char *storageName = "_test";

    drwnThreadPool pool;
    vector<PersistentStorageJob *> jobs;

    DRWN_LOG_VERBOSE("starting thread pool with " << drwnThreadPool::MAX_THREADS << " threads");

    // write phase
    DRWN_LOG_VERBOSE("write phase...");
    pool.start();
    for (int i = 0; i < 1000; i++) {
        string name = string(storageName) + drwn::padString(toString(i), 4, '0');
        jobs.push_back(new PersistentStorageJob(name));
        pool.addJob(jobs.back());
    }
    pool.finish(false);

    // read phase
    pool.start();
    DRWN_LOG_VERBOSE("read phase...");
    for (int i = 0; i < 1000; i++) {
        pool.addJob(jobs[i]);
    }
    pool.finish(true);

    for (int i = 0; i < (int)jobs.size(); i++) {
        delete jobs[i];
    }
}
