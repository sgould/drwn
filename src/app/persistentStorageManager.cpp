/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    persistentStorageManager.cpp
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

// PersistentRecordBuffer ---------------------------------------------------

class PersistentRecordBuffer : public drwnPersistentRecord {
 protected:
    size_t _numBytes;
    char *_buffer;

 public:
    PersistentRecordBuffer(size_t b) : drwnPersistentRecord() {
        _buffer = new char[_numBytes = b];
    }
    ~PersistentRecordBuffer() {
        delete[] _buffer;
    }

    size_t numBytesOnDisk() const { return _numBytes; }
    bool write(ostream& os) const { os.write(_buffer, _numBytes); return true; }
    bool read(istream& is) { is.read(_buffer, _numBytes); return true; }
};

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./persistentStorageManager [OPTIONS] <storage> [ACTION]\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "ACTIONS:\n"
         << "  stats             :: storage statistics\n"
         << "  records           :: list records\n"
         << "  merge <dir> <ext> :: merge binary files from a directory\n"
         << "  split <dir> <ext> :: split into a directory of binary files\n"
         << "  delete (<rec>)+   :: delete one or more records\n"
         << "  defrag            :: defragment storage\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC < 1) {
        usage();
        return -1;
    }

    const char *storageName = DRWN_CMDLINE_ARGV[0];
    const char *action = (DRWN_CMDLINE_ARGC == 1) ? "stats" : DRWN_CMDLINE_ARGV[1];

    drwnPersistentStorage storage;
    storage.open(storageName);

    // open storage and perform action
    if (!strcasecmp(action, "stats")) {
        if (DRWN_CMDLINE_ARGC > 2) {
            usage();
            return -1;
        }

        drwnLogLevel level = drwnLogger::setAtLeastLogLevel(DRWN_LL_MESSAGE);
        DRWN_LOG_MESSAGE("Statistics for " << storageName);
        DRWN_LOG_MESSAGE("  " << storage.numRecords() << " records");
        DRWN_LOG_MESSAGE("  " << drwn::bytesToString(storage.numTotalBytes()) << " total space");
        DRWN_LOG_MESSAGE("  " << drwn::bytesToString(storage.numUsedBytes()) << " used space");
        drwnLogger::setLogLevel(level);

    } else if (!strcasecmp(action, "records")) {
        if (DRWN_CMDLINE_ARGC != 2) {
            usage();
            return -1;
        }

        drwnLogLevel level = drwnLogger::setAtLeastLogLevel(DRWN_LL_MESSAGE);
        const set<string> keys = storage.getKeys();
        if (keys.empty()) {
            DRWN_LOG_MESSAGE(storageName << " has no records");
        } else {
            DRWN_LOG_MESSAGE("Records in " << storageName);
            for (set<string>::const_iterator it = keys.begin(); it != keys.end(); ++it) {
                DRWN_LOG_MESSAGE("  " << *it << "\t" << drwn::bytesToString(storage.bytes(it->c_str())));
            }
        }
        drwnLogger::setLogLevel(level);
        
    } else if (!strcasecmp(action, "split")) {
        if (DRWN_CMDLINE_ARGC != 4) {
            usage();
            return -1;
        }

        const char *dir = DRWN_CMDLINE_ARGV[2];
        const char *ext = DRWN_CMDLINE_ARGV[3];

        if (drwnDirExists(dir)) {
            DRWN_LOG_WARNING("split directory " << dir << " already exists");
        } else {
            DRWN_LOG_VERBOSE("creating split directory " << dir);
            drwnCreateDirectory(dir);            
        }
        
        const set<string> keys = storage.getKeys();
        for (set<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
            PersistentRecordBuffer record(storage.bytes(it->c_str()));
            storage.read(it->c_str(), &record);
            
            string filename = string(dir) + DRWN_DIRSEP + (*it) + string(ext);
            ofstream ofs(filename.c_str(), ios::binary);
            record.write(ofs);
            ofs.close();
        }

    } else if (!strcasecmp(action, "merge")) {
        if (DRWN_CMDLINE_ARGC != 4) {
            usage();
            return -1;
        }

        const char *dir = DRWN_CMDLINE_ARGV[2];
        const char *ext = DRWN_CMDLINE_ARGV[3];

        DRWN_ASSERT_MSG(drwnDirExists(dir), "merge directory " << dir << " does not exist");
        const vector<string> keys = drwnDirectoryListing(dir, ext, false, false);
        
        for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); ++it) {
            if (storage.hasKey(it->c_str())) {
                DRWN_LOG_WARNING("replacing record " << *it);
            } else {
                DRWN_LOG_VERBOSE("adding record " << *it);
            }

            string filename = string(dir) + DRWN_DIRSEP + (*it) + string(ext);
            PersistentRecordBuffer record(drwnFileSize(filename.c_str()));
            DRWN_LOG_VERBOSE("merging record " << *it << " of size " 
                << drwn::bytesToString(record.numBytesOnDisk()));
            ifstream ifs(filename.c_str(), ios::binary);            
            record.read(ifs);
            ifs.close();
            storage.write(it->c_str(), &record);
        }

    } else if (!strcasecmp(action, "delete")) {
        if (DRWN_CMDLINE_ARGC < 3) {
            usage();
            return -1;
        }
        
        for (int i = 2; i < DRWN_CMDLINE_ARGC; i++) {
            const char *key = DRWN_CMDLINE_ARGV[i];
            DRWN_ASSERT_MSG(storage.hasKey(key), "no record with key \"" << key 
                << "\" in storage " << storageName);

            storage.erase(key);
            DRWN_LOG_VERBOSE("...storage has " << drwn::bytesToString(storage.numFreeBytes())
                << " free bytes after deleting record " << key);
        }

    } else if (!strcasecmp(action, "defrag")) {
        if (DRWN_CMDLINE_ARGC != 2) {
            usage();
            return -1;
        }

        DRWN_LOG_MESSAGE("...defrag will save " << drwn::bytesToString(storage.numFreeBytes()));
        storage.defragment();
        DRWN_LOG_VERBOSE("...storage requires " << drwn::bytesToString(storage.numTotalBytes())
            << " after defragmentation");

    } else {
        usage();
        DRWN_LOG_FATAL("unknown action \"" << action << "\"");
    } 

    // close storage and print profile information
    storage.close();
    drwnCodeProfiler::print();
    return 0;
}
