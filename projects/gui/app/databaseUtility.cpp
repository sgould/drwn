/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    databaseUtility.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Commandline application for managing databases.
**
*****************************************************************************/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// prototypes ----------------------------------------------------------------



// main ----------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./databaseUtility [OPTIONS] <command> <db> [<args> ...]\n";
    cerr << "COMMANDS:\n"
         << "  defrag            :: defragment database (specific tables if given)\n"
         << "  statistics        :: print database statistics\n"
         << "  unlock            :: unlocks a database\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // process commandline propertys
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        // TODO
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC < 2) {
        usage();
        return -1;
    }

    const char *COMMAND = DRWN_CMDLINE_ARGV[0];
    const char *DBNAME = DRWN_CMDLINE_ARGV[1];

    // execute command
    if (!strcasecmp(COMMAND, "defrag")) {
        drwnDatabase db(DBNAME);
        db.read();

        vector<string> tableNames;
        if (DRWN_CMDLINE_ARGC > 2) {
            // compress specific tables
            for (int i = 2; i < DRWN_CMDLINE_ARGC; i++) {
                const char *TBLNAME = DRWN_CMDLINE_ARGV[i];
                drwnDataTable *tbl = db.getTable(TBLNAME);
                if (tbl == NULL) {
                    DRWN_LOG_ERROR("table " << TBLNAME << " does not exist in database");
                }
                tableNames.push_back(string(TBLNAME));
            }
        } else {
            // compress all tables
            tableNames = db.getTableNames();
        }

        for (int i = 0; i < (int)tableNames.size(); i++) {
            drwnDataTable *tbl = db.getTable(tableNames[i].c_str());
            DRWN_ASSERT(tbl != NULL);
            DRWN_LOG_VERBOSE("defragmenting table \"" << tableNames[i] << "\"...");
            int originalBytes = tbl->size();
            tbl->defragment();
            int defragmentedBytes = tbl->size();
            DRWN_LOG_VERBOSE("...from " << drwn::bytesToString(originalBytes)
                << " to " << drwn::bytesToString(defragmentedBytes));
        }

    } else if (!strcasecmp(COMMAND, "statistics") || !strcasecmp(COMMAND, "stats")) {
        drwnDatabase db(DBNAME);
        db.read();

        vector<string> names = db.getTableNames();
        int totalRecords = 0;
        double totalBytes = 0.0;
        cout << " RECORDS     SIZE  TABLE NAME\n";
        for (vector<string>::const_iterator it = names.begin(); it != names.end(); it++) {
            drwnDataTable *tbl = db.getTable(*it);
            totalRecords += tbl->numRecords();
            totalBytes += (double)tbl->size();
            cout << "  " << setw(6) << tbl->numRecords()
                 << "  " << setw(7) << drwn::bytesToString(tbl->size())
                 << "  " << tbl->name() << "\n";
        }
        cout << "  ------  -------\n";
        cout << "  " << setw(6) << totalRecords
             << "  " << setw(7) << drwn::bytesToString((int)totalBytes) << "\n";

    } else if (!strcasecmp(COMMAND, "unlock")) {
        string lockFilename = string(DBNAME) + DRWN_DIRSEP + string("locked");
        DRWN_LOG_DEBUG("checking for file " << lockFilename);
        if (drwnFileExists(lockFilename.c_str())) {
            // TODO: print locked date
            DRWN_LOG_VERBOSE("unlocking database \"" << DBNAME << "\"");
            remove(lockFilename.c_str());
        } else {
            DRWN_LOG_ERROR("database \"" << DBNAME << "\" does not exist or is not locked");
        }

    } else {
        DRWN_LOG_ERROR("unrecognized command \"" << COMMAND << "\"");
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print(cerr);
    return 0;
}


