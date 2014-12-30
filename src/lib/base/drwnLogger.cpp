/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLogger.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstring>

#include "drwnConstants.h"
#include "drwnCompatibility.h"
#include "drwnConfigManager.h"
#include "drwnLogger.h"

using namespace std;

// logging callbacks
void (*drwnLogger::showFatalCallback)(const char *message) = NULL;
void (*drwnLogger::showErrorCallback)(const char *message) = NULL;
void (*drwnLogger::showWarningCallback)(const char *message) = NULL;
void (*drwnLogger::showStatusCallback)(const char *message) = NULL;
void (*drwnLogger::showMessageCallback)(const char *message) = NULL;

void (*drwnLogger::showProgressCallback)(const char *status, double p) = NULL;

ofstream drwnLogger::_log;
drwnLogLevel drwnLogger::_logLevel = DRWN_LL_MESSAGE;
drwnLogLevel drwnLogger::_lastMessageLevel = DRWN_LL_MESSAGE;
string drwnLogger::_cmdLineString;

bool drwnLogger::_bRunning = true;
string drwnLogger::_progressStatus;
int drwnLogger::_lastProgress = 0;
int drwnLogger::_progressLimit = 100;

drwnLogger::drwnLogger()
{
    // do nothing
}

drwnLogger::~drwnLogger()
{
    // do nothing
}

void drwnLogger::initialize(const char *filename, bool bOverwrite)
{
    if (_log.is_open()) {
        _log.close();
    }

    if ((filename != NULL) && (strlen(filename) > 0)) {
        _log.open(filename, bOverwrite ? ios_base::out : ios_base::out | ios_base::app);
        assert(!_log.fail());

        time_t t = time(NULL);
        struct tm *lt = localtime(&t);
        char buffer[256];
        if (strftime(&buffer[0], sizeof(buffer), "%c", lt) == 0) {
            buffer[0] = '\0';
        }

        _log << "-" << setw(2) << setfill('0') << lt->tm_hour
             << ":" << setw(2) << setfill('0') << lt->tm_min
             << ":" << setw(2) << setfill('0') << lt->tm_sec
             << "--- log opened: " << buffer << " (version: " << DRWN_VERSION << ") --- \n";

        if (!_cmdLineString.empty()) {
            _log << "-" << setw(2) << setfill('0') << lt->tm_hour
                 << ":" << setw(2) << setfill('0') << lt->tm_min
                 << ":" << setw(2) << setfill('0') << lt->tm_sec
                 << "--- command line: " << _cmdLineString << "\n";
        }
    }
}

void drwnLogger::initialize(const char *filename,
    bool bOverwrite, drwnLogLevel level)
{
    _logLevel = level;
    initialize(filename, bOverwrite);
}

void drwnLogger::cacheCommandLine(int argc, char **argv)
{
    _cmdLineString.clear();
    for (int i = 0; i < argc; i++) {
        if (i != 0) _cmdLineString += ' ';
        _cmdLineString += string(argv[i]);
    }
}

void drwnLogger::logMessage(drwnLogLevel level, const string& msg)
{
    if (level > _logLevel) return;

    char prefix[4] = "---";
    switch (level) {
      case DRWN_LL_FATAL:   prefix[1] = '*'; break;
      case DRWN_LL_ERROR:   prefix[1] = 'E'; break;
      case DRWN_LL_WARNING: prefix[1] = 'W'; break;
      case DRWN_LL_MESSAGE: prefix[1] = '-'; break;
      case DRWN_LL_STATUS:  prefix[1] = '-'; break;
      case DRWN_LL_VERBOSE: prefix[1] = '-'; break;
      case DRWN_LL_METRICS: prefix[1] = '-'; break;
      case DRWN_LL_DEBUG:   prefix[1] = 'D'; break;
    }

    // write message to file
    if (_log.is_open() && (level != DRWN_LL_STATUS)) {
        time_t t = time(NULL);
        struct tm *lt = localtime(&t);

        _log << "-" << setw(2) << setfill('0') << lt->tm_hour
             << ":" << setw(2) << setfill('0') << lt->tm_min
             << ":" << setw(2) << setfill('0') << lt->tm_sec
             << prefix << ' ' << msg << "\n";
    }

    // display the message
    if ((_lastMessageLevel == DRWN_LL_STATUS) && (level != DRWN_LL_STATUS)) {
        cerr << "\n";
    }

    switch (level) {
    case DRWN_LL_FATAL:
        if (showFatalCallback == NULL) {
            cerr << prefix << ' ' << msg << endl;
        } else {
            showFatalCallback(msg.c_str());
        }
        break;
    case DRWN_LL_ERROR:
        if (showErrorCallback == NULL) {
            cerr << prefix << ' ' << msg << endl;
        } else {
            showErrorCallback(msg.c_str());
        }
        break;
    case DRWN_LL_WARNING:
        if (showWarningCallback == NULL) {
            cerr << prefix << ' ' << msg << endl;
        } else {
            showWarningCallback(msg.c_str());
        }
        break;
    case DRWN_LL_STATUS:
        if (showStatusCallback == NULL) {
            cerr << prefix << ' ' << msg << '\r';
            cerr.flush();
        } else {
            showStatusCallback(msg.c_str());
        }
        break;
    default:
        if (showMessageCallback == NULL) {
            cout << prefix << ' ' << msg << endl;
        } else {
            showMessageCallback(msg.c_str());
        }
    }

    _lastMessageLevel = level;

    // abort on fatal errors
    if (level == DRWN_LL_FATAL) {
        abort();
    }
}

// drwnLoggerConfig ---------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnLogger
//! \b logLevel :: verbosity level. Can be one of (ERROR, WARNING,
//!    MESSAGE (default), STATUS, VERBOSE, METRICS or DEBUG)\n
//! \b logFile :: name of file for logging message

class drwnLoggerConfig : public drwnConfigurableModule {
public:
    drwnLoggerConfig() : drwnConfigurableModule("drwnLogger") { }
    ~drwnLoggerConfig() { }

    void usage(ostream &os) const {
        os << "      logLevel      :: verbosity level. Can be one of (ERROR, WARNING,\n"
           << "                       MESSAGE (default), STATUS, VERBOSE, METRICS or DEBUG)\n"
           << "      logFile       :: name of file for logging message\n";
    }

    void setConfiguration(const char *name, const char *value) {
        // log level
        if (!strcmp(name, "logLevel")) {
            if (!strcasecmp(value, "ERROR")) {
                drwnLogger::setLogLevel(DRWN_LL_ERROR);
            } else if (!strcasecmp(value, "WARNING")) {
                drwnLogger::setLogLevel(DRWN_LL_WARNING);
            } else if (!strcasecmp(value, "MESSAGE")) {
                drwnLogger::setLogLevel(DRWN_LL_MESSAGE);
            } else if (!strcasecmp(value, "STATUS")) {
                drwnLogger::setLogLevel(DRWN_LL_STATUS);
            } else if (!strcasecmp(value, "VERBOSE")) {
                drwnLogger::setLogLevel(DRWN_LL_VERBOSE);
            } else if (!strcasecmp(value, "METRICS")) {
                drwnLogger::setLogLevel(DRWN_LL_METRICS);
            } else if (!strcasecmp(value, "DEBUG")) {
                drwnLogger::setLogLevel(DRWN_LL_DEBUG);
            } else {
                DRWN_LOG_FATAL("invalid configuration value " << value << " for logLevel");
            }

        // log file
        } else if (!strcmp(name, "logFile")) {
            drwnLogger::initialize(value);

        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnLoggerConfig gLoggerConfig;
