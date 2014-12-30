/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLogger.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#define DRWN_LOG(L, M) \
  if ((L) > drwnLogger::getLogLevel()) { } \
  else { std::stringstream __s; if (L == DRWN_LL_FATAL) { __s << "(" << __FILE__ << ", " << __LINE__ << ") "; } \
      __s << M; drwnLogger::logMessage(L, __s.str()); }

#define DRWN_LOG_ONCE(L, M) \
  static bool _drwnLoggedMessage ## __LINE__ = false; \
  if (((L) > drwnLogger::getLogLevel()) || _drwnLoggedMessage ## __LINE__) { } \
  else { std::stringstream __s; if (L == DRWN_LL_FATAL) { __s << "(" << __FILE__ << ", " << __LINE__ << ") "; } \
      __s << M; drwnLogger::logMessage(L, __s.str()); _drwnLoggedMessage ## __LINE__ = true;}

#define DRWN_LOG_FATAL(M)   DRWN_LOG(DRWN_LL_FATAL, M)
#define DRWN_LOG_ERROR(M)   DRWN_LOG(DRWN_LL_ERROR, M)
#define DRWN_LOG_WARNING(M) DRWN_LOG(DRWN_LL_WARNING, M)
#define DRWN_LOG_MESSAGE(M) DRWN_LOG(DRWN_LL_MESSAGE, M)
#define DRWN_LOG_STATUS(M)  DRWN_LOG(DRWN_LL_STATUS, M)
#define DRWN_LOG_VERBOSE(M) DRWN_LOG(DRWN_LL_VERBOSE, M)
#define DRWN_LOG_METRICS(M) DRWN_LOG(DRWN_LL_METRICS, M)
#define DRWN_LOG_DEBUG(M)   DRWN_LOG(DRWN_LL_DEBUG, M)

#define DRWN_LOG_ERROR_ONCE(M)   DRWN_LOG_ONCE(DRWN_LL_ERROR, M)
#define DRWN_LOG_WARNING_ONCE(M) DRWN_LOG_ONCE(DRWN_LL_WARNING, M)
#define DRWN_LOG_MESSAGE_ONCE(M) DRWN_LOG_ONCE(DRWN_LL_MESSAGE, M)
#define DRWN_LOG_VERBOSE_ONCE(M) DRWN_LOG_ONCE(DRWN_LL_VERBOSE, M)
#define DRWN_LOG_METRICS_ONCE(M) DRWN_LOG_ONCE(DRWN_LL_METRICS, M)
#define DRWN_LOG_DEBUG_ONCE(M)   DRWN_LOG_ONCE(DRWN_LL_DEBUG, M)

#define DRWN_ASSERT(C) \
  if (!(C)) { DRWN_LOG(DRWN_LL_FATAL, #C); }

#define DRWN_ASSERT_MSG(C, M) \
  if (!(C)) { DRWN_LOG(DRWN_LL_FATAL, #C << ": " << M); }

#define DRWN_TODO DRWN_LOG_FATAL("not implemented yet");

#define DRWN_DEPRECATED(C) \
    {DRWN_LOG_WARNING_ONCE("deprecated code (" << __FILE__ << ", " << __LINE__ << ")"); C;}

#define DRWN_START_PROGRESS(S, M) drwnLogger::initProgress(S, M)
#define DRWN_INC_PROGRESS         if (!drwnLogger::incrementProgress()) break;
#define DRWN_END_PROGRESS         drwnLogger::initProgress()
#define DRWN_SET_PROGRESS(P)      drwnLogger::updateProgress(P)

// drwnLogLevel ---------------------------------------------------------------
//! Verbosity level in logging.

typedef enum _drwnLogLevel {
    DRWN_LL_FATAL = 0,  //!< An unrecoverable error has occurred and the code will terminate
    DRWN_LL_ERROR,      //!< A recoverable error has occurred, e.g., a missing file.
    DRWN_LL_WARNING,    //!< Something unexpected happened, e.g., a parameter is zero.
    DRWN_LL_MESSAGE,    //!< Standard messages, e.g., application-level progress information.
    DRWN_LL_STATUS,     //!< Status messages, e.g., image names and sizes during loading.
    DRWN_LL_VERBOSE,    //!< Verbose messages, e.g., intermediate performance results.
    DRWN_LL_METRICS,    //!< Metrics messages, e.g., detailed process statistics.
    DRWN_LL_DEBUG	//!< Debugging messages, e.g., matrix inversion results, etc.
} drwnLogLevel;

// drwnLogger -----------------------------------------------------------------
//! Message and error logging. This class is not thread-safe in the interest
//! of not having to flush the log on every message.
//!
//! \sa \ref drwnLoggerDoc
//!
class drwnLogger
{
 public:
    //! callback for fatal errors
    static void (*showFatalCallback)(const char *message);
    //! callback for non-fatal errors
    static void (*showErrorCallback)(const char *message);
    //! callback for warnings
    static void (*showWarningCallback)(const char *message);
    //! callback for status updates
    static void (*showStatusCallback)(const char *message);
    //! callback for messages (standard, verbose and debug)
    static void (*showMessageCallback)(const char *message);

    //! callback for displaying progress
    static void (*showProgressCallback)(const char *status, double p);

 private:
    static ofstream _log;
    static drwnLogLevel _logLevel;
    static drwnLogLevel _lastMessageLevel;
    static string _cmdLineString;

    static string _progressStatus;
    static int _lastProgress;
    static int _progressLimit;
    static bool _bRunning;

 public:
    drwnLogger();
    ~drwnLogger();

    //! open a log file
    static void initialize(const char *filename,
        bool bOverwrite = false);
    //! open a log file with particular logging level
    static void initialize(const char *filename,
        bool bOverwrite, drwnLogLevel level);
    //! set the current verbosity level
    static inline void setLogLevel(drwnLogLevel level) {
        _logLevel = level;
    }
    //! set the current verbosity level unless already at a lower level
    //! and return previous log level
    static inline drwnLogLevel setAtLeastLogLevel(drwnLogLevel level) {
        const drwnLogLevel lastLevel = _logLevel;
        if (_logLevel < level) _logLevel = level;
        return lastLevel;
    }
    //! get the current verbosity level
    static inline drwnLogLevel getLogLevel() {
        return _logLevel;
    }
    //! check whether current verbosity level is above given level
    static inline bool checkLogLevel(drwnLogLevel level) {
        return (_logLevel >= level);
    }

    //! save the command line string (will be logged when the log file is opened)
    static void cacheCommandLine(int argc, char **argv);

    //! log a message (see also \p DRWN_LOG_* macros)
    static void logMessage(drwnLogLevel level, const string& msg);

    // progress feedback
    static void setRunning(bool bRunning) {
        _bRunning = bRunning;
    }
    static bool isRunning() {
        return _bRunning;
    }
    static void initProgress(const char *status = NULL, int maxProgress = 100) {
        _progressStatus = status == NULL ? string("") : string(status);
        _lastProgress = 0; _progressLimit = maxProgress; updateProgress(0.0);
    }
    static void setProgressStatus(const char *status) {
        if (status == NULL) {
            _progressStatus.clear();
        } else {
            _progressStatus = string(status);
        }
        if (showProgressCallback != NULL) {
            showProgressCallback(_progressStatus.c_str(),
                (double)_lastProgress / (double)_progressLimit);
        }
    }
    static bool incrementProgress() {
        if (_lastProgress < _progressLimit) {
            _lastProgress += 1;
            updateProgress((double)_lastProgress / (double)_progressLimit);
        }
        return _bRunning;
    }
    static bool updateProgress(double p) {
        if (showProgressCallback != NULL) {
            showProgressCallback(_progressStatus.c_str(), p);
        }
        return _bRunning;
    }
};
