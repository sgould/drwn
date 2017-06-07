/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCommandLine.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnCommandLine.h
** \anchor drwnCommandLine
** \brief Command line processing macros. Applications should use these
**  macros to present a consistent interface.
**
**  The following is an example usage
**  \code
**  int main(int argc, char* argv[])
**  {
**     int integerVariable = 0;
**     const char *stringVariable = NULL;
**
**     DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
**         DRWN_CMDLINE_INT_OPTION("-intgerOption", integerVariable)
**         DRWN_CMDLINE_STR_OPTION("-stringOption", stringVariable)
**         DRWN_CMDLINE_OPTION_BEGIN("-longOption", p)
**             cerr << p[0] << "\n";
**             cerr << p[1] << "\n";
**         DRWN_CMDLINE_OPTION_END(2)
**     DRWN_END_CMDLINE_PROCESSING();
**
**     return 0;
**  }
**  \endcode
**
**  \sa \ref drwnCommandLineDoc
*/

#pragma once

#define DRWN_STANDARD_OPTIONS_USAGE                           \
    "  -help             :: display application usage\n"      \
    "  -config <xml>     :: configure Darwin from XML file\n" \
    "  -set <m> <n> <v>  :: set (configuration) <m>::<n> to value <v>\n" \
    "  -profile          :: profile code\n"                   \
    "  -quiet            :: only show warnings and errors\n"  \
    "  -verbose          :: show verbose messages\n"          \
    "  -debug            :: show debug messages\n"            \
    "  -log <filename>   :: log filename\n"                   \
    "  -threads <max>    :: set maximum number of threads\n"  \
    "  -randseed <n>     :: seed random number generators rand and drand48\n"

#define DRWN_PROCESS_STANDARD_OPTIONS(ARGS, ARGC)             \
    if (!strcmp(*ARGS, "-config")) {                          \
        if (ARGC == 1) {                                      \
            drwnConfigurationManager::get().showRegistry(true); \
            return 0;                                         \
        }                                                     \
        drwnConfigurationManager::get().configure(*(++ARGS)); \
        ARGC -= 1;                                            \
    } else if (!strcmp(*ARGS, "-set")) {                      \
        if (ARGC == 1) {                                      \
            drwnConfigurationManager::get().showRegistry(false); \
            return 0;                                         \
        }                                                     \
        if (ARGC == 2) {                                      \
            drwnConfigurationManager::get().showModuleUsage(ARGS[1]); \
            return 0;                                         \
        }                                                     \
        DRWN_ASSERT_MSG(ARGC > 3, "not enough arguments for -set"); \
        drwnConfigurationManager::get().configure(ARGS[1], ARGS[2], ARGS[3]); \
        ARGC -= 3; ARGS += 3;                                 \
    } else if (!strcmp(*ARGS, "-profile")) {                  \
        drwnCodeProfiler::enabled = true;                     \
    } else if (!strcmp(*ARGS, "-quiet")) {                    \
        drwnLogger::setLogLevel(DRWN_LL_WARNING);             \
    } else if (!strcmp(*ARGS, "-verbose") || !strcmp(*ARGS, "-v")) { \
        if (drwnLogger::getLogLevel() < DRWN_LL_VERBOSE)      \
            drwnLogger::setLogLevel(DRWN_LL_VERBOSE);         \
    } else if (!strcmp(*ARGS, "-debug")) {                    \
        drwnLogger::setLogLevel(DRWN_LL_DEBUG);               \
    } else if (!strcmp(*ARGS, "-log")) {                      \
        drwnLogger::initialize(*(++ARGS));                    \
        ARGC -= 1;                                            \
    } else if (!strcmp(*ARGS, "-threads")) {                  \
        drwnThreadPool::MAX_THREADS = atoi(*(++ARGS));        \
        ARGC -= 1;                                            \
    } else if (!strcmp(*ARGS, "-randseed")) {                 \
        const unsigned n = atoi(*(++ARGS));                   \
        srand(n); srand48(n);                                 \
        ARGC -= 1;                                            \
    }

#define DRWN_BEGIN_CMDLINE_PROCESSING(ARGC, ARGV)              \
    drwnLogger::cacheCommandLine(ARGC, ARGV);                  \
    char **_drwn_args = ARGV + 1;                              \
    int _drwn_argc = ARGC;                                     \
    while (--_drwn_argc > 0) {                                 \
        DRWN_LOG_DEBUG("processing cmdline arg " << *_drwn_args << " (" << _drwn_argc << ")"); \
        DRWN_PROCESS_STANDARD_OPTIONS(_drwn_args, _drwn_argc) \

#define DRWN_CMDLINE_STR_OPTION(OPTSTR, VAR)                  \
    else if (!strcmp(*_drwn_args, OPTSTR)) {                  \
        VAR = *(++_drwn_args); _drwn_argc -= 1; }

#define DRWN_CMDLINE_INT_OPTION(OPTSTR, VAR)                  \
    else if (!strcmp(*_drwn_args, OPTSTR)) {                  \
        VAR = atoi(*(++_drwn_args)); _drwn_argc -= 1; }

#define DRWN_CMDLINE_REAL_OPTION(OPTSTR, VAR)                 \
    else if (!strcmp(*_drwn_args, OPTSTR)) {                  \
        VAR = atof(*(++_drwn_args)); _drwn_argc -= 1; }

#define DRWN_CMDLINE_BOOL_OPTION(OPTSTR, VAR)                 \
    else if (!strcmp(*_drwn_args, OPTSTR)) { VAR = true; }

#define DRWN_CMDLINE_BOOL_TOGGLE_OPTION(OPTSTR, VAR)          \
    else if (!strcmp(*_drwn_args, OPTSTR)) { VAR = !VAR; }

#define DRWN_CMDLINE_VEC_OPTION(OPTSTR, VAR)                  \
  else if (!strcmp(*_drwn_args, OPTSTR)) {                    \
    VAR.push_back(*(++_drwn_args)); _drwn_argc -= 1; }

#define DRWN_CMDLINE_OPTION_BEGIN(OPTSTR, PTR)                \
    else if (!strcmp(*_drwn_args, OPTSTR)) {                  \
        const char **PTR = (const char **)(_drwn_args + 1);

#define DRWN_CMDLINE_OPTION_END(N)                            \
        _drwn_args += (N); _drwn_argc -= (N); }

#define DRWN_CMDLINE_FLAG_BEGIN(OPTSTR)                       \
    else if (!strcmp(*_drwn_args, OPTSTR)) {

#define DRWN_CMDLINE_FLAG_END }

#define DRWN_END_CMDLINE_PROCESSING(USAGE)                    \
        else if (!strcmp(*_drwn_args, "-help")) {             \
            USAGE;                                            \
            return 0;                                         \
        } else if ((*_drwn_args)[0] == '-') {                 \
            USAGE;                                            \
	    DRWN_LOG_ERROR("unrecognized option " << *_drwn_args); \
	    return -1;                                        \
        } else {                                              \
            DRWN_LOG_DEBUG(_drwn_argc << " commandline arguments remaining"); \
            break;                                            \
        }                                                     \
        _drwn_args++;                                         \
    }

#define DRWN_CMDLINE_ARGV _drwn_args
#define DRWN_CMDLINE_ARGC _drwn_argc
