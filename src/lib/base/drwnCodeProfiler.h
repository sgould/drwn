/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCodeProfiler.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <ctime>
#include <vector>
#include <map>
#include <string>
#include <iostream>

// macro for function tic/toc
#define DRWN_FCN_TIC {drwnCodeProfiler::tic(drwnCodeProfiler::getHandle(__PRETTY_FUNCTION__));}
#define DRWN_FCN_TOC {drwnCodeProfiler::toc(drwnCodeProfiler::getHandle(__PRETTY_FUNCTION__));}

/*!
** \brief Static class for providing profile information on functions.
**
** drwnCodeProfiler is typically used to accumulate information on entire
** functions (or within subroutines). Wrap the function or code block in
** \code
**    drwnCodeProfiler::tic(handle = getHandle("functionName"));
**    drwnCodeProfiler::toc(handle);
** \endcode
** to accumulate timing and number of calls for a given function.
** The timer accumulates the amount of processor and real (wall clock)
** time used between \ref tic and \ref toc calls (child processes, such
** as file I/O, are not counted in this time). Processor times may be
** inaccurate for functions that take longer than about 1 hour.
** 
** By default profiling is turned off and must be enabled in \b main() with
** \code
**    drwnCodeProfiler::enabled = true;
** \endcode
** Most \b Darwin applications use the standard command line option \b
** -profile to enable profiling. Call the function \p
** drwnCodeProfiler::print() before exiting \b main() to log profiling
** information.
** 
** The code can also be used to setting time limits or recursive call limits
** within instrumented functions. Use \ref time and \ref calls to get the total
** running time or total number of calls for a given handle.
** 
** The macros \b DRWN_FCN_TIC and \b DRWN_FCN_TOC can be used at the entry and
** exit of your functions to instrument the entire function. Make sure you put
** \b DRWN_FCN_TOC \a before all \b return statements within the function.
** 
** \warning 
** Profiling provided by drwnCodeProfiler should be used as an estimate only.
** Specifically it is not accurate for small functions. In those cases you are
** better off using \b gprof and the \b "gcc -pg" option or Microsoft's Visual
** C++ profiling software.
** Note also that instrumenting code for profiling will unavoidably add a small
** overhead to a function's running time, so do not use \ref tic and \ref toc
** within tight loops. Unlike compiling with \b -pg for \b gprof, functions
** \ref tic and \ref toc are \a always compiled into the code.
*/

class drwnCodeProfiler {
public:
    static bool enabled;  //!< set \b true to enable profiling

private:
    //@cond
    class drwnCodeProfilerEntry {
    public:
        clock_t startClock;
	time_t startTime;
	unsigned long totalClock;
        double totalTime;
        int totalCalls;

        drwnCodeProfilerEntry() { clear(); };
        ~drwnCodeProfilerEntry() { /* do nothing */ };

        inline void clear() {
            startClock = clock();
            totalClock = 0L;

	    startTime = std::time(NULL);
	    totalTime = 0L;

            totalCalls = 0;
        }
        inline void tic() {
            startClock = clock();
	    startTime = std::time(NULL);
        }
        inline void toc() {
            clock_t endClock = clock();
	    time_t endTime;
	    endTime = std::time(NULL);

	    if (endClock >= startClock) {
		totalClock += (endClock - startClock);		
	    } else {
		totalClock += ((clock_t)(-1) - startClock) + endClock;
	    }
            startClock = endClock;
 
	    totalTime += difftime(endTime, startTime);
	    
            totalCalls += 1;
        }
    };
    //@endcond

    static std::vector<drwnCodeProfilerEntry> _entries;
    static std::map<std::string, int> _names;

public:
    drwnCodeProfiler() { /* do nothing */ }
    ~drwnCodeProfiler() { /* do nothing */ }

    //! return a \b handle for profiling a code block called \b name
    static int getHandle(const char *name);

    //! clear all profiling information for \b handle
    static inline void clear(int handle) {
        if (enabled) _entries[handle].clear();
    }
    //! starting timing execution of \b handle
    static inline void tic(int handle) {
        if (enabled) _entries[handle].tic();
    }
    //! stop timing execution of \b handle (and update number of calls)
    static inline void toc(int handle) {
        if (enabled) _entries[handle].toc();
    }
    //! return total real-world running time of \b handle (in seconds)
    static inline double walltime(int handle) {
        if (!enabled) return -1.0;
	return _entries[handle].totalTime;
    }
    //! return total CPU running time of \b handle (in seconds)
    static inline double time(int handle) {
	if (!enabled) return -1.0;
	return (double)_entries[handle].totalClock / (double)CLOCKS_PER_SEC;
    }
    //! return number of times \b handle has been profiled
    static inline int calls(int handle) {
	if (!enabled) return -1;
	return _entries[handle].totalCalls;
    }

    //! display profile information for all handles (to message logger)
    static void print();
};

