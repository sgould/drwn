/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCodeProfiler.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "drwnCompatibility.h"
#include "drwnConfigManager.h"
#include "drwnLogger.h"
#include "drwnCodeProfiler.h"

using namespace std;

// drwnCodeProfiler -----------------------------------------------------------

bool drwnCodeProfiler::enabled = false;

vector<drwnCodeProfiler::drwnCodeProfilerEntry> drwnCodeProfiler::_entries;
map<string, int> drwnCodeProfiler::_names;

int drwnCodeProfiler::getHandle(const char *name)
{
    if (!enabled) return -1;

    // if entry exists return it
    map<string, int>::const_iterator i;
    if ((i = _names.find(string(name))) != _names.end()) {
        return i->second;
    }

    // otherwise create a new entry for profiling
    int handle = (int)_entries.size();
    _names[string(name)] = handle;
    _entries.push_back(drwnCodeProfiler::drwnCodeProfilerEntry());

    return handle;
}

void drwnCodeProfiler::print()
{
    if (!enabled || _entries.empty())
        return;

    DRWN_LOG_MESSAGE("  CALLS        CPU TIME   WALL TIME      TIME PER   FUNCTION");
    for (map<string, int>::const_iterator it = _names.begin(); it != _names.end(); it++) {
	int handle = it->second;
        std::stringstream s;
        s << setw(7) << setfill(' ') << _entries[handle].totalCalls << "   ";

	// cpu time
	double mseconds = 1000.0 * (double)_entries[handle].totalClock / (double)CLOCKS_PER_SEC;
	int seconds = (int)floor(mseconds / 1000.0);
	mseconds -= 1000.0 * seconds;
	int minutes = (int)floor(seconds / 60.0);
	seconds -= 60 * minutes;
	int hours = (int)floor(minutes / 60.0);
	minutes -= 60 * hours;
	s << setw(3) << setfill(' ') << hours << ":"
          << setw(2) << setfill('0') << minutes << ":"
          << setw(2) << setfill('0') << seconds << "."
          << setw(3) << setfill('0') << (int)mseconds << "   ";

	// wall time
	seconds =  (int) _entries[handle].totalTime;
	minutes = (int)floor(seconds / 60.0);
	seconds -= 60 * minutes;
	hours = (int)floor(minutes / 60.0);
	minutes -= 60 * hours;
	s << setw(3) << setfill(' ') << hours << ":"
          << setw(2) << setfill('0') << minutes << ":"
          << setw(2) << setfill('0') << seconds << "   ";

	// time per method
	s << " ";
	if ( _entries[handle].totalCalls == 0 ) {
            s << "      0  s   ";
	} else {
            float time_per_s = (float)_entries[handle].totalClock / (float)CLOCKS_PER_SEC / float(_entries[handle].totalCalls);
            if ( time_per_s < 1e-6 ) {
                // Print in nanoseconds.
                s << setw(3) << setfill(' ') << int(1e9*time_per_s) << "."
                  << setw(3) << setfill('0') << int(1e12*time_per_s)%1000 << " ns   ";
            } else if ( time_per_s < 1e-3 ) {
                // Print in microseconds.
                s << setw(3) << setfill(' ') << int(1e6*time_per_s) << "."
                  << setw(3) << setfill('0') << int(1e9*time_per_s)%1000 << " us   ";
            } else if ( time_per_s < 1.0 ) {
                // Print in milliseconds.
                s << setw(3) << setfill(' ') << int(1e3*time_per_s) << "."
                  << setw(3) << setfill('0') << int(1e6*time_per_s)%1000 << " ms   ";
            } else if ( time_per_s < 1e3 ) {
                // Print in seconds.
                s << setw(3) << setfill(' ') << int(time_per_s) << "."
                  << setw(3) << setfill('0') << int(1e3*time_per_s)%1000 << "  s   ";
            } else {
                // Print saturated.
                s << "   >999  s   ";
            }
	}

        DRWN_LOG_MESSAGE(s.str() << it->first.c_str());
    }
}

// drwnCodeProfilerConfig ---------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnCodeProfiler
//! \b enabled :: enable profiling (default: false)

class drwnCodeProfilerConfig : public drwnConfigurableModule {
public:
    drwnCodeProfilerConfig() : drwnConfigurableModule("drwnCodeProfiler") { }
    ~drwnCodeProfilerConfig() { }

    void usage(ostream &os) const {
        os << "      enabled       :: enable profiling (default: false)\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "enabled")) {
            drwnCodeProfiler::enabled =
                (!strcasecmp(value, "TRUE") || !strcmp(value, "1"));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnCodeProfilerConfig gCodeProfilerConfig;
