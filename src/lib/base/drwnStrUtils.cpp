/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStrUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <string>
#include <iostream>
#include <sstream>
#if !(defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||(__VISUALC__))
#include <strings.h>
#endif

#include "drwnCompatibility.h"
#include "drwnLogger.h"
#include "drwnStrUtils.h"

using namespace std;

// constants and globals ----------------------------------------------------

string DRWN_COL_SEP("\t");
string DRWN_ROW_BEG("\t");
string DRWN_ROW_END("");

// functions ----------------------------------------------------------------

// toString() routines.
string toString(const map<string, string>& p)
{
    std::stringstream s;
    for (map<string, string>::const_iterator i = p.begin();
	 i != p.end(); i++) {
	if (i != p.begin()) {
	    s << ", ";
	}
	s << i->first << "=" << i->second;
    }
    return s.str();
}

// Case insensitive comparison
int drwn::strNoCaseCompare(const string& A, const string& B)
{
    string::const_iterator itA = A.begin();
    string::const_iterator itB = B.begin();

    while ((itA != A.end()) && (itB != B.end())) {
        if (::toupper(*itA) != ::toupper(*itB))
            return (::toupper(*itA)  < ::toupper(*itB)) ? -1 : 1;
        ++itA;
        ++itB;
    }

    if (A.size() == B.size())
        return 0;
    return (A.size() < B.size()) ? -1 : 1;
}

// Function to break string of "<name>\s*=\s*<value>[,; ]" pairs
// into an stl map. If the value part does not exist then sets to
// "true".
map<string, string> drwn::parseNameValueString(string str)
{
    // first tokenize into <name>=<value> pairs
    vector<string> tokens;
    string::size_type lastPos = str.find_first_not_of(" ", 0);
    string::size_type pos = str.find_first_of(",; ", lastPos);

    while ((string::npos != pos) || (string::npos != lastPos)) {
        // found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // skip delimiters
        lastPos = str.find_first_not_of(",; ", pos);
        // find next "non-delimiter"
        pos = str.find_first_of(",; ", lastPos);
    }

    // now break tokens into name and value pairs
    map<string, string> output;
    for (unsigned i = 0; i < tokens.size(); i++) {
	pos = tokens[i].find_first_of("=", 0);
	if (pos != string::npos) {
	    output[tokens[i].substr(0, pos)] =
		tokens[i].substr(pos + 1, tokens[i].length() - pos);
	} else {
	    output[tokens[i]] = "true";
	}
    }

    return output;
}

bool drwn::trueString(const string& str) {
    return ((!strcasecmp(str.c_str(), "true")) ||
        (!strcasecmp(str.c_str(), "yes")) ||
        (!strcasecmp(str.c_str(), "1")));
}

string drwn::padString(const string& str, int padLength,
                 unsigned char padChar)
{
    if (str.size() >= (unsigned)padLength) {
        return str;
    }

    string padString((unsigned)padLength - str.size(), padChar);
    return (padString + str);
}

list<string> drwn::breakString(const std::string& str, unsigned lineLength)
{
    DRWN_ASSERT(lineLength > 3);
    list<string> lines;

    vector<string> tokens;
    drwn::parseString<string>(str, tokens);

    // truncate strings that are too long
    for (unsigned i = 0; i < tokens.size(); i++) {
        if (tokens[i].size() > lineLength) {
            tokens[i].resize(lineLength - 3);
            tokens[i] += string("...");
        }
    }
    
    // concatenate strings together to form lines
    string currentLine;
    for (unsigned i = 0; i < tokens.size(); i++) {
        if (currentLine.size() + tokens[i].size() < lineLength) {
            if (!currentLine.empty()) currentLine += string(" ");
            currentLine += tokens[i];
        } else if (tokens[i].size() <= lineLength) {
            lines.push_back(currentLine);
            currentLine = tokens[i];
        } else {
            DRWN_LOG_FATAL("error truncating token " << tokens[i]);
        }
    }

    if (!currentLine.empty()) {
        lines.push_back(currentLine);
    }

    return lines;
}

string& drwn::trim(string& str) {
    str.erase(str.find_last_not_of(" ") + 1);
    return str.erase(0, str.find_first_not_of(" "));
}

string drwn::strReplaceSubstr(const string & str, const string & substr, const string & rep)
{
  string rval;
  size_t searchPos = 0, prevPos;

  while ((searchPos != string::npos) && (searchPos < str.size())) {
      prevPos = searchPos;
      searchPos = str.find(substr, searchPos);
      if (searchPos == string::npos) {
          rval += str.substr(prevPos);
      } else {
	  rval += str.substr(prevPos, searchPos - prevPos);
	  rval += rep;

	  searchPos += substr.length();
      }
  }

  return rval;
}

string drwn::strSpacifyCamelCase(const string& str)
{
    string newStr;
    newStr.reserve(str.length());
    string::const_iterator it = str.begin();

    bool bNeedSpace = false;
    while (it != str.end()) {
        string::const_iterator jt = it++;
        if (bNeedSpace || ((it != str.end()) && (jt != str.begin()) && 
                isupper(*jt) && islower(*it))) {
            newStr.push_back(' ');
        }
        newStr.push_back(*jt);
        bNeedSpace = (it != str.end()) && (islower(*jt) && isupper(*it));
    }

    return newStr;
}

// Conversion of units to strings
string drwn::bytesToString(unsigned b)
{
    unsigned minorUnits = 0;
    unsigned majorUnits = b;
    unsigned nSteps = 0;
    while (majorUnits > 1000) {
        minorUnits = majorUnits % 1000;
        majorUnits /= 1000;
        nSteps += 1;
    }

    string units;
    switch (nSteps) {
      case 0: units = string("B "); break;
      case 1: units = string("kB"); break;
      case 2: units = string("MB"); break;
      case 3: units = string("GB"); break;
      case 4: units = string("TB"); break;
      case 5: units = string("PB"); break;
      default: units = string("?B");
    }

    return toString(majorUnits) + '.' + toString(minorUnits / 100) + units;
}

string drwn::millisecondsToString(unsigned ms)
{
    int seconds = ms / 1000;
    ms -= 1000 * seconds;
    int minutes = seconds / 60;
    seconds -= 60 * minutes;
    int hours = minutes / 60;
    minutes -= 60 * hours;

    char buffer[11];
    sprintf(buffer, ":%02d:%02d.%03d", minutes, seconds, ms);
    DRWN_ASSERT(buffer[10] == '\0');

    return toString(hours) + string(buffer);
}

string drwn::strBaseName(const string &fullPath)
{
    string baseName;

    // strip directory name
    string::size_type pos = fullPath.find_last_of("/\\");
    if (pos == string::npos) {
	baseName = fullPath;
    } else {
	baseName = fullPath.substr(pos + 1, fullPath.length() - pos);
    }

    // strip extension
    return drwn::strWithoutExt(baseName);
}

string drwn::strFilename(const string &fullPath)
{
    // strip directory name
    string::size_type pos = fullPath.find_last_of("/\\");
    if (pos == string::npos) {
	return fullPath;
    }

    return fullPath.substr(pos + 1, fullPath.length() - pos);
}

string drwn::strDirectory(const string &fullPath)
{
    string::size_type pos = fullPath.find_last_of("/\\");
    if (pos == string::npos) {
	return string(".");
    }

    return fullPath.substr(0, pos);
}

string drwn::strExtension(const string &fullPath)
{
    string filename = drwn::strFilename(fullPath);
    string::size_type pos = filename.find_last_of(".");
    if (pos != string::npos) {
	return filename.substr(pos + 1, filename.length() - pos);
    }

    return string("");
}

string drwn::strReplaceExt(const string &fullPath, const string &ext)
{
    string oldExt = drwn::strExtension(fullPath);
    size_t len = oldExt.length() == 0 ? 0 : oldExt.length() + 1;
    return (fullPath.substr(0, fullPath.length() - len) + ext);
}

string drwn::strWithoutExt(const string &fullPath)
{
  string filename = fullPath;

  // strip extension
  string::size_type pos = filename.find_last_of(".");
  if (pos != string::npos) {
    filename = filename.substr(0, pos);
  }
  return filename;
}

string drwn::strWithoutEndSlashes(const string &fullPath)
{
  string filename = fullPath;
  string::size_type pos = filename.find_last_not_of("/");
  return filename.substr(0, pos+1);
}

// Returns index from filenames with the form <base><index>.<ext>
int drwn::strFileIndex(const string &fullPath)
{
    string baseName = drwn::strBaseName(fullPath);
    string::size_type ib = baseName.find_first_of("0123456789");
    string::size_type ie = baseName.find_last_of("0123456789");

    if ((ib == string::npos) || (ie == string::npos)) {
	return -1;
    }

    string index = baseName.substr(ib, ie - ib + 1);
    return atoi(index.c_str());
}

