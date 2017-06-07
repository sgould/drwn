/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStrUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnStrUtils.h
** \anchor drwnStrUtils
** \brief Generic string utilities.
*/

#pragma once

#include <string>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <sstream>
#include <map>
#include <deque>
#include <stdlib.h>
#include <limits>

using namespace std;

// constants and globals ----------------------------------------------------

extern std::string DRWN_COL_SEP;  //!< column separator when printing tables
extern std::string DRWN_ROW_BEG;  //!< row beginning when printing tables
extern std::string DRWN_ROW_END;  //!< row ending when printing tables

// functions ----------------------------------------------------------------

//! Templated function to make conversion from simple data types like
//! \p int and \p double to strings easy for printing and debugging.
//! Avoids messy char buffers.
template<typename T>
std::string toString(const T& v);

template<typename T>
std::string toString(const std::vector<T>& v);

template<typename T>
std::string toString(const std::list<T>& v);

template<typename T>
std::string toString(const std::set<T>& v);

template<typename T>
std::string toString(const std::deque<T>& q);

template<typename T, typename U>
std::string toString(const std::pair<T, U>& p);

//! specialized toString() routine for name/value pairs
std::string toString(const map<std::string, std::string>& m);

namespace drwn {

    //! case-insensitive comparison of two strings
    int strNoCaseCompare(const std::string& A, const std::string& B);

    //! parses a string representation of a vector
    template<typename T>
    int parseString(const std::string& str, std::vector<T>& v);
};

    //@cond
    template<typename T, bool B>
    struct parseInfToken {
        static bool apply(const std::string& token, T& value);
    };
    //@endcond

namespace drwn {

    //! parses a string of the form "name=value ..." into (name, value) pairs
    map<string, string> parseNameValueString(std::string str);

    //! returns \b true if string matches one of "1", "true" or "yes"
    bool trueString(const std::string& str);

    //! pads \b str with \b padChar up to \b padLength size
    std::string padString(const std::string& str, int padLength,
        unsigned char padChar = '0');

    //! breaks long strings into rows
    std::list<string> breakString(const std::string& str, unsigned lineLength);

    //! trim leading and trailing spaces (modifies \b str inline and returns it)
    std::string& trim(std::string& str);

    //! replaces any occurrences in \b str of \b substr with \b rep
    string strReplaceSubstr(const string& str, const string& substr, const string& rep);

    //! inserts spaces into camelCase string \b str
    //! (e.g., \a strSpacifyCamelCase becomes "str spacify camel case")
    string strSpacifyCamelCase(const string& str);

    //! conversion of bytes into a pretty string with units
    string bytesToString(unsigned b);
    //! conversion of milliseconds into a pretty string with units
    string millisecondsToString(unsigned ms);

    //! returns a base filename with directory and extension stripped
    string strBaseName(const string &fullPath);
    //! returns a filename with directory stripped
    string strFilename(const string &fullPath);
    //! returns the directory for a full path (filename stripped)
    string strDirectory(const string &fullPath);
    //! returns the extension for a given filename
    string strExtension(const string &fullPath);
    //! replaces the extension of a given file with \p ext
    string strReplaceExt(const string &fullPath, const string &ext);
    //! returns the full path of a file with extension stripped off
    string strWithoutExt(const string &fullPath);
    //! strips all terminating directory separators from a path
    string strWithoutEndSlashes(const string &fullPath);
    //! returns the index of a file with format base<nnn>.ext
    int strFileIndex(const string &fullPath);
};

// Implementation -----------------------------------------------------------

template<typename T>
std::string toString(const T& v)
{
    std::stringstream s;
    s << v;
    return s.str();
}

template<typename T>
std::string toString(const std::vector<T>& v)
{
    std::stringstream s;
    for (unsigned i = 0; i < v.size(); i++) {
        s << " " << v[i];
    }
    return s.str();
}

template<typename T>
std::string toString(const std::list<T>& v)
{
    std::stringstream s;
    for (typename std::list<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        if (it != v.begin()) s << " ";
        s << *it;
    }
    return s.str();
}

template<typename T>
std::string toString(const std::set<T>& v)
{
    std::stringstream s;
    s << "{";
    for (typename std::set<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        s << " " << *it;
    }
    s << " }";
    return s.str();
}

template<typename T>
std::string toString(const std::deque<T>& q)
{
    std::stringstream s;
    for (typename std::deque<T>::const_iterator it = q.begin(); it != q.end(); ++it) {
        s << " " << *it;
    }
    return s.str();
}

template<typename T, typename U>
std::string toString(const std::pair<T, U>& p)
{
    std::stringstream s;
    s << "(" << p.first << ", " << p.second << ")";
    return s.str();
}

// Conversion from a string
template<typename T>
int drwn::parseString(const std::string& str, std::vector<T>& v)
{
    std::stringstream buffer;
    T data;
    int count;

    buffer << str;

    count = 0;
    while (1) {
        int lastPosition = buffer.tellg();
        buffer >> data;
        if (buffer.fail()) {
            // try to parse special token
            buffer.clear();
            buffer.seekg(lastPosition, ios::beg);
            string token;
            buffer >> token;
            if (!parseInfToken<T, numeric_limits<T>::has_infinity>::apply(token, data)) {
                break;
            }
        }
        v.push_back(data);
        count++;

        if (buffer.eof()) break;
    }

    return count;
}

//@cond
template<typename T>
struct parseInfToken<T, true>  {
    static bool apply(const std::string& token, T& value) {
        if (token.compare("-inf") == 0) {
            value = -numeric_limits<T>::infinity();
        } else if (token.compare("inf") == 0) {
            value = numeric_limits<T>::infinity();
        } else {
            return false;
        }

        return true;
    }
};

template<typename T>
struct parseInfToken<T, false>  {
    static bool apply(const std::string& token, T& value) {
        return false;
    }
};
//@condend
