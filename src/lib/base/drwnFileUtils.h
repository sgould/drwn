/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFileUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnFileUtils.h
** \anchor drwnFileUtils
** \brief File and directory processing utilities.
*/

#pragma once

#include <set>
#include <string>
#include <string.h>
#include <vector>
#include <fstream>

using namespace std;

//! returns a sorted list of filenames in \b directory filtered
//! by \b extension
vector<string> drwnDirectoryListing(const char *directory,
    const char *extension = NULL, bool bIncludeDir = true,
    bool bIncludeExt = true);

//! returns a sorted list of filenames in \b directory with
//! extensions matching one of \b extensions
vector<string> drwnDirectoryListing(const char *directory,
    const set<const char *>& extensions, bool bIncludeDir = true);

//! returns the current directory path
string drwnGetCurrentDir();
//! changes the current directory (returns \b true if successful)
bool drwnChangeCurrentDir(const char *directoryPath);

//! creates a new directory path
bool drwnCreateDirectory(const char *directoryPath);
//! removes a directory and its contents
bool drwnRemoveDirectory(const char *directoryPath);

//! removes a file
bool drwnRemoveFile(const char *filename);

//! returns \b true if the directory path is absolute (starts from root)
bool drwnIsAbsolutePath(const char *path);
//! returns \b true if the directory path is relative to the current directory
bool drwnIsRelativePath(const char *path);

//! counts the number of fields per line (separated by a single character delimiter)
int drwnCountFields(ifstream *ifs, char delimiter = ' ', bool bSkipRepeated = true);

//! read strings (separated by whitespace) from a file
vector<string> drwnReadFile(const char *filename);
//! read complete lines from a file
vector<string> drwnReadLines(const char *filename);
//! read complete file into a single string
string drwnReadAll(const char *filename);
//! counts the number of non-empty lines in a text file
int drwnCountLines(const char *filename);

//! checks if a path (directory or file) exists
bool drwnPathExists(const char *pathname);
//! checks if a file exists
bool drwnFileExists(const char *filename);
//! checks if a directory exists
bool drwnDirExists(const char *dirname);

//! returns the size of a file in bytes
unsigned int drwnFileSize(const char *filename);

//! resizes a file to \b size bytes
bool drwnFileResize(const char *filename, unsigned int size);

//! returns the number of files and subdirectories in \b directory
//! (excluding . and ..)
int drwnDirSize(const char *dirname);
