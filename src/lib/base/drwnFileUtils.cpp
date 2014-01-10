/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFileUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <string>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
#include "win32/dirent.h"
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#define access _access
#else
#include <dirent.h>
#include <unistd.h>
#endif

#include "drwnConstants.h"
#include "drwnCompatibility.h"
#include "drwnLogger.h"
#include "drwnFileUtils.h"

using namespace std;

// build sorted list of filenames
vector<string> drwnDirectoryListing(const char *directory,
    const char *extension, bool bIncludeDir, bool bIncludeExt)
{
    DRWN_ASSERT(directory != NULL);

    vector<string> filenames;
    DIR *dir = opendir(directory);
    if (dir == NULL) {
        DRWN_LOG_ERROR("could not open directory " << directory);
        return filenames;
    }

    string prefix;
    if (bIncludeDir) {
        prefix = string(directory) + DRWN_DIRSEP;
    }
    
    int extLength = 0;
    if (extension != NULL) {
        extLength = strlen(extension);
    }

    struct dirent *e = readdir(dir);
    while (e != NULL) {
        // skip . and ..
        if (!strncmp(e->d_name, ".", 1) || !(strncmp(e->d_name, "..", 2))) {
            e = readdir(dir);
            continue;
        }
            
        bool bIncludeFile = false;
        if (extension == NULL) {
            bIncludeFile = true;
        } else {
            const char *p = strstr(e->d_name, extension);
            if ((p != NULL) && (*(p + strlen(extension)) == '\0')) {
                bIncludeFile = true;
            }
        }

        if (bIncludeFile) {
            string filename = string(e->d_name);
            if (!bIncludeExt) {
                filename.erase(filename.length() - extLength);
            }
            filenames.push_back(prefix + filename);
        }
            
        e = readdir(dir);
    }    
    closedir(dir);

    sort(filenames.begin(), filenames.end());
    return filenames;
}

vector<string> drwnDirectoryListing(const char *directory,
    const set<const char *>& extensions, bool bIncludeDir)
{
    vector<string> filenames;

    for (set<const char *>::const_iterator it = extensions.begin(); 
         it != extensions.end(); it++) {
        vector<string> names = drwnDirectoryListing(directory, *it, bIncludeDir, true);
        filenames.insert(filenames.end(), names.begin(), names.end());
    }

    sort(filenames.begin(), filenames.end());
    return filenames;
}

// current directory
string drwnGetCurrentDir()
{
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    char buffer[_MAX_PATH];
    getcwd(buffer, _MAX_PATH);
#else
    char buffer[PATH_MAX];
    getcwd(buffer, PATH_MAX);
#endif
    return string(buffer);
}

bool drwnChangeCurrentDir(const char *directoryPath)
{
    DRWN_ASSERT(directoryPath != NULL);

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    bool success = (SetCurrentDirectory(directoryPath) == TRUE);
#else
    bool success = (chdir(directoryPath) == 0);
#endif

    if (!success) {
        DRWN_LOG_WARNING("failed to change to directory \"" << directoryPath << "\"");
    }

    return success;
}

// create or remove directory
bool drwnCreateDirectory(const char *directoryPath)
{
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    bool success = (CreateDirectory(directoryPath, NULL) == TRUE);
#else
    bool success = (mkdir(directoryPath, 0777) == 0);
#endif

    if (!success) {
        DRWN_LOG_WARNING("failed to create directory \"" << directoryPath << "\"");
    }

    return success;
}

bool drwnRemoveDirectory(const char *directoryPath)
{
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    // recursive directory removal
    char szDir[MAX_PATH+1];  // +1 for the double null terminate
    SHFILEOPSTRUCT fos = {0};

    strncpy_s(szDir, directoryPath, MAX_PATH);
    int len = lstrlen(szDir);
    szDir[len+1] = 0; // double null terminate for SHFileOperation

    // delete the folder and everything inside
    fos.wFunc = FO_DELETE;
    fos.pFrom = szDir;
    fos.fFlags = FOF_SILENT | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_NOCONFIRMMKDIR;
    bool success = (SHFileOperation(&fos) == 0);
#else
    bool success = (remove(directoryPath) == 0);
#endif

    if (!success) {
        DRWN_LOG_WARNING("failed to remove directory \"" << directoryPath << "\"");
    }

    return success;
}

bool drwnRemoveFile(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    bool success = (DeleteFile(filename) == TRUE);
#else
    bool success = (unlink(filename) == 0);
#endif
    return success;
}

// directory utililities
bool drwnIsAbsolutePath(const char *path)
{
    DRWN_ASSERT(path != NULL);
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    return ((strlen(path) > 2) && (path[1] == ':'));
#else
    return (path[0] == '/');
#endif
}

bool drwnIsRelativePath(const char *path)
{
    DRWN_ASSERT(path != NULL);
    return !drwnIsAbsolutePath(path);
}

// count fields
int drwnCountFields(ifstream *ifs, char delimiter, bool bSkipRepeated)
{
    DRWN_ASSERT(ifs != NULL);

    int numFields = 1;

    unsigned p = ifs->tellg();
    char ch;
    bool lastCharDelimiter = true;
    while (!ifs->eof() && (ifs->peek() != '\n')) {
        ifs->read(&ch, 1);
        if (ch == delimiter) {
            if (!bSkipRepeated || !lastCharDelimiter)
                numFields += 1;
            lastCharDelimiter = true;
        } else {
            lastCharDelimiter = false;
        }
    }
    ifs->seekg(p, ios::beg);

    return numFields;
}

// read strings from a file
vector<string> drwnReadFile(const char *filename)
{
    ifstream ifs(filename);
    DRWN_ASSERT(!ifs.fail());

    vector<string> fileLines;
    while (!ifs.eof()) {
        string str;
        ifs >> str;
        if (ifs.fail()) break;
        if (str.empty()) continue;
        fileLines.push_back(str);
    }    
    ifs.close();

    return fileLines;
}

// read lines from a file
vector<string> drwnReadLines(const char *filename)
{
    ifstream ifs(filename);
    DRWN_ASSERT(!ifs.fail());

    vector<string> fileLines;
    while (!ifs.eof()) {
        string str;
        getline(ifs, str);
        if (ifs.fail()) break;
        if (str.empty()) continue;
        fileLines.push_back(str);
    }    
    ifs.close();

    return fileLines;
}

// read file into a single string
string drwnReadAll(const char *filename)
{
    string data;
    data.reserve(drwnFileSize(filename));

    ifstream ifs(filename);
    DRWN_ASSERT(!ifs.fail());

    while (!ifs.eof()) {
        string str;
        getline(ifs, str);
        if (ifs.fail()) break;
        if (str.empty()) continue;
        data += str;
    }    
    ifs.close();

    return data;
}

// count lines
int drwnCountLines(const char *filename)
{
    ifstream ifs(filename);
    DRWN_ASSERT(!ifs.fail());

    int lineCount = 0;
    while (!ifs.eof()) {
        string str;
        getline(ifs, str);
        if (ifs.fail()) break;
        if (str.empty()) continue;
        lineCount += 1;
    }    
    ifs.close();

    return lineCount;
}

// check for file or directory existance
bool drwnPathExists(const char *pathname)
{
    DRWN_ASSERT(pathname != NULL);
    return (access(pathname, 0) == 0);
}

bool drwnFileExists(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    FILE *f = fopen(filename, "r");
    if (f == NULL) return false;

    fclose(f);
    return true;
#else
    // first check for directory
    DIR *dir;
    if ((dir = opendir(filename)) != NULL) {
        closedir(dir);
        return false;
    }

    // next check read access on file    
    return (access(filename, R_OK) == 0);
#endif
}

bool drwnDirExists(const char *dirname)
{
    DRWN_ASSERT(dirname != NULL);

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
	const size_t n = strlen(dirname);
	DRWN_ASSERT(n > 0);
	if ((dirname[n - 1] == '\\') || (dirname[n - 1] == '/')) {
		char *buffer = new char[n + 1];
		strcpy(buffer, dirname);
		buffer[n - 1] = '\0';
		bool b = drwnDirExists(buffer);
		delete[] buffer;
		return b;
	}
#endif

	if (access(dirname, 0) == 0) {
        struct stat status;
        stat(dirname, &status);        
        if (status.st_mode & S_IFDIR)
            return true;
    }

    return false;
}

// file sizing and resizing
unsigned int drwnFileSize(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    struct stat status;
    if (stat(filename, &status) != 0) {
        return 0;
    }

    return status.st_size;
}

bool drwnFileResize(const char *filename, unsigned int size)
{
    DRWN_ASSERT(filename != NULL);

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
    int fh;
    bool success = (_sopen_s(&fh, filename, _O_RDWR | _O_CREAT, _SH_DENYNO, _S_IREAD | _S_IWRITE) == 0);
    DRWN_ASSERT(success);
    success = (_chsize(fh, size) == 0);
    _close( fh );
#else
    bool success = (truncate(filename, size) == 0);
#endif

    return success;
}

// directory size (excluding . and ..) or -1 if directory doesn't exist
int drwnDirSize(const char *dirname)
{
    int numEntities = 0;
    
    DIR *dir = opendir(dirname);
    if (dir == NULL) {
        return -1;
    }

    struct dirent *e = readdir(dir);
    while (e != NULL) {
        // skip . and ..
        if (!strncmp(e->d_name, ".", 1) || !(strncmp(e->d_name, "..", 2))) {
            e = readdir(dir);
            continue;
        }

        numEntities += 1;
        e = readdir(dir);
    }    
    closedir(dir);

    return numEntities;
}
