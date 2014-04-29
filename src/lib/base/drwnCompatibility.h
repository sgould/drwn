/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCompatibility.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnCompatibility.h
** \anchor drwnCompatibility
** \brief Windows/linux compatibility layer.
*/

#pragma once

// Microsoft Visual Studio (Win32)
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||defined(__VISUALC__)
#pragma warning(disable: 4018) // signed/unsigned mismatch
#pragma warning(disable: 4267) // conversion from size_t to int
#pragma warning(disable: 4244) // conversion from double to float
#pragma warning(disable: 4355) // this used in base member constructor
#pragma warning(disable: 4996) // unsafe functions
#define __PRETTY_FUNCTION__ __FUNCTION__
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE 1
#endif
#define _USE_MATH_DEFINES
#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max
#define strcasecmp _stricmp
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
#ifndef isnan
#define isnan(x) (_isnan(x))
#endif
#ifndef isinf
#define isinf(x) (!_finite(x))
#endif
#ifndef isfinite
#define isfinite(x) (_finite(x))
#endif
#define drand48() ((double) rand() / (double)(RAND_MAX + 1))
#define srand48(seed) (srand(seed))
#define round(x) (((x) < 0) ? ceil((x)-0.5) : floor((x)+0.5))
#define DRWN_DIRSEP '\\'
#endif

// Linux
#if defined(__LINUX__)
#define DRWN_DIRSEP '/'
#endif

// Mac OS X
#if defined(__APPLE__)
#define DRWN_DIRSEP '/'
#endif
