/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCompressionBuffer.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard library
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <limits>
#include <iomanip>

// Eigen library
#include "Eigen/Core"

// zlib library
#ifdef DRWN_HAS_ZLIB
#include "zlib.h"
#endif

// Darwin headers
#include "drwnBase.h"
#include "drwnCompressionBuffer.h"

using namespace std;
using namespace Eigen;

// globals and constants ----------------------------------------------------

#ifdef DRWN_DEBUG_STATISTICS
unsigned drwnCompressionBuffer::_dbRefCount = 0;
unsigned drwnCompressionBuffer::_dbOriginalBytes = 0;
unsigned drwnCompressionBuffer::_dbCompressedBytes = 0;
#endif

// drwnCompressionBuffer ----------------------------------------------------

drwnCompressionBuffer::drwnCompressionBuffer() :
    _data(NULL), _bytesAllocated(0), _bytesOriginal(0), _bytesCompressed(0)
{
#ifdef DRWN_DEBUG_STATISTICS
    _dbRefCount += 1;
#endif
}

drwnCompressionBuffer::~drwnCompressionBuffer()
{
    free();
#ifdef DRWN_DEBUG_STATISTICS
    _dbRefCount -= 1;
    if ((_dbRefCount == 0) && ((_dbCompressedBytes > 0) || (_dbOriginalBytes > 0))) {
        cerr << "drwnCompressionBuffer compressed " << drwn::bytesToString(_dbOriginalBytes)
             << " to " << drwn::bytesToString(_dbCompressedBytes) << "\n";
        cerr << "drwnCompressionBuffer compression ratio "
             << (double)_dbCompressedBytes / (double)_dbOriginalBytes << "\n";
    }
#endif
}

void drwnCompressionBuffer::compress(const unsigned char *data, unsigned int size)
{
    DRWN_ASSERT((data != NULL) && (size > 0));

#ifndef DRWN_HAS_ZLIB
    DRWN_LOG_WARNING_ONCE("zlib not installed: compression not implemented");
    if (size > _bytesAllocated) {
        free();
        _data = new unsigned char[size];
        _bytesAllocated = size;
    }
    _bytesOriginal = size;
    memcpy(_data, data, size);
    _bytesCompressed = size;
#else
    // allocate buffer
    uLong bufferLen = compressBound((uLong)size);
    if (bufferLen > _bytesAllocated) {
        free();
        _data = new unsigned char[bufferLen];
        _bytesAllocated = bufferLen;
    }

    // compress data
    _bytesOriginal = size;
    int result = ::compress((Bytef *)_data, &bufferLen, (const Bytef *)data, (uLong)size);
    DRWN_ASSERT(result == Z_OK);

    // update buffer size
    _bytesCompressed = bufferLen;
    DRWN_LOG_DEBUG("compressed " << _bytesOriginal << " bytes to "
        << _bytesCompressed << " (estimated " << _bytesAllocated << ")");
#endif

#ifdef DRWN_DEBUG_STATISTICS
    _dbOriginalBytes += _bytesOriginal;
    _dbCompressedBytes += _bytesCompressed;
#endif
}

void drwnCompressionBuffer::decompress(unsigned char *data) const
{
    DRWN_ASSERT((data != NULL) && (_bytesCompressed > 0));
    DRWN_LOG_DEBUG("decompressing " << _bytesCompressed << " into "
        << _bytesOriginal << " bytes");

#ifndef DRWN_HAS_ZLIB
    DRWN_LOG_WARNING_ONCE("zlib not installed: decompression not implemented");
    memcpy(data, _data, _bytesOriginal);
#else
    uLongf uncompressedBytes = _bytesOriginal;
    int result = uncompress((Bytef *)data, &uncompressedBytes,
        (const Bytef *)_data, (uLong)_bytesCompressed);
    DRWN_ASSERT(result == Z_OK);
    DRWN_ASSERT_MSG(uncompressedBytes == _bytesOriginal,
        "o: " << _bytesOriginal << ", c: " << _bytesCompressed << ", u: " << uncompressedBytes);
#endif
}

size_t drwnCompressionBuffer::numBytesOnDisk() const
{
    return (2 * sizeof(unsigned int) + _bytesCompressed * sizeof(unsigned char));
}

bool drwnCompressionBuffer::write(ostream& os) const
{
    os.write((char *)&_bytesOriginal, sizeof(unsigned int));
    os.write((char *)&_bytesCompressed, sizeof(unsigned int));
    if (_bytesCompressed > 0) {
        os.write((char *)_data, _bytesCompressed * sizeof(unsigned char));
    }

    return (!os.fail());
}

bool drwnCompressionBuffer::read(istream& is)
{
    is.read((char *)&_bytesOriginal, sizeof(unsigned int));
    is.read((char *)&_bytesCompressed, sizeof(unsigned int));
    if (_bytesCompressed > _bytesAllocated) {
        if (_data != NULL) {
            delete[] _data;
        }
        _data = new unsigned char[_bytesAllocated = _bytesCompressed];
        is.read((char *)_data, _bytesCompressed * sizeof(unsigned char));
    }

    return (!is.fail());
}

drwnCompressionBuffer drwnCompressionBuffer::clone() const
{
    drwnCompressionBuffer buffer;
    if (_bytesAllocated > 0) {
        buffer._data = new unsigned char[_bytesAllocated];
        memcpy(buffer._data, _data, _bytesAllocated);
        buffer._bytesAllocated = _bytesAllocated;
        buffer._bytesOriginal = _bytesOriginal;
        buffer._bytesCompressed = _bytesCompressed;
    }
    return buffer;
}

void drwnCompressionBuffer::free()
{
    if (_data != NULL) {
        delete[] _data;
    }
    _data = NULL;
    _bytesAllocated = 0;
    _bytesOriginal = 0;
    _bytesCompressed = 0;
}

// compression utility functions --------------------------------------------

drwnCompressionBuffer drwnCompressVector(const VectorXd& x)
{
    drwnCompressionBuffer buffer;
    buffer.compress((unsigned char *)x.data(), x.rows() * sizeof(double));
    return buffer;
}

drwnCompressionBuffer drwnCompressMatrix(const MatrixXd& x)
{
    drwnCompressionBuffer buffer;
    buffer.compress((unsigned char *)x.data(), x.rows() * x.cols() * sizeof(double));
    return buffer;
}

VectorXd drwnDecompressVector(const drwnCompressionBuffer& buffer, int rows)
{
    DRWN_ASSERT_MSG(rows * sizeof(double) == buffer.originalBytes(),
        rows << " * " << sizeof(double) << " != " << buffer.originalBytes());
    VectorXd x(rows);
    buffer.decompress((unsigned char *)x.data());
    return x;
}

MatrixXd drwnDecompressMatrix(const drwnCompressionBuffer& buffer, int rows, int cols)
{
    DRWN_ASSERT_MSG(rows * cols * sizeof(double) == buffer.originalBytes(),
        rows << " * " << cols << " * " << sizeof(double) << " != " << buffer.originalBytes());
    MatrixXd x(rows, cols);
    buffer.decompress((unsigned char *)x.data());
    return x;
}

