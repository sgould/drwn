/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCompressionBuffer.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include "Eigen/Core"

using namespace std;
using namespace Eigen;

// drwnCompressionBuffer ----------------------------------------------------
//! \brief Utility class for compressing data using the zlib library.
//!
//! The class holds compressed data. Copying the buffer does not copy the
//! data (i.e., a shallow copy is performed). To copy the data use the clone
//! member function. When decompressing data enough space must be allocated
//! in output buffer. The following code snippet shows example usage:
//!
//! \code
//!    // compress a string
//!    string inputStr = getStringToCompress();
//!    drwnCompressionBuffer buffer;
//!    buffer.compress((unsigned char *)inputStr.c_str(), inputStr.size());
//!
//!    DRWN_LOG_MESSAGE("compressed " << buffer.originalBytes() << " bytes to " 
//!        << buffer.compressedBytes() << " bytes");
//!
//!    // decompress a string
//!    string recoveredStr;
//!    recoveredStr.resize(buffer.originalBytes());
//!    buffer.decompress((unsigned char *)&recoveredStr[0]);
//! \endcode
//!
//! \warning This class will simply copy data without compressing if
//! compiled without DRWN_HAS_ZLIB defined.

class drwnCompressionBuffer {
 protected:
    unsigned char *_data;
    unsigned int _bytesAllocated;
    unsigned int _bytesOriginal;
    unsigned int _bytesCompressed;

#ifdef DRWN_DEBUG_STATISTICS
    static unsigned _dbRefCount;
    static unsigned _dbOriginalBytes;
    static unsigned _dbCompressedBytes;
#endif
        
 public:
    drwnCompressionBuffer();
    ~drwnCompressionBuffer();

    //! number of uncompressed bytes
    unsigned int originalBytes() const { return _bytesOriginal; }
    //! number of compressed bytes
    unsigned int compressedBytes() const { return _bytesCompressed; }

    //! compress \p size bytes of \p data into buffer
    void compress(const unsigned char *data, unsigned int size);
    //! decompress buffer into \p data
    void decompress(unsigned char *data) const;
    
    //! number of bytes required to store buffer contents on disk
    size_t numBytesOnDisk() const;
    //! write compression buffer to output stream
    bool write(ostream& os) const;
    //! read compression buffer from input stream
    bool read(istream& is);

    //! clone the buffer
    drwnCompressionBuffer clone() const;
    //! free all memory associated with the buffer
    void free();
};

// compression utility functions --------------------------------------------

//! compress a vector
drwnCompressionBuffer drwnCompressVector(const VectorXd& x);
//! compress a matrix
drwnCompressionBuffer drwnCompressMatrix(const MatrixXd& x);
//! decompress a vector of known size
VectorXd drwnDecompressVector(const drwnCompressionBuffer& buffer, int rows);
//! decompress a matrix of known size
MatrixXd drwnDecompressMatrix(const drwnCompressionBuffer& buffer, int rows, int cols);

