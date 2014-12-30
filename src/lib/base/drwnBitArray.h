/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBitArray.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Paul Baumstarck <pbaumstarck@stanford.edu>
** DESCRIPTION:
**  Implements a packed array of bits.
**
*****************************************************************************/

#pragma once

#include <stdlib.h>
#include <iostream>
#include <cstring>

using namespace std;

#define DRWN_INT_BIT_SIZE (8*sizeof(int))
#define DRWN_BIT_GET(x,b) ( ( (x)[(b)/DRWN_INT_BIT_SIZE] & (1<<((b)%DRWN_INT_BIT_SIZE)) ) != 0 )
#define DRWN_BIT_SET(x,b) ( (x)[(b)/DRWN_INT_BIT_SIZE] |= (1<<((b)%DRWN_INT_BIT_SIZE)) )
#define DRWN_BIT_CLEAR(x,b) ( (x)[(b)/DRWN_INT_BIT_SIZE] &= (unsigned int)(-1) - (1<<((b)%DRWN_INT_BIT_SIZE)) )
#define DRWN_BIT_FLIP(x,b) ( (x)[(b)/DRWN_INT_BIT_SIZE] ^= (1<<((b)%DRWN_INT_BIT_SIZE)) )
#define DRWN_BITMAP_SIZE(x) int( ((x+DRWN_INT_BIT_SIZE-1)/DRWN_INT_BIT_SIZE) )

/*!
** \brief Implements an efficient packed array of bits.
**
** For example the following code creates two arrays of size eight and counts
** the number of bits set when one array is xor'd with the other.
** \code
**   drwnBitArray a(8), b(8);
**   a.ones(); b.zeros();
**   cout << a.bitwisexor(b).count();
** \endcode
*/
class drwnBitArray {
 protected:
    static const int NUMSETLOOKUP[256]; //!< lookup table for counting the number of \a set bits
    int _sz;                            //!< size of the array in bits
    int _map_sz;                        //!< size of the array in words
    int *_map;                          //!< storage for the array
    int _mask;                          //!< mask for the last word in \ref _map

 public:
    drwnBitArray();
    //! creates an array with \b sz bits
    drwnBitArray(int sz);
    //! copy constructor
    drwnBitArray(const drwnBitArray &m);
    virtual ~drwnBitArray();

    //! returns true if the array is empty
    inline bool empty() const { return (_sz == 0); }
    //! returns the number of bits in the array
    inline int size() const { return _sz; }
    //! counts the number of bits in the array that are \a set
    int count() const;

    //! returns \b true if the \a i-th bit of the array is set
    inline bool get(int i) const { return DRWN_BIT_GET(_map, i); }
    //! sets the \a i-th bit of the array to \b true
    inline void set(int i) { DRWN_BIT_SET(_map, i); }
    //! sets the \a i-th bit of the array to \b false
    inline void clear(int i) { DRWN_BIT_CLEAR(_map, i); }
    //! negates the \a i-th bit of the array
    inline void flip(int i) { DRWN_BIT_FLIP(_map, i); }

    //! sets all bits in the array
    drwnBitArray& ones();
    //! clears all bits in the array
    drwnBitArray& zeros();
    //! flips all bits in the array
    drwnBitArray& negate();
    //! performs a bitwise \b and with an array of equal size
    drwnBitArray& bitwiseand(const drwnBitArray& c);
    //! performs a bitwise \b or with an array of equal size
    drwnBitArray& bitwiseor(const drwnBitArray& c);
    //! performs a bitwise \b nand with an array of equal size
    drwnBitArray& bitwisenand(const drwnBitArray& c);
    //! performs a bitwise \b nor with an array of equal size
    drwnBitArray& bitwisenor(const drwnBitArray& c);
    //! performs a bitwise \b xor with an array of equal size
    drwnBitArray& bitwisexor(const drwnBitArray& c);

    //! assignment operator
    drwnBitArray& operator=(const drwnBitArray &c);
    //! equality operator (returns \b true if arrays are the same size and all bits are equal)
    bool operator==(const drwnBitArray& c) const;
    //! not-equality operator (returns \b true if arrays are different sizes or some bits are not equal)
    bool operator!=(const drwnBitArray& c) const;
    //! less-than operator (returns \b true if this array preceeds \b c lexigraphically when printed)
    bool operator<=(const drwnBitArray& c) const;
    //! returns \b true if the \a i-th bit of the array is set (see drwnBitArray::get)
    inline bool operator[](int i) const { return DRWN_BIT_GET(_map, i); }

    //! prints the array as a string of 1's and 0's
    void print(ostream &os = cout, int stride = -1) const;
};
