/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBitArray.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Paul Baumstarck <pbaumstarck@stanford.edu>
**
*****************************************************************************/

#include "drwnLogger.h"
#include "drwnBitArray.h"

// a lookup table for the number of bits set in a char
///@cond
const int drwnBitArray::NUMSETLOOKUP[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};
///@endcond

drwnBitArray::drwnBitArray() :
    _sz(0), _map_sz(0), _map(NULL), _mask(0)
{
    // do nothing
}

drwnBitArray::drwnBitArray(int sz) :
    _sz(sz), _map_sz(DRWN_BITMAP_SIZE(sz))
{
    _map = new int[_map_sz];
    if ((_sz % DRWN_INT_BIT_SIZE) == 0) {
        _mask = -1;
    } else {
        _mask = (1 << (_sz % DRWN_INT_BIT_SIZE)) - 1;
    }
    memset(_map, 0, _map_sz * sizeof(int));
}

drwnBitArray::drwnBitArray(const drwnBitArray &m) :
    _sz(m._sz), _map_sz(m._map_sz), _mask(m._mask)
{
    _map = new int[_map_sz];
    memcpy(_map, m._map, _map_sz * sizeof(int));
}

drwnBitArray::~drwnBitArray()
{
    if (_map != NULL)
        delete[] _map;
}

// calculate number of bits set via lookup table
int drwnBitArray::count() const
{
    unsigned char *ptr = (unsigned char *)_map;
    int ret = 0;
    for (int i = 0; i < (int)(_map_sz * sizeof(int)); i++) {
        ret += NUMSETLOOKUP[*ptr++];
    }

    return ret;
}

// logical operations
drwnBitArray& drwnBitArray::ones()
{
    memset(_map, 0xff, _map_sz * sizeof(int));
    if (_map_sz != 0) {
        _map[_map_sz - 1] &= _mask;
    }
    return *this;
}

drwnBitArray& drwnBitArray::zeros()
{
    memset(_map, 0, _map_sz * sizeof(int));
    return *this;
}

drwnBitArray& drwnBitArray::negate()
{
    for (int i = 0; i < _map_sz; i++) {
        _map[i] ^= -1;
    }
    if (_map_sz != 0) {
        _map[_map_sz - 1] &= _mask;
    }
    return *this;
}

drwnBitArray& drwnBitArray::bitwiseand(const drwnBitArray& c)
{
    DRWN_ASSERT(c._sz == _sz);
    for (int i = 0; i < _map_sz; i++) {
        _map[i] &= c._map[i];
    }
    return *this;
}

drwnBitArray& drwnBitArray::bitwiseor(const drwnBitArray& c)
{
    DRWN_ASSERT(c._sz == _sz);
    for (int i = 0; i < _map_sz; i++) {
        _map[i] |= c._map[i];
    }
    return *this;
}

drwnBitArray& drwnBitArray::bitwisenand(const drwnBitArray& c)
{
    DRWN_ASSERT(c._sz == _sz);
    for (int i = 0; i < _map_sz; i++) {
        _map[i] = (_map[i] & c._map[i]) ^ -1;
    }
    if (_map_sz != 0) {
        _map[_map_sz - 1] &= _mask;
    }
    return *this;
}

drwnBitArray& drwnBitArray::bitwisenor(const drwnBitArray& c)
{
    DRWN_ASSERT(c._sz == _sz);
    for (int i = 0; i < _map_sz; i++) {
        _map[i] = (_map[i] | c._map[i]) ^ -1;
    }
    if (_map_sz != 0) {
        _map[_map_sz - 1] &= _mask;
    }
    return *this;
}


drwnBitArray& drwnBitArray::bitwisexor(const drwnBitArray& c)
{
    DRWN_ASSERT(c._sz == _sz);
    for (int i = 0; i < _map_sz; i++) {
        _map[i] ^= c._map[i];
    }
    if (_map_sz != 0) {
        _map[_map_sz - 1] &= _mask;
    }
    return *this;
}

// copy one bitmap to another
drwnBitArray &drwnBitArray::operator=(const drwnBitArray &c)
{
    if (_map == c._map) {
        return *this;
    }

    // check if enough space in bitmap
    if (_map_sz < c._map_sz) {
        if (_map) delete[] _map;
        _map = new int[c._map_sz];
        _sz = c._sz;
        _map_sz = c._map_sz;
    }
    memcpy(_map, c._map, c._map_sz * sizeof(int));
    _mask = c._mask;

    return *this;
}

bool drwnBitArray::operator==(const drwnBitArray& c) const
{
    if (c._sz != _sz) return false;
    for (int i = 0; i < _map_sz; i++) {
        if (c._map[i] != _map[i]) {
            return false;
        }
    }
    return true;
}

bool drwnBitArray::operator!=(const drwnBitArray& c) const
{
    return !(this->operator==(c));
}

bool drwnBitArray::operator<=(const drwnBitArray& c) const
{
    if (c._sz < _sz) return false;
    if (c._sz > _sz) return true;

    for (int i = 0; i < _map_sz; i++) {
        if (c._map[i] > _map[i]) {
            return true;
        }
    }
    return false;    
}

// printing
void drwnBitArray::print(ostream &os, int stride) const
{
    for ( int i=0; i<_sz; ++i ) {
        os << (DRWN_BIT_GET(_map,i) ? 1 : 0 );
        if ( stride > 0 && (i+1)%stride == 0 )
            os << "\n";
    }
    if ( !( stride > 0 && _sz%stride == 0 ) )
        os << "\n";
}


