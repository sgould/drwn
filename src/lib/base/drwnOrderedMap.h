/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOrderedMap.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <map>

using namespace std;

// drwnOrderedMap -----------------------------------------------------------
//! Provides a datastructure for that can be indexed by a \c KeyType (usually a
//! string) or unsigned integer, i.e., the index.
//!
//! The unsigned integer variant indexes entries by insertion order. \c KeyType
//! must be default constructable and cannot be an unsigned integer.
//!
//! Example usage:
//! \code
//!     drwnOrderedMap<string, int> map;
//!     map.insert(string("bart"), 100);
//!     map.insert(string("lisa"), 20);
//!     map.insert(string("maggie"), 300);
//!
//!     cout << "'bart' has value " << map[string("bart")] << endl;
//!     cout << "first entry has value " << map[0] << endl;
//!     cout << "first entry has key " << map.find(string("bart")) << endl;
//! \endcode

template<typename KeyType, typename ValueType>
class drwnOrderedMap {
 protected:
    vector<pair<KeyType, ValueType> *> _data;
    map<KeyType, unsigned> _index;

 public:
    //! default constructor
    drwnOrderedMap();
    //! copy constructor
    drwnOrderedMap(const drwnOrderedMap<KeyType, ValueType>& m);
    //! destructor
    ~drwnOrderedMap();

    //! clear all entries from the map
    void clear();
    //! returns the number of entries in the map
    size_t size() const { return _data.size(); };
    //! Inserts value \p v into the map at the end position with key \p key.
    //! If the key already exists then the value for the entry is replaced
    //! and it's index is unchanged.
    void insert(const KeyType& key, const ValueType& v);
    //! remove the entry in the map corresponding to key \p key
    void erase(const KeyType& key);
    //! find the index of an entry in the map (-1 if the key does not exist)
    int find(const KeyType& key) const;

    //! assignment operator
    drwnOrderedMap<KeyType, ValueType>& operator=(const drwnOrderedMap<KeyType, ValueType>& m);

    //! index the \p indx-th entry in the map
    const ValueType& operator[](unsigned int indx) const;
    //! index the \p indx-th entry in the map
    ValueType& operator[](unsigned int indx);
    //! index the entry in the map with key \p key
    const ValueType& operator[](const KeyType& key) const;
    //! index the entry in the map with key \p key
    ValueType& operator[](const KeyType& key);
};

// implementation -----------------------------------------------------------

template<typename KeyType, typename ValueType>
drwnOrderedMap<KeyType, ValueType>::drwnOrderedMap()
{
    // do nothing
}

template<typename KeyType, typename ValueType>
drwnOrderedMap<KeyType, ValueType>::drwnOrderedMap(const drwnOrderedMap<KeyType, ValueType>& m) :
    _index(m._index)
{
    // deep copy data
    _data.reserve(m._data.size());
    for (typename vector<pair<KeyType, ValueType> *>::const_iterator it = m._data.begin();
         it != m._data.end(); it++) {
        _data.push_back(new pair<KeyType, ValueType>(*it));
    }
}

template<typename KeyType, typename ValueType>
drwnOrderedMap<KeyType, ValueType>::~drwnOrderedMap()
{
    clear(); // delete entries
}

template<typename KeyType, typename ValueType>
void drwnOrderedMap<KeyType, ValueType>::clear()
{
    for (typename vector<pair<KeyType, ValueType> *>::iterator it = _data.begin();
         it != _data.end(); it++) {
        delete *it;
    }
    _data.clear();
    _index.clear();
}


template<typename KeyType, typename ValueType>
void drwnOrderedMap<KeyType, ValueType>::insert(const KeyType& key, const ValueType& v)
{
    typename map<KeyType, unsigned>::iterator it = _index.find(key);
    if (it != _index.end()) {
        _data[it->second]->second = v;
    } else {
        _index.insert(it, make_pair(key, (unsigned)_data.size()));
        _data.push_back(new pair<KeyType, ValueType>(key, v));
    }
}

template<typename KeyType, typename ValueType>
void drwnOrderedMap<KeyType, ValueType>::erase(const KeyType& key)
{
    typename map<KeyType, unsigned>::iterator it = _index.find(key);
    DRWN_ASSERT(it != _index.end());

    // re-order vector
    delete _data[it->second];
    for (unsigned i = it->second + 1; i < _data.size(); i++) {
        _data[i - 1] = _data[i];
    }
    _data.pop_back();

    // re-assign mapping
    for (typename map<KeyType, unsigned>::iterator jt = _index.begin();
         jt != _index.end(); jt++) {
        if (jt->second > it->second)
            jt->second -= 1;
    }
    _index.erase(it);
}

template<typename KeyType, typename ValueType>
int drwnOrderedMap<KeyType, ValueType>::find(const KeyType& key) const
{
    typename map<KeyType, unsigned>::const_iterator it = _index.find(key);
    if (it == _index.end()) {
        return -1;
    } else {
        return (int)it->second;
    }
}

template<typename KeyType, typename ValueType>
drwnOrderedMap<KeyType, ValueType>& drwnOrderedMap<KeyType, ValueType>::operator=(const drwnOrderedMap<KeyType, ValueType>& m)
{
    if (&m != this) {
        clear();
        _data.reserve(m._data.size());
        for (typename vector<pair<KeyType, ValueType> *>::const_iterator it = m._data.begin();
             it != m._data.end(); it++) {
            _data.push_back(new pair<KeyType, ValueType>(*it));
        }
        _index = m._index;
    }

    return *this;
}

template<typename KeyType, typename ValueType>
const ValueType& drwnOrderedMap<KeyType, ValueType>::operator[](unsigned int indx) const
{
    return _data[indx]->second;
}

template<typename KeyType, typename ValueType>
ValueType& drwnOrderedMap<KeyType, ValueType>::operator[](unsigned int indx)
{
    return _data[indx]->second;
}

template<typename KeyType, typename ValueType>
const ValueType& drwnOrderedMap<KeyType, ValueType>::operator[](const KeyType& key) const
{
    typename map<KeyType, unsigned>::const_iterator it = _index.find(key);
    return _data[it->second]->second;
}

template<typename KeyType, typename ValueType>
ValueType& drwnOrderedMap<KeyType, ValueType>::operator[](const KeyType& key)
{
    typename map<KeyType, unsigned>::iterator it = _index.find(key);
    if (it == _index.end()) {
        _index.insert(it, make_pair(key, (unsigned)_data.size()));
        _data.push_back(new pair<KeyType, ValueType>(key, ValueType(0)));
        return _data.back()->second;
    }

    return _data[it->second]->second;
}

