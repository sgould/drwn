/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSparseVector.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>
#include <map>

#include "drwnBase.h"

using namespace std;

// drwnSparseVec ------------------------------------------------------------
//! Quick-and-dirty sparse vector class as a plugin replacement for std::vector.
//!
//! The class has the same interface as std::vector, but does not explicitly
//! store zeros. The class typecasts between std::vector objects when needed so
//! is very computationally inefficient. However, code modification should be
//! minimal. The dot product member function should be used where possible to
//! mitigate computational inefficiency.
//!
//! The folllowing code snippet demonstrates example usage:
//! \code
//!    drwnSparseVec<double> x(1000);
//!    for (size_t i = 0; i < x.size(); i++) {
//!        if (drand48() < 0.1)
//!            x[i] = 1.0;
//!    }
//!    
//!    DRWN_LOG_MESSAGE("sparse vector has " << x.nnz() << " non-zero entries");
//! \endcode

template<typename T>
class drwnSparseVec {
 protected:
    static const T _zero;
    size_t _size;
    map<size_t, T> _data;

 public:
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;

    class iterator;

    class reference {
    protected:
        friend class drwnSparseVec;
        friend class drwnSparseVec<T>::iterator;
        drwnSparseVec<T> *_p;
        size_t _indx;

        inline reference();
        inline reference(drwnSparseVec<T>& v, size_t indx);

    public:
        inline ~reference();

        inline operator T() const;
        inline reference& operator=(const T& x);
        inline reference& operator=(const reference& x);
        inline reference& operator+=(const T& x);
        inline reference& operator-=(const T& x);
        inline reference& operator*=(const T& x);
        inline reference& operator/=(const T& x);
    };
    friend class reference;

    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    class iterator {
    protected:
        friend class drwnSparseVec;
        drwnSparseVec<T> *_p;
        size_t _indx;

        inline iterator(drwnSparseVec<T>& v, size_t indx);

    public:
        typedef random_access_iterator_tag iterator_category;
        typedef T value_type;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef T *pointer;
        typedef const T *const_pointer;
        typedef typename drwnSparseVec<T>::reference reference;
        typedef const T& const_reference;

    public:
        inline iterator();
        inline ~iterator();

        inline iterator& operator=(const iterator& it);
        inline bool operator==(const iterator& it) const;
        inline bool operator!=(const iterator& it) const;
        inline bool operator<(const iterator& it) const;
        inline bool operator>(const iterator& it) const;
        inline bool operator<=(const iterator& it) const;
        inline bool operator>=(const iterator& it) const;
        inline const_reference operator*() const;
        inline reference operator*();
        //inline pointer operator->();
        inline iterator& operator++();
        inline iterator& operator--();
        inline iterator operator++(int);
        inline iterator operator--(int);
        inline iterator operator+(difference_type n);
        inline iterator& operator+=(difference_type n);
        inline iterator operator-(difference_type n);
        inline iterator& operator-=(difference_type n);
        inline const T& operator[](difference_type n) const;
        inline reference operator[](difference_type n);

        inline difference_type operator-(const iterator& it) const {
            return (_indx - it._indx);
        }
        inline difference_type operator+(const iterator& it) const {
            return (_indx + it._indx);
        }

        //operator size_t() const { return _indx; }
    };
    friend class iterator;

    /*
    class const_iterator;
    friend class const_iterator;
    */

 public:
    drwnSparseVec();
    drwnSparseVec(size_t size);
    drwnSparseVec(const drwnSparseVec<T>& v);
    drwnSparseVec(const vector<T>& v);
    ~drwnSparseVec();

    bool empty() const { return (_size == 0); }
    void clear() { _size = 0; _data.clear(); }
    void resize(size_t n);
    size_t size() const { return _size; }
    size_t capacity() const { return _size; }
    size_t max_size() const { return size_type(-1) / sizeof(value_type); }
    void reserve(size_t size) { /* do nothing */ }
    void swap(drwnSparseVec<T>& v);

    size_t nnz() const { return _data.size(); }

    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, _size); }

    // modification
    void push_back(const T& x);
    void pop_back();
    void insert(size_t position, const T& x);
    void insert(size_t position, typename vector<T>::const_iterator first,
        typename vector<T>::const_iterator last);
    void insert(iterator position, const T& x) {
        assert(position._p == this);
        return insert(position._indx, x);
    }
    void insert(iterator position, typename vector<T>::const_iterator first,
        typename vector<T>::const_iterator last) {
        assert(position._p == this);
        return insert(position._indx, first, last);
    }

    // operators
    drwnSparseVec& operator=(const drwnSparseVec<T>& v);
    drwnSparseVec& operator=(const vector<T>& v);
    const T& operator[](size_t indx) const;
    reference operator[](size_t indx);

    // typecasting
    operator vector<T>() const { return decode(); }

    //! efficient dot product between a vector and a sparse vector
    static T dot(const vector<T>& x, const drwnSparseVec<T>& y);
    //! efficient dot product between a sparse vector and a vector
    static T dot(const drwnSparseVec<T>& x, const vector<T>& y);
    //! efficient dot product between two sparse vectors
    static T dot(const drwnSparseVec<T>& x, const drwnSparseVec<T>& y);   
    
 protected:
    // decode into an stl vector
    vector<T> decode() const;

    // encode from an stl vector
    void encode(const vector<T>& v);
};

// drwnSparseVec implementation ---------------------------------------------

template<typename T>
const T drwnSparseVec<T>::_zero(0);

template<typename T>
drwnSparseVec<T>::drwnSparseVec() : _size(0)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::drwnSparseVec(size_t size) : _size(size)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::drwnSparseVec(const drwnSparseVec<T>& v) :
    _size(v._size), _data(v._data)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::drwnSparseVec(const vector<T>& v)
{
    encode(v);
}

template<typename T>
drwnSparseVec<T>::~drwnSparseVec()
{
    // do nothing
}


template<typename T>
void drwnSparseVec<T>::resize(size_t n)
{
    _size = n;
    for (typename map<size_t, T>::iterator it = _data.begin(); it != _data.end(); it++) {
        if (it->first >= _size) {
            it = _data.erase(it);
        }
    }
}

template<typename T>
void drwnSparseVec<T>::swap(drwnSparseVec<T>& v)
{
    std::swap(_size, v._size);
    _data.swap(v._data);
}

template<typename T>
void drwnSparseVec<T>::push_back(const T& x)
{
    if (x != T(0)) {
        _data[_size] = x;
    }
    _size += 1;
}

template<typename T>
void drwnSparseVec<T>::pop_back()
{
    DRWN_ASSERT(_size > 0);
    typename map<size_t, T>::iterator it = _data.find(_size - 1);
    if (it != _data.end()) {
        _data.erase(it);
    }
    _size -= 1;
}

template<typename T>
void drwnSparseVec<T>::insert(size_t position, const T& x)
{
    // TODO: this is ugly; fix
    DRWN_ASSERT((position >= this->begin()._indx) && (position <= this->end()._indx));

    if (position != this->end()._indx) {
        map<size_t, T> oldVec(_data);
        _data.clear();

        for (typename map<size_t, T>::const_iterator it = oldVec.begin(); it != oldVec.end(); it++) {
            if (it->first >= position) {
                _data[it->first + 1] = it->second;
            } else {
                _data[it->first] = it->second;
            }
        }
    }

    if (x != T(0)) {
        _data[position] = x;
    }

    _size += 1;
}

template<typename T>
void drwnSparseVec<T>::insert(size_t position, typename vector<T>::const_iterator first,
    typename vector<T>::const_iterator last)
{
    // TODO: this is ugly; fix
    DRWN_ASSERT((position >= this->begin()._indx) && (position <= this->end()._indx));

    map<size_t, T> oldVec;
    if (position != this->end()._indx) {
        _data.swap(oldVec);
    }

    size_t delta = 0;
    for (typename vector<T>::const_iterator it = first; it != last; it++) {
        if (*it != T(0)) {
            _data[position + delta] = *it;
        }
        delta += 1;
    }

    for (typename map<size_t, T>::const_iterator it = oldVec.begin(); it != oldVec.end(); it++) {
        if (it->first >= position) {
            _data[it->first + delta] = it->second;
        } else {
            _data[it->first] = it->second;
        }
    }

    _size += delta;
}

// operators
template<typename T>
drwnSparseVec<T>& drwnSparseVec<T>::operator=(const drwnSparseVec<T>& v)
{
    if (&v != this) {
        _size = v._size;
        _data = v._data;
    }
    return *this;
}

template<typename T>
drwnSparseVec<T>& drwnSparseVec<T>::operator=(const vector<T>& v)
{
    encode(v);
    return *this;
}

template<typename T>
const T& drwnSparseVec<T>::operator[](size_t indx) const
{
    typename map<size_t, T>::const_iterator it = _data.find(indx);
    if (it != _data.end()) {
        return it->second;
    }

    return _zero;
}

template<typename T>
typename drwnSparseVec<T>::reference drwnSparseVec<T>::operator[](size_t indx)
{
    /*
    typename map<size_t, T>::iterator it = _data.find(indx);
    if (it == _data.end()) {
        it = _data.insert(make_pair(indx, T(0))).first;
    }

    return it->second;
    */
    return typename drwnSparseVec<T>::reference(*this, indx);
}

template <typename T>
T drwnSparseVec<T>::dot(const vector<T>& x, const drwnSparseVec<T>& y)
{
    T d(0);
    for (typename map<size_t, T>::const_iterator it = y._data.begin(); it != y._data.end(); ++it) {
        d += x[it->first] * it->second;
    }

    return d;
}

template <typename T>
T drwnSparseVec<T>::dot(const drwnSparseVec<T>& x, const vector<T>& y)
{
    T d(0);
    for (typename map<size_t, T>::const_iterator it = x._data.begin(); it != x._data.end(); ++it) {
        d += y[it->first] * it->second;
    }

    return d;
}

template <typename T>
T drwnSparseVec<T>::dot(const drwnSparseVec<T>& x, const drwnSparseVec<T>& y)
{ 
    T d(0);

    if (x.nnz() < y.nnz()) {
        for (typename map<size_t, T>::const_iterator it = x._data.begin(); it != x._data.end(); ++it) {
            d += y[it->first] * it->second;
        }
    } else {
        for (typename map<size_t, T>::const_iterator it = y._data.begin(); it != y._data.end(); ++it) {
            d += x[it->first] * it->second;
        }
    }

    return d;
}

// decode into an stl vector
template<typename T>
vector<T> drwnSparseVec<T>::decode() const
{
    vector<T> v(_size, T(0));
    for (typename map<size_t, T>::const_iterator it = _data.begin(); it != _data.end(); it++) {
        v[it->first] = it->second;
    }
    return v;
}

// encode from an stl vector
template<typename T>
void drwnSparseVec<T>::encode(const vector<T>& v)
{
    _size = v.size();
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] != T(0)) {
            _data[i] = v[i];
        }
    }
}

// drwnSparseVec::reference implementation ----------------------------------

template<typename T>
drwnSparseVec<T>::reference::reference(drwnSparseVec<T>& v, size_t indx) :
    _p(&v), _indx(indx)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::reference::~reference()
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::reference::operator T() const
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it == _p->_data.end()) {
        return T(0);
    }

    return it->second;
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator=(const T& x)
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it == _p->_data.end()) {
        if (x != T(0)) {
            _p->_data.insert(make_pair(_indx, x));
        }
    } else {
        if (x == T(0)) {
            _p->_data.erase(it);
        } else {
            it->second = x;
        }
    }

    return *this;
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator=(const reference& x)
{
    this->operator=((T)x);
    return *this;
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator+=(const T& x)
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it != _p->_data.end()) {
        return (*this = (it->second + x));
    }

    return (*this = x);
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator-=(const T& x)
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it != _p->_data.end()) {
        return (*this = (it->second - x));
    }

    return (*this = (T(0) - x));
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator*=(const T& x)
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it != _p->_data.end()) {
        return (*this = (it->second * x));
    }

    return (*this = (T(0) * x));
}

template<typename T>
typename drwnSparseVec<T>::reference& drwnSparseVec<T>::reference::operator/=(const T& x)
{
    typename map<size_t, T>::iterator it = _p->_data.find(_indx);
    if (it != _p->_data.end()) {
        return (*this = (it->second / x));
    }

    return (*this = (T(0) / x));
}

// drwnSparseVec::iterator implementation ------------------------------------

template<typename T>
drwnSparseVec<T>::iterator::iterator(drwnSparseVec<T>& v, size_t indx) :
    _p(&v), _indx(indx)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::iterator::iterator() :
    _p(NULL), _indx(0)
{
    // do nothing
}

template<typename T>
drwnSparseVec<T>::iterator::~iterator()
{
    // do nothing
}

template<typename T>
typename drwnSparseVec<T>::iterator& drwnSparseVec<T>::iterator::operator=(const iterator& it)
{
    _p = it._p;
    _indx = it._indx;
    return *this;
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator==(const iterator& it) const
{
    return ((_p == it._p) && (_indx == it._indx));
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator!=(const iterator& it) const
{
    return ((_p != it._p) || (_indx != it._indx));
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator<(const iterator& it) const
{
    return (_indx < it._indx);
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator>(const iterator& it) const
{
    return (_indx > it._indx);
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator<=(const iterator& it) const
{
    return (_indx <= it._indx);
}

template<typename T>
bool drwnSparseVec<T>::iterator::operator>=(const iterator& it) const
{
    return (_indx >= it._indx);
}

template<typename T>
typename drwnSparseVec<T>::const_reference drwnSparseVec<T>::iterator::operator*() const
{
    DRWN_ASSERT(_p != NULL);
    return (*_p)[_indx];
}

template<typename T>
typename drwnSparseVec<T>::reference drwnSparseVec<T>::iterator::operator*()
{
    return typename drwnSparseVec<T>::reference(*_p, _indx);
}

template<typename T>
typename drwnSparseVec<T>::iterator& drwnSparseVec<T>::iterator::operator++()
{
    _indx = std::min(_indx + 1, _p->_size);
    return *this;
}

template<typename T>
typename drwnSparseVec<T>::iterator& drwnSparseVec<T>::iterator::operator--()
{
    if (_indx > 0) _indx -= 1;
    return *this;
}

template<typename T>
typename drwnSparseVec<T>::iterator drwnSparseVec<T>::iterator::operator++(int)
{
    iterator copy(*this);
    ++(*this);
    return copy;
}

template<typename T>
typename drwnSparseVec<T>::iterator drwnSparseVec<T>::iterator::operator--(int)
{
    iterator copy(*this);
    --(*this);
    return copy;
}

template<typename T>
typename drwnSparseVec<T>::iterator drwnSparseVec<T>::iterator::operator+(difference_type n)
{
    iterator tmp(*this);
    tmp._indx += n;
    return tmp;
}

template<typename T>
typename drwnSparseVec<T>::iterator& drwnSparseVec<T>::iterator::operator+=(difference_type n)
{
    _indx += n;
    *this;
}

template<typename T>
typename drwnSparseVec<T>::iterator drwnSparseVec<T>::iterator::operator-(difference_type n)
{
    iterator tmp(*this);
    tmp._indx -= n;
    return tmp;
}

template<typename T>
typename drwnSparseVec<T>::iterator& drwnSparseVec<T>::iterator::operator-=(difference_type n)
{
    _indx -= n;
    *this;
}

template<typename T>
const T& drwnSparseVec<T>::iterator::operator[](difference_type n) const
{
    return (*_p)[_indx + n];
}

template<typename T>
typename drwnSparseVec<T>::reference drwnSparseVec<T>::iterator::operator[](difference_type n)
{
    return drwnSparseVec<T>::reference(*_p, _indx + n);
}
