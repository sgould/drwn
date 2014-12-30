/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSmartPointer.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**   Provides a smart pointer interface to avoid the need to deep copy large
**   constant (shared) objects. Example use:
**      drwnSmartPointer<TImage> A(new TImage());
**      drwnSmartPointer<TImage> B(A);
**      drwnSmartPointer<TImage> C = A;
**      DRWN_ASSERT((C == A) && (B == A));
**      C->showImage();
**   Unlike the STL's auto_ptr, it is safe to use smart pointers in STL
**   containers.
**
**   By default the smart pointer takes ownership of the object (and will
**   delete it when the reference count reaches zero). However, the smart
**   pointer can also be configured to access statically allocated objects
**   (e.g. on the stack). In this case the object is not deleted, but care
**   must be taken to ensure that the object does not go out of scope before
**   all smart pointers have been destroyed. Here's an example:
**      TImage I;
**      drwnSmartPointer<TImage> A(&I, false);
**
*****************************************************************************/

#pragma once

// drwnSmartPointer class ---------------------------------------------------

/*!
** \brief Implements a shared pointer interface to avoid the need to deep
** copy constant (shared) objects.
**
** Example use:
** \code
**   drwnSmartPointer<TImage> A(new TImage());
**   drwnSmartPointer<TImage> B(A);
**   drwnSmartPointer<TImage> C = A;
**   DRWN_ASSERT((C == A) && (B == A));
**   C->showImage();
** \endcode
**
** Unlike the STL's auto_ptr, it is safe to use smart pointers in STL
** containers.
**
** By default the smart pointer takes ownership of the object (and will
** delete it when the reference count reaches zero). However, the smart
** pointer can also be configured to access statically allocated objects
** (e.g. on the stack). In this case the object is not deleted, but care
** must be taken to ensure that the object does not go out of scope before
** all smart pointers have been destroyed. Here's an example:
** \code
**   TImage I;
**   drwnSmartPointer<TImage> A(&I, false);
** \endcode
*/
template <typename T>
class drwnSmartPointer {
 protected:
    T* _objPtr;            //!< pointer to the shared object
    unsigned *_refCount;   //!< number of drwnSmartPointer objects that share \p _objPtr
    bool _bOwner;          //!< \b true if drwnSmartPointer is responsible for destroying \p _objPtr

 public:
    //! default constructor (NULL smart pointer)
    inline drwnSmartPointer() : 
        _objPtr(NULL), _refCount(NULL), _bOwner(true) {
	// do nothing
    }

    //! create a smart pointer of type \p T to \p obj 
    inline explicit drwnSmartPointer(T* obj, bool bOwner = true) : 
        _objPtr(obj), _bOwner(bOwner) {
	_refCount = (_objPtr == NULL) ? NULL : new unsigned(1);
    }

    //! copy constructor
    inline drwnSmartPointer(const drwnSmartPointer<T>& p) : 
        _objPtr(p._objPtr), _refCount(p._refCount), _bOwner(p._bOwner) {
	if (_refCount != NULL) {
	    ++*_refCount;
        }
    }

    inline ~drwnSmartPointer() {
	if ((_refCount != NULL) && (--*_refCount == 0)) {
            if (_bOwner) {
                delete _objPtr;
            }
	    delete _refCount;
	    _objPtr = NULL;
	    _refCount = NULL;
	}
    }

    //! assignment operator
    inline drwnSmartPointer<T>& operator=(const drwnSmartPointer<T>& p) {
	if (p._refCount) {
	    ++*p._refCount;
        }

	if ((_refCount != NULL) && (--*_refCount == 0)) {
            if (_bOwner) {
                delete _objPtr;
            }
	    delete _refCount;
	}

	_objPtr = p._objPtr;
	_refCount = p._refCount;
        _bOwner = p._bOwner;

	return *this;
    }

    //! compare two smart pointers for equality
    inline bool operator==(const drwnSmartPointer<T>& p) {
	return (_refCount == p._refCount);
    }
    //! compare two smart pointers for inequality
    inline bool operator!=(const drwnSmartPointer<T>& p) {
	return (_refCount != p._refCount);
    }
    //! compare smart pointers with object pointer for equality
    inline bool operator==(const T* o) {
        return (_objPtr == o);
    }
    //! compare smart pointers with object pointer for inequality
    inline bool operator!=(const T* o) {
        return (_objPtr != o);
    }

    //! de-reference the smart pointer
    inline T* operator->() { return _objPtr; }
    //! de-reference the smart pointer
    inline const T* operator->() const { return _objPtr; }

    //! de-reference the smart pointer
    inline operator T*() { return _objPtr; }
    //! de-reference the smart pointer
    inline operator const T*() const { return _objPtr; }
};

// drwnSmartPointer comparison classes -------------------------------------

//! comparison operator for objects held in a drwnSmartPointer
template <typename T>
struct drwnSmartPointerCmpLessThan {
    bool operator()(drwnSmartPointer<T>& a, drwnSmartPointer<T>& b) const {
        return (*a < *b);
    }
};
