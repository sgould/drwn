/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnProperties.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**
*****************************************************************************/

#pragma once

#include <string>
#include <string.h>
#include <vector>
#include <map>
#include <list>

#include "Eigen/Core"

#include "drwnSmartPointer.h"
#include "drwnXMLParser.h"

using namespace std;

// drwnPropertyType -------------------------------------------------------

typedef enum _drwnPropertyType {
    DRWN_INVALID_PROPERTY, DRWN_BOOLEAN_PROPERTY, DRWN_INTEGER_PROPERTY,
    DRWN_DOUBLE_PROPERTY, DRWN_STRING_PROPERTY, DRWN_FILENAME_PROPERTY,
    DRWN_DIRECTORY_PROPERTY, DRWN_LIST_PROPERTY, DRWN_SELECTION_PROPERTY, 
    DRWN_VECTOR_PROPERTY, DRWN_MATRIX_PROPERTY, DRWN_USER_PROPERTY
} drwnPropertyType;

// drwnPropertyInterface --------------------------------------------------

class drwnPropertyInterface {
 protected:
    bool _bReadOnly;   // read-only property
    bool _bSerialize;  // serializeable property

    drwnPropertyInterface() : _bReadOnly(false), _bSerialize(true) { };
    drwnPropertyInterface(bool bReadOnly, bool bSerialize = true) : 
        _bReadOnly(bReadOnly), _bSerialize(bSerialize) { }

 public:
    virtual ~drwnPropertyInterface() { }

    // type info
    virtual drwnPropertyType type() const = 0;
    virtual string asString() const = 0;

    inline bool isReadOnly() const { return _bReadOnly; }
    inline bool isSerializeable() const { return _bSerialize; }

    // set/cast functions
    virtual bool setProperty(bool value);
    virtual bool setProperty(int value);
    virtual bool setProperty(double value);
    virtual bool setProperty(const string& value);
    virtual bool setProperty(const char *value);
    virtual bool setProperty(const Eigen::VectorXd& value);
    virtual bool setProperty(const Eigen::MatrixXd& value);

    // i/o
    virtual void read(drwnXMLNode& xml);
    virtual void write(drwnXMLNode& xml) const;
    virtual drwnPropertyInterface* clone() const = 0;
    inline drwnPropertyInterface* clone(bool bReadOnly, bool bSerialize) const {
        drwnPropertyInterface *iface = this->clone();
        iface->_bReadOnly |= bReadOnly;
        iface->_bSerialize = bSerialize;
        return iface;
    }
};

// drwnStoragePropertyInterface -------------------------------------------

template <typename T>
class drwnStoragePropertyInterface : public drwnPropertyInterface {
 protected:
    T* _storage;

 public:
    drwnStoragePropertyInterface(T *storage, bool bReadOnly = false) :
        drwnPropertyInterface(bReadOnly), _storage(storage) { }

    inline const T& getValue() const { return *_storage; }
    inline bool setValue(const T& value) { *_storage = value; return true; }

    //virtual bool setProperty(const T& value) { *_storage = value; return true; }
};

// standard drwnProperty types --------------------------------------------

class drwnBooleanProperty : public drwnStoragePropertyInterface<bool>
{
 public:
    drwnBooleanProperty(bool *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<bool>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_BOOLEAN_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnBooleanProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);
};

class drwnIntegerProperty : public drwnStoragePropertyInterface<int>
{
 public:
    drwnIntegerProperty(int *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<int>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_INTEGER_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnIntegerProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);
};

class drwnRangeProperty : public drwnIntegerProperty
{
 protected:
    int _lowerBound;
    int _upperBound;

 public:
    drwnRangeProperty(int *storage, bool bReadOnly = false) :
        drwnIntegerProperty(storage, bReadOnly), _lowerBound(0), _upperBound(100) { }
    drwnRangeProperty(int *storage, int lb, int ub, bool bReadOnly = false) :
        drwnIntegerProperty(storage, bReadOnly), _lowerBound(lb), _upperBound(ub) { }

    drwnPropertyInterface* clone() const { return new drwnRangeProperty(*this); }

    bool setProperty(int value);
};

class drwnDoubleProperty : public drwnStoragePropertyInterface<double>
{
 public:
    drwnDoubleProperty(double *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<double>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_DOUBLE_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnDoubleProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);
};

class drwnDoubleRangeProperty : public drwnDoubleProperty
{
 protected:
    double _lowerBound;
    double _upperBound;

 public:
    drwnDoubleRangeProperty(double *storage, bool bReadOnly = false) :
        drwnDoubleProperty(storage, bReadOnly), _lowerBound(0.0), _upperBound(1.0) { }
    drwnDoubleRangeProperty(double *storage, double lb, double ub, bool bReadOnly = false) :
        drwnDoubleProperty(storage, bReadOnly), _lowerBound(lb), _upperBound(ub) { }

    drwnPropertyInterface* clone() const { return new drwnDoubleRangeProperty(*this); }

    bool setProperty(double value);
};

class drwnStringProperty : public drwnStoragePropertyInterface<string>
{
 public:
    drwnStringProperty(string *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<string>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_STRING_PROPERTY; }
    string asString() const;

    drwnPropertyInterface* clone() const { return new drwnStringProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);
};

class drwnFilenameProperty : public drwnStringProperty
{
 public:
    drwnFilenameProperty(string *storage, bool bReadOnly = false) :
        drwnStringProperty(storage, bReadOnly) { }
    drwnPropertyType type() const { return DRWN_FILENAME_PROPERTY; }
    drwnPropertyInterface* clone() const { return new drwnFilenameProperty(*this); }
};

class drwnDirectoryProperty : public drwnStringProperty
{
 public:
    drwnDirectoryProperty(string *storage, bool bReadOnly = false) :
        drwnStringProperty(storage, bReadOnly) { }
    drwnPropertyType type() const { return DRWN_DIRECTORY_PROPERTY; }
    drwnPropertyInterface* clone() const { return new drwnDirectoryProperty(*this); }
};

class drwnListProperty : public drwnStoragePropertyInterface<list<string> >
{
 public:
    drwnListProperty(list<string> *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<list<string> >(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_LIST_PROPERTY; }
    string asString() const;

    drwnPropertyInterface* clone() const { return new drwnListProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);
};

class drwnSelectionProperty : public drwnStoragePropertyInterface<int> {
 protected:
    const vector<string> *_choices;

 public:
    drwnSelectionProperty(int *storage, const vector<string>* choices,
        bool bReadOnly = false) : drwnStoragePropertyInterface<int>(storage, bReadOnly),
        _choices(choices) { }

    drwnPropertyType type() const { return DRWN_SELECTION_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnSelectionProperty(*this); }

    const vector<string> *getChoices() const { return _choices; }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);

    // i/o
    void read(drwnXMLNode& xml);
    void write(drwnXMLNode& xml) const;
};

class drwnVectorProperty : public drwnStoragePropertyInterface<Eigen::VectorXd>
{
 public:
    drwnVectorProperty(Eigen::VectorXd *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<Eigen::VectorXd>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_VECTOR_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnVectorProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);

    // i/o
    void read(drwnXMLNode& xml);
    void write(drwnXMLNode& xml) const;
};

class drwnMatrixProperty : public drwnStoragePropertyInterface<Eigen::MatrixXd>
{
 public:
    drwnMatrixProperty(Eigen::MatrixXd *storage, bool bReadOnly = false) :
        drwnStoragePropertyInterface<Eigen::MatrixXd>(storage, bReadOnly) { }

    drwnPropertyType type() const { return DRWN_MATRIX_PROPERTY; }
    string asString() const;
    drwnPropertyInterface* clone() const { return new drwnMatrixProperty(*this); }

    bool setProperty(bool value);
    bool setProperty(int value);
    bool setProperty(double value);
    bool setProperty(const string& value);
    bool setProperty(const Eigen::VectorXd& value);
    bool setProperty(const Eigen::MatrixXd& value);

    // i/o
    void read(drwnXMLNode& xml);
    void write(drwnXMLNode& xml) const;
};

// drwnProperties -------------------------------------------------------------

/*!
** \brief Provides an abstract interface for dynamic properties.
** 
** Defines an abstract interface for setting arbitrary properties used by
** other classes. The class provides a unified interface making it easy
** to set properties via XML or GUI interface. It also does lots of error
** checking. The XML schema has one of the following forms:
** \code
**    <property name="..." value="..." />
**    <property name="...">
**      ...
**    </property>
** \endcode
** The class is related to the dynamic properties design pattern.
**
** Property values are stored in a class member variable.
**
** \todo Could also store within property interface object (which
** would make for slower access within class members, but easier
** declaration of lots of properties).
**
** \warning This interface is not stable.
*/

class drwnProperties {
 private:
    // Properties. This data member is private, it cannot be accessed directly
    // from derived classes. Instead derived classes should use the access
    // functions: declareProperty and undeclareProperty. All properties need
    // to be declared before they can be set. This is usually done in the
    // derived class's constructor.

    // TODO: replace below with templated drwnOrderedMap class with:
    //   vector<pair<map<string, unsigned>::iterator, T> > _entries;
    //   map<string, unsigned> _index;
    //typedef drwnSmartPointer<drwnPropertyInterface> drwnPropertyInterfacePtr;
    typedef drwnPropertyInterface* drwnPropertyInterfacePtr;
    vector<pair<string, drwnPropertyInterfacePtr> > _properties;
    map<string, unsigned> _propertiesIndex;

 public:
    drwnProperties() { /* do nothing */ }
    virtual ~drwnProperties();

    // Returns the number of properties and mapping from property
    // names to index.
    unsigned numProperties() const { return _properties.size(); }
    bool hasProperty(const string& name) const;
    bool hasProperty(const char *name) const { return hasProperty(string(name)); }
    unsigned findProperty(const string& name) const;
    unsigned findProperty(const char *name) const { return findProperty(string(name)); }

    // Called by owner of the derived class to set properties. Incorrect
    // types are cast into the appropriate form.
    void setProperty(unsigned indx, bool value);
    void setProperty(unsigned indx, int value);
    void setProperty(unsigned indx, double value);
    void setProperty(unsigned indx, const string& value);
    void setProperty(unsigned indx, const char *value);
    void setProperty(unsigned indx, const Eigen::VectorXd& value);
    void setProperty(unsigned indx, const Eigen::MatrixXd& value);

    void setProperty(const char *name, bool value);
    void setProperty(const char *name, int value);
    void setProperty(const char *name, double value);
    void setProperty(const char *name, const string& value);
    void setProperty(const char *name, const char *value);
    void setProperty(const char *name, const Eigen::VectorXd& value);
    void setProperty(const char *name, const Eigen::MatrixXd& value);

    // Used to access property settings.
    string getPropertyAsString(unsigned indx) const;
    drwnPropertyType getPropertyType(unsigned indx) const;
    bool isReadOnly(unsigned indx) const;

    const drwnPropertyInterface *getProperty(unsigned indx) const;
    const drwnPropertyInterface *getProperty(const char *name) const;
    bool getBoolProperty(unsigned indx) const;
    int getIntProperty(unsigned indx) const;
    double getDoubleProperty(unsigned indx) const;
    const string& getStringProperty(unsigned indx) const;
    const list<string>& getListProperty(unsigned indx) const;
    int getSelectionProperty(unsigned indx) const;
    const Eigen::VectorXd& getVectorProperty(unsigned indx) const;
    const Eigen::MatrixXd& getMatrixProperty(unsigned indx) const;

    // Return list of all property names.
    const string& getPropertyName(unsigned indx) const;
    vector<string> getPropertyNames() const;

    // Serialization/deserialization.
    void readProperties(drwnXMLNode& xml, const char *tag = "property");
    void writeProperties(drwnXMLNode& xml, const char *tag = "property") const;
    void printProperties(ostream &os) const;

 protected:
    // Declare properties in the constructor for the class. An undeclare
    // method is provided to be able to remove properties in derived classes.
    // Ownership of the drwnPropertyInterface object is transferred to this
    // class and will be destoryed when no longer needed.
    void declareProperty(const string& name, drwnPropertyInterface *optif);

    // Remove an property declared in a base class. Allows class to remove
    // exposure to certain base class properties.
    void undeclareProperty(const string& name);

    // Mirror properties of a contained class. Since the property interface points
    // to contained data members, this should only be used with static data
    // members.
    void exposeProperties(drwnProperties *opts, const string& prefix = string(""),
        bool bSerializable = false);

    // Called whenever an property changes (i.e. setProperty or setProperties
    // is called). Should be overridden by derived classed. Can be used to
    // validate properties or setup internal datastructures. The default
    // behaviour is to do nothing. This is a simpler interface than defining
    // a new drwnPropertyInterface type, but is only called *after* the property
    // is set, which may be problematic in some cases.
    virtual void propertyChanged(const string& name) {
        // do nothing
    }

 private:
    // properties cannot be copied because they point to member variables
    drwnProperties(const drwnProperties& o) { DRWN_ASSERT(false); }
};

// drwnPropertiesCopy ------------------------------------------------------

class drwnPropertiesCopy : public drwnProperties {
 protected:
    // copy of property value and modification flag
    vector<pair<void *, bool> > _data;

 public:
    drwnPropertiesCopy(const drwnProperties *o);
    ~drwnPropertiesCopy();

    void copyBack(drwnProperties *o) const;

 protected:
    void propertyChanged(const string& name);
};

