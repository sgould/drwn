/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPropertys.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <cassert>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "drwnConstants.h"
#include "drwnCompatibility.h"
#include "drwnLogger.h"
#include "drwnStrUtils.h"
#include "drwnProperties.h"
#include "drwnXMLUtils.h"

#include "Eigen/Core"

using namespace std;

#define PRINT_ERRORS
//#undef PRINT_ERRORS

// drwnPropertyInterface --------------------------------------------------

// set/cast functions
bool drwnPropertyInterface::setProperty(bool value)
{
    DRWN_ASSERT_MSG(false, "property does not implement bool");
    return false;
}

bool drwnPropertyInterface::setProperty(int value)
{
    DRWN_ASSERT_MSG(false, "property does not implement int");
    return false;
}

bool drwnPropertyInterface::setProperty(double value)
{
    DRWN_ASSERT_MSG(false, "property does not implement double");
    return false;
}

bool drwnPropertyInterface::setProperty(const string& value)
{
    DRWN_ASSERT_MSG(false, "property does not implement string");
    return false;
}

bool drwnPropertyInterface::setProperty(const char *value)
{
    return setProperty(string(value));
}

bool drwnPropertyInterface::setProperty(const Eigen::VectorXd& value)
{
    DRWN_ASSERT_MSG(false, "property does not implement vector");
    return false;
}

bool drwnPropertyInterface::setProperty(const Eigen::MatrixXd& value)
{
    DRWN_ASSERT_MSG(false, "property does not implement matrix");
    return false;
}

// i/o
void drwnPropertyInterface::read(drwnXMLNode& xml)
{
    drwnXMLAttr *a = xml.first_attribute("value");
    const char *value = (a == NULL) ? xml.value() : a->value();

    DRWN_ASSERT_MSG(value != NULL, "XML node is missing value attribute");
    bool bSuccess = setProperty(string(value));
    DRWN_ASSERT_MSG(bSuccess, "invalue property string " << value);
}

void drwnPropertyInterface::write(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "value", this->asString().c_str(), false);
}

// drwnBooleanProperty ------------------------------------------------------

string drwnBooleanProperty::asString() const
{
    return (*_storage) ? string("1") : string("0");
}

bool drwnBooleanProperty::setProperty(bool value)
{
    *_storage = value;
    return true;
}

bool drwnBooleanProperty::setProperty(int value)
{
    return this->setProperty(value != 0);
}

bool drwnBooleanProperty::setProperty(double value)
{
    return this->setProperty(value != 0.0);
}

bool drwnBooleanProperty::setProperty(const string& value)
{
    return this->setProperty((!strcasecmp(value.c_str(), "true")) ||
        (!strcasecmp(value.c_str(), "yes")) ||
        (!strcasecmp(value.c_str(), "1")));
}

bool drwnBooleanProperty::setProperty(const Eigen::VectorXd& value)
{
    return this->setProperty(value.size() != 0);
}

bool drwnBooleanProperty::setProperty(const Eigen::MatrixXd& value)
{
    return this->setProperty(value.size() != 0);
}

// drwnIntegerProperty ------------------------------------------------------

string drwnIntegerProperty::asString() const
{
    return ::toString(*_storage);
}

bool drwnIntegerProperty::setProperty(bool value)
{
    return this->setProperty(value ? 1 : 0);
}

bool drwnIntegerProperty::setProperty(int value)
{
    *_storage = value;
    return true;
}

bool drwnIntegerProperty::setProperty(double value)
{
    return this->setProperty((int)value);
}

bool drwnIntegerProperty::setProperty(const string& value)
{
    return this->setProperty(atoi(value.c_str()));
}

bool drwnIntegerProperty::setProperty(const Eigen::VectorXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnIntegerProperty::setProperty(const Eigen::MatrixXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

// drwnRangeProperty --------------------------------------------------------

bool drwnRangeProperty::setProperty(int value)
{
    *_storage = std::min(std::max(value, _lowerBound), _upperBound);
    return true;
}

// drwnDoubleProperty -------------------------------------------------------

string drwnDoubleProperty::asString() const
{
    return ::toString(*_storage);
}

bool drwnDoubleProperty::setProperty(bool value)
{
    return this->setProperty(value ? 1.0 : 0.0);
}

bool drwnDoubleProperty::setProperty(int value)
{
    return this->setProperty((double)value);
}

bool drwnDoubleProperty::setProperty(double value)
{
    *_storage = value;
    return true;
}

bool drwnDoubleProperty::setProperty(const string& value)
{
    return this->setProperty(atof(value.c_str()));
}

bool drwnDoubleProperty::setProperty(const Eigen::VectorXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnDoubleProperty::setProperty(const Eigen::MatrixXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

// drwnDoubleRangeProperty --------------------------------------------------

bool drwnDoubleRangeProperty::setProperty(double value)
{
    *_storage = std::min(std::max(value, _lowerBound), _upperBound);
    return true;
}

// drwnStringProperty -------------------------------------------------------

string drwnStringProperty::asString() const
{
    return *_storage;
}

bool drwnStringProperty::setProperty(bool value)
{
    return this->setProperty(value ? string("1") : string("0"));
}

bool drwnStringProperty::setProperty(int value)
{
    return this->setProperty(::toString(value));
}

bool drwnStringProperty::setProperty(double value)
{
    return this->setProperty(::toString(value));
}

bool drwnStringProperty::setProperty(const string& value)
{
    *_storage = value;
    return true;
}

bool drwnStringProperty::setProperty(const Eigen::VectorXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnStringProperty::setProperty(const Eigen::MatrixXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

// drwnListProperty ---------------------------------------------------------

string drwnListProperty::asString() const
{
    return ::toString(*_storage);
}

bool drwnListProperty::setProperty(bool value)
{
    return this->setProperty(value ? string("1") : string("0"));
}

bool drwnListProperty::setProperty(int value)
{
    return this->setProperty(::toString(value));
}

bool drwnListProperty::setProperty(double value)
{
    return this->setProperty(::toString(value));
}

bool drwnListProperty::setProperty(const string& value)
{
    _storage->clear();

    stringstream buffer;
    string token;

    buffer << value;

    while (1) {
        buffer >> token;
        if (buffer.fail()) {
            break;
        }
        _storage->push_back(token);
        if (buffer.eof()) break;
    }

    return true;
}

bool drwnListProperty::setProperty(const Eigen::VectorXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnListProperty::setProperty(const Eigen::MatrixXd& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

// drwnSelectionProperty ----------------------------------------------------

string drwnSelectionProperty::asString() const
{
    return (*_storage < 0) ? string("") : (*_choices)[*_storage];
}

bool drwnSelectionProperty::setProperty(bool value)
{
    if (!value) *_storage = -1;
    return true;
}

bool drwnSelectionProperty::setProperty(int value)
{
    *_storage = value;
    return true;
}

bool drwnSelectionProperty::setProperty(double value)
{
    return this->setProperty((int)value);
}

bool drwnSelectionProperty::setProperty(const string& value)
{
    *_storage = -1;
    for (int n = 0; n < (int)_choices->size(); n++) {
        if (!strcasecmp(value.c_str(), (*_choices)[n].c_str())) {
            *_storage = n;
            break;
        }
    }

    return true;
}

// i/o
void drwnSelectionProperty::read(drwnXMLNode& xml)
{
    drwnXMLAttr *a = xml.first_attribute("value");
    const char *value = (a == NULL) ? drwnGetXMLText(xml) : a->value();

    *_storage = atoi(value);
    if ((*_storage < -1) || (*_storage >= (int)_choices->size())) {
        DRWN_LOG_ERROR("invalid choice " << *_storage << " for selection property");
        *_storage = -1;
    }
}

void drwnSelectionProperty::write(drwnXMLNode& xml) const
{
    string value = ::toString(*_storage);
    for (int i = 0; i < (int)_choices->size(); i++) {
        string textComment((*_choices)[i]);
        if (i == *_storage) {
            textComment = string("[") + textComment + string("]");
        }
        xml.append_node(xml.document()->allocate_node(rapidxml::node_comment,
                xml.document()->allocate_string(textComment.c_str())));
    }
    drwnAddXMLAttribute(xml, "value", value.c_str(), false);
}

// drwnVectorProperty ------------------------------------------------------

string drwnVectorProperty::asString() const
{
    std::stringstream s;
    s << _storage->rows() << "-vector";
    return s.str();
}

bool drwnVectorProperty::setProperty(bool value)
{
    *_storage = Eigen::VectorXd(1);
    (*_storage)[0] = value ? 1.0 : 0.0;
    return true;
}

bool drwnVectorProperty::setProperty(int value)
{
    *_storage = Eigen::VectorXd(1);
    (*_storage)[0] = (double)value;
    return true;
}

bool drwnVectorProperty::setProperty(double value)
{
    *_storage = Eigen::VectorXd(1);
    (*_storage)[0] = value;
    return true;
}

bool drwnVectorProperty::setProperty(const string& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnVectorProperty::setProperty(const Eigen::VectorXd& value)
{
    *_storage = value;
    return true;
}

bool drwnVectorProperty::setProperty(const Eigen::MatrixXd& value)
{
    *_storage = value;
    return true;
}

// i/o
void drwnVectorProperty::read(drwnXMLNode& xml)
{
    drwnXMLUtils::deserialize(xml, *_storage);
}

void drwnVectorProperty::write(drwnXMLNode& xml) const
{
    drwnXMLUtils::serialize(xml, *_storage);
}

// drwnMatrixProperty -------------------------------------------------------

string drwnMatrixProperty::asString() const
{
    std::stringstream s;
    s << "(" << _storage->rows() << "-by-" << _storage->cols() << ")-matrix";
    return s.str();
}

bool drwnMatrixProperty::setProperty(bool value)
{
    *_storage = Eigen::MatrixXd(1, 1);
    (*_storage)(0) = value ? 1.0 : 0.0;
    return true;
}

bool drwnMatrixProperty::setProperty(int value)
{
    *_storage = Eigen::MatrixXd(1, 1);
    (*_storage)(0) = (double)value;
    return true;
}

bool drwnMatrixProperty::setProperty(double value)
{
    *_storage = Eigen::MatrixXd(1, 1);
    (*_storage)(0) = value;
    return true;
}

bool drwnMatrixProperty::setProperty(const string& value)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return true;
}

bool drwnMatrixProperty::setProperty(const Eigen::VectorXd& value)
{
    *_storage = value;
    return true;
}

bool drwnMatrixProperty::setProperty(const Eigen::MatrixXd& value)
{
    *_storage = value;
    return true;
}

// i/o
void drwnMatrixProperty::read(drwnXMLNode& xml)
{
    drwnXMLUtils::deserialize(xml, *_storage);
}

void drwnMatrixProperty::write(drwnXMLNode& xml) const
{
    drwnXMLUtils::serialize(xml, *_storage);
}

// drwnProperties public members ---------------------------------------------

drwnProperties::~drwnProperties()
{
#if 1
    // NOW SMART POINTERS

    // delete interface objects
    for (vector<pair<string, drwnPropertyInterface *> >::iterator jt = _properties.begin();
         jt != _properties.end(); jt++) {
        delete jt->second;
    }
#endif
}

bool drwnProperties::hasProperty(const string& name) const
{
    return (_propertiesIndex.find(name) != _propertiesIndex.end());
}

unsigned drwnProperties::findProperty(const string& name) const
{
    map<string, unsigned>::const_iterator it = _propertiesIndex.find(name);
#ifdef PRINT_ERRORS
    if (it == _propertiesIndex.end()) {
	cerr << "ERROR: property \"" << name.c_str() << "\" not declared" << endl;
        cerr << "Properties are:";
        for (vector<pair<string, drwnPropertyInterfacePtr> >::const_iterator jt = _properties.begin();
             jt != _properties.end(); jt++) {
            cerr << " " << jt->first;
        }
        cerr << endl;
    }
#endif
    DRWN_ASSERT(it != _propertiesIndex.end());

    return it->second;
}

void drwnProperties::setProperty(unsigned indx, bool value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(unsigned indx, int value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(unsigned indx, double value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(unsigned indx, const string& value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(unsigned indx, const Eigen::VectorXd& value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(unsigned indx, const Eigen::MatrixXd& value)
{
    DRWN_ASSERT(indx < _properties.size());
    drwnPropertyInterface *e = _properties[indx].second;
    if (e->isReadOnly()) {
        DRWN_LOG_ERROR("attempted to set read-only property \"" << _properties[indx].first << "\"");
        return;
    }

    // change property and notify derived class
    e->setProperty(value);
    propertyChanged(_properties[indx].first);
}

void drwnProperties::setProperty(const char *name, bool value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, int value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, double value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, const string& value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, const char *value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, const Eigen::VectorXd& value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(const char *name, const Eigen::MatrixXd& value)
{
    setProperty(findProperty(name), value);
}

void drwnProperties::setProperty(unsigned indx, const char *value)
{
    setProperty(indx, string(value));
}

string drwnProperties::getPropertyAsString(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    return _properties[indx].second->asString();
}

drwnPropertyType drwnProperties::getPropertyType(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    return _properties[indx].second->type();
}

bool drwnProperties::isReadOnly(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    return _properties[indx].second->isReadOnly();
}

const drwnPropertyInterface *drwnProperties::getProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    return _properties[indx].second;
}

const drwnPropertyInterface *drwnProperties::getProperty(const char *name) const
{
    DRWN_ASSERT(name != NULL);
    return _properties[findProperty(name)].second;
}

bool drwnProperties::getBoolProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_BOOLEAN_PROPERTY);
    return ((drwnBooleanProperty *)&(*_properties[indx].second))->getValue();
}

int drwnProperties::getIntProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT((_properties[indx].second->type() == DRWN_INTEGER_PROPERTY) ||
        (_properties[indx].second->type() == DRWN_SELECTION_PROPERTY));
    return ((drwnIntegerProperty *)&(*_properties[indx].second))->getValue();
}

double drwnProperties::getDoubleProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_DOUBLE_PROPERTY);
    return ((drwnDoubleProperty *)&(*_properties[indx].second))->getValue();
}

const string& drwnProperties::getStringProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT((_properties[indx].second->type() == DRWN_STRING_PROPERTY) ||
        (_properties[indx].second->type() == DRWN_FILENAME_PROPERTY) ||
        (_properties[indx].second->type() == DRWN_DIRECTORY_PROPERTY));
    return ((drwnStringProperty *)&(*_properties[indx].second))->getValue();
}

const list<string>& drwnProperties::getListProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_LIST_PROPERTY);
    return ((drwnListProperty *)&(*_properties[indx].second))->getValue();
}

int drwnProperties::getSelectionProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_SELECTION_PROPERTY);
    return ((drwnSelectionProperty *)&(*_properties[indx].second))->getValue();
}

const Eigen::VectorXd& drwnProperties::getVectorProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_VECTOR_PROPERTY);
    return ((drwnVectorProperty *)&(*_properties[indx].second))->getValue();
}

const Eigen::MatrixXd& drwnProperties::getMatrixProperty(unsigned indx) const
{
    DRWN_ASSERT(indx < _properties.size());
    DRWN_ASSERT(_properties[indx].second->type() == DRWN_MATRIX_PROPERTY);
    return ((drwnMatrixProperty *)&(*_properties[indx].second))->getValue();
}

const string& drwnProperties::getPropertyName(unsigned indx) const
{
#ifdef PRINT_ERRORS
    if (indx >= _properties.size()) {
	cerr << "ERROR: property " << indx << " not declared (" << numProperties() << ")" << endl;
        cerr << "Properties are:";
        for (vector<pair<string, drwnPropertyInterfacePtr> >::const_iterator jt = _properties.begin();
             jt != _properties.end(); jt++) {
            cerr << " " << jt->first;
        }
        cerr << endl;
    }
#endif
    DRWN_ASSERT_MSG(indx < _properties.size(), indx << " !< " << _properties.size());
    return _properties[indx].first;
}

vector<string> drwnProperties::getPropertyNames() const
{
    vector<string> keys;

    keys.reserve(_properties.size());
    for (vector<pair<string, drwnPropertyInterfacePtr> >::const_iterator it = _properties.begin();
         it != _properties.end(); it++) {
	keys.push_back(it->first);
    }

    return keys;
}

void drwnProperties::readProperties(drwnXMLNode& xml, const char *tag)
{
    DRWN_ASSERT(tag != NULL);

    int i = 0;
    for (drwnXMLNode *nd = xml.first_node(tag); nd != NULL; nd = nd->next_sibling(tag), i++) {
        drwnXMLNode& node = *nd;
        drwnXMLAttr *a = nd->first_attribute("name");
        const char *nodeName = (a == NULL) ? NULL : a->value();

        DRWN_ASSERT_MSG(nodeName != NULL,
            "missing " << (i + 1) << "-th property name");

        // get the property
        map<string, unsigned>::iterator it = _propertiesIndex.find(nodeName);
        if (it == _propertiesIndex.end()) {
            DRWN_LOG_WARNING("property \"" << nodeName << "\" is not defined");
            continue;
        }
        drwnPropertyInterface *e = _properties[it->second].second;

        // skip read-only properties
        if (e->isReadOnly()) {
            DRWN_LOG_WARNING("skipping read-only property \"" << it->first << "\"");
            continue;
        }

        // deserialize the property
        e->read(node);
    }
}

void drwnProperties::writeProperties(drwnXMLNode& xml, const char *tag) const
{
    // add properties
    for (unsigned i = 0; i < _properties.size(); i++) {
        const drwnPropertyInterface *e = _properties[i].second;

        // skip read-only and non-serializeable properties
        if (e->isReadOnly() || !e->isSerializeable()) {
            continue;
        }

        // add xml node
        drwnXMLNode *node = drwnAddXMLChildNode(xml, tag);
        drwnAddXMLAttribute(*node, "name", _properties[i].first.c_str(), false);
        e->write(*node);
    }
}

void drwnProperties::printProperties(ostream &os) const
{
    for (unsigned i = 0; i < _properties.size(); i++) {
        const drwnPropertyInterface *e = _properties[i].second;

        cout << setw(12) << _properties[i].first << " :: ";
        cout << (e->isReadOnly() ? "R" : " ") << " ::  ";
        cout << e->asString() << endl;
    }
}

// drwnProperties protected members -------------------------------------------

void drwnProperties::declareProperty(const string& name, drwnPropertyInterface *optif)
{
    // check that property hasn't already been declared
    DRWN_ASSERT_MSG((_propertiesIndex.find(name) == _propertiesIndex.end()),
        "property \"" << name << "\" already declared");

    _propertiesIndex[name] = _properties.size();
    _properties.push_back(make_pair(name, optif));
}

void drwnProperties::undeclareProperty(const string& name)
{
    // check that property has been declared
    map<string, unsigned>::iterator it = _propertiesIndex.find(name);
    DRWN_ASSERT_MSG((it != _propertiesIndex.end()), "property \"" << name << "\" not declared");

    _properties.erase(_properties.begin() + it->second);
    _propertiesIndex.erase(it);
}

void drwnProperties::exposeProperties(drwnProperties *opts, const string& prefix, bool bSerialize)
{
    DRWN_ASSERT(opts != NULL);
    for (unsigned i = 0; i < opts->_properties.size(); i++) {
        string name = prefix + opts->_properties[i].first;

        DRWN_ASSERT_MSG((_propertiesIndex.find(name) == _propertiesIndex.end()),
            "property \"" << name << "\" already declared");

        _propertiesIndex[name] = _properties.size();
        drwnPropertyInterfacePtr p = opts->_properties[i].second->clone(false, bSerialize);
        //pair<string, drwnPropertyInterfacePtr> p(name, opts->_properties[i].second);
        _properties.push_back(make_pair(name, p));
    }
}

// drwnPropertiesCopy ------------------------------------------------------

drwnPropertiesCopy::drwnPropertiesCopy(const drwnProperties *o) : drwnProperties()
{
    DRWN_ASSERT(o != NULL);
    _data.reserve(o->numProperties());
    for (unsigned i = 0; i < o->numProperties(); i++) {
        void *p = NULL;
        switch (o->getPropertyType(i)) {
        case DRWN_INVALID_PROPERTY:
            break;
        case DRWN_BOOLEAN_PROPERTY:
            p = (void *)new bool(o->getBoolProperty(i));
            declareProperty(o->getPropertyName(i), new drwnBooleanProperty((bool *)p, o->isReadOnly(i)));
            break;
        case DRWN_INTEGER_PROPERTY:
        case DRWN_SELECTION_PROPERTY:
            p = (void *)new int(o->getIntProperty(i));
            declareProperty(o->getPropertyName(i), new drwnIntegerProperty((int *)p, o->isReadOnly(i)));
            break;
        case DRWN_DOUBLE_PROPERTY:
            p = (void *)new double(o->getDoubleProperty(i));
            declareProperty(o->getPropertyName(i), new drwnDoubleProperty((double *)p, o->isReadOnly(i)));
            break;
        case DRWN_STRING_PROPERTY:
            p = (void *)new string(o->getStringProperty(i));
            declareProperty(o->getPropertyName(i), new drwnStringProperty((string *)p, o->isReadOnly(i)));
            break;
        case DRWN_FILENAME_PROPERTY:
            p = (void *)new string(o->getStringProperty(i));
            declareProperty(o->getPropertyName(i), new drwnFilenameProperty((string *)p, o->isReadOnly(i)));
            break;
        case DRWN_DIRECTORY_PROPERTY:
            p = (void *)new string(o->getStringProperty(i));
            declareProperty(o->getPropertyName(i), new drwnDirectoryProperty((string *)p, o->isReadOnly(i)));
            break;
        case DRWN_VECTOR_PROPERTY:
            p = (void *)new Eigen::VectorXd(o->getVectorProperty(i));
            declareProperty(o->getPropertyName(i), new drwnVectorProperty((Eigen::VectorXd *)p, o->isReadOnly(i)));
            break;
        case DRWN_MATRIX_PROPERTY:
            p = (void *)new Eigen::MatrixXd(o->getMatrixProperty(i));
            declareProperty(o->getPropertyName(i), new drwnMatrixProperty((Eigen::MatrixXd *)p, o->isReadOnly(i)));
            break;
        default:
            DRWN_LOG_FATAL("unrecognized type for property \"" << getPropertyName(i) << "\"");
        }
        DRWN_ASSERT(p != NULL);

        _data.push_back(make_pair(p, false));
    }
}

drwnPropertiesCopy::~drwnPropertiesCopy()
{
    DRWN_ASSERT(_data.size() == numProperties());
    for (unsigned i = 0; i < _data.size(); i++) {
        switch (getPropertyType(i)) {
        case DRWN_BOOLEAN_PROPERTY:
            delete (bool *)_data[i].first;
            break;
        case DRWN_INTEGER_PROPERTY:
            delete (int *)_data[i].first;
            break;
        case DRWN_DOUBLE_PROPERTY:
            delete (double *)_data[i].first;
            break;
        case DRWN_STRING_PROPERTY:
        case DRWN_FILENAME_PROPERTY:
        case DRWN_DIRECTORY_PROPERTY:
            delete (string *)_data[i].first;
            break;
        case DRWN_SELECTION_PROPERTY:
            delete (int *)_data[i].first;
            break;
        case DRWN_VECTOR_PROPERTY:
            delete (Eigen::VectorXd *)_data[i].first;
            break;
        case DRWN_MATRIX_PROPERTY:
            delete (Eigen::MatrixXd *)_data[i].first;
            break;
        default:
            DRWN_LOG_FATAL("unrecognized type for property \"" << getPropertyName(i) << "\"");
        }
    }
    _data.clear();
}

void drwnPropertiesCopy::copyBack(drwnProperties *o) const
{
    DRWN_ASSERT(o != NULL);

    for (unsigned i = 0; i < _data.size(); i++) {
        if (!_data[i].second) continue;
        DRWN_ASSERT(o->getPropertyType(i) == this->getPropertyType(i));
        switch (o->getPropertyType(i)) {
        case DRWN_BOOLEAN_PROPERTY:
            o->setProperty(i, this->getBoolProperty(i));
            break;
        case DRWN_SELECTION_PROPERTY:
        case DRWN_INTEGER_PROPERTY:
            o->setProperty(i, this->getIntProperty(i));
            break;
        case DRWN_DOUBLE_PROPERTY:
            o->setProperty(i, this->getDoubleProperty(i));
            break;
        case DRWN_STRING_PROPERTY:
        case DRWN_FILENAME_PROPERTY:
        case DRWN_DIRECTORY_PROPERTY:
            o->setProperty(i, this->getStringProperty(i));
            break;
        case DRWN_VECTOR_PROPERTY:
            o->setProperty(i, this->getVectorProperty(i));
            break;
        case DRWN_MATRIX_PROPERTY:
            o->setProperty(i, this->getMatrixProperty(i));
            break;
        default:
            DRWN_LOG_FATAL("unrecognized type for property \"" << o->getPropertyName(i) << "\"");
        }
    }
}

void drwnPropertiesCopy::propertyChanged(const string &name)
{
    unsigned indx = findProperty(name);
    _data[indx].second = true;
}

