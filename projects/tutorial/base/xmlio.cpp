/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    xmlio.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;

// person class --------------------------------------------------------------

class Person : public drwnWriteable {
protected:
    string _name;
    int _age;

public:
    Person() : _age(0) { /* do nothing */ }
    Person(const string& name, int age) :
        _name(name), _age(age) { /* do nothing */ }

    const char* type() const { return "Person"; }

    bool save(drwnXMLNode& xml) const {
        drwnAddXMLAttribute(xml, "name", _name.c_str(), false);
        drwnAddXMLAttribute(xml, "age", toString(_age).c_str(), false);
        return true;
    }

    bool load(drwnXMLNode& xml) {
        _name = string(drwnGetXMLAttribute(xml, "name"));
        _age = atoi(drwnGetXMLAttribute(xml, "age"));
        return true;
    }
};

// group class ---------------------------------------------------------------

class Group : public drwnWriteable {
protected:
    vector<Person> _people;

public:
    Group() { /* do nothing */ }

    const char* type() const { return "Group"; }

    bool save(drwnXMLNode& xml) const {
        drwnXMLUtils::save(xml, "Person", _people);
        return true;
    }

    bool load(drwnXMLNode& xml) {
        _people.clear();
        drwnXMLUtils::load(xml, "Person", _people);
        return true;
    }

    void addMember(const Person& person) {
        _people.push_back(person);
    }
};

// main ----------------------------------------------------------------------

int main()
{
    Group group;

    group.addMember(Person("Bart", 10));
    group.addMember(Person("Lisa", 8));
    group.addMember(Person("Maggie", 1));

    group.write("TheSimpsons.xml");
    group.dump();

    return 0;
}
