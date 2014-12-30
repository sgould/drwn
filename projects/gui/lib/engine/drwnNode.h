/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Defines the interface for all data processing nodes. The database table
**  corresponding to an output port will have name "<nodeName>.<portName>"
**  in the database corresponding to the owning graph. The initialize and
**  finalize functions are used for setting up interfaces to external
**  processes such as Matlab.
**
*****************************************************************************/

#pragma once

using namespace std;

// forward declarations ------------------------------------------------------

class drwnGraph;
class drwnDataPort;
class drwnInputPort;
class drwnOutputPort;
class drwnGUIWindow;

// drwnNode ------------------------------------------------------------------

class drwnNode : public drwnStdObjIface, public drwnProperties {
 protected:
    unsigned _nVersion;     // version of this node (*100)
    string _desc;           // description of this node

    drwnGraph *_owner;      // graph which owns this node
    string _name;           // unique user assigned name for this node
    string _notes;          // user assigned notes for this node
    int _x, _y;             // location (in graphical layout)
    drwnGUIWindow *_window; // window (in graphical interface)

    // connectivity
    vector<drwnInputPort *> _inputPorts;
    vector<drwnOutputPort *> _outputPorts;

    // option callbacks
    class SetNameInterface : public drwnStringProperty {
    protected:
        drwnNode *_owner;

    public:
        SetNameInterface(drwnNode *owner) : drwnStringProperty(&owner->_name),
            _owner(owner) { }
        bool setProperty(const string& value) {
            _owner->setName(value);
            return true;
        }
    };

 public:
    drwnNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnNode(const drwnNode& node);
    virtual ~drwnNode();

    // accessors
    unsigned int getVersion() const { return _nVersion; }
    const char *getDescription() const { return _desc.c_str(); }

    drwnGraph *getOwner() const { return _owner; }
    void setOwner(drwnGraph *owner) { _owner = owner; };
    const string& getName() const { return _name; }
    void setName(const string& name);
    const string& getNotes() const { return _notes; }
    void setNotes(const string& notes) { _notes = notes; }
    int getLocationX() const { return _x; }
    int getLocationY() const { return _y; }
    void setLocation(int x, int y) { _x = x; _y = y; }

    // i/o
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    // gui
    virtual void showWindow();
    virtual void hideWindow();
    virtual bool isShowingWindow() const;
    virtual void updateWindow();

    // connectivity
    int numInputPorts() const { return (int)_inputPorts.size(); }
    int numOutputPorts() const { return (int)_outputPorts.size(); }
    drwnInputPort *getInputPort(int indx) const {
        DRWN_ASSERT(indx < (int)_inputPorts.size());
        return _inputPorts[indx];
    }
    drwnInputPort *getInputPort(const char *name) const;
    drwnOutputPort *getOutputPort(int indx) const {
        DRWN_ASSERT(indx < (int)_outputPorts.size());
        return _outputPorts[indx];
    }
    drwnOutputPort *getOutputPort(const char *name) const;

    // clear data tables
    void clearOutput();

    // forward evaluation
    virtual void initializeForwards(bool bClearOutput = true);
    virtual void evaluateForwards(); // overrides existing output
    virtual void updateForwards(); // only evaluates missing output
    virtual void finalizeForwards();

    // backward propagation
    virtual void initializeBackwards();
    virtual void propagateBackwards();
    virtual void finalizeBackwards();

    // learning local objective (adaptive nodes)
    virtual void resetParameters();
    virtual void initializeParameters();

    // learning global objective (adaptive nodes)
    /*
    virtual int getNumParameters() const;
    virtual void getParameters(double *x) const;
    virtual void setParameters(const double *x);
    virtual double getObjective() const;
    virtual void getGradient(double *df) const;
    virtual double getObjectiveAndGradient(double *df);
    */
};

// drwnSimpleNode ------------------------------------------------------------
// Interface for a simple data processing node having a single input and single
// output port. Each record is processed independently.

class drwnSimpleNode : public drwnNode {
 public:
    drwnSimpleNode(const char *name = NULL, drwnGraph *owner = NULL);
    ~drwnSimpleNode();

    // processing
    void evaluateForwards();
    void updateForwards();
    void propagateBackwards();

 protected:
    virtual bool forwardFunction(const string& key, const drwnDataRecord *src,
        drwnDataRecord *dst);
    virtual bool backwardGradient(const string& key, drwnDataRecord *src,
        const drwnDataRecord *dst);
};

// drwnMultiIONode -----------------------------------------------------------
// Interface for a simple data processing node having a multiple input and
// output ports. Each record is processed independently. Derived class must
// declare the ports (unlike drwnSimpleNode).

class drwnMultiIONode : public drwnNode {
 public:
    drwnMultiIONode(const char *name = NULL, drwnGraph *owner = NULL);
    ~drwnMultiIONode();

    // processing
    void evaluateForwards();
    void updateForwards();
    void propagateBackwards();

 protected:
    virtual bool forwardFunction(const string& key, 
        const vector<const drwnDataRecord *>& src,
        const vector<drwnDataRecord *>& dst);
    virtual bool backwardGradient(const string& key,
        const vector<drwnDataRecord *>& src,
        const vector<const drwnDataRecord *>& dst);
};

// drwnAdaptiveNode ----------------------------------------------------------
// Interface for a standard adaptive node with parameters that can be learned.
// Inherits from drwnOptimizer for parameter optimization, so derived class
// should also implement objectiveAndDerivative() private member function.

class drwnAdaptiveNode : public drwnNode {
 protected:
    int _trainingColour;   // colour used for training (-1 for all data)
    int _subSamplingRate;  // sub-sampling rate for training (e.g., every n-th)
    int _maxIterations;    // maximum number of iterations (for iterative algorithms)

    // local parameter regularization
    static vector<string> _regularizationChoices;
    int _regularizer;      // regularization option
    double _lambda;        // regularization strength

 public:
    drwnAdaptiveNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnAdaptiveNode(const drwnAdaptiveNode& node);
    ~drwnAdaptiveNode();

 protected:
    // computes regularization part of objective on parameter vector
    // and (optionally) updates gradient (not zeroed internally). Assumes
    // member variable _n holds the length of x and df.
    //double addRegularization(const double *x, double *df = NULL);

};
