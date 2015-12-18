#include "drwnGeneralFactor.h"
#include "drwnNumber.h"

using namespace std::chrono;

void drwnGeneralFactor::addVariable(int var)
{	
    DRWN_ASSERT((var >= 0) && (var < _pUniverse->numVariables()));
	DRWN_ASSERT(find(_variables.begin(), _variables.end(), var) == _variables.end());
    _variables.push_back(var);
}

void drwnGeneralFactor::setOps(const drwnGeneralFactor *f1, const drwnGeneralFactor *f2)
{
	_storageType = EVAL;
	_op1 = f1;
	_op2 = f2;
}

double drwnGeneralFactor::getValueOf(const drwnPartialAssignment& y) const
{
	if (_storageType == EVAL) {
		return _op1->getValueOf(y) + _op2->getValueOf(y);
	}

	if (_storageType == DENSE) {
		return _tblFac->getValueOf(y);
	}

	return _sparseFac->getValueOf(y);
}

void drwnGeneralFactor::setValueOf(const drwnFullAssignment& y, double val) {
	if (_storageType == DENSE) {
		_tblFac->setValueOf(y, val);
		return;
	}

	DRWN_ASSERT(_storageType == SPARSE);	// can't assign to evaluation factors
	_sparseFac->setValueOf(y, val);
}

void drwnGeneralFactor::setValueOf(const drwnPartialAssignment& y, double val) {
	if (_storageType == DENSE) {
		_tblFac->setValueOf(y, val);
		return;
	}
	
	DRWN_ASSERT(_storageType == SPARSE);	// can't assign to evaluation factors
	_sparseFac->setValueOf(y, val);
}

drwnTableFactor *drwnGeneralFactor::getTableFactor() const
{
	DRWN_ASSERT(_tblFac);
	return _tblFac;
}

drwnSparseFactor *drwnGeneralFactor::getSparseFactor() const
{
	DRWN_ASSERT(_sparseFac);
	return _sparseFac;
}

void drwnGeneralFactor::subtractMostCommonVal()
{	
    map<double, int> tally;
    double mostFreqVal;
    int maxFreq = 0;

    for (int i = 0; i < _tblFac->entries(); i++) {
		drwnPartialAssignment pa;
        _tblFac->assignmentOf(i, pa);
        double val = _tblFac->getValueOf(pa);

        if (tally.find(val) == tally.end()) {
			tally[val] = 1;
        } else {
			tally[val]++;
        }

        if (maxFreq < tally[val]) {
			maxFreq = tally[val];
            mostFreqVal = val;
        }
    }

	for (int i = 0; i < _tblFac->entries(); i++) {
		(*_tblFac)[i] -= mostFreqVal;
	}
}

void drwnGeneralFactor::convertIfSparse()
{
	DRWN_ASSERT(_storageType == DENSE);
	int numNonZeros = 0;

	for (int ent = 0; ent < _tblFac->entries(); ent++) {
		if (!drwn::eq((*_tblFac)[ent], 0)) {
			numNonZeros++;
		}
	}

	if (drwn::lt(THRESHOLD, (double) numNonZeros / _tblFac->entries())) {
		return;
	}

	_storageType = SPARSE;	/* store using coordinate format */
    vector<int> vars = _tblFac->getOrderedVars();

    for (vector<int>::iterator it = vars.begin(); it != vars.end(); it++) {
		_sparseFac->addVariable(_pUniverse->varName(*it).c_str());
    }

	for (int ass_idx = 0; ass_idx < _tblFac->entries(); ass_idx++) {
		drwnPartialAssignment pa;
        _tblFac->assignmentOf(ass_idx, pa);
        double cost = _tblFac->getValueOf(pa);

        if (!drwn::eq(cost, 0)) {   
            _sparseFac->setValueOf(pa, cost);
        }
    }
}
