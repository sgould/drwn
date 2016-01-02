#include "factorAdditionOp.h"

// FactorAdditionOp  ----------------------------------------------------------
FactorAdditionOp::FactorAdditionOp(drwnGeneralFactor *target, const drwnGeneralFactor *A,
		const drwnGeneralFactor *B) : drwnFactorNAryOp(target, A, B)
{
	target->setOps(A, B);
	vector<int> ordTargetVars = target->getOrderedVars();

	if (ordTargetVars.empty()) {
		vector<int> varsA = A->getOrderedVars();
		vector<int> varsB = B->getOrderedVars();
		set<int> AunionB;

		set_union(varsA.begin(), varsA.end(), varsB.begin(), varsB.end(),
			inserter(AunionB, AunionB.begin()));

		for (set<int>::iterator si = AunionB.begin(); si != AunionB.end(); si++) {
			target->addVariable(*si);
		}
	}

	vector<int> ordVarsA = A->getOrderedVars();
	vector<int> ordVarsB = B->getOrderedVars();
	set<int> varsA(ordVarsA.begin(), ordVarsA.end());
	set<int> varsB(ordVarsB.begin(), ordVarsB.end());

	if ((A->getStorageType() == EVAL) || (B->getStorageType() == EVAL) ||
		(varsA != varsB)) {
		return;
	}

	if ((A->getStorageType() == SPARSE) && (B->getStorageType() == SPARSE)) {
		set<int> targetVars(ordTargetVars.begin(), ordTargetVars.end());	

		for (set<int>::iterator si = targetVars.begin(); si != targetVars.end(); si++) {
			vector<int>::iterator it = find(ordVarsA.begin(), ordVarsA.end(), *si);
			_targetToA.push_back(it - ordVarsA.begin());
			it = find(ordVarsB.begin(), ordVarsB.end(), *si);
			_targetToB.push_back(it - ordVarsB.begin());
		}

		return;
	}
}
<<<<<<< HEAD

void FactorAdditionOp::execute()
{
	if (_target->getStorageType() == EVAL) {
		return;
	} else if (_target->getStorageType() == SPARSE) {
		vector<int> vars = _target->getOrderedVars();
		drwnSparseFactor *fac1 = _A->getSparseFactor();
		drwnSparseFactor *fac2 = _B->getSparseFactor();
		drwnSparseFactor *target = _target->getSparseFactor();
		map<vector<int>, double> assignmtsA = fac1->getAssignments();
		map<vector<int>, double> assignmtsB = fac2->getAssignments();

		for (map< vector<int>, double >::iterator mi = assignmtsA.begin();
				mi != assignmtsA.end(); mi++) {
			vector<int> vals(vars.size());

			for (int i = 0; i < vars.size(); i++) {
				vals[_targetToA[i]] = (*mi).first[_targetToA[i]];
			}

			drwnPartialAssignment dpa;
			target->setValueOf(dpa, (*mi).second);
		}

		for (map< vector<int>, double >::iterator mi = assignmtsB.begin();
				mi != assignmtsA.end(); mi++) {
			vector<int> vals(vars.size());

			for (int i = 0; i < vars.size(); i++) {
				vals[_targetToB[i]] = (*mi).first[_targetToB[i]];
			}

			drwnPartialAssignment dpa;
			target->setValueOf(dpa, target->getValueOf(dpa) + (*mi).second);
		}

		return;
	}

	/* dealing with dense factors */
	drwnTableFactor *target = _target->getTableFactor();
	drwnFactorAdditionOp dfao(target, _A->getTableFactor(), _B->getTableFactor());
	dfao.execute();
}
=======
>>>>>>> 5477d5d4f80732bdb8e6f07f513126e864b9fc40
