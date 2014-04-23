/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnADLPInference.cpp
** AUTHOR(S):   Hendra Gunadi <u4971560@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <cstdlib>

#include "drwnBase.h"
#include "drwnPGM.h"

#include "drwnADLPInference.h"

using namespace std;

// drwnADLPInference -----------------------------------------------------------

int drwnADLPInference::MAX_ITERATIONS = 1000;
double drwnADLPInference::EPSILON = 1e-6;
double drwnADLPInference::PENALTY_PARAMETER = 1.0;

double drwnADLPInference::TRIM(drwnTableFactor* v, double z) {
    int vsize = v->entries(), k;
    vector<int> U, G, L;
    for (int i = 0; i < vsize; i++) U.push_back(i);
    double s = 0.0, p = 0.0;
    double delta_s, delta_p;

    srand(time(NULL));

    // while U is not empty
    while(U.size() > 0) {
        // pick k in U at random
        k = U[rand() % U.size()];

        // Partition U:
        //      G = {j in U | v_j >= v_k}
        //      L = {j in U | v_j < v_k}
        G.clear(); L.clear();
        delta_p = 0.0;
        delta_s = 0.0;
        for (unsigned int j = 0; j < U.size(); j++) {
            if (U[j] == k) {
                delta_p += 1.0;
                delta_s += (*v)[k];
            }
            else if ((*v)[U[j]] >= (*v)[k]) {
                G.push_back(U[j]);
                delta_p += 1.0;
                delta_s += (*v)[U[j]];
            }
            // in this implementation, I skipped the v_k as it will be removed later because it is easier than to remove element from vector
            else {
                L.push_back(U[j]);
            }
        }

        // Calculate delta_p = |G| ; delta_s = sum_{j in G} v_j; Already done in the loop

        U.clear();

        // If (s + delta_s) - (p + delta_p) * v_k < z
        if ((s + delta_s) - (p + delta_p) * (*v)[k] < z)
        {
        //      s = s + delta_s; p = p + delta_p; U = L
            s = s + delta_s;
            p = p + delta_p;
            U = L;
        }
        // Else
        else
        {
        //      U = G \ {k}
            U = G;
        }
        // End If
    }

    // Set theta = (s - z) / p;
    double theta = (s - z) / p;

    return theta;
}

void drwnADLPInference::buildComputationGraph()
{
    _updateLambdaOp.resize(_cliqueSize);
    _updateDeltaBarOp.resize(_cliqueSize);
    _updateMuOp.resize(_cliqueSize);
    _decodeOp.resize(_cliqueSize);

    int tempSize;

    for (int i = 0; i < _cliqueSize; i++) {
        tempSize = _clique[i]->size();
        _updateDeltaBarOp[i].resize(tempSize);
        for (int j = 0; j < tempSize; j++) {
            //=============================================================================================
            //  Update lambda
            _updateLambdaOp[i].push_back(new drwnFactorMinusEqualsOp(_clique_bar[i], _message_clique_bar[i][j]));

            //=============================================================================================
            //  Update delta_bar
            _updateDeltaBarOp[i][j].push_back(new drwnFactorMarginalizeOp(_margin_result_lambda[i][j], _lambda[i], _marginalizer[i][j]));
            _updateDeltaBarOp[i][j].push_back(new drwnFactorMarginalizeOp(_margin_result_mu[i][j], _mu[i], _marginalizer[i][j]));

            //=============================================================================================
            //  Update the multipliers: mu
            _updateMuOp[i].push_back(new drwnFactorMinusEqualsOp(_tempMu[i], _message_clique_bar[i][j]));

            //=============================================================================================
            //  Decode the assignment
            _decodeOp[i].push_back(new drwnFactorMinusEqualsOp(_clique_bar[i], _message_clique[i][j]));
            _decodeOp[i].push_back(new drwnFactorMinusEqualsOp(_tempMu[i], _message_clique_bar[i][j]));
        }
    }
}

drwnADLPInference::drwnADLPInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph), _numNodes(graph.numVariables())
{
    // Temporary Variables
    int numFactors = graph.numFactors();
    const drwnTableFactor* temp;
    drwnTableFactor* factor;
    drwnTableFactor* factor_1;
    drwnTableFactor* factor_c;
    drwnTableFactor* factor_lamda;
    drwnTableFactor* factor_mu;
    drwnTableFactor* factor_tempMu;
    drwnTableFactorStorage storage, storage_2;
    drwnVarUniversePtr universe(graph.getUniverse());

    _unary.resize(_numNodes);
    _unary_bar.resize(_numNodes);
    _message_unary.resize(_numNodes);
    _message_unary_bar.resize(_numNodes);
    _gamma.resize(_numNodes);
    _flag.resize(_numNodes);

    for (int i = 0; i < numFactors; i++) {
        temp = _graph.getFactor(i);
        if (temp->size() == 1) {
            _unary[temp->varId(0)] = temp;

            factor = new drwnTableFactor(universe);
            factor->addVariable(temp->varId(0));
            _unary_bar[temp->varId(0)] = factor;

            _flag[temp->varId(0)] = 1;
        }
        else {
            factor_c = new drwnTableFactor(universe);
            factor_lamda = new drwnTableFactor(universe);
            factor_mu = new drwnTableFactor(universe);
            factor_tempMu = new drwnTableFactor(universe);
            vector<drwnTableFactor* > tempMessageCliqueBar;
            vector<drwnTableFactor* > tempMessageClique;
            vector<drwnTableFactor* > tempGammaClique;
            vector<drwnTableFactor* > tempMarginLambda;
            vector<drwnTableFactor* > tempMarginMu;
            vector<set<int> > tempMarginalizer;

            int tempSize = temp->size();
            for (int j = 0; j < tempSize; j++) {
                factor = new drwnTableFactor(universe);
                factor->addVariable(temp->varId(j));
                _message_unary[temp->varId(j)].push_back(factor);
                tempMessageClique.push_back(factor);

                factor_1 = new drwnTableFactor(universe);
                factor_1->addVariable(temp->varId(j));
                _message_unary_bar[temp->varId(j)].push_back(factor_1);
                tempMessageCliqueBar.push_back(factor_1);

                drwnTableFactor* tempGamma = new drwnTableFactor(universe);
                tempGamma->addVariable(temp->varId(j));
                _gamma[temp->varId(j)].push_back(tempGamma);
                tempGammaClique.push_back(tempGamma);

                factor_c->addVariable(temp->varId(j));
                factor_lamda->addVariable(temp->varId(j));
                factor_mu->addVariable(temp->varId(j));
                factor_tempMu->addVariable(temp->varId(j));

                set<int> tempSet;
                for (int k = 0; k < tempSize; k++) {
                    if (j != k) tempSet.insert(temp->varId(k));
                }
                tempMarginalizer.push_back(tempSet);

                tempMarginLambda.push_back(new drwnTableFactor(universe));
                tempMarginMu.push_back(new drwnTableFactor(universe));
            }

            _clique.push_back(temp);
            _clique_bar.push_back(factor_c);
            _message_clique.push_back(tempMessageClique);
            _message_clique_bar.push_back(tempMessageCliqueBar);
            _gamma_clique.push_back(tempGammaClique);
            _lambda.push_back(factor_lamda);
            factor_mu->fill(1.0 / factor_mu->entries());
            _mu.push_back(factor_mu);
            _tempMu.push_back(factor_tempMu);
            _marginalizer.push_back(tempMarginalizer);
            _margin_result_lambda.push_back(tempMarginLambda);
            _margin_result_mu.push_back(tempMarginMu);
        }
    }

   // Check whether there's variable missing then free the flag to indicate missing factor
	for (int i = 0; i < _numNodes; i++) {
		// Insert 0 table factor if not yet initialized
		if (_flag[i] == 0) {
            // Add unary table factor
            factor = new drwnTableFactor(universe);
            factor->addVariable(i);
            _unary[i] = factor;

            // Add space for calibrated table factor
            factor = new drwnTableFactor(universe);
            factor->addVariable(i);
            _unary_bar[i] = factor;
		}
	}

    _cliqueSize = _clique.size();

    buildComputationGraph();
}

drwnADLPInference::~drwnADLPInference()
{
    clear();
}

void drwnADLPInference::clear()
{
    int size, j;
	for (int i = 0; i < _numNodes; i++)
	{
		delete(_unary_bar[i]);

	    if (_flag[i] == 0) delete(_unary[i]);

		size = _message_unary[i].size();
		for (j = 0; j < size; j++) {
            delete(_message_unary[i][j]);
            delete(_message_unary_bar[i][j]);
            delete(_gamma[i][j]);
        }
        _message_unary[i].clear();
        _message_unary_bar[i].clear();
        _gamma[i].clear();
	}
    _unary.clear();
    _unary_bar.clear();
    _message_unary.clear();
    _message_unary_bar.clear();
    _gamma.clear();
    _marginalizer.clear();

	for (int i = 0; i < _cliqueSize; i++)
	{
		delete(_clique_bar[i]);
        delete(_lambda[i]);
        delete(_mu[i]);
        delete(_tempMu[i]);

        _message_clique[i].clear();
        _message_clique_bar[i].clear();
        _gamma_clique[i].clear();

        for (unsigned int j = 0; j < _margin_result_lambda[i].size(); j++) {
            delete(_margin_result_lambda[i][j]);
            delete(_margin_result_mu[i][j]);
        }

        for (unsigned int j = 0; j < _updateLambdaOp[i].size(); j++) {
            delete(_updateLambdaOp[i][j]);
        }
        for (unsigned int j = 0; j < _updateDeltaBarOp[i].size(); j++) {
            delete(_updateDeltaBarOp[i][j][0]);
            delete(_updateDeltaBarOp[i][j][1]);
        }
        for (unsigned int j = 0; j < _updateMuOp[i].size(); j++) {
            delete(_updateMuOp[i][j]);
        }
        for (unsigned int j = 0; j < _decodeOp[i].size(); j++) {
            delete(_decodeOp[i][j]);
        }
	}
    _clique.clear();
    _clique_bar.clear();
    _lambda.clear();
    _mu.clear();
    _tempMu.clear();
    _message_clique.clear();
    _message_clique_bar.clear();
    _gamma_clique.clear();
    _margin_result_lambda.clear();
    _margin_result_mu.clear();

    _updateLambdaOp.clear();
    _updateDeltaBarOp.clear();
    _updateMuOp.clear();
    _decodeOp.clear();

    _flag.clear();
}

pair<double, double> drwnADLPInference::inference(drwnFullAssignment& mapAssignment)
{
    int iteration = 0;
    drwnTableFactorStorage storage, storage_2;
    drwnVarUniversePtr universe(_graph.getUniverse());

    double factorDiff = 100 ;
    double incrFactor = 1.25 ;

    mapAssignment.clear();
    mapAssignment.resize(_numNodes);
    double bestEnergy = _graph.getEnergy(mapAssignment);
    double bestDualEnergy = -DRWN_DBL_MAX;
    double sumResidual, sumPrimalUpdate;
    bool notConverged = true;

    // for t = 1 to T do
    while((iteration < MAX_ITERATIONS) && (notConverged)) {
    //=============================================================================================
    //      Update delta: for all i = 1, ..., n
        for (int i = 0; i < _numNodes; i++)
        {
    //          Set theta_bar_i = theta_i + sum_{c,i in C} (delta_bar_{ci} - 1 / p * gamma_{ci})
            int tempSize = _message_unary[i].size();
            int entries = _unary_bar[i]->entries();
            for (int k = 0; k < entries; k++) {
                (*_unary_bar[i])[k] = -(*_unary[i])[k];
            }
            for (int j = 0; j < tempSize; j++) {
                _gamma[i][j]->scale(1.0 / PENALTY_PARAMETER); // Only need to scale once
                for (int k = 0; k < entries; k++) {
                    (*_unary_bar[i])[k] += (*_message_unary_bar[i][j])[k] - (*_gamma[i][j])[k];
                }
            }

            double theta = TRIM(_unary_bar[i], (double)tempSize / PENALTY_PARAMETER);
            for (int k = 0; k < entries; k++) {
                (*_unary_bar[i])[k] = ((*_unary_bar[i])[k] > theta) ? ((*_unary_bar[i])[k] - theta) / (double)tempSize : 0.0;
            }

    //          Update delta_{ci} = delta_bar_{ci} - 1 / p * gamma_{ci} - q, forall_c : i in c
            for (int j = 0; j < tempSize; j++) {
                for (int k = 0; k < entries; k++) { // Note : This is possible because there is only one variable
                    (*_message_unary[i][j])[k] = (*_message_unary_bar[i][j])[k] - (*_gamma[i][j])[k] - (*_unary_bar[i])[k];
                }
            }
        }

    //=============================================================================================
    //      Update lambda: for all c in C
        for (int i = 0; i < _cliqueSize; i++) {
    //          Set theta_bar_c = theta_c - sum_{i:i in c} delta_bar_{ci} + 1 / p * mu_c
            int entries = _lambda[i]->entries();
            _mu[i]->scale(1.0 / PENALTY_PARAMETER); // Only need to scale once
            for (int j = 0; j < entries; j++) { // Note : this is possible due to the way mu and clique_bar are constructed, taking the same variable order as the clique
                (*_clique_bar[i])[j] = (*_mu[i])[j] - (*_clique[i])[j];
            }

            int tempSize = _updateLambdaOp[i].size();
            for (int j = 0; j < tempSize; j++) {
                _updateLambdaOp[i][j]->execute();   // theta_bar_c - delta_bar
            }

            double theta = TRIM(_clique_bar[i], 1.0 / PENALTY_PARAMETER);
            for (int j = 0; j < entries; j++) { // Note : this is possible due to the way lambda and clique_bar are constructed, taking the same variable order as the clique
                (*_lambda[i])[j] = ((*_clique_bar[i])[j] > theta) ?  -(*_clique[i])[j] - theta : -(*_clique[i])[j] - (*_clique_bar[i])[j];
            }
        }

    //=============================================================================================
    //      Update delta_bar: for all c in C, i : i in c, x_i
        sumPrimalUpdate = 0.0;
        for (int i = 0; i < _cliqueSize; i++) {
            // Set v_{ci} = delta_{ci} + 1 / p * gamma_{ci} + sum_{x_c\i} lambda_c + 1 / p * sum_{x_c\i} mu_c
            int tempSize = _message_clique[i].size();
            vector<vector<double> > v(tempSize);
            vector<double> sum;
            double totalSum = 0.0;
            int sumCard = 0;
            sum.resize(tempSize);
            for (int j = 0; j < tempSize; j++) {
                _updateDeltaBarOp[i][j][0]->execute();  // Marginalize Lambda
                _updateDeltaBarOp[i][j][1]->execute();  // Marginalize Mu
                sum[j] = 0.0;
                int entries = _message_clique[i][j]->entries();
                v[j].resize(entries, 0.0);
                for(int k = 0; k < entries; k++) {
                    v[j][k] = (*_message_clique[i][j])[k] + (*_gamma_clique[i][j])[k] + (*_margin_result_lambda[i][j])[k] + (*_margin_result_mu[i][j])[k];
                    sum[j] += v[j][k];
                }
                sumCard += _clique[i]->entries() / universe->varCardinality(_clique[i]->varId(j));
                totalSum += sum[j] * (double)(_clique[i]->entries() / universe->varCardinality(_clique[i]->varId(j)));
            }

            // v_bar_{c} = 1 / {1 + sum_{k : k in c} |X_{c\k}|} * sum_{k : k in c} |X_{c\k}| * sum_{x_k} v_{ck}(x_k)
            double v_bar = (1.0 / (double)(1 + sumCard)) * totalSum;

            // delta_bar_{ci} = 1 / {1 + |X_{c\i}|} * [v_{ci} - sum_{j : j in c, j != i} |X_{c\ji}| (sum_{x_j} v_{cj}(x_j) - v_bar_{c})]
            for (int j = 0; j < tempSize; j++) {
                totalSum = 0.0;
                for (int k = 0; k < tempSize; k++) {
                    if (j != k) {
                        totalSum += (double)(_clique[i]->entries() / universe->varCardinality(_clique[i]->varId(j)) / universe->varCardinality(_clique[i]->varId(k))) * (sum[k] - v_bar);
                    }
                }

                int entries = _message_clique_bar[i][j]->entries();
                double denominator = (double)(1 + _clique[i]->entries() / universe->varCardinality(_clique[i]->varId(j)));
                for(int k = 0; k < entries; k++) {
                    double result = (v[j][k] - totalSum) / denominator;
                    sumPrimalUpdate += pow(result - (*_message_clique_bar[i][j])[k], 2);
                    (*_message_clique_bar[i][j])[k] = result;
                }
            }

            v.clear();
        }

    //=============================================================================================
    //      Update the multipliers:
    //          gamma_{ci} = gamma_{ci} + p * (delta_{ci} - delta_bar_{ci}) for all c in C, i : i in c, x_i
        sumResidual = 0.0;
        for (int i = 0; i < _numNodes; i++) {
            int tempSize = _message_unary[i].size();
            for (int j = 0; j < tempSize; j++) {
                int entries = _message_unary[i][j]->entries();
                for (int k = 0; k < entries; k++) {
                    double result = (*_message_unary[i][j])[k] - (*_message_unary_bar[i][j])[k]; // Note : this is possible because there is only one variable
                    sumResidual += result * result;
                    (*_gamma[i][j])[k] = (*_gamma[i][j])[k] * PENALTY_PARAMETER + result * PENALTY_PARAMETER;
                }
            }
        }

    //          mu_c = mu_c + p * (lambda_c - sum{i:i in c} delta_bar_{ci}) for all c in C, x_c
        for (int i = 0; i < _cliqueSize; i++) {
            int entries = _lambda[i]->entries();
            for (int k = 0; k < entries; k++) {
                (*_tempMu[i])[k] = (*_lambda[i])[k];
            }

            _tempMu[i]->dataCompareAndCopy(*_lambda[i]);
            int tempSize = _updateMuOp[i].size();
            for (int j = 0; j < tempSize; j++) {
                _updateMuOp[i][j]->execute();   // tempMu - delta_bar
            }
            for (int k = 0; k < entries; k++) {
                sumResidual += std::pow((*_tempMu[i])[k], 2);
                (*_mu[i])[k] = (*_mu[i])[k] * PENALTY_PARAMETER + (*_tempMu[i])[k] * PENALTY_PARAMETER;
            }
        }

    //      Check for Convergence : based on the author's code in SVL
        double dualObjDelta = 0.0, dualObjDeltaBar = 0.0;
        drwnFullAssignment deltaAssignment(_numNodes), deltaBarAssignment(_numNodes);
        drwnFullAssignment dashAssignment(_numNodes);
        for (int i = 0; i < _numNodes; i++) {
            drwnTableFactor thetaBar(universe, &storage);
            thetaBar.addVariable(i);
            drwnTableFactor thetaBarBar(universe, &storage_2);
            thetaBarBar.addVariable(i);
            int entries = thetaBar.entries();
            for (int k = 0; k < entries; k++) {
                thetaBar[k] = 0.0;
                thetaBarBar[k] = 0.0;
            }

            int tempSize = _message_unary[i].size();
            for (int j = 0; j < tempSize; j++) {
                for (int k = 0; k < entries; k++) { // Note : this is possible because there is only one variable
                    thetaBar[k] += (*_message_unary[i][j])[k];
                    thetaBarBar[k] += (*_message_unary_bar[i][j])[k];
                }
            }

            if (_flag[i] != 0) {
                for (int k = 0; k < entries; k++) {
                    thetaBar[k] -= (*_unary[i])[k];
                    thetaBarBar[k] -= (*_unary[i])[k];
                }
            }

            deltaAssignment[i] = thetaBar.valueOf(i, thetaBar.indexOfMax());
            dualObjDelta += thetaBar[thetaBar.indexOfMax()];
            deltaBarAssignment[i] = thetaBarBar.valueOf(i, thetaBarBar.indexOfMax());
            dualObjDeltaBar += thetaBarBar[thetaBarBar.indexOfMax()];
        }
        for (int i = 0; i < _cliqueSize; i++) {
            int entries = _clique[i]->entries();
            for (int k = 0; k < entries; k++) {
                (*_clique_bar[i])[k] = -(*_clique[i])[k];
                (*_tempMu[i])[k] = -(*_clique[i])[k];
            }

            int tempSize = _decodeOp[i].size();
            for (int j = 0; j < tempSize; j++) {
                _decodeOp[i][j]->execute(); // clique - delta & clique - delta_bar
            }
            dualObjDelta += (*_clique_bar[i])[_clique_bar[i]->indexOfMax()];
            dualObjDeltaBar += (*_tempMu[i])[_tempMu[i]->indexOfMax()];
        }

        double deltaEnergy = _graph.getEnergy(deltaAssignment);
        double deltaBarEnergy = _graph.getEnergy(deltaBarAssignment);
        if (deltaEnergy < bestEnergy) {
            bestEnergy = deltaEnergy;
            mapAssignment = deltaAssignment;
        }
        if (deltaBarEnergy < bestEnergy) {
            bestEnergy = deltaBarEnergy;
            mapAssignment = deltaBarAssignment;
        }

        if ((dualObjDelta <= -bestEnergy) || (dualObjDeltaBar <= -bestEnergy)
            || ((sqrt(sumResidual) < EPSILON) && (sqrt(sumPrimalUpdate) < EPSILON)) ){
            notConverged = false;
        }

        bestDualEnergy = -dualObjDelta;
        DRWN_LOG_VERBOSE("...iteration " << iteration
            << "; dual objective " << bestDualEnergy << "; best energy " << bestEnergy);

        iteration++;
        if (sqrt(sumResidual) > factorDiff*sqrt(sumPrimalUpdate)) {
            PENALTY_PARAMETER *= incrFactor ; // residual is large -> increase penalty
        }
        else if (sqrt(sumPrimalUpdate) > factorDiff*sqrt(sumResidual)) {
            PENALTY_PARAMETER /= incrFactor ; // residual small -> decrease penalty
        }
    }
    // end for

    return make_pair(bestEnergy, bestDualEnergy);
}

// drwnADLPInferenceConfig ---------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnADLPInference
//! \b maxIterations :: maximum number of iterations (default: 1000)\n
//! \b epsilon :: smallest difference in value to be considered as equal (default: 1.0e-6)\n
//! \b rho :: initial penalty parameter (rho) value (default: 1.0)

class drwnADLPInferenceConfig : public drwnConfigurableModule {
public:
    drwnADLPInferenceConfig() : drwnConfigurableModule("drwnADLPInference") { }
    ~drwnADLPInferenceConfig() { }

    void usage(ostream &os) const {
        os << "      maxIterations :: maximum number of iterations (default: "
           << drwnADLPInference::MAX_ITERATIONS << ")\n";
        os << "      epsilon       :: smallest difference in value to be considered as equal (default: "
           << drwnADLPInference::EPSILON << ")\n";
        os << "      rho           :: initial penalty parameter (rho) value (default: "
           << drwnADLPInference::PENALTY_PARAMETER << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxIterations")) {
            drwnADLPInference::MAX_ITERATIONS = std::max(0, atoi(value));
        } else if (!strcmp(name, "epsilon")) {
            drwnADLPInference::EPSILON = std::max(1.0e-12, atof(value));
        } else if (!strcmp(name, "rho")) {
            drwnADLPInference::PENALTY_PARAMETER = atof(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnADLPInferenceConfig gADLPInferenceConfig;
