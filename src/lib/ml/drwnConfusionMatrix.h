/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfusionMatrix.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** 
*****************************************************************************/

#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "drwnDataset.h"
#include "drwnClassifier.h"

// drwnConfusionMatrix ------------------------------------------------------
//! Utility class for computing and printing confusion matrices.
//!
//! A negative actual/predicted class is considered unknown and not counted.
//! The Jaccard coefficient for a class is TP / (TP + FP + FN).
//!
//! \sa \ref drwnTutorialML "drwnML Tutorial"

class drwnConfusionMatrix {
 public:
    static std::string COL_SEP;   //!< string for separating columns when printing
    static std::string ROW_BEGIN; //!< string for starting a row when printing
    static std::string ROW_END;   //!< string for ending a row when printing

 protected:
    std::vector<std::vector<unsigned> > _matrix;

 public:
    //! construct a confusion matrix for \p n classes
    drwnConfusionMatrix(int n);
    //! construct a confusion matrix for \p n actual and \p m predicted classes
    drwnConfusionMatrix(int n, int m);
    virtual ~drwnConfusionMatrix();

    //! returns the number of rows
    int numRows() const;
    //! returns the number of columns
    int numCols() const;

    //! clear all counts in the confusion matrix
    void clear();
    //! accumulate a prediction/actual pair
    void accumulate(int actual, int predicted);
    //! accumulate a set of prediction/actual pairs
    void accumulate(const std::vector<int>& actual,
	const std::vector<int>& predicted);
    //! accumulate classification results
    void accumulate(const drwnClassifierDataset& dataset,
        drwnClassifier const *classifier);
    //! accumulate counts from a different confusion matrix of the same size
    void accumulate(const drwnConfusionMatrix& confusion);

    //! print the confusion matrix
    void printCounts(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print the confusion normalized by row
    void printRowNormalized(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print the confusion normalized by column
    void printColNormalized(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print the confusion matrix normalized by total count
    void printNormalized(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print precision and recall for each class
    void printPrecisionRecall(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print the F1-score for each class
    void printF1Score(std::ostream &os = std::cout,
        const char *header = NULL) const;
    //! print the Jaccard (intersection-over-union) score for each class
    void printJaccard(std::ostream &os = std::cout,
        const char *header = NULL) const;

    //! write the confusion matrix
    void write(std::ostream &os) const;
    //! read a confusion matrix
    void read(std::istream &is);

    //! returns the sum of entries along a row
    double rowSum(int n) const;
    //! returns the sum of entries down a column
    double colSum(int m) const;
    //! returns the sum of entries along the diagonal
    double diagSum() const;
    //! returns the sum of all entries in the confusion matrix
    double totalSum() const;
    //! diagSum / totalSum
    double accuracy() const;
    //! average of TP / (TP + FP); diagonal / colSum
    double avgPrecision() const;
    //! average of TP / (TP + FN); diagonal / rowSum 
    double avgRecall() const;
    //! average of TP / (TP + FP + FN); diagonal / (rowSum + colSum - diagonal)
    double avgJaccard() const;

    //! precision for class n
    double precision(int n) const;
    //! recall for class n
    double recall(int n) const;
    //! jaccard coefficient for class n
    double jaccard(int n) const;

    //! returns the counts at location \p (x,y)
    const unsigned& operator()(int x, int y) const;
    //! accesses the counts at location \p (x,y)
    unsigned& operator()(int x, int y);
};


