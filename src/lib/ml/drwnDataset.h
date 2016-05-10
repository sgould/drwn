/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDataset.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include "drwnBase.h"

using namespace std;

//! \file

// drwnDataset --------------------------------------------------------------
//! Implements a cacheable dataset containing feature vectors, labels and
//! optional weights.
//!
//! The dataset can be used by various machine learning algorithms. The dataset
//! is stored on disk as
//! \code
//!    <version, flags (int32)> <num features (int32)>
//!    <label> <feature 1> ... <feature n> (<weight>)? (<index>)?
//!    ...
//! \endcode
//!
//! \warning
//! The API for drwnDataset is not stable.

template <typename XType, typename YType, typename WType>
class drwnDataset {
 public:
    vector<vector<XType> > features;  //!< feature vectors
    vector<YType> targets;            //!< target labels
    vector<WType> weights;            //!< weights (optional)
    vector<int> indexes;              //!< external indices (optional)

 public:
    //! default constructor
    drwnDataset();
    //! copy constructor
    drwnDataset(const drwnDataset<XType, YType, WType>& d);
    //! construct and load dataset from file
    drwnDataset(const char *filename);
    ~drwnDataset();

    // dataset properties
    //! return true if the dataset is empty
    inline bool empty() const { return features.empty(); }
    //! return the number of samples in the dataset
    inline int size() const { return (int)features.size(); }
    //! return true if the dataset contains weighted samples
    inline bool hasWeights() const { return !weights.empty(); }
    //! return true if the dataset has external indices associated with each sample
    inline bool hasIndexes() const { return !indexes.empty(); }
    //! return true if the dataset is valid (e.g., number of targets equals number
    //! of feature vectors)
    inline bool valid() const;

    //! returns the number of samples with a given target label
    int count(const YType& label) const;
    //! pre-allocate memory for storing samples (feature vectors and targets)
    void reserve(int reserveSize);
    //! returns the number of features in the feature vector
    inline int numFeatures() const;
    //! returns the minimum target value in the dataset
    inline YType minTarget() const;
    //! returns the maximum target value in the dataset
    inline YType maxTarget() const;

    // stored dataset properties (without loading)
    //! returns the size of a dataset stored on disk
    static int size(const char *filename);
    //! returns the number of features in the feature vectors of a dataset stored on disk
    static int numFeatures(const char *filename);
    //! returns true if the dataset stored on disk contains weighted samples
    static bool hasWeights(const char *filename);
    //! returns true if the dataset stored on disk has indexes associated with each sample
    static bool hasIndexes(const char *filename);

    // i/o
    //! clears all data in the dataset
    void clear();
    //! writes the current dataset to disk (optionally appending to an existing dataset)
    int write(const char *filename, bool bAppend = false) const;
    //! writes a range of samples to disk (optionally appending to an existing dataset)
    int write(const char *filename, int startIndx, int endIndx, bool bAppend = false) const;
    //! reads a dataset from disk (optionally appending to the current dataset)
    int read(const char *filename, bool bAppend = false);
    //! reads a range of samples from disk (optionally appending to the current dataset)
    int read(const char *filename, int startIndx, int endIndx, bool bAppend = false);

    // modification
    //! appends the samples from another dataset to this dataset
    int append(const drwnDataset<XType, YType, WType>& d);
    //! appends a sample (feature vector and target) to the dataset
    int append(const vector<XType>& x, const YType& y);
    //! appends a weighted sample (feature vector and target) to the dataset
    int append(const vector<XType>& x, const YType& y, const WType& w);
    //! appends a weighted sample with associated external index to the dataset
    int append(const vector<XType>& x, const YType& y, const WType& w, int indx);

    //! subsample a dataset (balanced is only valid for discrete target types)
    //! if \p bBalanced is \p true then sampleRate is applied to most abundant
    //! target
    int subSample(int sampleRate, bool bBalanced = false);
};

// standard datasets --------------------------------------------------------

//! standard dataset for supervised classification algorithms
typedef drwnDataset<double, int, double> drwnClassifierDataset;
//! standard dataset for supervised regression algorithms
typedef drwnDataset<double, double, double> drwnRegressionDataset;

// implementation -----------------------------------------------------------

template <typename XType, typename YType, typename WType>
drwnDataset<XType, YType, WType>::drwnDataset()
{
    // do nothing
}

template <typename XType, typename YType, typename WType>
drwnDataset<XType, YType, WType>::drwnDataset(const drwnDataset<XType, YType, WType>& d) :
    features(d.features), targets(d.targets), weights(d.weights), indexes(d.indexes) {
    // do nothing
}

template <typename XType, typename YType, typename WType>
drwnDataset<XType, YType, WType>::drwnDataset(const char *filename)
{
    read(filename);
}

template <typename XType, typename YType, typename WType>
drwnDataset<XType, YType, WType>::~drwnDataset()
{
    // do nothing
}

template <typename XType, typename YType, typename WType>
bool drwnDataset<XType, YType, WType>::valid() const
{
    size_t nFeatures = (size_t)numFeatures();
    for (typename vector<vector<XType> >::const_iterator it = features.begin();
         it != features.end(); it++) {
        if (it->size() != nFeatures) return false;
    }

    return (features.size() == targets.size()) &&
        (weights.empty() || (weights.size() == targets.size())) &&
        (indexes.empty() || (indexes.size() == targets.size()));
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::count(const YType& label) const
{
    int c = 0;
    for (typename vector<YType>::const_iterator it = targets.begin();
         it != targets.end(); it++) {
        if (*it == label)
            c += 1;
    }

    return c;
}

template <typename XType, typename YType, typename WType>
void drwnDataset<XType, YType, WType>::reserve(int reserveSize)
{
    features.reserve(reserveSize);
    targets.reserve(reserveSize);
    weights.reserve(reserveSize);
    indexes.reserve(reserveSize);
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::numFeatures() const
{
    return features.empty() ? 0 : (int)features[0].size();
}

template <typename XType, typename YType, typename WType>
YType drwnDataset<XType, YType, WType>::minTarget() const
{
    return targets.empty() ? (YType)0 :
        *std::min_element(targets.begin(), targets.end());
}

template <typename XType, typename YType, typename WType>
YType drwnDataset<XType, YType, WType>::maxTarget() const
{
    return targets.empty() ? (YType)0 :
        *std::max_element(targets.begin(), targets.end());
}

// stored dataset properties (without loading)
template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::size(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    // open file and read header
    ifstream ifs(filename, ifstream::in | ifstream::binary);
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not find file " << filename);
        return 0;
    }

    if (ifs.eof()) {
        DRWN_LOG_WARNING("empty file " << filename);
        return 0;
    }

    unsigned flags;
    ifs.read((char *)&flags, sizeof(unsigned));
    DRWN_ASSERT_MSG((flags & 0xffff0000) == 0x00010000, "unrecognized file version");

    int nFeatures;
    ifs.read((char *)&nFeatures, sizeof(int));

    ifs.seekg(0, ios::end);
    int len = (int)ifs.tellg() - 2 * sizeof(int);
    ifs.close();

    // determine number of records
    int bytesPerRecord = sizeof(YType) + nFeatures * sizeof(XType);
    if ((flags & 0x00000001) == 0x00000001) bytesPerRecord += sizeof(WType);
    if ((flags & 0x00000002) == 0x00000002) bytesPerRecord += sizeof(int);

    DRWN_ASSERT_MSG(len % bytesPerRecord == 0, "corrupt file " << filename
        << " (len: " << len << ", bytes/record = " << bytesPerRecord << ")");
    return (int)(len / bytesPerRecord);
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::numFeatures(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    // open file and read header
    ifstream ifs(filename, ifstream::in | ifstream::binary);
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not open file " << filename);
        return 0;
    }

    if (ifs.eof()) {
        DRWN_LOG_WARNING("empty file " << filename);
        return 0;
    }

    unsigned flags;
    ifs.read((char *)&flags, sizeof(unsigned));
    DRWN_ASSERT_MSG((flags & 0xffff0000) == 0x00010000, "unrecognized file version");

    int nFeatures;
    ifs.read((char *)&nFeatures, sizeof(int));
    ifs.close();

    return nFeatures;
}

template <typename XType, typename YType, typename WType>
bool drwnDataset<XType, YType, WType>::hasWeights(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    // open file and read header
    ifstream ifs(filename, ifstream::in | ifstream::binary);
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not open file " << filename);
        return false;
    }

    if (ifs.eof()) {
        DRWN_LOG_WARNING("empty file " << filename);
        return false;
    }

    unsigned flags;
    ifs.read((char *)&flags, sizeof(unsigned));
    DRWN_ASSERT_MSG((flags & 0xffff0000) == 0x00010000, "unrecognized file version");

    return ((flags & 0x00000001) == 0x00000001);
}

template <typename XType, typename YType, typename WType>
bool drwnDataset<XType, YType, WType>::hasIndexes(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    // open file and read header
    ifstream ifs(filename, ifstream::in | ifstream::binary);
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not open file " << filename);
        return false;
    }

    if (ifs.eof()) {
        DRWN_LOG_WARNING("empty file " << filename);
        return false;
    }

    unsigned flags;
    ifs.read((char *)&flags, sizeof(unsigned));
    DRWN_ASSERT_MSG((flags & 0xffff0000) == 0x00010000, "unrecognized file version");

    return ((flags & 0x00000002) == 0x00000002);
}

// i/o

template <typename XType, typename YType, typename WType>
void drwnDataset<XType, YType, WType>::clear()
{
    features.clear();
    targets.clear();
    weights.clear();
    indexes.clear();
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::write(const char *filename, bool bAppend) const
{
    if (this->empty()) return 0;
    return write(filename, 0, this->size() - 1, bAppend);
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::write(const char *filename, int startIndx, int endIndx, bool bAppend) const
{
    DRWN_ASSERT(filename != NULL);
    DRWN_ASSERT(this->valid());
    DRWN_ASSERT_MSG((startIndx >= 0) && (endIndx < this->size()) && (startIndx <= endIndx),
        "startIndx = " << startIndx << ", endIndx = " << endIndx << ", size() = " << this->size());

    // open file
    unsigned flags = 0x00010000;
    if (hasWeights()) flags |= 0x00000001;
    if (hasIndexes()) flags |= 0x00000002;

    int nFeatures = numFeatures();
    fstream ofs;
    if (bAppend && drwnFileExists(filename)) {
        unsigned fileFlags;
        int fileNumFeatures;

        ofs.open(filename, ios::in | ios::out | ios::binary);
        ofs.seekg(0, ios::beg);
        ofs.read((char *)&fileFlags, sizeof(unsigned));
        DRWN_ASSERT(fileFlags == flags);
        ofs.read((char *)&fileNumFeatures, sizeof(int));
        DRWN_ASSERT(fileNumFeatures == nFeatures);

        ofs.seekp(0, ios::end);
    } else {
        ofs.open(filename, ios::out | ios::binary);
        ofs.write((char *)&flags, sizeof(unsigned));
        ofs.write((char *)&nFeatures, sizeof(int));
    }
    DRWN_ASSERT_MSG(!ofs.fail(), filename);

    // write data
    for (int i = startIndx; i <= endIndx ; i++) {
        ofs.write((char *)&targets[i], sizeof(YType));
        ofs.write((char *)&features[i][0], nFeatures * sizeof(XType));
        if (!weights.empty()) {
            ofs.write((char *)&weights[i], sizeof(WType));
        }
        if (!indexes.empty()) {
            ofs.write((char *)&indexes[i], sizeof(int));
        }
    }

    int len = (int)ofs.tellp() - 2 * sizeof(int);
    ofs.close();

    int bytesPerRecord = sizeof(YType) + nFeatures * sizeof(XType);
    if (hasWeights()) bytesPerRecord += sizeof(WType);
    if (hasIndexes()) bytesPerRecord += sizeof(int);

    DRWN_ASSERT_MSG(len % bytesPerRecord == 0, "corrupt file " << filename
        << " (len: " << len << ", bytes/record = " << bytesPerRecord << ")");
    return (int)(len / bytesPerRecord);
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::read(const char *filename, bool bAppend)
{
    return read(filename, 0, numeric_limits<int>::max(), bAppend);
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::read(const char *filename, int startIndx, int endIndx, bool bAppend)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_ASSERT((startIndx >= 0) && (endIndx >= startIndx));
    if (!bAppend) clear();

    // open file
    ifstream ifs(filename, ifstream::in | ifstream::binary);
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not find file " << filename);
        return size();
    }

    if (ifs.eof()) {
        DRWN_LOG_WARNING("empty file " << filename);
        return size();
    }

    unsigned flags;
    ifs.read((char *)&flags, sizeof(unsigned));
    DRWN_ASSERT_MSG((flags & 0xffff0000) == 0x00010000, "unrecognized file version: " << flags);
    DRWN_ASSERT(empty() || ((flags & 0x00000001) == (hasWeights() ? 0x00000001 : 0x00000000)));
    DRWN_ASSERT(empty() || ((flags & 0x00000002) == (hasIndexes() ? 0x00000002 : 0x00000000)));

    int nFeatures;
    ifs.read((char *)&nFeatures, sizeof(int));
    DRWN_ASSERT_MSG(empty() || (nFeatures == numFeatures()), nFeatures << " != " << numFeatures());

    int bytesPerRecord = sizeof(YType) + nFeatures * sizeof(XType);
    if ((flags & 0x00000001) == 0x00000001) bytesPerRecord += sizeof(WType);
    if ((flags & 0x00000002) == 0x00000002) bytesPerRecord += sizeof(int);

    // goto start index
    ifs.seekg(startIndx * bytesPerRecord, ios::cur);
    if (ifs.fail()) {
        ifs.close();
        DRWN_LOG_WARNING("less than " << startIndx << " record in file " << filename);
        return size();
    }

    // read until end of file or end index
    YType y;
    vector<XType> x(nFeatures);
    WType w;
    int index;

    int recordCount = startIndx;
    while (recordCount <= endIndx) {
        ifs.read((char *)&y, sizeof(YType));
        ifs.read((char *)&x[0], nFeatures * sizeof(XType));
        if ((flags & 0x00000001) == 0x00000001) {
            ifs.read((char *)&w, sizeof(WType));
        }
        if ((flags & 0x00000002) == 0x00000002) {
            ifs.read((char *)&index, sizeof(int));
        }

        if (ifs.fail()) break;
        targets.push_back(y);
        features.push_back(x);
        if ((flags & 0x00000001) == 0x00000001) {
            weights.push_back(w);
        }
        if ((flags & 0x00000002) == 0x00000002) {
            indexes.push_back(index);
        }

        recordCount += 1;
    }

    // close file
    ifs.close();

    return size();
}

// modification
template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::append(const drwnDataset<XType, YType, WType>& d)
{
    if (d.empty()) return size();
    if (empty()) {
        features = d.features();
        targets = d.targets();
        weights = d.weights();
        indexes = d.indexes();
        return size();
    }

    DRWN_ASSERT(d.numFeatures() == numFeatures());
    DRWN_ASSERT(d.hasWeights() == hasWeights());
    DRWN_ASSERT(d.hasIndexes() == hasIndexes());

    features.insert(features.end(), d.features.begin(), d.features.end());
    targets.insert(targets.end(), d.targets.begin(), d.targets.end());
    if (hasWeights()) {
        weights.insert(weights.end(), d.weights.begin(), d.weights.end());
    }
    if (hasIndexes()) {
        indexes.insert(indexes.end(), d.indexes.begin(), d.indexes.end());
    }

    return size();
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::append(const vector<XType>& x, const YType& y)
{
    if (!empty()) {
        DRWN_ASSERT(!hasWeights() && !hasIndexes());
        DRWN_ASSERT((int)x.size() == numFeatures());
    }

    features.push_back(x);
    targets.push_back(y);

    return size();
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::append(const vector<XType>& x, const YType& y, const WType& w)
{
    if (!empty()) {
        DRWN_ASSERT(hasWeights() && !hasIndexes());
        DRWN_ASSERT((int)x.size() == numFeatures());
    }

    features.push_back(x);
    targets.push_back(y);
    weights.push_back(w);

    return size();
}


template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::append(const vector<XType>& x, const YType& y, const WType& w, int indx)
{
    if (!empty()) {
        DRWN_ASSERT(hasWeights() && hasIndexes());
        DRWN_ASSERT((int)x.size() == numFeatures());
    }

    features.push_back(x);
    targets.push_back(y);
    weights.push_back(w);
    indexes.push_back(indx);

    return size();
}

template <typename XType, typename YType, typename WType>
int drwnDataset<XType, YType, WType>::subSample(int sampleRate, bool bBalanced)
{
    DRWN_ASSERT_MSG(sampleRate > 0, "sampleRate must be greater than one");
    //! \todo consider sample weighting
    //! \todo enforce balanced for discrete targets only

    // construct random permutation of indices
    vector<int> indx = drwn::randomPermutation(features.size());

    if (bBalanced) {
        map<YType, vector<int> > stratified;
        for (size_t i = 0; i < indx.size(); i++) {
            typename map<YType, vector<int> >::iterator it = stratified.find(targets[indx[i]]);
            if (it == stratified.end()) {
                stratified.insert(make_pair(targets[indx[i]], vector<int>(1, indx[i])));
            } else {
                it->second.push_back(indx[i]);
            }
        }

        size_t maxSamplesPerTarget = 1;
        for (typename map<YType, vector<int> >::const_iterator it = stratified.begin();
             it != stratified.end(); ++it) {
            maxSamplesPerTarget = std::max(maxSamplesPerTarget, it->second.size());
        }
        maxSamplesPerTarget = (maxSamplesPerTarget + sampleRate - 1) / sampleRate;

        // reconstruct indx vector
        indx.clear();
        for (typename map<YType, vector<int> >::iterator it = stratified.begin();
             it != stratified.end(); ++it) {
            if (it->second.size() > maxSamplesPerTarget) {
                it->second.resize(maxSamplesPerTarget);
            }
            indx.insert(indx.end(), it->second.begin(), it->second.end());
        }

    } else {
        // resize indx vector to first n samples
        indx.resize((features.size() + sampleRate - 1) / sampleRate);
    }

    // construct new samples according to permutation
    vector<vector<XType> > nfeatures(indx.size());
    vector<YType> ntargets(nfeatures.size());
    vector<WType> nweights(hasWeights() ? nfeatures.size() : 0);
    vector<int> nindexes(hasIndexes() ? nfeatures.size() : 0);

    for (size_t i = 0; i < nfeatures.size(); i++) {
        std::swap(nfeatures[i], features[indx[i]]);
        std::swap(ntargets[i], targets[indx[i]]);
        if (!nweights.empty()) {
            std::swap(nweights[i], weights[indx[i]]);
        }
        if (!nindexes.empty()) {
            nindexes[i] = indexes[indx[i]];
        }
    }

    std::swap(features, nfeatures);
    std::swap(targets, ntargets);
    std::swap(weights, nweights);
    std::swap(indexes, nindexes);

    return (int)indx.size();
}
