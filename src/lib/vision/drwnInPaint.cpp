/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInPaint.cpp
** AUTHOR(S):   Robin Liang <robin.gnail@gmail.com>
**
*****************************************************************************/

#include <cstdlib>

#include "drwnInPaint.h"

// private macros -----------------------------------------------------------

#define halfWinSize ((windowSize - 1) / 2)

// drwnInPaint static members -----------------------------------------------

bool drwnInPaint::STEP_ITERATION = false;
bool drwnInPaint::WRITE_PROGRESS = false;
float drwnInPaint::CULL_FACTOR = 0.8f;
int drwnInPaint::MIN_WIN_SIZE = 13;
int drwnInPaint::MAX_WIN_SIZE = 29;
float drwnInPaint::LAMBDA = 0.8f;
float drwnInPaint::MIN_SIZE_FAC = 10.0f;
const Mat drwnInPaint::G_X = (Mat_<float>(3,3) << 3,  0, -3, 10, 0, -10, 3,  0, -3);
const Mat drwnInPaint::G_Y = (Mat_<float>(3,3) <<  3,  10,  3, 0,   0,  0, -3, -10, -3);

// drwnInPaint --------------------------------------------------------------

Mat drwnInPaint::inPaint(const Mat& source, Mat& output, const Mat& fillMask)
{
    // sourceMask is what is not inpainted if it is not supplied
    return inPaint(source, output, fillMask, 255 - fillMask,
        Mat(source.rows, source.cols, CV_8UC1, Scalar(0xFF)));
}

Mat drwnInPaint::inPaint(const Mat& source, Mat& output,
    const Mat& fillMask, const Mat& sourceMask) {
    return inPaint(source, output, fillMask, sourceMask,
        Mat(source.rows, source.cols, CV_8UC1, Scalar(0xFF)));
}

Mat drwnInPaint::inPaint(const Mat& source, Mat& output,
    const Mat& fillMaskParam, const Mat& sourceMask, const Mat& validMask) {

    DRWN_ASSERT_MSG((source.cols == fillMaskParam.cols) && (source.rows == fillMaskParam.rows) &&
        (fillMaskParam.cols == sourceMask.cols) && (fillMaskParam.rows == sourceMask.rows),
        "inPaint(): Image size does not match the masks' sizes");

    // Make a copy of the fillMask since we'll be modifying it
    Mat fillMask = fillMaskParam.clone();

    output = source.clone();

    float lambda = 0.0;

    // Try downsample everything and get a low res inpainted skeleton first
    DRWN_LOG_STATUS("Downsampling to " << source.cols/2 << "x" << source.rows/2 << endl);
    Mat smallSource, smallFillMask, smallSourceMask, smallValidMask;
    pyrDown(source, smallSource, Size(source.cols/2, source.rows/2));
    pyrDown(fillMask, smallFillMask, Size(source.cols/2, source.rows/2));
    pyrDown(sourceMask, smallSourceMask, Size(source.cols/2, source.rows/2));
    pyrDown(validMask, smallValidMask, Size(source.cols/2, source.rows/2));
    // Threshold the masks to get a clear edge
    threshold(smallFillMask, smallFillMask, 1, 255, THRESH_BINARY);
    threshold(smallSourceMask, smallSourceMask, 254, 255, THRESH_BINARY);
    threshold(smallValidMask, smallValidMask, 1, 255, THRESH_BINARY);
    Mat result;

    if (min(source.cols/2, source.rows/2) > MIN_SIZE_FAC * MIN_WIN_SIZE) {
        inPaint(smallSource, result, smallFillMask, smallSourceMask, smallValidMask);
        lambda = LAMBDA;
    } else {
        cv::inpaint(smallSource, smallFillMask, result, 2 * MIN_WIN_SIZE, INPAINT_TELEA);
        lambda = 0.05f;
    }

    DRWN_LOG_STATUS("Inpainting at " << source.cols/2 << "x" << source.rows/2 << " complete" << endl);

    // Upscale the result
    pyrUp(result, result, Size(source.cols, source.rows));
    // Copy over the patches that are in the mask
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            if (fillMask.at<uint8_t>(y, x) != 0 &&
                validMask.at<uint8_t>(y, x) != 0) {
                output.at<Vec3b>(y, x) = result.at<Vec3b>(y, x);
            }
        }
    }

    if (STEP_ITERATION) {
        // write all the parameters out for debugging
        imwrite("temp/source.png", source);
        imwrite("temp/fillMask.png", fillMask);
        imwrite("temp/sourceMask.png", sourceMask);
        imwrite("temp/validMask.png", validMask);
    }

    // Convert image to L*a*b colour space for perceptual-based processing
    cvtColor(output, output, CV_RGB2Lab);

    // This function expects the masks to have a range from 0x00 to 0xFF
    // where 0xFF = region to be inpainted and 0x00 = known pixels for a
    // fillMask. We'll be calculating confidence values around the
    // edge of the fillMask so we need to make sure that confidence
    // values we have right now cover all the area that aren't filled
    // (which doesn't necessarily = sourceMask)
    Mat confidence(source.rows, source.cols, CV_32FC1);
    // Scale it to a float matrix with range 0 - 1 and flip it around
    // since 0x00 = known pixels
    fillMask.convertTo(confidence, CV_32F, 1.0/255.0);
    confidence = 1.0 - confidence;

    // Vector of pixel locations on the edge of the fill region (del Omega)
    vector<_Pixel> fillFront;
    // Keep count of number of iterations
    int iterations = 0;

    // starting file sequence for progress output in ./progress/
    int fileID = 0;

    if (WRITE_PROGRESS) {
        ifstream file;
        char filename[50];

        // Find the last number in the file sequence
        while (sprintf(filename, "progress/%u.png", fileID),
                file.open(filename), file.is_open()) {
            file.close();
            fileID++;
        }
    }

    drwnThreadPool threadPool;
    queue<_findExemplarJob *> jobs;

    // Update the fill front (del omega) and keep looping until it's empty
    while (findFillFront(fillMask, validMask, fillFront), !fillFront.empty()) {
        DRWN_LOG_STATUS("Iteration " << ++iterations);

        for (vector<_Pixel>::iterator it = fillFront.begin();
             it != fillFront.end(); it++) {
            // "Confidence" terms
            it->confidence = findConfidence(*it, confidence);
            // "Data" terms
            it->data = findData(*it, fillMask, output);
            // Priority is just the product of the two
            it->priority = it->confidence * it->data;
        }

        if (STEP_ITERATION) {
            // Normalise and write out the internal priority data
            Mat confidenceOutput(output.rows, output.cols, CV_32FC1, Scalar(0));
            Mat dataOutput = confidenceOutput.clone();
            Mat priorityOutput = confidenceOutput.clone();

            for (vector<_Pixel>::iterator it = fillFront.begin();
                    it != fillFront.end(); it++) {
                 dataOutput.at<float>(it->y, it->x) = it->data;
                 priorityOutput.at<float>(it->y, it->x) = it->priority;
            }

            normalize(confidence, confidenceOutput, 255, 0, NORM_INF);
            imwrite("temp/confidence.png", confidenceOutput);
            normalize(dataOutput, dataOutput, 255, 0, NORM_INF);
            imwrite("temp/data.png", dataOutput);
            normalize(priorityOutput, priorityOutput, 255, 0, NORM_INF);
            imwrite("temp/priority.png", priorityOutput);
        }

        // --- Find the patch with the highest priority ---
        sort(fillFront.begin(), fillFront.end());

        // Multi-threaded exemplar search
        // Mask to keep track of which thread is calculating replacement patches
        // for which area
        Mat lock(source.rows, source.cols, CV_8UC1, Scalar(0));

        // Set a limit on the minimum priority for processing to be 5%
        // of the highest priority so that we don't end up processing
        // all the points with extremely low priority early on
        for (vector<_Pixel>::reverse_iterator it = fillFront.rbegin();
                it->priority > fillFront.rbegin()->priority * 0.05 &&
                threadPool.numJobsRemaining() < threadPool.numThreads() &&
                it != fillFront.rend(); it++) {

            // Convolve the lock with a box filter the size of the maximum window so that
            // if the window is not entirely unlocked it would not have value 0
            Mat averagedLock;
            boxFilter(lock, averagedLock, -1, Size(MAX_WIN_SIZE, MAX_WIN_SIZE));

            // There may be high priority pixels tightly clustered together
            // so we look at the lock mask and make sure that the area
            // for each thread do not overlap
            if (averagedLock.at<uint8_t>(it->y, it->x) != 0) {
                continue;
            }

            DRWN_LOG_DEBUG("Inpaint target: " << *it << " with priority value " << it->priority);

            // Update the lock mask with the location for this job
            rectangle(lock, *it - Point(MAX_WIN_SIZE / 2, MAX_WIN_SIZE / 2),
                *it + Point(MAX_WIN_SIZE / 2, MAX_WIN_SIZE / 2), Scalar(0xFF), -1);

            jobs.push(new _findExemplarJob(*it, output, fillMask, sourceMask, validMask, lambda));
            threadPool.addJob(jobs.back());
        }
        threadPool.start();

        // Twiddle our thumbs while the threads do the hard work
        threadPool.finish(false);

        vector< pair< pair<Point, Point>, int> > replacements;

        while (!jobs.empty()) {
            // Fill in each patch
            inPaintPatch(jobs.front()->px, jobs.front()->replacement, validMask, output,
                fillMask, confidence, jobs.front()->difference);
            replacements.push_back(pair< pair<_Pixel, _Pixel>, int>(
                pair<_Pixel, _Pixel>(jobs.front()->sourcePx, jobs.front()->px),
                jobs.front()->replacement.cols));
            delete jobs.front();
            jobs.pop();
        }

        if (WRITE_PROGRESS) {
            Mat output_RGB;
            char filename[50];
            cvtColor(output, output_RGB, CV_Lab2RGB);
            MatIterator_<Vec3b> it;
            MatIterator_<uint8_t> mit;

            for (it = output_RGB.begin<Vec3b>(),
                  mit = fillMask.begin<uint8_t>();
                  it != output_RGB.end<Vec3b>(); it++, mit++) {
                if (*mit != 0) {
                    *it /= 3;
                }
            }

            for (float i = 0; i < replacements.size(); i++) {
                Point src = replacements[i].first.first;
                Point dest = replacements[i].first.second;
                Point halfWindow = Point(replacements[i].second / 2, replacements[i].second / 2);
                Point window = Point(replacements[i].second, replacements[i].second);
                Vec3b color = heatMap((i + 1)/ (float) replacements.size());
                Scalar colorScalar = Scalar(color[0], color[1], color[2]);
                rectangle(output_RGB, src, src + window, colorScalar, 1);
                rectangle(output_RGB, dest - halfWindow, dest + halfWindow, colorScalar, 1);
            }

            sprintf(filename, "progress/%u.png", fileID++);
            imwrite(filename, output_RGB);
        }

        if (STEP_ITERATION) {
            Mat output_RGB;
            cvtColor(output, output_RGB, CV_Lab2RGB);
            imwrite("temp/output.png", output_RGB);
            imwrite("temp/fillMask.png", fillMask);
            imwrite("temp/lock.png", lock);

            // Wait for user input
            getchar();
        }
    }

    // Convert it back to RGB and write it out
    cvtColor(output, output, CV_Lab2RGB);
    if (STEP_ITERATION) {
        imwrite("temp/output.png", output);
    }

    DRWN_LOG_STATUS("Image inpainting finished. Total iterations " << iterations << "." << endl);

    return output;
}

void drwnInPaint::inPaintPatch(const _Pixel& px, const Mat& replacement, const Mat& validMask,
    Mat& output, Mat& fillMask, Mat& confidence, const float& diff) {

    // need to set windowSize for the macro "halfWinSize"
    int windowSize = replacement.cols;

    // Copy the replacement patch into the output and update the
    // fill mask and confidence values of the newly inpainted patch

    // patchConfidence reflects the exemplar accuracy as per Zhang & Zhou
    float patchConfidence = exp(-diff) * px.confidence;

    for (int y = px.y - halfWinSize; y <= px.y + halfWinSize; y++) {
        for (int x = px.x - halfWinSize; x <= px.x + halfWinSize; x++) {
            if (x >= 0 && y >= 0 && x < output.cols && y < output.rows) {
                // Only modify the pixels that need to be inpainted and
                // are in the valid area
                if (fillMask.at<uint8_t>(y, x) != 0 &&
                    validMask.at<uint8_t>(y, x) != 0) {
                    confidence.at<float>(y, x) = patchConfidence;

                    output.at<Vec3b>(y, x) = replacement.at<Vec3b>(y -
                        (px.y - halfWinSize), x - (px.x - halfWinSize));
                    fillMask.at<uint8_t>(y, x) = 0;
                }
            }
        }
    }
}

Mat drwnInPaint::extract(const Mat& source, const int& x, const int& y,
    const int& width, const int& height) {
    // The requested area is outside of the source's bound - need to extrapolate
    if (x < 0 || y < 0 || x + width >= source.cols || y + height >= source.rows) {
        // The width of padding on top and left corner = pixel offset
        // needed to access the same region after padding
        int xOffset = x < 0 ? -x : 0;
        int yOffset = y < 0 ? -y : 0;

        Mat paddedSource;
        copyMakeBorder(source, paddedSource, yOffset,
          y + height >= source.rows ? y + height - source.rows + 1 : 0,
          xOffset, x + width >= source.cols ? x + width - source.cols + 1 : 0,
          BORDER_REPLICATE);

        return paddedSource(Range(y + yOffset, y + yOffset + height),
          Range(x + xOffset, x + xOffset + width));
    } else {
        return source(Range(y, y + height), Range(x, x + width));
    }
}

float drwnInPaint::calcDifference(const Mat& source1, const Mat& source2,
    const Mat& mask, const float& lambda, const Mat& validMask = Mat()) {
    // Can't use openCV's template matching since we have a mask

    DRWN_ASSERT_MSG((source1.cols == source2.cols) && (source1.rows == source2.rows),
        "calcDifference(): images have different dimensions: " << source1.cols << "x"
        << source1.rows << " vs " << source2.cols << "x" << source2.rows);
    DRWN_ASSERT_MSG((source1.depth() == CV_8U) && (source2.depth() == CV_8U),
      "calcDifference(): images' bit depth isn't unsigned 8-bit");

    // Mask2 is the inverse mask - or the pixels that are valid and are
    // to be inpainted. It's useful if the region were previously inpainted
    // (i.e. at a lower resolution or with another algorithm) and it's
    // beneficial to give it some weight in the final difference calculation
    Mat mask1, mask2;

    if (!validMask.empty()) {
        bitwise_and(mask, validMask, mask1);
        bitwise_and(255 - mask, validMask, mask2);
    } else {
        mask1 = mask;
        mask2 = 255 - mask;
    }

    const int mask1px = countNonZero(mask1);
    const int mask2px = countNonZero(mask2);
    DRWN_ASSERT(mask1px != 0);

    // Calculate the mean of squared differences (MSD) between the two
    // sources using the L2 norm. The differences are scaled by lambda
    // (0 < lambda < 1) for pixel differences that are in mask2.
    float MSD = norm(source1, source2, NORM_L2, mask1) / mask1px;

    if (mask2px != 0) {
        MSD += lambda * norm(source1, source2, NORM_L2, mask2) / mask2px;
    }

    return MSD;
}

void drwnInPaint::findFillFront(const Mat& fillMask, const Mat& validMask, vector<_Pixel>& fillFront) {

    // /TODO Currently discards previous results -> quite wasteful, could optimise a bit more
    fillFront.clear();

    // Find the edges of the inpaint mask by only selecting the pixels that
    // are white and are surrounded by at least 1 dark pixel - so we
    // convolve it with a 3x3 box filter
    Mat fillMaskFiltered;
    boxFilter(fillMask, fillMaskFiltered, -1, Size(3, 3));

    for (int y = 0; y < fillMask.rows; y++) {
        for (int x = 0; x < fillMask.cols; x++) {

            // And test if the convolution result is somewhere inbetween.
            // Also makes sure that the fill front is within the mask
            // and that it is within the valid area if the inpainting
            // is split
            if (fillMask.at<uint8_t>(y, x) != 0 &&
                validMask.at<uint8_t>(y, x) != 0 &&
                fillMaskFiltered.at<uint8_t>(y, x) > 0.0 &&
                fillMaskFiltered.at<uint8_t>(y, x) < 255.0) {

                // Extract the masks and make sure that we have valid patches
                // for comparison later on
                Mat psi_p_mask = 255 - extract(fillMask, x - (MIN_WIN_SIZE / 2),
                    y - (MIN_WIN_SIZE / 2), MIN_WIN_SIZE, MIN_WIN_SIZE);
                Mat validExtract = extract(validMask, x - (MIN_WIN_SIZE / 2),
                    y - (MIN_WIN_SIZE / 2), MIN_WIN_SIZE, MIN_WIN_SIZE);
                bitwise_and(psi_p_mask, validExtract, psi_p_mask);

                // The overall mask must have at least 1 white pixel
                // otherwise exemplar search will fail since we have nothing
                // to compare against.
                if (countNonZero(psi_p_mask) > 0) {
                    fillFront.push_back((_Pixel) Point(x, y));
                }
            }
        }
    }

    if (STEP_ITERATION)  {
        // Output the fillFront for debug purpose
        Mat fillFrontOutput(fillMask.rows, fillMask.cols, CV_8UC1, Scalar(0));
        for (vector<_Pixel>::iterator it = fillFront.begin();
             it != fillFront.end(); it++) {
            fillFrontOutput.at<uint8_t>(it->y, it->x) = 0xFF;
        }
        imwrite("temp/fillFront.png", fillFrontOutput);
    }
}

// Comparison function for ordering pixel-difference pairs according
// to their difference values
bool compFunc(const pair<Point, float>& a, const pair<Point, float>& b) 
{
    return a.second < b.second;
}

void drwnInPaint::_findExemplarJob::operator()() 
{
    // reset the size of template window to a small one and then
    // progressively enlarge it when finding exemplar patches
    int windowSize = MIN_WIN_SIZE;

    // --- Extract the highest priority window ---
    Mat psi_p = extract(_source, px.x - halfWinSize, px.y - halfWinSize, windowSize, windowSize);
    // Invert the mask because we want the area inside the window
    // that is valid (i.e. not to be inpainted) and AND it with the
    // valid mask to ignore pixels that are not valid to be propagated
    Mat psi_p_mask = 255 - extract(_fillMask, px.x - halfWinSize, px.y - halfWinSize, windowSize, windowSize);
    Mat validExtract = extract(_validMask, px.x - halfWinSize, px.y - halfWinSize, windowSize, windowSize);

    if (STEP_ITERATION) {
        Mat psi_p_output;
        cvtColor(psi_p, psi_p_output, CV_Lab2RGB);
        imwrite("temp/extract.png", psi_p_output);
        imwrite("temp/validExtract.png", validExtract);
        imwrite("temp/extractMask.png", psi_p_mask);
    }

    // Calculate the difference between this patch and every other
    // patch in the image - difference is inifinite initially
    // the Mat is for outputting as a png if required
    Mat diff(_source.rows, _source.cols, CV_32FC1, Scalar(numeric_limits<double>::infinity()));
    // otherwise just store the point of max difference and the difference value
    double lowestDiffValue = numeric_limits<double>::infinity();
    Point lowestDiff;

    // Filtered sourceMask to figure out wheter the current patch is
    // entirely within the source region
    Mat sourceMaskFiltered(_sourceMask.rows, _sourceMask.cols, CV_8UC1);

    // Convolve the mask with a box filter the size of the window so that
    // if the entire window is 0xFF then the output will still be 0xFF
    boxFilter(_sourceMask, sourceMaskFiltered, -1, Size(windowSize, windowSize));

    // Downscale everything by 2 and find a preliminary difference map first
    Mat smallSource, small_psi_p, small_psi_p_mask, smallSourceMaskFiltered;
    pyrDown(_source, smallSource, Size(_source.cols/2, _source.rows/2));
    pyrDown(psi_p, small_psi_p, Size(psi_p.cols/2, psi_p.rows/2));
    pyrDown(psi_p_mask, small_psi_p_mask, Size(psi_p.cols/2, psi_p.rows/2));
    pyrDown(sourceMaskFiltered, smallSourceMaskFiltered, Size(_source.cols/2, _source.rows/2));
    // Normalize the mask first since there may be isolated pixels that
    // get turn into < 128 when downsampled and thus can result in a
    // completely 0 downscaled mask if a simple threshold is used.
    normalize(small_psi_p_mask, small_psi_p_mask, 255, 0, NORM_INF);
    // Threshold the masks to get a clear edge
    threshold(small_psi_p_mask, small_psi_p_mask, 128, 255, THRESH_BINARY);

    // Pixel/Difference pairs for potential matching patches
    vector< pair<Point, float> > candidates;

    // Mat for downscaled difference for png output
    Mat smallDiff(smallSource.rows, smallSource.cols, CV_32FC1, Scalar(numeric_limits<double>::infinity()));

    for (int y = 0; y <= smallSource.rows - halfWinSize; y++) {
        for (int x = 0; x <= smallSource.cols - halfWinSize; x++) {
            if (smallSourceMaskFiltered.at<uint8_t>(y + (halfWinSize/2),
                      x + (halfWinSize/2)) == 0xFF) {
                Mat window = extract(smallSource, x, y, halfWinSize, halfWinSize);
                float diff = calcDifference(small_psi_p, window, small_psi_p_mask, _lambda);

                if (STEP_ITERATION) {
                    smallDiff.at<float>(y, x) = diff;
                }

                // The coordinates in candidates are for full sized image
                // so they're twice of the coords from the downscaled img
                candidates.push_back(pair<Point, float>(Point(x, y) * 2, diff));
            }
        }
    }

    if (STEP_ITERATION) {
        double maxDifference = 0.0;
        // Scale diff back to 255 and export the image
        // Can't use normalize since there are INFs in diff
        for (MatIterator_<float> it = smallDiff.begin<float>(); it != smallDiff.end<float>(); it++) {
            if ((*it > maxDifference) && (*it != numeric_limits<double>::infinity())) {
                maxDifference = *it;
            }
        }
        Mat smallDiff_output = smallDiff / maxDifference * 255.0;
        imwrite("temp/smallDiff.png", smallDiff_output);
    }

    DRWN_ASSERT(candidates.size() > 0);


    // Get the lowest (CULL_FACTOR * candidates) number of candidates
    // at the start of the vector (but are otherwise not sorted, nor do
    // they need to be)
    nth_element(candidates.begin(), candidates.begin() + ceil((1 - CULL_FACTOR) * candidates.size()), candidates.end(), compFunc);

    // Process the lowest difference points in full resolution up to
    // cullFactor * candidates
    for (vector< pair<Point, float> >::const_iterator it = candidates.begin();
            it != candidates.begin() + ceil((1 - CULL_FACTOR) * candidates.size()); it++) {

        for (int y = it->first.y; y < it->first.y + 2; y++) {
            for (int x = it->first.x; x < it->first.x + 2; x++) {
            // Check if the current region is entirely within source region
            // (i.e. all the terms are 0xFF and so when summed and then
            // normalised it is still 0xFF)
            // Need to shift the coordinates a bit though since the x & y
            // in Mat diff is top-left coordinate while sourceMaskFiltered
            // is centre coordinate.
                if (sourceMaskFiltered.at<uint8_t>(y + halfWinSize, x + halfWinSize) == 0xFF) {

                    Mat window = extract(_source, x, y, windowSize, windowSize);

                    float difference = calcDifference(psi_p, window, psi_p_mask, _lambda, validExtract);
                    if (STEP_ITERATION) {
                        diff.at<float>(y, x) = difference;
                    }

                    if (difference < lowestDiffValue) {
                        lowestDiffValue = difference;
                        lowestDiff.x = x;
                        lowestDiff.y = y;
                    }
                }
            }
        }

        // No need to keep searching if we already have a perfect match
        if (lowestDiffValue == 0) {
            break;
        }
    }

    if (STEP_ITERATION) {
        double maxDifference = 0.0;
        // Scale diff back to 255 and export the image
        // Can't use normalize since there are INFs in diff
        for (MatIterator_<float> it = diff.begin<float>(); it != diff.end<float>(); it++)
        {
            if (*it > maxDifference && *it != numeric_limits<double>::infinity())
            {
                maxDifference = *it;
            }
        }
        Mat diff_output = diff / maxDifference * 255.0;
        imwrite("temp/difference.png", diff_output);
    }

    // Make sure our match is valid (uninitialised Point is Point(-1, -1))
    DRWN_ASSERT((lowestDiff.x >= 0) && (lowestDiff.y >= 0) && (lowestDiffValue < numeric_limits<double>::infinity()));

    Rect window(lowestDiff.x, lowestDiff.y, MIN_WIN_SIZE, MIN_WIN_SIZE);
    // Variable window size per "An improved scheme for Criminisi's
    // inpainting algorithm" by Zhang and Zhou
    if (MAX_WIN_SIZE != MIN_WIN_SIZE) {
        // Try bigger window sizes at the same coordinate
        float newDiff = lowestDiffValue;
        int i = 0;
        Mat sourceMaskExtract;
        int sourceMaskSum;

        Rect oldWindow;

        // Keep increasing the window until the difference is worse
        // than the previous smaller window
        do {
            i++;
            lowestDiffValue = newDiff;
            oldWindow = window;
            windowSize += 2;
            // Try expand the window - by default it expands towards
            // the bottom right corner of the image.
            window += Size(2, 2);
            // But we want to expand from the centre, so we move the
            // x&y coordinates back towards the top left corner
            // unelss they are already at the top left corner in which
            // case we can only expand towards the centre of the image
            if (window.x > 0) {
                window.x--;
            }
            if (window.y > 0) {
                window.y--;
            }
            // And also if the patch was already touching the bottom and
            // right border of the image, then we need to move the x&y
            // coordinates back and expand towards the centre of the
            // image too.
            if (window.x + window.width >= _source.cols) {
                window.x -= 2;
            }
            if (window.y + window.height >= _source.rows) {
                window.y -= 2;
            }

            psi_p = extract(_source, px.x - halfWinSize, px.y - halfWinSize,
                window.width, window.height);
            psi_p_mask = 255-extract(_fillMask, px.x - halfWinSize, px.y - halfWinSize,
                window.width, window.height);
            validExtract = extract(_validMask, px.x - halfWinSize,
                px.y - halfWinSize, window.width, window.height);

            replacement = extract(_source, window.x, window.y, window.width, window.height);

            if (STEP_ITERATION) {
                Mat psi_p_output;
                cvtColor(psi_p, psi_p_output, CV_Lab2RGB);
                imwrite("temp/extract2.png", psi_p_output);
                imwrite("temp/extractMask2.png", psi_p_mask);
            }

            newDiff = calcDifference(psi_p, replacement, psi_p_mask, _lambda, validExtract);

            // Make sure the new region is still completely within the source region
            sourceMaskExtract = extract(_sourceMask, window.x, window.y, window.width, window.height);
            // the source mask extract should be non-zero for all elements
            sourceMaskSum = countNonZero(sourceMaskExtract);
        } while ((newDiff <= lowestDiffValue) && (i <= (MAX_WIN_SIZE - MIN_WIN_SIZE)/2) &&
            (window.x >= 0) && (window.x + window.width < _source.cols) &&
            (window.y >= 0) && (window.y + window.height < _source.rows) &&
            (sourceMaskSum == window.width * window.height));

        // Undo last iteration's changes since it was the iteration
        // that failed
        window = oldWindow;
    }

    // Get the replacement patch
    replacement = extract(_source, window.x, window.y, window.width, window.height);
    if (STEP_ITERATION) {
        Mat replacement_output;
        cvtColor(replacement, replacement_output, CV_Lab2RGB);
        imwrite("temp/replacement.png", replacement_output);
    }

    DRWN_LOG_DEBUG("Replacement: [" << window.x << ", " << window.y
        << "] with difference of " << lowestDiffValue <<
        " and a window size of " << window.width << " px");
    sourcePx = Point(window.x, window.y);
    difference = lowestDiffValue;
}

float drwnInPaint::findConfidence(const Point& px, Mat& confidence)
{
    int windowSize = MIN_WIN_SIZE;
    // Calculate the Confidence terms of the pixels
    Mat confidenceExtract = extract(confidence, px.x - halfWinSize,
      px.y - halfWinSize, windowSize, windowSize);
    // confidence = average confidence within the surrounding region
    float c = sum(confidenceExtract)[0] / (windowSize * windowSize);
    confidence.at<float>(px.y, px.x) = c;

    return c;
}

Vec2f drwnInPaint::findN_p(const Point& px, const Mat& fillMask)
{
    int x = px.x;
    int y = px.y;

    // Perform Sobel derivative on the region to find
    // the fillMask gradient in the region.
    Point grad(0, 0);
    Mat patch = extract(fillMask, x - 1, y - 1, 3, 3);
    patch.convertTo(patch, CV_32F);

    Mat patch_x = patch.mul(G_X);
    grad.x = sum(patch_x)[0];
    Mat patch_y = patch.mul(G_Y);
    grad.y = sum(patch_y)[0];

    return Vec2f(sqrt(double(grad.x*grad.x + grad.y*grad.y)), atan2(double(grad.y), double(grad.x)));
}

Vec2f drwnInPaint::findI_p(const Point& px, const Mat& fillMask, const Mat& source)
{
    // Find the source gradient
    // By definition the fillFront pixel is unknown
    // and could be surrounded by areas to be inpainted.
    // We need to find a 3x3 patch that's has as much
    // source region as possible to get a better result.
    // So we start with a larger (i.e. 7x7) window, box
    // filter it to find the average value of each 3x3
    // segment, and find the lowest 3x3 block sum since
    // source region is fillMask = 0;
    Point grad(0, 0);

    //int searchSize = 7;
    int x = px.x, y = px.y;
    //Point minMask;
    //Mat mask = extract(fillMask, x - searchSize/2,
    //    y - searchSize/2, searchSize, searchSize);
    //Mat filteredMask;
    //boxFilter(mask, filteredMask, CV_32F, Size(3, 3));
    vector<Mat> sourceSplit;
    split(source, sourceSplit);

    // Go through the middle n-2 x n-2 region in the output
    // to find the lowest value and ignore the results
    // at the border
    //minMaxLoc(filteredMask.colRange(Range(1, searchSize - 1))
    //    .rowRange(Range(1, searchSize - 1)), NULL, NULL,
    //    &minMask, NULL);

    // Translate the coordinates back into absolute coords
    // +1 because we skipped the border pixel in the
    // convolution results
    //minMask.x += x - searchSize/2 + 1;
    //minMask.y += y - searchSize/2 + 1;

    // Find gradient in all 3 channels
    for (int i = 0; i < 3; i++)
    {
        Mat patch = extract(sourceSplit[i], x - 1, y - 1, 3, 3);
        patch.convertTo(patch, CV_32F);

        Mat patch_x = patch.mul(G_X);
        grad.x += sum(patch_x)[0];
        Mat patch_y = patch.mul(G_Y);
        grad.y += sum(patch_y)[0];
    }

    // We want a vector that's orthogonal to the gradient
    return Vec2f(sqrt(double(grad.x*grad.x + grad.y*grad.y)), atan2(double(grad.x), double(-grad.y)));
}

float drwnInPaint::findData(const Point& px, const Mat& fillMask, const Mat& source)
{
    Vec2f n_p = findN_p(px, fillMask);
    Vec2f I_p = findI_p(px, fillMask, source);

    // Dot product = |a|*|b|*cos(theta), + 0.1 so that it doesn't go to 0
    return fabs(I_p[0] * n_p[0] * cos(n_p[1] - I_p[1])) + 0.1;
}

Vec3b drwnInPaint::heatMap(const double& val)
{
    // We define that val=0 -> 240 deg hue in HSV colour space (blue)
    // and val=1 -> 0 deg hue (red). Value and saturation are both 1
    // (maximum) to generate a variable colour depending on val
    // Formula from http://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV

    double H_p = val * 4;     // H' is hue (in deg) divided by 60

    if (H_p > 4.0) {
        H_p = 4.0;
    } else if (H_p < 0.0) {
        H_p = 0.0;
    }

    // C is 1 but we want to scale everything to 0..255 instead of 0..1
    double X = 255 * (1 - abs(fmod(H_p, 2) - 1));

    switch ((int) floor(H_p)) {
      case 0:
        return Vec3b(255, X, 0);
      case 1:
        return Vec3b(X, 255, 0);
      case 2:
        return Vec3b(0, 255, X);
      case 3:
      case 4:
        return Vec3b(0, X, 255);
      default:
        return Vec3b(0, 0, 0);
    }
}

// drwnInPaintConfig --------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnInPaint
//! \b stepIteration   :: whether to write out results at each iteration and wait for user input (default: false)\n
//! \b writeProgress   :: whether to output progress to ./progress/ at each iteration (default: false)\n
//! \b cullFactor      :: the ratio of high difference candidates that are removed before full resolution comparison (default: 0.8)\n
//! \b minWindowSize   :: the initial window size for patch matching (must be odd) (default: 13)\n
//! \b maxWindowSize   :: the maximum window size for patch matching (must be odd) (default: 29)\n
//! \b lambda          :: weighting for difference in area that were previously inpainted at a lower resolution (default: 0.8)\n
//! \b minSizeFactor   :: the minimum ratio between image size and minWindowSize for downscaled inpainting (default: 10)

class drwnInPaintConfig : public drwnConfigurableModule {
    public:
    friend class drwnInPaint;

    drwnInPaintConfig() : drwnConfigurableModule("drwnInPaint") { };
    ~drwnInPaintConfig() { };

    void usage(ostream &os) const {
        os << "      stepIteration :: whether to write out results at each iteration and wait for user input (default: "
           << drwnInPaint::STEP_ITERATION << ")\n";
        os << "      writeProgress :: whether to output progress to ./progress/ at each iteration (default: "
           << drwnInPaint::WRITE_PROGRESS << ")\n";
        os << "      cullFactor    :: the ratio of high difference candidates that are removed before full resolution comparison (default: "
           << drwnInPaint::CULL_FACTOR << ")\n";
        os << "      minWindowSize :: the initial window size for patch matching (must be odd) (default: "
           << drwnInPaint::MIN_WIN_SIZE << ")\n";
        os << "      maxWindowSize :: the maximum window size for patch matching (must be odd) (default: "
           << drwnInPaint::MAX_WIN_SIZE << ")\n";
        os << "      lambda        :: weighting for difference in area that were previously inpainted at a lower resolution (default: "
           << drwnInPaint::LAMBDA << ")\n";
        os << "      minSizeFactor :: the minimum ratio between image size and minWindowSize for downscaled inpainting (default: "
           << drwnInPaint::MIN_SIZE_FAC << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "stepIteration")) {
            drwnInPaint::STEP_ITERATION = trueString(value);
        } else if (!strcmp(name, "writeProgress")) {
            drwnInPaint::WRITE_PROGRESS = trueString(value);
        } else if (!strcmp(name, "cullFactor")) {
            drwnInPaint::CULL_FACTOR = std::min(std::max(0.0, atof(value)), 1.0);
        } else if (!strcmp(name, "minWindowSize")) {
            int val = std::max(5, atoi(value));
            if (val % 2 == 1) {
                drwnInPaint::MIN_WIN_SIZE = val;
            } else {
                DRWN_LOG_FATAL("minWindowSize must be odd.");
            }
        } else if (!strcmp(name, "maxWindowSize")) {
            int val = std::max(5, atoi(value));
            if (val % 2 == 1) {
                drwnInPaint::MAX_WIN_SIZE = val;
            } else {
                DRWN_LOG_FATAL("maxWindowSize must be odd.");
            }
        } else if (!strcmp(name, "lambda")) {
            drwnInPaint::LAMBDA = std::max(0, atoi(value));
        } else if (!strcmp(name, "minSizeFactor")) {
            drwnInPaint::MIN_SIZE_FAC = std::max(1, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnInPaintConfig gInPaintConfig;
