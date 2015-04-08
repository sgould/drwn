% BUILDLINUX    Script for compiled mex interface to Darwin for Linux
% Stephen Gould <stephen.gould@anu.edu.au>

% check that Darwin libraries have been compiled
if (~exist('../../bin/libdrwnBase.a') || ~exist('../../bin/libdrwnIO.a') || ...
    ~exist('../../bin/libdrwnML.a') || ~exist('../../bin/libdrwnPGM.a')),
    error('YOU NEED TO COMPILE THE DARWIN LIBRARIES FIRST');
end;

% setup build string
mexBuildOptions = '-D__LINUX__ -I../../external -I../../include -L../../bin -ldrwnPGM -ldrwnML -ldrwnIO -ldrwnBase -outdir ../../bin';

% compile test application
eval(['mex ', mexBuildOptions, ' mexDarwinTest.cpp']);

% compile machine learning applications
eval(['mex ', mexBuildOptions, ' mexLearnClassifier.cpp']);
eval(['mex ', mexBuildOptions, ' mexEvalClassifier.cpp']);
eval(['mex ', mexBuildOptions, ' mexAnalyseClassifier.cpp']);
eval(['mex ', mexBuildOptions, ' mexKMeans.cpp']);
eval(['mex ', mexBuildOptions, ' mexGetLinearTransform.cpp']);
eval(['mex ', mexBuildOptions, ' mexSetLinearTransform.cpp']);

% build graphical models applications
eval(['mex ', mexBuildOptions, ' mexFactorGraphInference.cpp']);
eval(['mex ', mexBuildOptions, ' mexMaxFlow.cpp']);

% build vision applications
if (exist('../../external/opencv', 'dir')),
    [status, mexOpenCVOptions] = system('pkg-config --cflags --libs ../../external/opencv/lib/pkgconfig/opencv.pc');
    mexVisionBuildOptions = ['-D__LINUX__ -I../../external -I../../include -L../../bin ', ...
        strtrim(mexOpenCVOptions), ' ', ...
        ' -ldrwnVision -ldrwnPGM -ldrwnML -ldrwnIO -ldrwnBase -outdir ../../bin', ...
        ' ', strtrim(mexOpenCVOptions)];
    
    eval(['mex ', mexVisionBuildOptions, ' mexImageCRF.cpp']);
    eval(['mex ', mexVisionBuildOptions, ' mexLoadSuperpixels.cpp']);
    eval(['mex ', mexVisionBuildOptions, ' mexSaveSuperpixels.cpp']);
    eval(['mex ', mexVisionBuildOptions, ' mexLoadPatchMatchGraph.cpp']);
end
