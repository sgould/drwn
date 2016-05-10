% DRWNLOADDATASET  Loads a Darwin dataset
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2016, Stephen Gould
% All rights reserved.
%

function [dataset] = drwnLoadDataset(filename, isIntegerLabel);

% create dataset
if (nargin == 1), isIntegerLabel = 0; end;
dataset = struct('features', [], 'targets', [], 'weights', [], 'indexes', []);

% open the file and read the header
fid = fopen(filename);
flags = fread(fid, 1, '*uint32');
if (bitand(flags, uint32(hex2dec('ffff0000'))) ~= uint32(hex2dec('00010000'))),
    error('invalid dataset file');
end;
hasWeights = bitand(flags, uint32(hex2dec('00000001'))) == uint32(hex2dec('00000001'));
hasIndexes = bitand(flags, uint32(hex2dec('00000002'))) == uint32(hex2dec('00000002'));

nFeatures = fread(fid, 1, '*int32');
disp(sprintf('...reading %d length feature vectors', nFeatures));

bytesPerRecord = nFeatures * 8;
if (isIntegerLabel),
    bytesPerRecord = bytesPerRecord + 4;
else
    bytesPerRecord = bytesPerRecord + 8;
end;
if (hasWeights), bytesPerRecord = bytesPerRecord + 8; end;
if (hasIndexes), bytesPerRecord = bytesPerRecord + 4; end;

% estimate dataset size
p = ftell(fid);
fseek(fid, 0, 'eof');
bytesForRecords = ftell(fid) - p;
fseek(fid, p, 'bof');

nRecords = bytesForRecords / bytesPerRecord;
disp(sprintf('...reading %d examples', nRecords));

dataset.features = zeros(nRecords, nFeatures);
dataset.targets = zeros(nRecords, 1);
if (hasWeights), dataset.weights = zeros(nRecords, 1); end;
if (hasIndexes), dataset.indexes = zeros(nRecords, 1); end;

% read all records
for n = 1:nRecords,
    if (isIntegerLabel),
        y = fread(fid, 1, '*int32');
    else
        y = fread(fid, 1, 'double');
    end;
    [x, count] = fread(fid, nFeatures, 'double');
    if (count ~= nFeatures),
        error('corrupt file');
    end;
    
    dataset.features(n, :) = x';
    dataset.targets(n) = y;
    if (hasWeights), 
        w = fread(fid, 1, 'double');
        dataset.weights(n) = w;
    end;
    if (hasIndexes),
        indx = fread(fid, 1, '*int32');
        dataset.indexes(n) = indx;
    end;
end;

fclose(fid);
