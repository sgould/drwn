% DRWNSAVEDATASET  Saves a Darwin dataset
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2017, Stephen Gould
% All rights reserved.
%

function drwnSaveDataset(dataset, filename);

% check dataset
if (~isstruct(dataset)), error('dataset must be a structure'); end;
if (~isfield(dataset, 'features')), error('dataset must have a features field'); end;
if (~isfield(dataset, 'targets')), error('dataset must have a targets field'); end;

[nRecords, nFeatures] = size(dataset.features);
isIntegerLabel = isinteger(dataset.targets);
hasWeights = isfield(dataset, 'weights') && (~isempty(dataset.weights));
hasIndexes = isfield(dataset, 'indexes') && (~isempty(dataset.indexes));

% open the file and write the header
fid = fopen(filename, 'w');
flags = int32(hex2dec('00010000'));
if (hasWeights), flags = bitor(flags, hex2dec('00000001')); end;
if (hasIndexes), flags = bitor(flags, hex2dec('00000002')); end;
fwrite(fid, flags, '*uint32');
fwrite(fid, nFeatures, '*uint32');

bytesPerRecord = nFeatures * 8;
if (isIntegerLabel),
    bytesPerRecord = bytesPerRecord + 4;
else
    bytesPerRecord = bytesPerRecord + 8;
end;
if (hasWeights), bytesPerRecord = bytesPerRecord + 8; end;
if (hasIndexes), bytesPerRecord = bytesPerRecord + 4; end;

% write all data
for i = 1:nRecords,
    if (isIntegerLabel),
        fwrite(fid, dataset.targets(i), '*int32');
    else
        fwrite(fid, dataset.targets(i), 'double');
    end;
    fwrite(fid, double(dataset.features(i, :)), 'double');
    if (hasWeights),
        fwrite(fid, double(dataset.weights(i)), 'double');
    end;
    if (hasIndexes),
        fwrite(fid, int32(dataset.indexes(i)), '*int32');
    end;
end;

fclose(fid);
