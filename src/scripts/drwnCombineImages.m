% DRWNCOMBINEIMAGES Combines a number of images into a single image.
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2014, Stephen Gould
% All rights reserved.
%
% Combines multiple images into a single big image. The image list can be
% specified as filenames or actual images.
%
%   [outImg] = drwnCombineImages(imageList, [options]);
%
%  imageList :: cell array of images or filenames
%  options   :: structure with fields 'width', 'height', 'spacing',
%               'border', 'bgcolor', 'rowwise', 'numrows'
%

function [outImg] = drwnCombineImages(images, options);

% check input arguments
if (nargin < 1), help combineImages; outImg = []; return; end;
if (~iscell(images)), error('"images" must be a cell array'); end;

if (nargin < 2), options = []; end;
if (~isfield(options, 'width')), options.width = 64; end;
if (~isfield(options, 'height')), options.height = 48; end;
if (~isfield(options, 'spacing')), options.spacing = 1; end;
if (~isfield(options, 'border')), options.border = 1; end;
if (~isfield(options, 'bgcolor')), options.bgcolor = [0, 0, 0]; end;
if (~isfield(options, 'rowwise')), options.rowwise = 1; end;
if (~isfield(options, 'numrows')), options.numrows = 0; end;

N = length(images);
if ((options.numrows > 0) && (options.numrows <= N)),
    Nr = options.numrows;
else
    Nr = ceil(sqrt(N));
end;
Nc = ceil(N / Nr);

outImg = uint8(zeros(Nr * options.height + (Nr - 1) * options.spacing + 2 * options.border, ...
    Nc * options.width + (Nc - 1) * options.spacing + 2 * options.border, 3));
outImg(:, :, 1) = options.bgcolor(1);
outImg(:, :, 2) = options.bgcolor(2);
outImg(:, :, 3) = options.bgcolor(3);

for i = 1:length(images),
    if (ischar(images{i})),
        img = imread(images{i});
    elseif (isa(images{i}, 'uint8')),
        img = images{i};
    else
        error(['unknown datatype for image ', int2str(i)]);
    end;
    % TODO: keep aspect ratio when resizing [H, W, C] = size(img);

    if (size(img, 3) == 1), img = repmat(img, [1, 1, 3]); end;
    img = imresize(img, [options.height, options.width], 'bilinear');
    
    if (options.rowwise),
        x = mod(i - 1, Nc) * (options.width + options.spacing) + options.border;
        y = floor((i - 1) / Nc) * (options.height + options.spacing) + options.border;
    else
        y = mod(i - 1, Nr) * (options.height + options.spacing) + options.border;
        x = floor((i - 1) / Nr) * (options.width + options.spacing) + options.border;
    end;
    
    outImg(y + (1:options.height), x + (1:options.width), :) = img;    
end;
