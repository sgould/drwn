% IMAGECRFEXAMPLE
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2014, Stephen Gould
% All rights reserved.
%

addpath('../../../bin/');

H = 32; % image height
W = 48; % image width
L = 5;  % number of labels

image = rand(H, W, 3);
unary = rand(H, W, L);

options = struct('verbose', 0, 'debug', 0, 'profile', 1);

% pairwise CRF
figure(1);
for lambda = 0:0.1:0.5,
    x = mexImageCRF(image, unary, lambda, options);

    subplot(1, 2, 1); imagesc(image); title('image');
    subplot(1, 2, 2); imagesc(x, [1, L]); title(['labels for \lambda = ', num2str(lambda)]);
    drawnow; pause(1);
end;

% higher-order CRF
regions = repmat(-1, [H, W]);
regions(H/4:3*H/4, W/4:3*W/4) = 0;

x = mexImageCRF(image, unary, 0.0, regions, 0.5, options);
figure(2);
subplot(1, 3, 1); imagesc(image); title('image');
subplot(1, 3, 2); imagesc(regions); title('regions');
subplot(1, 3, 3); imagesc(x, [1, L]); title('labels with robust potts');

