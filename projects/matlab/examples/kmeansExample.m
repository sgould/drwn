% K-MEANS EXAMPLE
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2017, Stephen Gould
% All rights reserved.
%

addpath('../../../bin/');

% generate some data
x = 3 * rand(600, 2) - 1;
x = [x; rand(500, 2) + repmat([1 0], 500, 1)];
x = [x; rand(200, 2) + repmat([-1 0], 200, 1)];
x = [x; rand(300, 2) + repmat([0 1], 300, 1)];
x = [x; rand(400, 2) + repmat([0 -1], 400, 1)];

% run k-means
options = struct('verbose', 1, 'maxiters', 100);
centroids = mexKMeans(5, x, [], options);

% plot results
figure;
plot(x(:, 1), x(:, 2), 'b.');
hold on;
plot(centroids(:, 1), centroids(:, 2), 'rx', 'MarkerSize', 5, 'LineWidth', 2);
hold off;
title('Darwin k-Means Example');
