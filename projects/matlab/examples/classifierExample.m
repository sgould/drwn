% CLASSIFIER EXAMPLE
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2016, Stephen Gould
% All rights reserved.
%

addpath('../../../bin/');

% load iris dataset
x = load('iris.data.txt');
y = load('iris.labels.txt');
nClasses = max(y) + 1;

% add some noise to the data
x = x + 0.5 * randn(size(x));

% learn a classifier
options = struct('verbose', 0, 'method', 'drwnMultiClassLogistic');
classifier = mexLearnClassifier(x, y, [], options);

% evaluate classifier (on training data)
options = struct('verbose', 1);
[predictions, scores] = mexEvalClassifier(classifier, x, options);

disp(['Accuracy: ', num2str(sum(predictions == y) / length(y))]);

% analyse classifier (on training data)
[c, pr] = mexAnalyseClassifier(scores, y);

figure;
for i = 1:nClasses,
    subplot(1, nClasses, i);
    plot(pr{i}(:, 1), pr{i}(:, 2), 'r-', 'LineWidth', 2);
    axis([0, 1, 0, 1]); grid on;
    title([int2str(i - 1), '-vs-all']);
    xlabel('recall'); ylabel('precision');
end;
