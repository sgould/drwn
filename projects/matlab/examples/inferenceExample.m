% INFERENCE EXAMPLE
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2016, Stephen Gould
% All rights reserved.
%
% This example demonstrates how to use the Darwin mex interfaces to run
% various graphical models inference algorithms.

addpath('../../../bin/');
rand('seed', -1);

% define variables A, B, and C with cardinalities 3, 2, and 2
universe = [3, 2, 2];

% define factors over (A, B) and (B, C) with random entries
factors = [];
factors(1).vars = [0 1];
factors(1).data = rand(prod(universe([1 2])), 1);
factors(2).vars = [1 2];
factors(2).data = rand(prod(universe([2 3])), 1);

% run junction tree inference
options = struct('verbose', 1, 'method', 'drwnJunctionTreeInference');
assignment = mexFactorGraphInference(universe, factors, [], options);
disp(assignment);

% run max-product inference
options = struct('verbose', 1, 'method', 'drwnMaxProdInference');
assignment = mexFactorGraphInference(universe, factors, [], options);
disp(assignment);

% run icm inference
options = struct('verbose', 1, 'method', 'drwnICMInference');
assignment = mexFactorGraphInference(universe, factors, [], options);
disp(assignment);

% run gemplp inference
options = struct('verbose', 1, 'method', 'drwnGEMPLPInference');
assignment = mexFactorGraphInference(universe, factors, [], options);
disp(assignment);
