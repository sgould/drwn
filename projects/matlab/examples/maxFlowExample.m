% MAXFLOW EXAMPLE
% Distributed under the terms of the BSD license (see the LICENSE file)
% Copyright (c) 2007-2017, Stephen Gould
% All rights reserved.
%
% This example demonstrates how to use the Darwin mex interfaces to find
% the minimum-cut/maximum-flow in a directed graph with positive edge
% weights.

addpath('../../../bin/');

% define the graph
edgeList = [
    -1  0  3;  % SOURCE -> 0
    -1  2  3;  % SOURCE -> 2
     0  1  4;  % 0 -> 1
     1  2  1;  % 1 -> 2
     1  3  2;  % 1 -> 3
     2  3  2;  % 2 -> 3
     2  4  6;  % 2 -> 4
     3 -1  1;  % 3 -> TARGET
     4 -1  9;  % 4 -> TARGET
];

% run max-flow
options = struct('verbose', 1);
[value, cut] = mexMaxFlow(edgeList, options);

disp(['Value of minimum cut is ', num2str(value)]);
