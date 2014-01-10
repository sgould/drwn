% ROSETTA2DRWN  Convert Rosetta Protein Design to Darwin Factor Graph
% Distributed under the terms of the BSD license (see the LICENSE file)
% Stephen Gould <stephen.gould@anu.edu.au>
%
function rosetta2drwn(modelName);

% read input
disp(['Reading ', modelName, '...']);
load([modelName, '.dee.mat']);

nVariables = length(Ei);
disp([int2str(nVariables), ' variables']);

% construct edges between pairwise and singleton nodes
indx = []; edges = [];
[srcNode, dstNode] = find(adjMatrix);
nPairwiseNodes = length(srcNode);
for i = 1:nPairwiseNodes,
    if (srcNode(i) >= dstNode(i)), continue; end;
    phi = Eij{srcNode(i), dstNode(i)};
    %if (all(phi == phi(1))), continue; end;
    indx = [indx; i];
    edges(end + 1, :) = [srcNode(i), nVariables + length(indx)];
    edges(end + 1, :) = [dstNode(i), nVariables + length(indx)];
end;

srcNode = srcNode(indx);
dstNode = dstNode(indx);
nPairwiseNodes = length(indx);
nEdges = size(edges, 1);

disp([int2str(nPairwiseNodes), ' pairwise cliques']);

% write output
disp(['Writing to ', modelName, '.graph.xml...']);
fid = fopen([modelName, '.graph.xml'], 'w');
fprintf(fid, '<drwnFactorGraph>\n');

fprintf(fid, '  <drwnVarUniverse nVariables="%d" uniformCards="0">\n', nVariables);
fprintf(fid, '    <varCards>\n     ');
varCards = zeros(nVariables, 1);
for i = 1:nVariables,
    fprintf(fid, ' %d', length(Ei{i}));
    varCards(i) = length(Ei{i});
end;
fprintf(fid, '\n    </varCards>\n');
fprintf(fid, '  </drwnVarUniverse>\n');

for i = 1:nVariables,
    fprintf(fid, '    <factor type="drwnTableFactor">\n');
    fprintf(fid, '      <vars>%d</vars>\n', i - 1);
    fprintf(fid, '      <data rows="%d" encoder="text">\n', length(Ei{i}));
    fprintf(fid, '        %12.8f\n', Ei{i});
    fprintf(fid, '      </data>\n');
    fprintf(fid, '    </factor>\n');
end;
for i = 1:nPairwiseNodes,
    phi = Eij{srcNode(i), dstNode(i)};
    fprintf(fid, '    <factor type="drwnTableFactor">\n');
    fprintf(fid, '      <vars>%d %d</vars>\n', srcNode(i) - 1, dstNode(i) - 1);
    if (any(phi(:))),
            fprintf(fid, '      <data rows="%d" encoder="text">\n', length(phi(:)));
            fprintf(fid, '        %12.8f\n', phi);
            fprintf(fid, '      </data>\n');
    end;
    fprintf(fid, '    </factor>\n');    
end;

fprintf(fid, '  <edges>\n');
for i = 1:nEdges,
    fprintf(fid, '    %d %d\n', edges(i, 1) - 1, edges(i, 2) - 1);
end;
fprintf(fid, '  </edges>\n');

fprintf(fid, '</drwnFactorGraph>\n');
fclose(fid);

disp('...done');
