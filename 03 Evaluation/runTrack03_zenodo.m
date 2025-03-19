
% IPIN Competition 2023
% Main file to evaluate competition estimations in csv files
close all; clear all; clc;

% Define team names, number of files(tries) and other parameters
edition        = '23';       % Edition of the conference (trial names)
trackNumber    = 3;          % Track number 3
nTrials        = 2;          % Max number of trials
distanceMetric = 'geodesic'; % geodesic or harvesine - harvesine by default
baseDir        = pwd;        % Location of files (GT & Trials) and results
folderTrials   = 'tr_zenodo' % folder with estimates 
PDF_generation = 2;          % No report at all
                             % 1 for text report and kml
                             % 2 for above plus PDF report

% Get teams from EST files
estFiles       = dir([baseDir filesep folderTrials filesep 'S' num2str(trackNumber) edition '*est']);
for i = 1:size(estFiles,1)
    estFilenamesShort(i) = string(estFiles(i).name(1:12)); % First 12 characters identify the team
end    
teams = unique(estFilenamesShort');

% Compute the scores for every trial
fprintf("\nTeamName-FileId     \tEvAAL  \tMEAN   \tRMSE   \tMEDIAN \tP95    \tFHR(%%) \tMPExy  \tPOINTS\n");
evalMetric_all=ones(size(teams,1),3)*inf;

if ~exist('Results','dir'); mkdir('Results'); end
for i =1:size(teams,1)  % for each team
    team_name=teams{i,1};
    for currentTrial = 1:nTrials   % for each try
        if ~exist(['Results' filesep team_name sprintf('%02d',currentTrial)],'dir'); mkdir(['Results' filesep team_name sprintf('%02d',currentTrial)]); end
        [results, evalMetric, estimatesFile,analysedPoints] = evaluateCompetitor2023( trackNumber, folderTrials, team_name, currentTrial, distanceMetric , PDF_generation, baseDir);
        fprintf('%20s:\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%d\n',estimatesFile,evalMetric,mean(results(:,1)+results(:,2)),...
                       sqrt(mean((results(:,1)+results(:,2)).^2)), median(results(:,1)+results(:,2)),...
                       quantile(results(:,1)+results(:,2),0.95),mean(results(:,2)==0)*100,mean(results(:,1)),analysedPoints);
                   evalMetric_all(i,currentTrial)=evalMetric;
        dlmwrite(['Results' filesep team_name sprintf('%02d',currentTrial) '.txt'],results(:,1)+results(:,2),'precision','%7.3f')
     end 
end

% Ranking
evalMetric_all_best=ones(size(teams,1),1)*100;
for i=1:size(teams,1) % each team
    evalMetric_all_best(i)=min(evalMetric_all(i,:));
end
[Metric,idx]=sort(evalMetric_all_best);
fprintf('\nRanking:\n Team name \t 3rd quantile \n');
for i=1:size(teams,1) % each team
    fprintf('%d %s \t %10.2f m \n',i,teams{idx(i),1},evalMetric_all_best(idx(i)));
end




