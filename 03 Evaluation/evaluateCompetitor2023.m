function [ results,evalmetric,estimatesFile,analysedPoints] = evaluateCompetitor2023( competitionTrack, folderTrials, teamName, trial, distanceMetric, PDF_generation, basedir)

close all

competitionEdition  = '2023';
altitudeForThisYear = 314;    % TO BE UPDATED EVERY YEAR
                              % 220 for DSI 2022
							  % 314 for Museum 2023

% Parameters for the PDF report
linewidth_IPIN      = 1;
MarkerSize_IPIN     = 15;
verboseMode         = 0; % 1 to show results in screen

%% Get the files

% Load ground truth corresponding to the trial
%if strcmp(teamName(1),'S')
%GT = dlmread([basedir filesep 'GT' filesep 'GT_Track' sprintf('%02d',competitionTrack) '_' competitionEdition '_Trial' sprintf('%02d',trial)  '.csv' ]);
%else
%    GT = dlmread([basedir filesep 'GT' filesep 'GT_Track' sprintf('%02d',competitionTrack) '_' competitionEdition '_Trial' sprintf('%02d',trial)  'Testing.csv' ]);
%end

%% New format for GT filename 2023
GT = dlmread([basedir filesep 'GT' filesep 'GT_IPIN' competitionEdition '_T' sprintf('%01d',competitionTrack)  '_ScoringTrial' sprintf('%02d',trial) '.csv' ]);
nsamples   = size(GT,1);

% Get the filename with results
track = 'Results';
estimatesFile  = [teamName , sprintf('%02d',trial)];

% Get alternative results file - Francesco appended 'a' to those files
if exist([basedir  filesep folderTrials filesep  estimatesFile 'b.est'],'file')
    estimatesFile = [estimatesFile, 'b'];
elseif exist([basedir  filesep folderTrials  filesep  estimatesFile 'a.est'],'file')
    estimatesFile = [estimatesFile, 'a'];
end

% Load position estimates, if they do not exist fill estimatins with +inf
escasos_datos=true;
try

    %if exists([basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) 'b.est'])
    %    filenameInput = [basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) 'b.est'];
    %elseif exists([basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) 'a.est'])
    %    filenameInput = [basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) 'a.est'];
    %else exists([basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) '.est'])
    %    filenameInput = [basedir filesep 'trials'  filesep  estimatesFile sprintf('%02d',trial) '.est'];
    %end

    CL_FULL =  dlmread([basedir filesep folderTrials  filesep  estimatesFile '.est'],',',2,0);
    if size(CL_FULL,2)>5   % deben tener 8 columnas los ficheros de los competidores
        escasos_datos=false;
    end
    %%%%%%%%%%%%%%%%%%%%
    if median(CL_FULL(:,6))<10   % Wrong order in coordinates, it's LON,LAT
        CL_FULL(:,[6,5]) = CL_FULL(:,[5,6]);
    end
    %%%%%%%%%%%%%%%%%%%%


catch % If problems opening the file, the trial is rejected
    results        = inf*ones(size(GT,1));
    evalmetric     = inf;
    analysedPoints = 0;
    return;
end
CL_FULL = CL_FULL(:,[1,5,6,7]); % time, LON. LAT, Floor


nsamples = size(GT,1);
for i = 1:nsamples
    temporalDiffTime = (CL_FULL(:,1) - GT(i,1));
    temporalDiffTime(temporalDiffTime>0.01) = -inf;
    [value,pos]      = max(temporalDiffTime);
    if temporalDiffTime(pos) == -inf
        disp('Error')
        return;
    end
    %CL_FULL(pos,5)= i;
    CL(i,:) = [CL_FULL(pos,1:4),i,abs(value)];
end

%% If competitor participates in Track 3, the evaluation points indexes have to be injected by organizers according to the timestamp
evaltimestamps = GT(GT(:,5) > 0,1);
%%for et = 1:size(evaltimestamps,1)
%%    CL_FULL((CL_FULL(:,1) == evaltimestamps(et)),5) = et;
%%end

nsamples = size(GT,1);
for i = 1:nsamples
    list_key_point_comp{i} = sprintf('%02d',i);
end

%%CL = CL_FULL(CL_FULL(:,5) > 0,:);

% If the logfile is incomplete, the trial is rejected
%analysedPoints = size(CL,1);

%if size(CL,1)<size(GT,1)
%    results    = inf*ones(size(GT,1));
%    evalmetric = inf;
%    %disp('Not complete')
%    return;
%end

% If timing 0.5Hz is not satisfied or not enough points

analysedPoints = sum(CL(:,6)<=0.5);
%if analysedPoints ~= nsamples;
%    results    = inf*ones(size(GT,1));
%    evalmetric = inf;
%    %disp('Not complete')
%    return;
%end



% Compute individual positioning errors X-Y
if strcmp(distanceMetric, 'geodesic')
    %% transform geo to ned
    %tangence point : lower left point of Ground Truth
    lat0  = min(GT(:,3));
    lon0  = min(GT(:,2));
    h0    = altitudeForThisYear; %220;%79.765;%default ~220 in DSI
    % altitude hyp : m�me que base
    h=ones(length(GT(:,1)),1)*h0;
    hfull=ones(length(CL_FULL(:,1)),1)*h0;
    % for GT
    [xNorth_GT,yEast_GT,zDown_GT] = geodetic2ned(GT(:,3),GT(:,2),h,lat0,lon0,h0,wgs84Ellipsoid('meters'));
    % for CL
    [xNorth_CL ,yEast_CL ,zDown_CL]  = geodetic2ned(CL(:,3),CL(:,2),h,lat0,lon0,h0,wgs84Ellipsoid('meters'));
    [xNorth_CL2,yEast_CL2,zDown_CL2] = geodetic2ned(CL_FULL(:,3),CL_FULL(:,2),hfull,lat0,lon0,h0,wgs84Ellipsoid('meters'));

    %% comptutation of Distance(Ri,Ei) and SampleError
    Distance_Ri_Ei = sqrt((xNorth_GT-xNorth_CL).^2 + (yEast_GT-yEast_CL).^2);
else
    Distance_Ri_Ei = zeros(size(evaltimestamps,1),1);
    for et = 1:size(evaltimestamps,1)
        Distance_Ri_Ei(et) = distance_harvesine(GT(et,[2,3]),CL(et,[2,3]))*1000;
    end
end

results(:,1) = Distance_Ri_Ei;
results(:,2) = 15*abs(GT(:,4)-CL(:,4));
SampleError=Distance_Ri_Ei + 15*abs(GT(:,4)-CL(:,4));

ThirdQuartil = quantile(SampleError,0.75);
evalmetric = ThirdQuartil;


% .........DOCUMENT the evaluation.........:
if ~escasos_datos && abs(mean(CL(:,2)))<90 && abs(mean(CL(:,3)))<90 && PDF_generation  % deben ser valores entre +/-90 grados
    % KML files:
    generate_KML_files(teamName, trial, nsamples, GT, CL, CL_FULL);
    % Plot and PDF:
    plot_and_pdfgeneration(results,teamName,trial,PDF_generation,nsamples, GT, CL, CL_FULL,xNorth_GT,yEast_GT,xNorth_CL,yEast_CL,xNorth_CL2,yEast_CL2);
end

end
