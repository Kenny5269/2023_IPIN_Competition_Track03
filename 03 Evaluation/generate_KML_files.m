function generate_KML_files(teamname, currentTrial, nsamples,GT,CL,CL_full)
% Create KML outputs (pins: GT, Estimations, Trajectory & Error lines)

Result_folder_name = ['Results'];

team  = [teamname sprintf('%02d',currentTrial)]; % string with "team + number of trial"

% for GT (ground-truth points)
points_names=cell(1,nsamples);       for i=1:nsamples, points_names{i}=['GT',num2str(i)]; end
kmlwritepoint([Result_folder_name,filesep,team,filesep,'GT_points.kml'], GT(:,3), GT(:,2),'Name', points_names,'IconScale',1,'Color','green');

% for CL (estimated points)
points_names=cell(1,nsamples);       for i=1:nsamples, points_names{i}=['E',num2str(i)]; end
kmlwritepoint([Result_folder_name,filesep,team,filesep,'Competitor_evaluation_points.kml'], CL(:,3), CL(:,2),'Name', points_names,'IconScale',0.6,'Color','magenta');

% for CL_trajectory (trajectory or full set of estimated points)
kmlwriteline([Result_folder_name,filesep,team,filesep,'Competitor_trajectory_full.kml'], CL_full(:,3), CL_full(:,2), 'Color','magenta', 'LineWidth', 3);

% error lines (connecting Gt with Estimated points)
error_lines=nan*zeros(3*nsamples,2);
for i=1:nsamples
    % pto 1:
    error_lines((i-1)*3+1,1)=GT(i,3);  % latitude
    error_lines((i-1)*3+1,2)=GT(i,2);  % longitude
    %pto2:
    error_lines((i-1)*3+2,1)=CL(i,3);  % latitude
    error_lines((i-1)*3+2,2)=CL(i,2);  % longitude
    % NaN:
    error_lines((i-1)*3+3,1)=NaN;  % latitude
    error_lines((i-1)*3+3,2)=NaN;  % longitude
end
kmlwriteline([Result_folder_name,filesep,team,filesep,'Competitor_ErrorLines.kml'], error_lines(:,1), error_lines(:,2), 'Color','red', 'LineWidth', 3);

