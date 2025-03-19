function [ ] = checkResults( filename )

disp('-----');
disp(['Report for estimates provided in file ', filename]);
if ~exist(['trials' filesep filename '.est'],'file')
    disp('Trial does not exist');
	disp('Quitting');
    disp('-----');
    return;
end


try
    CL_FULL =  dlmread([ 'trials' filesep filename '.est'],',',0,0);   
catch % If problems opening the file, the trial is rejected
    disp('Problem reading first line')
    try
    CL_FULL =  dlmread(['trials' filesep filename '.est'],',',1,0);   
    catch % If problems opening the file, the trial is rejected
        disp('Problem reading second line')
            try
            CL_FULL =  dlmread(['trials'  filesep  filename '.est'],',',2,0);   
            catch % If problems opening the file, the trial is rejected
                disp('Problem reading third line')
                disp('Quitting')
                disp('-----');
                return
            end
    end
end

disp(sprintf('%2.6f,',CL_FULL(1,:)));
disp(sprintf('%2.6f,',CL_FULL(round((1+size(CL_FULL,1))/2),:)));
disp(sprintf('%2.6f,',CL_FULL(end,:)));
disp('-----');
end

