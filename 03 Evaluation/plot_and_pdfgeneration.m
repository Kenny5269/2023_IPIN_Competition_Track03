function plot_and_pdfgeneration(results,teamname,outputfile,PDF_generation,nsamples,GT,CL,CL_full,North_GT,East_GT,North_CL,East_CL,North_CL_full,East_CL_full)

linewidth_IPIN = 1;
MarkerSize_IPIN = 15;
verboseMode   = 0; % 1 to show results on screen
prec_str='2'; % resolution in stats ('1': 1 decimeter,  '2': 1 centiometer) 
Result_folder_name = ['Results'];
team  = [teamname sprintf('%02d',outputfile)]; % string with "team + number of trial"

Horizontal_Position_Error =results(:,1);
Floor_Position_Error= results(:,2);  % 15 meters floor penalty
SampleError=Horizontal_Position_Error + Floor_Position_Error;
Errors(:,1) = Horizontal_Position_Error;
Errors(:,2) = Floor_Position_Error;

%---------75 percentile calculation---------------
ThirdQuartil = quantile(SampleError,0.75);
%ThirdQuartil2=SampleError_sort(ceil(0.75*nsamples));
%ThirdQuartil3= prctile(SampleError,75);

%% --------------------show error stats (in txt file & matlab prompt)------------------------
if verboseMode == 1
disp(['error stat (m)']);
disp(['median: ', num2str(median(SampleError),['%10.',prec_str,'f\n']), ' m']);
disp(['mean: ', num2str(mean(SampleError),['%10.',prec_str,'f\n']), ' m']);
disp(['rms: ', num2str(rms(SampleError),['%10.',prec_str,'f\n']), ' m']);
disp(['90th perc.: ', num2str(quantile(SampleError,0.90),['%10.',prec_str,'f\n']), ' m']);
disp(['75th perc.: ', num2str(ThirdQuartil,['%10.',prec_str,'f\n']), ' m']);
disp(['75th perc.(!!!): ', num2str(ThirdQuartil2,['%10.',prec_str,'f\n']), ' m']);
end

%result.txt
fid = fopen([Result_folder_name,filesep,team,filesep,'results_report.txt'],'w');
fprintf(fid,'error stat (m)\n');
fprintf(fid,['median: ', num2str(median(SampleError),['%10.',prec_str,'f\n']), ' m\n']);
fprintf(fid,['mean: ', num2str(mean(SampleError),['%10.',prec_str,'f\n']), ' m\n']);
fprintf(fid,['rms: ', num2str(rms(SampleError),['%10.',prec_str,'f\n']), ' m\n']);
fprintf(fid,['90th perc.: ', num2str(quantile(SampleError,0.90),['%10.',prec_str,'f\n']), ' m\n']);
fprintf(fid,['75th perc.: ', num2str(ThirdQuartil,['%10.',prec_str,'f\n']), ' m']);
fprintf(fid,'\n');
fprintf(fid,'\n');
fprintf(fid,'Errors per ref position\n');
fprintf(fid,'%02d\t%08.3f\n',[1:size(SampleError,1);SampleError']);
fclose(fid);


%% --------------------- create PDF-------------------------------
if (PDF_generation == 2)

    
    % ---------------------Plot errors---------------------------
    %........ plot Horizontal error..............:
    h4=figure;
    hh=sgtitle(sprintf("IPIN2023 Competition Track 3\n%s  Team (file#%i)", teamname,outputfile));  
    set(hh,'FontSize',23);
    subplot(3,2,6);
    plot(Horizontal_Position_Error, '-o', 'linewidth', linewidth_IPIN); hold on;
    plot(SampleError, '-x', 'linewidth', linewidth_IPIN);
    mytitle = ['Position Error'];
    legend({'Horizontal','Horiz. + floor penalty'});
    ylabel('Error (meters)'); xlabel('sample number');
    title(mytitle)
    
    %if (PDF_generation == 0) print(gcf,[Result_folder_name,filesep,team,filesep,mytitle],'-dmeta'); end
    % %......... plot Total position error...........:
    % if (PDF_generation == 1) subplot(3,2,5);else figure; end;
    % plot(SampleError, '-o', 'linewidth', linewidth_IPIN);
    % mytitle = ['Position Error (horizontal + floor penalty)'];
    % ylabel('Error (meters)'); xlabel('sample number');
    % title(mytitle)
    % if (PDF_generation == 0) print(gcf,[Result_folder_name,filesep,team,filesep,mytitle],'-dmeta'); end;
    
    % ..........plot floor error......:
    subplot(3,2,4);
    plot(GT(:,4),'.g','MarkerSize',MarkerSize_IPIN)
    hold on;
    plot(CL(:,4),'-b','linewidth', linewidth_IPIN);
    plot(abs(GT(:,4)-CL(:,4)),'.r','MarkerSize',MarkerSize_IPIN*0.5);
    mytitle = ['Floor evaluation'];
    xlabel('sample number'); ylabel('Floor number'); 
    % set tick numbers according to real number of foors:
      Tick_numbers=[min(GT(:,4)):1:max(GT(:,4))];  num_floors=size(Tick_numbers,2);
      Tick_names=cell(1,3); for i=1:num_floors, Tick_names{i}=num2str(Tick_numbers(i)); end
      set(gca,'YTick',Tick_numbers,'YTickLabel',Tick_names);
    legend('ground truth', 'competitor','floor error');title(mytitle)
  
    %......... plot graph2D Competitor vs GroundTruth (All floor together).......:
    % create number label:
    for i = 1:nsamples
        list_key_point_comp{i} = sprintf('%02d',i);
    end
    %plot:
    subplot(3,2,[3,5])
    plot(East_GT,North_GT,'.g','MarkerSize',MarkerSize_IPIN);
    hold on
    idx_North_good=find(abs(North_CL_full)<=25000);   % more than 25000 m probably is a wrong estimation
    idx_East_good=find(abs(East_CL_full)<=25000);
    plot(East_CL_full(idx_East_good),North_CL_full(idx_North_good),'-b','linewidth', linewidth_IPIN);
    axis equal
    
    %draw each local error
    for nbpts = 1:length(East_GT)
        l = line([East_CL(nbpts), East_GT(nbpts)],[North_CL(nbpts), North_GT(nbpts)]);
        set(l, 'Color', 'r','linewidth', linewidth_IPIN);
    end
    legend_text = {'ground truth', 'competitor', 'error'};
    legend(legend_text,'location','southeast');
    xlabel('Easting (m)');
    ylabel('Northing (m)');
    
    
    mytitle = ['2D position estimation vs GT (all floor together)'];
    title(mytitle);
     
    %............ plot CDF .....................
    subplot(3,2,2)
    mytitle = ['CDF- Position Error distribution'];
    title(mytitle);
    SampleError_sort = sort(SampleError);
    hold on%utile pourla trace de l'asymptote
    plot(SampleError_sort,(1:size(SampleError_sort,1))/size(SampleError_sort,1),'kx-','linewidth',linewidth_IPIN)%plot ou  semilog
    
    % hold on;
    % j=1;
    % for err=[0:0.001:max(SampleError_sort)]  % error posicion metros
    %         idx=find(SampleError_sort<err);
    %         porcentaje_error_pos(j,2)=length(idx)/nsamples;
    %         porcentaje_error_pos(j,1)=err;
    %         j=j+1;
    %     end
    % plot(porcentaje_error_pos(:,1),porcentaje_error_pos(:,2),'g.-');
    
    % ergonomie d'affichage
    ylabel('CDF (%)');
    %title(sprintf('receiver and antenna : %s',char(candidat_Type)));
    xlabel('3D (Horizontal+Floor) Position Error (m)');
    %axis([0 max_axe_metre 0 1.1]);
    grid on;
    set(gca,'YTick',[0.25 0.50 0.75 1.00])
    set(gca,'YTickLabel',{'25';'50';'75';'100'})
    % set(fig_CDF, 'Name', 'Errors distribution');
    % set(fig_CDF, 'NumberTitle', 'off');
    set(gcf, 'Name', 'Errors distribution');
    set(gcf, 'NumberTitle', 'off');
    

    
    %draw 75%
    l = line([0, max(SampleError)], [0.75, 0.75]);
    set(l, 'LineStyle', '--', 'Color', 'r', 'linewidth', linewidth_IPIN);
    %draw Score
    l = line([ThirdQuartil, ThirdQuartil], [0, 0.75]);
    set(l, 'Color', 'r', 'linewidth', linewidth_IPIN);
    
    % %draw Score2
    % l = line([ThirdQuartil2, ThirdQuartil2], [0, 0.75],'LineStyle','--');
    % set(l, 'Color', 'g', 'linewidth', linewidth_IPIN);
    % 
    % %draw Score3
    % l = line([ThirdQuartil3, ThirdQuartil3], [0, 0.75],'LineStyle',':');
    % set(l, 'Color', 'b', 'linewidth', linewidth_IPIN);
    
    %save
    if (PDF_generation == 0) print(gcf,[Result_folder_name,filesep,team,filesep,mytitle],'-dmeta'); end;


    %print result inside PDF as a legend
    subplot(3,2,1);
    plot(NaN);
    axis off
    text(0,0.90,['Error statistics:'],'FontSize',20)
    text(0,0.75,[' - Correct Floor detection: ', num2str(mean(Errors(:,2)==0)*100,['%10.',prec_str,'f\n']), ' %'],'FontSize',20)
    text(0,0.60,[' - Median: ', num2str(median(SampleError),['%10.',prec_str,'f\n']), ' m'],'FontSize',20)
    text(0,0.45,[' - Mean: ', num2str(mean(SampleError),['%10.',prec_str,'f\n']), ' m'],'FontSize',20)
    text(0,0.30,[' - RMS: ', num2str(rms(SampleError),['%10.',prec_str,'f\n']), ' m'],'FontSize',20)
    text(0,0.15,[' - 90th perc.: ', num2str(quantile(SampleError,0.90),['%10.',prec_str,'f\n']), ' m'],'FontSize',20)
    text(0,0.0,[' - 75th perc.: ', num2str(ThirdQuartil,['%10.',prec_str,'f\n']), ' m'],'FontSize',25)
    %print PDF
    set(h4,'PaperPositionMode','Auto')
    set(h4,'PaperSize',[29.7 42]); %set the paper size to A3
    print(h4,[Result_folder_name,filesep,team,filesep,'all-results'],'-fillpage','-dpdf') % then print it
    
    %........draw All Floor together in a single PDF..............
    h5=figure;
    plot(East_GT,North_GT,'.g','MarkerSize',MarkerSize_IPIN);
    text(East_GT,North_GT,list_key_point_comp,'FontSize',12);
    hold on
    idx_North_good=find(abs(North_CL_full)<=25000);   % more than 25000 m probably is a wrong estimation
    idx_East_good=find(abs(East_CL_full)<=25000);
    plot(East_CL_full(idx_East_good),North_CL_full(idx_North_good),'-b','linewidth', linewidth_IPIN);
    axis equal
    
    %draw each local error
    for nbpts = 1:length(East_GT)
        l = line([East_CL(nbpts), East_GT(nbpts)],[North_CL(nbpts), North_GT(nbpts)]);
        set(l, 'Color', 'm','linewidth', linewidth_IPIN);
    end
    legend_text = {'key points', 'competitor', 'error'};
    legend(legend_text,'location','southeast','FontSize',12);
    xlabel('Easting (m)');
    ylabel('Northing (m)');
    
    mytitle = ['2D position estimation vs GT (all floors together)',' - ',strrep(teamname,'_',' ')];
    title(mytitle);
    

    xlim([-5,137])
    ylim([-25,105])
    xticks(0:10:140);
    yticks(-20:10:100);

    set(gca,'fontsize', 9*120/120);

    %print PDF
    set(h5,'PaperPositionMode','Auto')
    set(h5,'PaperSize',[29.7 42]); %set the paper size to A3
    print(h5,[Result_folder_name,filesep,team,filesep,'all-floors'],'-fillpage','-dpdf') % then print it
    
    
     %.........draw different Floors in a separate PDFs.............
    idxs = intersect([([1;GT(2:end,4)-GT(1:(end-1),4)]~=0).*[1:size(GT,1)]';size(GT,1)+1],[1:(size(GT,1)+1)]);
    totalParts = (sum([GT(2:end,4)-GT(1:(end-1),4)]~=0)+1);
    string_pdf_comand="";
    for trackPart = 1:totalParts
        idxsGT      = idxs(trackPart):(idxs(trackPart+1)-1);        
        idxsCL2     = (CL_full(:,1) >= GT(idxsGT(1),1)) .* (CL_full(:,1) <= GT(idxsGT(end),1)) > 0;
        if trackPart < totalParts
            idxsGTnext  = (idxs(trackPart+1)-1):idxs(trackPart+1);
            idxsCL2next = (CL_full(:,1) >= GT(idxsGTnext(1),1)) .* (CL_full(:,1) <= GT(idxsGTnext(end),1)) > 0;
        end
        h5=figure;
        plot(East_GT(idxsGT),North_GT(idxsGT),'.g','MarkerSize',MarkerSize_IPIN);
        text(East_GT(idxsGT),North_GT(idxsGT),list_key_point_comp(idxsGT),'FontSize',8);
        hold on
        plot(East_CL_full(idxsCL2),North_CL_full(idxsCL2),'-b','linewidth', linewidth_IPIN);
        if trackPart < totalParts
        plot(East_CL_full(idxsCL2next),North_CL_full(idxsCL2next),':b','linewidth', linewidth_IPIN);
        end
        axis equal
        %draw each local error
        for nbpts = idxsGT
            l = line([East_CL(nbpts), East_GT(nbpts)],[North_CL(nbpts), North_GT(nbpts)]);
            set(l, 'Color', 'm','linewidth', linewidth_IPIN);
        end
        legend_text = {'ground truth', 'competitor', 'competitor with floor transition' , 'error'};
        warning('off','MATLAB:legend:IgnoringExtraEntries')
        legend(legend_text,'location','southeast');
        warning('on','MATLAB:legend:IgnoringExtraEntries')
        xlabel('Easting (m)');
        ylabel('Northing (m)');
        mytitle = ['2D position estimation vs GT (separate floors). Part ', sprintf('%02d',trackPart) ' at floor ',...
                   sprintf('%02d', GT(idxsGT(1),4))  ,'.  Team: ',teamname,' (file#',num2str(outputfile),')'];
        title(mytitle);
        %print PDF
        set(h5,'PaperPositionMode','Auto')
        set(h5,'PaperSize',[29.7 42]); %set the paper size to A3
        axis([-10 120 -10 75])
        print(h5,[Result_folder_name,filesep,team,filesep,'part_' sprintf('%02d', trackPart) ],'-fillpage','-dpdf') % then print it
        
        % prepare string command to join all pdfs. Try to create a command like this: append_pdfs('all_parts.pdf','part_01.pdf','part_02.pdf')
        string_pdf_comand=string_pdf_comand+"'"+Result_folder_name+"\"+team+"\"+"part_"+sprintf('%02d',trackPart)+".pdf'";
        if trackPart<totalParts,  string_pdf_comand=string_pdf_comand+","; end % add a comma if not last
           %print(h5,[Result_folder_name,filesep,team,filesep,'all'],'-fillpage','-append','-dpsc') % append all in ps format
    end
    % print individual pdfs together in one single pdf file
    if isfile([Result_folder_name+"\"+team+"\all_parts.pdf"])
        delete([Result_folder_name+"\"+team+"\all_parts.pdf"]);
    else
       0;%disp("No encontré para borrar all_parts.pdf. No pasa nada.") 
    end 
    string_pdf_comand_full="append_pdfs('"+Result_folder_name+"\"+team+"\"+"all_parts.pdf',"+string_pdf_comand+");";

end