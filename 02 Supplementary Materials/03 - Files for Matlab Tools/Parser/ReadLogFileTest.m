function ReadLogFileTest()

linea = 'WIFI;0.651;65094.18;SSID2023_0001;20:23:00:00:00:01;2437;-60';
cell_array=textscan(linea,'%*s %f %f %*s %s %f %f','delimiter',';');
%MAC_str=cell_array{1,3}{1,1}; % MAC
%mac_clean = strrep(MAC_str, ':', '');
%mac_number = str2double(mac_clean);
%MAC_dec_array=sscanf(MAC_str,'%s:%s:%s:%s:%s:%s'); % quitar ":" y convertir a numero
%MAC_dec=MAC_dec_array(1)*256^5+MAC_dec_array(2)*256^4+MAC_dec_array(3)*256^3+MAC_dec_array(4)*256^2+MAC_dec_array(5)*256+MAC_dec_array(6);
%disp(cell_array{1});
disp(class(cell_array{1,3}{1,1}));