function [ distance ] = distance_harvesine( p1, p2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

R = 6371;   % Earth's radius in km 

%P1 in LON, LAT
%P2 in LON, LAT

% medir distancia
delta_lat =  deg2rad(p2(:,2)) -  deg2rad(p1(:,2));        % differences in latitude
delta_lon =  deg2rad(p2(:,1)) -  deg2rad(p1(:,1));        % differences in longitude
a = sin(delta_lat/2).^2 + cos( deg2rad(p1(:,2))) .* cos( deg2rad(p2(:,2))) .* ...
        sin(delta_lon/2).^2;
c = 2 * atan2(sqrt(a), sqrt(1-a));
distance = R * c;  

end

