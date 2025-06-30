close all
clear
clc
N_t = 51;
N_r = 51;

u0 = 50;   %
T =  200;    % day
re = 2.5; % m
rw = 0.2; % m
rs = 0.2; % m
h = 10;   % m
% kv=0.007/365; % m/day
% kh=0.0021/365;% m/day
% kw=1E6/365;   % m/day
% ks=kh/3/365;  % m/day
ch = 25/365; % m2/day
% cv = ch/3;% m2/day
t = linspace(0,T,N_t); 
r = linspace(rw,re,N_r);
n = re/rw;         % re/rw
Fn = n^2/(n^2-1)*log(n)-(3*n^2-1)/(4*n^2);
for i=1:length(r)
    for j=1:length(t)
        Th = ch*t(j)/(4*re^2);
        lambda = -8*Th/Fn;
        u_e = u0*exp(lambda);        
        u(i,j) = 4*u_e/(4*re^2*Fn)*(re^2*log(r(i)/rw)-(r(i)^2-rw^2)/2);     
    end
end
u(1,1) = 0;
u(2:end,1) = u0;
figure(1)
subplot(1,2,1)
pcolor(t/T,r/re,u/u0); shading interp, axis tight, colormap(jet)
colorbar
caxis([0 1])

subplot(1,2,2)
N_snapshot = [1, 10, 25, 51];
set(gcf,'position',[50,50,2000,300])
for i=1:4       
    plot(r/re, u(:, N_snapshot)/u0); hold on
end
save('../Data/RC_1D.mat','t','r','u')

% set(gca, 'YDir','reverse')
