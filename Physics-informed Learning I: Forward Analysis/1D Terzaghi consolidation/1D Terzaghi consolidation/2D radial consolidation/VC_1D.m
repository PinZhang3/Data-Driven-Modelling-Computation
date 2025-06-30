%% 1D_consolidation Terzaghi
clc;clear
format long
u0 = 50;     % pre-loading
ch = 25/365; % m2/day
cv = ch/2;% m2/day
T =  200;    % day
N_t = 51;
N_z = 51;
h = 5;
t = linspace(0,T,N_t); 
z = linspace(0,h,N_z);
for i=1:length(z)
    for j=1:length(t)
        sum_sub=0;        
        for n=0:5
            % subcharge            
            sub_term_sub(n+1)=4/(pi*(2*n+1))*sin((2*n+1)*pi*z(i)/(2*h))*...
                        exp(-((2*n+1)/2)^2*pi^2*cv*t(j)/(h^2));
            sum_sub=sum_sub+sub_term_sub(n+1);
        end
        u(i,j)=u0*sum_sub;       
    end
end
u(1,1)=0;
u(2:end,1)=u0;

figure(1)
subplot(1,2,1)
pcolor(t/T,z/h,u/u0); shading interp, axis tight, colormap(jet)
colorbar
caxis([0 1])
subplot(1,2,2)
N_snapshot = [1, 10, 25, 51];
set(gcf,'position',[50,50,2000,300])
for i=1:4
    plot(z/h, u(:, N_snapshot)/u0); hold on
end
save('../Data/VC_1D.mat','t','z','u')
    
