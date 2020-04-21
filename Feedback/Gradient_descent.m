% t = 0:1/500:20-1/500;                     
% x = (sin(2*pi*10*t-80)).^2-0.5;
% x1 = exp(-(t-80)/80).*((sin(2*pi*10*t-80)).^2-0.5);
% y = fft(x);
% y1 = fft(x1);
% n = length(x);    
% fshift = (-n/2:n/2-1)*(500/n);
% yshift = fftshift(y);
% yshift1 = fftshift(y1);
% plot(fshift,abs(yshift))
c = 0.01;
T = 6*10^6;
wo = 2.87*10^9;
w = linspace(0.94*wo,1.06*wo,1000);
f = 1 - c*((T/2)^2)./((T/2)^2 + (w-wo).^2);
y = awgn(f,72,'measured');
plot(w,y);

%% Gradient descent
fr = [];
er = [];
fre = [];
for j = 2.858:0.001:2.882
    fnl = fun(j*10^9);
    fr = [fr fnl];
    fre = [fre j*10^9];
end
plot(fre,fr)
%plot(1:25,sl)
fun(2.86*10^9)

%%
function final_w = fun(wi);
c = 0.01;
T = 6*10^6;
wo = 2.87*10^9;
f =@(x) 1 - c*((T/2)^2)/((T/2)^2 + (x-wo)^2);
m_max = (f(1.0005*wi)-f(wi))/(0.0005*wi);
wc = wi-nthroot((0.5*c*(T^2)/m_max),3);
m_max = (f(1.0005*wc)-f(wc))/(0.0005*wc);
iterations = 5;
step = 0.001;
wi=wc;
for i = 1:iterations
    m = (f(1.0005*wi)-f(wi))/(0.0005*wi);
    wi=wi-m*step*10^9/abs(m_max);
end
error=abs(wi-wo)*100/wo;
final_w = wc;
end


