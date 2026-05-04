clear;close all;

fontName = 'Times New Roman';
FontSize = 15;
%%
waveconfig = prach_config();
Deltaf = waveconfig.carrier.SubcarrierSpacing;
Bandwidth = waveconfig.carrier.Bandwidth;
FR = waveconfig.carrier.FR;

Deltaf_RA = waveconfig.PRACH.SubcarrierSpacing;
LRA = waveconfig.PRACH.LRA;
format = waveconfig.PRACH.Format;
[N_RB,mu,N_SLOT,N_u,N_1ms,N_RA_CP,gap,~] = N_RB_Cal(Bandwidth,FR,Deltaf,Deltaf_RA,LRA,format);

prach = waveconfig.PRACH;
N_subframe_slot = 2^mu;

[~,y_uv] = nrPRACH(waveconfig);

Basebandsignalgenerate_request.Nfft = N_u;
Basebandsignalgenerate_request.N_RB = N_RB;
Basebandsignalgenerate_request.Deltaf = Deltaf;
Basebandsignalgenerate_request.Deltaf_RA = Deltaf_RA;
Basebandsignalgenerate_request.LRA  = waveconfig.PRACH.LRA;
table = Basebandsignalgenerate(Basebandsignalgenerate_request);

%%

load('./dataCapture/prach_ota_P1.mat');%46084
% load('./dataCapture/prach_ota_P2.mat');%23445
% load('./dataCapture/prach_ota_P3.mat');%35970

data_rx = DataCapture;
figure(2);plot((abs(((data_rx)))));axis tight
figure(3);plot((abs(fftshift(fft(data_rx)))));axis tight
%%

offset = 46084;
% offset = 23445;
% offset = 35970;

data_rx_sync = data_rx(offset+1:offset+122880);
figure(4);plot((abs(((data_rx_sync)))));axis tight
figure(5);plot((abs(fftshift(fft(data_rx_sync)))));axis tight
%%

wave1ms_rx_removeCP = data_rx_sync(N_RA_CP+1:N_RA_CP+98304);
% 

data_freq = zeros(30720,9);
chanl_response_f = zeros(839,9);
chanl_response_t = zeros(98304,9);
for i = 0:8
    data_rx_1slot = data_rx(offset+1+i*122880:offset+122880+i*122880);

    data_rx_resample = resample(data_rx_1slot,1,4);
    data_freq(:,i+1) = fftshift(fft(data_rx_resample));

    wave1ms_rx_removeCP = data_rx_1slot(N_RA_CP+1:N_RA_CP+98304);
    [chanl_response_f(:,i+1),chanl_response_t(:,i+1)] = CONS_Cal(y_uv, wave1ms_rx_removeCP, table);
end

%% 
ifpicsave = 0;
path = 3;
channel_plot(data_freq, chanl_response_t, chanl_response_f, ifpicsave, path);

