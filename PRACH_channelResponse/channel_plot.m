function channel_plot(data_freq, chanl_response_t, chanl_response_f, ifpicsave, P)

% ifpicsave = 0;

x = linspace(0, 839, 839);
% x1 = linspace(0, 30720, 30720);

activeSub = -(30720-1)/2 : (30720-1)/2;          % -419,...,419
scs_prach = 1.25e3;            % PRACH subcarrier spacing for Format 0
fActive = activeSub(:) * scs_prach/1e6;
%% P1
% load('chanl_response_nlos2.mat')
figure('Units', 'pixels', 'Position', [200 200 560 200]);
plot_mean_sigma_Hn(x, abs(((chanl_response_f)).'), [0 0.35 0.65], [0.35 0.65 0.85]);
if(ifpicsave)
    exportgraphics(gcf, strcat('figure/Hf',num2str(P),'.pdf'), 'ContentType', 'vector');
end

figure('Units', 'pixels', 'Position', [200 200 560 200]);

t_us = (0:98304-1).' / 122.88e6 * 1e6;
plot_mean_sigma_ht(t_us', abs(chanl_response_t.'), [1 0.2 0.05], [1 0.55 0.45], 1);
if(ifpicsave)
    exportgraphics(gcf, strcat('figure/Hn',num2str(P),'.pdf'), 'ContentType', 'vector');
end

% load('data_freq_los.mat')
figure('Units', 'pixels', 'Position', [200 200 560 200]);
plot_mean_sigma_spectrum(fActive', 20*log10(abs(((data_freq(1:1:end,:))).')), [0.00 0.45 0.20], [0.65 0.90 0.70]);
if(ifpicsave)
    exportgraphics(gcf, strcat('figure/Spectrum',num2str(P),'.pdf'), 'ContentType', 'vector');
end