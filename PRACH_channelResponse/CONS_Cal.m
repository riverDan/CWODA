function [chanl,hhat] = CONS_Cal(y_uv,data_iq,table1)


ideal_prach = y_uv;
% data_freq = fftshift(fft(data_iq,98302));
data_freq = (fft(data_iq.*table1.',98304));

data_freq_valid = data_freq(34465:34465+838);
% ideal_prach_valid = ideal_prach(44306:44306+838);

chanl = data_freq_valid./ideal_prach;


% Put estimated active-subcarrier channel back to full frequency grid
activeIdx = 34465:34465+838;
% Hfull = zeros(98304, 1);
% Hfull(activeIdx) = chanl;

% Interpolate inactive subcarriers for impulse response display
allIdx = (1:98304).';
Hinterp = interp1(activeIdx(:), chanl(:), allIdx, 'linear', 'extrap');

% Convert centered frequency response to impulse response
hhat = ifft(ifftshift(Hinterp), 98304);

hhat = hhat/max(abs(hhat));

