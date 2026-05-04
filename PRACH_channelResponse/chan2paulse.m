function hhat = chan2paulse(chanl)

% Put estimated active-subcarrier channel back to full frequency grid
activeIdx = 34465:34465+838;
% Hfull = zeros(98304, 1);
% Hfull(activeIdx) = chanl;

% Interpolate inactive subcarriers for impulse response display
allIdx = (1:98304).';
[M,N] = size(chanl);
hhat = zeros(98304,N);
for i = 1:M
    Hinterp = interp1(activeIdx(:), chanl(:,i), allIdx, 'linear', 'extrap');

    % Convert centered frequency response to impulse response
    hhat(:,i) = ifft(ifftshift(Hinterp), 98304);
end