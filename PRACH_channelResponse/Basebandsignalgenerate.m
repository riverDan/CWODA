function [table] = Basebandsignalgenerate(request)

N_RB_sc = 12;
Nfft = request.Nfft;
N_RB = request.N_RB;
% prachGrid = request.prachGrid;
Deltaf = request.Deltaf;
Deltaf_RA = request.Deltaf_RA;
LRA = request.LRA;

Tables = prach_table();
SupportedSCSCombinations = Tables.SupportedSCSCombinations;


% table = zeros(1,Nfft);
table = zeros(1,Nfft);
k_mu_0 = 0;
nRA = 0;

[kbar,N_RA_RB] = kbar_nRaRB_cal(SupportedSCSCombinations,Deltaf,Deltaf_RA,LRA);

K = Deltaf/Deltaf_RA;
k1 = k_mu_0 + nRA*N_RB_sc*N_RA_RB - N_RB*N_RB_sc/2;
% kbar = 2;
for n = 1:Nfft

    table(n) = exp(-1i * 2 * pi /Nfft*((K*k1+kbar)* (n-1)));
end

