function seq = LowPAPR_Sequence(u,v,alpha,m,delta)
%TS 38.211, Section 5.2.2.

N_RB_sc = 12;
Mzc = m * N_RB_sc / 2^delta;
mD = Mzc;
vD = v;
uD = u;
alphaD = alpha;

% Get the base sequence
nIndex = (0:mD-1)';
if(Mzc<30)
    Fai = getFai(u,Mzc);
    baseSeq = exp(1j.*Fai.*pi/4);
elseif(Mzc==30)
    baseSeq = exp(-1j*pi*(uD+1)*(nIndex+1).*(nIndex+2)/31);
else
    % Get the largest prime number less than the length of sequence
    n = primes(mD-1);
    nzc = n(end);
    qBar = nzc*(uD+1)/31;
    q = floor(qBar+0.5) + vD*(-1)^floor(2*qBar);
    % Get the index values modulo nzc
    mIndex = [0:nzc-1 0:mD-nzc-1]';
    % Get the cyclic extended Zadoff-Chu sequence
    baseSeq = exp(-1j*pi*q*mIndex.*(mIndex+1)/nzc);
end

% Get the low-PAPR sequence from the base sequence
seq = exp(1j*nIndex*alphaD).*repmat(baseSeq,1,length(alphaD));


% % if(Mzc>=3*N_RB_sc)
% %     for n = 0:Mzc-1
% %         m = mod(n,Nzc);
% %         qhat = Nzc*(u+1)/31;
% %         q = floor(qhat+1/2)+v*(-1)^(floor(2*qhat));
% %         r_uv_hat = exp(-1j*(pi*q*m*(m+1)/Nzc));
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % elseif(Mzc==6)
% %     for n = 0:Mzc1
% %         Fai = Table5_2_2_2_1(u+1,n+1);
% %         r_uv_hat = exp(1j*Fai)*pi/4;
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % elseif(Mzc==12)
% %     for n = 0:Mzc-1
% %         Fai = Table5_2_2_2_2(u+1,n+1);
% %         r_uv_hat = exp(1j*Fai)*pi/4;
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % elseif(Mzc==18)
% %     for n = 0:Mzc-1
% %         Fai = Table5_2_2_2_3(u+1,n+1);
% %         r_uv_hat = exp(1j*Fai)*pi/4;
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % elseif(Mzc==24)
% %     for n = 0:Mzc-1
% %         Fai = Table5_2_2_2_4(u+1,n+1);
% %         r_uv_hat = exp(1j*Fai)*pi/4;
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % elseif(Mzc==30)
% %     for n = 0:Mzc-1
% %         r_uv_hat = exp((-1j*pi*(u+1)*(n+1)*(n+2))/31);
% %         r_uv(n+1) = exp(1j*afa*n)*r_uv_hat;
% %     end
% % end
% % 
% % 
% % end
