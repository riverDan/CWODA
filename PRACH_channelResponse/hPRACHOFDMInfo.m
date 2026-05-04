function info = hPRACHOFDMInfo(waveconfig)
    carrier = waveconfig.carrier;
    prach = waveconfig.PRACH;

    internalinfo = OFDMInfo(carrier,prach);
    
    info.SamplingRate = internalinfo.SamplingRate;
    info.Nfft = internalinfo.Nfft;
    info.Windowing = 0;
    info.CyclicPrefixLengths = internalinfo.CyclicPrefixLengths;
    info.GuardLengths = internalinfo.GuardLengths;
    info.SymbolLengths = internalinfo.CyclicPrefixLengths + internalinfo.Nfft + internalinfo.GuardLengths;
    info.OffsetLength = internalinfo.OffsetLength;
    info.NSubcarriers = internalinfo.K;
    info.SubcarrierSpacing = prach.SubcarrierSpacing;
    info.TotSubframes = internalinfo.TotSubframes;
    info.PRACHSlotsPerPeriod = internalinfo.PRACHSlotsPerPeriod;
    info.l_start = internalinfo.l_start;
    info.N_u = internalinfo.N_u;
end

function info = OFDMInfo(carrier,prach)
    
    % Sampling rate corresponding to 1/T_s where T_s is defined in TS
    % 38.211 Section 4.1. This will be adopted as the nominal sampling rate
    % for calculations here
    sr_nominal = 61.44e6;

    
    % Nominal IDFT size (i.e. in terms of the sampling rate above). This
    % corresponds to parts of the expressions for N_u in TS 38.211 Tables
    % 6.3.3.1-1 and 6.3.3.1-2:
    % For formats 0, 1 and 2: 24576
    % For format 3: 6144
    % For short formats: 2048 * 2^-mu
%     n_idft_nominal = 2048 * 15 / prach.SubcarrierSpacing;
      n_idft_nominal = 2048;
    
    % Get PRACH slot grid size and record the number of subcarriers K and
    % the number of OFDM symbols L
%     siz = size(nrPRACHGrid(carrier,prach));
%     K = siz(1);
%     L = siz(2);
    K = 52*12;
    L = 14;
    
    % Calculate OFDM information for TS 38.211 Section 5.3.1 "OFDM
    % basedband signal generation for all channels except PRACH", for the
    % numerology described in TS 38.211 Section 5.3.2
    LRA = prach.LRA;
    if (LRA==839)
        % For long sequences (LRA=839) numerology mu=0 is assumed
        deltaf_RA = 15;
    else % LRA==139
        % For short sequences (LRA=139) the numerology given by the PRACH
        % subcarrier spacing is used
        deltaf_RA = prach.SubcarrierSpacing;
    end
    uemu.NRB = K/12;
    uemu.SubcarrierSpacing = deltaf_RA;
    uemu.CyclicPrefix = 'Normal';
% %     muinfo = hOFDMInfo(uemu);
    
    % Calculate OFDM symbol start samples n_mu_start_l. These are the
    % sample indices corresponding to t_mu_start_l described in TS 38.211
    % Section 5.3.2, calculated at the nominal sampling rate. N_mu_u and
    % N_mu_CP are also defined in Section 5.3.2. n_mu_tot_l is the number
    % of samples to the end of OFDM symbol 'l' and is used subsequently for
    % calculating guard lengths
    K_pi = 2;
    mu = log2(prach.SubcarrierSpacing/15);
    N_mu_u = 2048 * K_pi * 15 / prach.SubcarrierSpacing;
%     mu = 0;
%     N_mu_u = 2048 * K_pi;
    N_mu_CP = N_cp_l_cal(mu,K_pi);

    nSlot = 0;
    start_instant = sum(N_mu_CP(1:nSlot*14) + N_mu_u);
    N_mu_CP = N_mu_CP(nSlot*14 + (1:14));
    n_mu_tot_l = cumsum(N_mu_CP + N_mu_u);
    n_mu_start_l = [0 n_mu_tot_l(1:end-1)];

    % In the case of format C0, each preamble has one active sequence
    % period (see Table 6.3.3.1-2) but including the guard and the cyclic
    % prefix, the preamble spans two OFDM symbols. For this reason, the
    % slot grid related to format C0 has 7 OFDM symbols, rather than 14.
    % Therefore, the symbol indices 'l_starts' and 'l_ends' that address
    % the PRACH slot grid must be adjusted by the function 'l_mu_fn' here
    % to correctly address the OFDM information for numerology mu (for
    % other formats, no adjustment is required)
    prachFormat = prach.Format;
    if (strcmpi(prachFormat,'C0'))
        l_mu_fn = @(x)(x*2 + 1);
    else
        l_mu_fn = @(x)x;
    end
    
    % Get starting OFDM symbol l_0 of the first PRACH time occasion,
    % defined in TS 38.211 Section 5.3.2
    
    Tables = prach_table();
    switch upper(prach.FrequencyRange)
        case 'FR1'
            if strcmpi(prach.DuplexMode,'TDD') % TS 38.211 Table 6.3.3.2-3
                configTable = Tables.ConfigurationsFR1Unpaired;
            else % TS 38.211 Table 6.3.3.2-2
                configTable = Tables.ConfigurationsFR1PairedSUL;
            end
        otherwise % TS 38.211 Table 6.3.3.2-4
            configTable = Tables.ConfigurationsFR2;
    end
    l_0 = configTable{prach.ConfigurationIndex+1,6};
%     SupportedSCSCombinations = Tables.SupportedSCSCombinations;
%     kbar = SupportedSCSCombinations;
    
    % For long sequences (LRA=839) and non-zero l_0, the starting position
    % is implemented as an initial guard length N_offset
    straddleSF = false;
    if (LRA==839 && l_0 > 0)
        N_offset = n_mu_tot_l(l_0);
        l_0 = 0;
        % For preamble formats 0 and 3, record that an active PRACH
        % preamble straddles an extra subframe due to the non-zero l_0
        if (any(strcmpi(prachFormat,{'0','3'})))
            straddleSF = true;
        end
    else
        N_offset = 0;
    end
    
    % Get the total number of PRACH time occasions n_RA_t, defined in TS
    % 38.211 Section 5.3.2
    n_RA_t = prach.NumTimeOccasions;
    
    % Get the duration N_RA_dur in OFDM symbols of one PRACH time occasion,
    % defined in TS 38.211 Section 5.3.2
%     N_RA_dur = prach.PRACHDuration;
    N_RA_dur = configTable{prach.ConfigurationIndex+1,9};
   
    if (strcmpi(prachFormat,'C0'))
        l_0 = l_0 / 2;
    end
  
    l_starts = l_0 + ((0:(n_RA_t-1)) * N_RA_dur);
      
    % Get the cyclic prefix length N_RA_CP and useful OFDM symbol period
    % N_u from TS 38.211 Table 6.3.3.1-1 or 6.3.3.1-2
    if (LRA==839)
        formatTable = Tables.LongPreambleFormats;
    else % LRA==139
        formatTable = Tables.ShortPreambleFormats;
    end
    
    N_RA_CP = K_pi*formatTable.N_CP(strcmpi(formatTable.Format,prachFormat))*2^(-mu);
    N_u = K_pi*formatTable.N_u(strcmpi(formatTable.Format,prachFormat))*2^(-mu);
    
    % For short sequences (LRA=139), adjust cyclic prefix lengths and
    % useful lengths to account for numerology (2^-mu term in TS 38.211
    % Table 6.3.3.1-2)
    if (LRA==139)
        N_RA_CP = N_RA_CP * n_idft_nominal / 2048;
        N_u = N_u * n_idft_nominal / 2048;
    end
    
    % Calculate cyclic prefix lengths N_RA_CP_l as described in TS 38.211
    % Section 5.3.2, at nominal sampling rate. Note that the vector
    % N_RA_CP_l only has non-zero values in the positions where 'l'
    % corresponds to the first OFDM symbol of a PRACH time occasion. Note
    % that format C2 has a cyclic prefix length equal to the nominal IDFT
    % size, which is implemented in nrPRACH and nrPRACHIndices as an extra
    % initial repetition of nominal sequence, so the cyclic prefix length
    % is zero here. The variable 'start_instant' takes into account the
    % position of the PRACH slot within the carrier subframe
    N_RA_CP_l = zeros([1 L]);
    N_GP_l = zeros([1 L]);
    
    sempty = l_starts*N_mu_u + sum(N_mu_CP(1:l_starts));
    gap = N_mu_u*15-sempty - N_RA_CP - N_u;
    
    
    % Calculate the number of samples per subframe
    samplesPerSubframe = (sr_nominal * 1e-3);
    
    SlotsPerSubframe = 2^mu;
    % Calculate total number of subframes spanned by a nominal PRACH slot
    if (LRA==139)
        totSubframes = 1 / SlotsPerSubframe;
    else
        totSubframes = (N_offset + sum(N_RA_CP_l + n_idft_nominal + N_GP_l)) / samplesPerSubframe;
        totSubframes = totSubframes - straddleSF;
    end
    
    % Calculate number of PRACH slots in the overall period that spans an
    % integer multiple of 'x' frames
    x = configTable.x(prach.ConfigurationIndex+1);
    nPRACHSlots = lcm(x*10,max([1 totSubframes])) / totSubframes;
    
    % Calculate carrier OFDM information
    ue.NRB = carrier.NSizeGrid;
    ue.SubcarrierSpacing = carrier.SubcarrierSpacing;
    ue.CyclicPrefix = carrier.CyclicPrefix;

    % Determine ratio 'R' between carrier sampling rate and nominal
    % sampling rate
    SamplingRate = carrier.SamplingRate;
%     R = SamplingRate / sr_nominal;
    R = 30/carrier.SubcarrierSpacing;
    
    info.K = K;
    info.Nfft = n_idft_nominal * R;
    info.SamplingRate = SamplingRate;
    N_RA_CP_l = N_mu_CP;
    N_RA_CP_l(l_starts+1) = N_RA_CP;
    N_RA_CP_l(l_starts+2:l_starts+N_RA_dur) = 0;
    
    N_GP_l(l_starts+N_RA_dur) = gap;

    info.CyclicPrefixLengths = N_RA_CP_l;
    info.GuardLengths = N_GP_l;
    info.OffsetLength = N_offset;
    info.TotSubframes = totSubframes;
    info.PRACHSlotsPerPeriod = nPRACHSlots;
    info.l_start = l_starts;
    info.N_u = N_u;
%     info.
    
end

% For long sequences (LRA=839), establish the adjustment to the number of
% subframes 'aSF' required to account for cases where the starting subframe
% of the PRACH preamble occurs partway through the nominal PRACH slot
% period. The nominal periods are defined here as intervals of
% 'totSubframes' duration, starting at subframe 0 of frame n_SFN mod x = 0
% where 'x' is given by the PRACH configuration table. Note that 'aSF' can
% be negative, indicating by how many subframes an inactive PRACH slot
% should be shortened. Example 2 for hPRACHOFDMInfo above shows the OFDM
% information for NPRACHSlot = 0, 1 and 2. The corresponding values of 
% 'aSF' produced by getSubframeAdjustment are 1, -3 and 2. 
function aSF = getSubframeAdjustment(prach,configTable,totSubframes,straddleSF)

    % Determine the set of subframe numbers where PRACH preambles start
    % within the period of 'x' frames
    y = configTable.y{prach.ConfigurationIndex+1};
    x = configTable.x(prach.ConfigurationIndex+1);
    startSFs = configTable.SubframeNumber{(prach.ConfigurationIndex+1)};
    startSFs = startSFs + (y * 10);
    startSFs = startSFs(:) + (x * 10 * (0:(totSubframes-1)));
    startSFs = startSFs(:).';
    
    % Determine the overall periodicity and starting subframes of the
    % nominal PRACH slots
    nPRACHSlots = lcm(x*10,totSubframes) / totSubframes;
    nominalSFs = [0 cumsum(repmat(totSubframes,1,nPRACHSlots-1))];
    
    % Establish which nominal PRACH slots are active, and their 
    % corresponding PRACH slot indices
    activeSlotFn = @(n)any(startSFs>=n & startSFs<=(n+totSubframes-1));
    activePRACHSlot = arrayfun(activeSlotFn,nominalSFs);
    activeIndex = find(activePRACHSlot);
    
    % Adjust the number of subframes in the PRACH slots:
    % (a) where the starting subframe of the PRACH preamble occurs 
    %     partway through the nominal PRACH slot period, and therefore the 
    %     PRACH slot needs lengthened to fully span the preamble
    % (b) where the lengthening of PRACH slots according to (a) above
    %     results in the PRACH slot timeline running behind the nominal
    %     timeline, such that a PRACH preamble cannot start "on time". In 
    %     this case, an earlier inactive PRACH slot needs to be shortened
    slotSFs = nominalSFs;
    aSF = zeros(size(slotSFs));
    % For each active PRACH slot index
    for i = 1:numel(activeIndex)
        % (a) Calculate extra slots required
        idxa = activeIndex(i);
        delta = startSFs(i) - slotSFs(idxa);
        if (delta < 0)
            % (b) If the number of extra slots is negative, adjust the
            % slot after the previous active PRACH slot to be empty
            idxb = activeIndex(i-1) + 1;
            [slotSFs,aSF] = adjust(slotSFs,aSF,idxb,-totSubframes);
            % (a) Recalculate extra slots after step (b) above
            delta = startSFs(i) - slotSFs(idxa);
        end 
        % (a) Adjust the current active PRACH slot by adding extra slots
        [slotSFs,aSF] = adjust(slotSFs,aSF,idxa,delta);
    end
    
    % Adjust additional PRACH slots to ensure that the total number of 
    % subframes matches the nominal period
    if (straddleSF)
        % If the PRACH preamble straddles an extra subframe due to a
        % non-zero starting symbol, remove the extra subframe from inactive
        % PRACH slots and shorten the slot following active PRACH slots by
        % one subframe
        aSF(~activePRACHSlot) = -1;
        aSF(activeIndex + 1) = aSF(activeIndex + 1) - 1;
    else
        % Establish the number of extra subframes 'eSF' by which the 
        % combined duration of all PRACH slots exceeds the nominal period
        last = slotSFs(end) + aSF(end) + totSubframes - 1;
        nominal_last = nPRACHSlots*totSubframes - 1;
        eSF = last - nominal_last;
        % If extra subframes are present
        if (eSF > 0)
            % If the last active PRACH slot is not the last slot of the
            % nominal period
            if(activeIndex(end) < numel(aSF))
                % Shorten the PRACH slot following the last active PRACH
                % slot by 'eSF'
                idx = activeIndex(end) + 1;
                aSF(idx) =  aSF(idx) - eSF;
            else % last active PRACH slot is the last slot
                % If the current PRACH slot is in the second or subsequent
                % nominal period
                if (prach.NPRACHSlot >= nPRACHSlots)
                    % shorten the first PRACH slot by 'eSF'
                    aSF(1) =  aSF(1) - eSF;
                else % current PRACH slot is in the first nominal period
                    % The final active PRACH slot of the first nominal
                    % period will have 'eSF' extra subframes beyond the end
                    % of the nominal period. This is unavoidable, because
                    % they contain part of an active PRACH preamble. The
                    % duration of the first period is therefore the nominal
                    % period plus 'eSF'. For the second period and
                    % subsequent periods, the first slot is shortened by
                    % 'eSF' above to make space for these extra subframes,
                    % so the second period (and subsequent periods) will be
                    % of nominal length
                end
            end
        end
    end

    % Select the element of the 'aSF' vector that corresponds to the 
    % current PRACH slot
    aSF = aSF(mod(prach.NPRACHSlot,nPRACHSlots) + 1);

end

% Adjust the PRACH slot timeline 'slotSFs' by adding 'extra' slots in
% positions idx+1 and beyond, and record the number of extra subframes in
% 'eSF'
function [slotSFs,eSF] = adjust(slotSFs,eSF,idx,extra)
    eSF(idx) = extra;
    slotSFs(idx+1:end) = slotSFs(idx+1:end) + eSF(idx);
end
