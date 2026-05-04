function [table] = prach_table()
%% Define TS 38.211 Table 6.3.3.2-1
% Each column represents the following:
%   Column 1: Length of the preamble sequence
%   Column 2: PRACH subcarrier spacing
%   Column 3: PUSCH subcarrier spacing
%   Column 4: Allocation expressed in number of RBs for PUSCH (N_RB^RA)
%   Column 5: kbar
t63321 = {
    839  1.25   15   6    7;
    839  1.25   30   3    1;
    839  1.25   60   2  133;
    839     5   15  24   12;
    839     5   30  12   10;
    839     5   60   6    7;
    139    15   15  12    2;
    139    15   30   6    2;
    139    15   60   3    2;
    139    30   15  24    2;
    139    30   30  12    2;
    139    30   60   6    2;
    139    60   60  12    2;
    139    60  120   6    2;
    139   120   60  24    2;
    139   120  120  12    2;
    };
% table63321 = cell2table(t63321,'VariableNames',{'LRA','PRACHSubcarrierSpacing','PUSCHSubcarrierSpacing','NRBAllocation','kbar'});
% table63321.Properties.Description = 'TS 38.211 Table 6.3.3.2-1: Supported combinations of subcarrier spacing between PRACH and PUSCH';
%% Define TS 38.211 Table 6.3.3.2-2
% Each column represents the following:
%   Column 1: Configuration index
%   Column 2: Preamble format
%   Column 3: x
%   Column 4: y
%   Column 5: Subframe number
%   Column 6: Starting symbol
%   Column 7: Number of PRACH slots within a subframe
%   Column 8: Number of time-domain PRACH occasions within a PRACH slot (Nslot_t)
%   Column 9: PRACH duration (N_dur)

load('table\LongPreambleFormats.mat');
load('table\ShortPreambleFormats.mat');
load('table\NCSFormat012.mat');
load('table\NCSFormat3.mat');
load('table\NCSFormatABC.mat');
load('table\SupportedSCSCombinations.mat');
load('table\ConfigurationsFR1PairedSUL.mat');
load('table\ConfigurationsFR1Unpaired.mat');
load('table\ConfigurationsFR2.mat');
%Table 6.3.3.1-1
table63311 = cell2table(LongPreambleFormats,'VariableNames',{'Format','LRA','SubcarrierSpacing','N_u','N_CP','RestrictedSets'});
table.LongPreambleFormats = table63311;

%Table 6.3.3.1-2
table63312 = cell2table(ShortPreambleFormats,'VariableNames',{'Format','LRA','SubcarrierSpacing','N_u','N_CP','RestrictedSets'});
table.ShortPreambleFormats = table63312;

%Table 6.3.3.1-5
table63315 = cell2table(NCSFormat012,'VariableNames',{'ZeroCorrelationZone','UnrestrictedSet','RestrictedSetTypeA','RestrictedSetTypeB'});
table.NCSFormat012 = table63315;
%Table 6.3.3.1-6
table63316 = cell2table(NCSFormat3,'VariableNames',{'ZeroCorrelationZone','UnrestrictedSet','RestrictedSetTypeA','RestrictedSetTypeB'});
table.NCSFormat3 = table63316;
%Table 6.3.3.1-7
table63317 = cell2table(NCSFormatABC,'VariableNames',{'ZeroCorrelationZone','UnrestrictedSet','RestrictedSetTypeA','RestrictedSetTypeB'});
table.NCSFormatABC = table63317;
%Table 6.3.3.2-1
table63321 = cell2table(SupportedSCSCombinations,'VariableNames',{'LRA','PRACHSubcarrierSpacing','PUSCHSubcarrierSpacing','NRBAllocation','kbar'});
table.SupportedSCSCombinations = table63321;
%Table 6.3.3.2-2
table63322 = cell2table(ConfigurationsFR1PairedSUL,'VariableNames',{'ConfigurationIndex','PreambleFormat','x','y','SubframeNumber','StartingSymbol','PRACHSlotsPerSubframe','NumTimeOccasions','PRACHDuration'});
table.ConfigurationsFR1PairedSUL = table63322;
%Table 6.3.3.2-3
table63323 = cell2table(ConfigurationsFR1Unpaired,'VariableNames',{'ConfigurationIndex','PreambleFormat','x','y','SubframeNumber','StartingSymbol','PRACHSlotsPerSubframe','NumTimeOccasions','PRACHDuration'});
table.ConfigurationsFR1Unpaired = table63323;
%Table 6.3.3.2-4
table63324 = cell2table(ConfigurationsFR2,'VariableNames',{'ConfigurationIndex','PreambleFormat','x','y','SlotNumber','StartingSymbol','PRACHSlotsPer60kHzSlot','NumTimeOccasions','PRACHDuration'});
table.ConfigurationsFR2 = table63324;