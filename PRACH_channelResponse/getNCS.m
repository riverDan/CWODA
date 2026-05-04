function NCS = getNCS(format,ZeroCorrelationZone,varname)
% getNCS gets the value of NumCyclicShifts (NCS) from TS 38.211 Tables
% 6.3.3.1-5, 6.3.3.1-6, and 6.3.3.1-7, given the PRACH preamble format, the
% value of restricted set, and ZeroCorrelationZone.
%
%    Note: This is an internal undocumented function and its API and/or
%    functionality may change in subsequent releases.
%
%    NCS = getNCS(FORMAT,ZEROCORRELATIONZONE,VARNAME) returns the value of
%    NCS corresponding to the given value of ZEROCORRELATIONZONE and
%    RestrictedSet in VARNAME. The table to use, amongst TS 38.211 Tables
%    6.3.3.1-5, 6.3.3.1-6, and 6.3.3.1-7, is decided based on the PRACH
%    preamble FORMAT.

%  Copyright 2019 The MathWorks, Inc.

%#codegen

    var = zeros(16,1);
    
    [table] = prach_table();
    
    switch format
        case {'0','1','2'} % Long preamble
            t63315 = table.NCSFormat012;
            if strcmpi(varname,'UnrestrictedSet')
                var = t63315.UnrestrictedSet;
            elseif strcmpi(varname,'RestrictedSetTypeA')
                var = t63315.RestrictedSetTypeA;
            elseif strcmpi(varname,'RestrictedSetTypeB')
                var = t63315.RestrictedSetTypeB;
            end
        case {'3'} % Long preamble
            t63316 = table.NCSFormat3;
            if strcmpi(varname,'UnrestrictedSet')
                var = t63316.UnrestrictedSet;
            elseif strcmpi(varname,'RestrictedSetTypeA')
                var = t63316.RestrictedSetTypeA;
            elseif strcmpi(varname,'RestrictedSetTypeB')
                var = t63316.RestrictedSetTypeB;
            end
        otherwise % Short preamble
            t63317 = table.NCSFormatABC;
            if strcmpi(varname,'UnrestrictedSet')
                var = t63317.UnrestrictedSet;
            elseif strcmpi(varname,'RestrictedSetTypeA')
                var = t63317.RestrictedSetTypeA;
            elseif strcmpi(varname,'RestrictedSetTypeB')
                var = t63317.RestrictedSetTypeB;
            end
    end
    
    NCS = var(ZeroCorrelationZone+1);
end