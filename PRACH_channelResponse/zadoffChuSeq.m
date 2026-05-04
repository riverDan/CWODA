function seq = zadoffChuSeq(u,LRA)
%ZADOFFCHUSEQ Generate root Zadoff-Chu sequence of complex symbols
%
%   SEQ = ZADOFFCHUSEQ(R,N) generates the Rth root Zadoff-Chu sequence of
%   length N. The output SEQ is an N-length column vector of complex
%   symbols.
%
%   Example:
%   % Generate the 25th root length-63 Zadoff-Chu sequence.
%   seq = zadoffChuSeq(25,63);
%
%   See also comm.PNSequence.

%   Copyright 2018 The MathWorks, Inc.

%#codegen

% Check number of inputs
narginchk(2,2);

% Check inputs
fcnName = 'zadoffChuSeq';
%   R: scalar, positive, finite integer
validateattributes(u,{'numeric'}, ...
    {'scalar','finite','positive','integer'},fcnName,'Root R');
%   N: scalar, positive, finite, odd, integer
validateattributes(LRA,{'numeric'}, ...
    {'scalar','finite','positive','integer','odd'},fcnName,'Sequence length N');

% Check for relative primeness between N and R
coder.internal.errorIf(gcd(LRA,u)~=1, ...
    'comm:zadoffChuSeq:NotRelativelyPrime',u,LRA);

i = (0:LRA-1).';
seq = exp( -1i * pi * u * i.*(i+1) / LRA );

% [EOF]
