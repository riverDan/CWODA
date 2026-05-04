function [x_uv,y_uv] = nrPRACH(waveconfig)
prach = waveconfig.PRACH;
[numCyclicShifts,rootSequence,cyclicShift,cyclicOffset] = getPreambleSeqParameters(prach);
x_u = zadoffChuSeq(rootSequence(1), prach.LRA);
C_v = cyclicShift;
x_uv = circshift(x_u, [-C_v 0]);
y_uv = fft(x_uv)/sqrt(prach.LRA);