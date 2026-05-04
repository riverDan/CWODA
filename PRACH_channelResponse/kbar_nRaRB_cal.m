function [kbar,N_RA_RB] = kbar_nRaRB_cal(SupportedSCSCombinations,Deltaf,Deltaf_RA,LRA)
ll1 = SupportedSCSCombinations.LRA==LRA;
ll2 = SupportedSCSCombinations.PRACHSubcarrierSpacing==Deltaf_RA;
ll3 = SupportedSCSCombinations.PUSCHSubcarrierSpacing==Deltaf;

index = intersect(intersect(find(ll1==1),find(ll2==1)),intersect(find(ll1==1),find(ll3==1)));

kbar = SupportedSCSCombinations.kbar(index);
N_RA_RB = SupportedSCSCombinations.NRBAllocation(index);
