function waveconfig = prach_config()
waveconfig.PRACH.FrequencyRange = 'FR1';
waveconfig.PRACH.DuplexMode = 'TDD';
waveconfig.PRACH.ConfigurationIndex = 0; 
waveconfig.PRACH.SubcarrierSpacing = 1.25;
waveconfig.PRACH.SequenceIndex = 5; % Logical root sequence(RS)
waveconfig.PRACH.PreambleIndex = 5;
waveconfig.PRACH.RestrictedSet = 'UnrestrictedSet'; 
waveconfig.PRACH.ZeroCorrelationZone = 5; 
waveconfig.PRACH.RBOffset = 0; 
waveconfig.PRACH.FrequencyStart = 0;
waveconfig.PRACH.FrequencyIndex = 0; 
waveconfig.PRACH.TimeIndex = 2;
waveconfig.PRACH.ActivePRACHSlot = 0;
waveconfig.PRACH.NPRACHSlot = 0;
waveconfig.PRACH.Format = '0'; 
waveconfig.PRACH.LRA = 839;
waveconfig.PRACH.NumTimeOccasions = 1; 
waveconfig.PRACH.PRACHDuration = 6;
waveconfig.PRACH.SymbolLocation = 8;

%%
waveconfig.carrier.NCellID = 1;
waveconfig.carrier.SubcarrierSpacing = 30;
waveconfig.carrier.CyclicPrefix = 'normal';
waveconfig.carrier.Bandwidth = 100;
waveconfig.carrier.FR = 'FR1';
