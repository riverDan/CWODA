function [N_RB,mu,N_SLOT,N_u,N_1ms,N_RA_CP,gap,sampleRate] = N_RB_Cal(BandWidth,FR,SCS,Deltaf_RA,LRA,format)
if(SCS==15)
    if(BandWidth==50)
        N_RB = 270;
    elseif(BandWidth==40)
        N_RB = 216;
    elseif(BandWidth==30)
        N_RB = 160;
    elseif(BandWidth==25)
        N_RB = 133;
    elseif(BandWidth==20)
        N_RB = 106;
    elseif(BandWidth==15)
        N_RB = 79;
    elseif(BandWidth==10)
        N_RB = 52;
    else
        N_RB = 25;
    end
    mu = 0;
    N_SLOT = 10;
    N_fft = 4096;
    sampleRate = 61.44e6;
elseif(SCS==30)
    if(BandWidth==100)
        N_RB = 273;
        N_fft = 4096;
    elseif(BandWidth==90)
        N_RB = 245;
        N_fft = 4096;
    elseif(BandWidth==80)
        N_RB = 217;
        N_fft = 4096;
    elseif(BandWidth==70)
        N_RB = 189;
        N_fft = 4096;
    elseif(BandWidth==60)
        N_RB = 162;
        N_fft = 2048;
    elseif(BandWidth==50)
        N_RB = 133;
        N_fft = 2048;
    elseif(BandWidth==40)
        N_RB = 106;
        N_fft = 2048;
    elseif(BandWidth==30)
        N_RB = 78;
    elseif(BandWidth==25)
        N_RB = 65;
    elseif(BandWidth==20)
        N_RB = 51;
    elseif(BandWidth==15)
        N_RB = 38;
    elseif(BandWidth==10)
        N_RB = 24;
    else
        N_RB = 11;
    end
    mu = 1;
    N_SLOT = 20;
    
    sampleRate = 122880000;
elseif(SCS==60)    
    if(FR=='FR1')
        if(BandWidth==10)
            N_RB = 11;
            sampleRate = 61440000;
        elseif(BandWidth==15)
            N_RB = 18;
            sampleRate = 61440000;
        elseif(BandWidth==20)
            N_RB = 24;
            sampleRate = 61440000;
        elseif(BandWidth==25)
            N_RB = 31;
            sampleRate = 61440000;
        elseif(BandWidth==30)
            N_RB = 38;
            sampleRate = 61440000;
        elseif(BandWidth==40)
            N_RB = 51;
            sampleRate = 61440000;
        elseif(BandWidth==50)
            N_RB = 65;
            sampleRate = 61440000;
        elseif(BandWidth==60)
            N_RB = 79;
            sampleRate = 122880000;
        elseif(BandWidth==70)
            N_RB = 93;
            sampleRate = 122880000;
        elseif(BandWidth==80)
            N_RB = 107;
            sampleRate = 122880000;
        elseif(BandWidth==90)
            N_RB = 121;
            sampleRate = 122880000;
        else
            N_RB = 135;
            sampleRate = 122880000;
        end
    else
        if(BandWidth==50)
            N_RB = 66;
            sampleRate = 122880000/2;
        elseif(BandWidth==100)
            N_RB = 132;
            sampleRate = 122880000;
        else
            N_RB = 264;
            sampleRate = 122880000*2;
        end
    end
    mu = 2;
    N_SLOT = 40;
elseif(SCS==120)
    if(BandWidth==50)
        N_RB = 32;
        N_fft = 512;
        sampleRate = 61440000;
    elseif(BandWidth==100)
        N_RB = 66;
        N_fft = 1024;
        sampleRate = 122880000;
    elseif(BandWidth==200)
        N_RB = 132;
        N_fft = 2048;
        sampleRate = 245760000;
    else
        N_RB = 264;
        N_fft = 4096;
        sampleRate = 491520000;
    end
    mu = 3;
    N_SLOT = 80;
end

N_fft = sampleRate/SCS/1000;

Tables = prach_table();
formatTable = Tables.LongPreambleFormats;
if(LRA==839)
    if(format=='0')
        k_pi = SCS*N_fft/Deltaf_RA/24576;
        N_RA_CP = k_pi*formatTable.N_CP(strcmpi(formatTable.Format,format));
        N_u = k_pi*formatTable.N_u(strcmpi(formatTable.Format,format));
        gap = SCS*N_fft - N_u - N_RA_CP;
    elseif(format=='1')
        k_pi = SCS*N_fft/Deltaf_RA/24576;
        N_RA_CP = k_pi*formatTable.N_CP(strcmpi(formatTable.Format,format));
        N_u = k_pi*formatTable.N_u(strcmpi(formatTable.Format,format))/2;
        gap = 3*SCS*N_fft - 2*N_u - N_RA_CP;
    elseif(format=='2')
        k_pi = SCS*N_fft/Deltaf_RA/24576;
        N_RA_CP = k_pi*formatTable.N_CP(strcmpi(formatTable.Format,format));
        N_u = k_pi*formatTable.N_u(strcmpi(formatTable.Format,format))/4;
        gap = 3.5*SCS*N_fft - 4*N_u - N_RA_CP;
    else
        k_pi = SCS*N_fft/Deltaf_RA/6144;
        N_RA_CP = k_pi*formatTable.N_CP(strcmpi(formatTable.Format,format));
        N_u = k_pi*formatTable.N_u(strcmpi(formatTable.Format,format))/4;
        gap = SCS*N_fft - 4*N_u - N_RA_CP;
    end
end
N_1ms = N_fft*SCS;