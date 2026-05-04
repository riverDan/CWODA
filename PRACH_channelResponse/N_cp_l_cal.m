function N_cp_l = N_cp_l_cal(mu,K_pi)
N_subframe_slot = 2^mu;
N_slot_symb = 14;
N_frame_subframe = 10;
N_cp_l = zeros(1,N_frame_subframe*N_slot_symb*N_subframe_slot);
for f = 0:N_frame_subframe-1
    for l = 0:N_slot_symb*N_subframe_slot-1
        if(l==0||l==7*2^mu)
            N_cp_l(f*(N_slot_symb*N_subframe_slot)+l+1) = 144*2^(-mu)*K_pi+16*K_pi;
        else
            N_cp_l(f*(N_slot_symb*N_subframe_slot)+l+1) = 144*2^(-mu)*K_pi;
        end
    end
end