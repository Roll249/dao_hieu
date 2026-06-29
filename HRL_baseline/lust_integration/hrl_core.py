"""
hrl_core.py
===========
Thuật toán HRL baseline được TRÍCH NGUYÊN VĂN từ HRL_baseline/Main_Simulation.ipynb.

KHÔNG sửa đổi bất kỳ logic/công thức/cấu trúc mạng nào của thuật toán HRL.
Chỉ gom các cell định nghĩa hàm + lớp + không gian trạng thái/hành động + tham số
mạng (network constants) vào một module để driver tái sử dụng.

Các cell được trích: 0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,19,20,21
và phần "network constants" của cell 9 (trước dòng V=200).

Điểm DUY NHẤT khác so với baseline: tốc độ xe (VN_spd) và số xe (V) sẽ do
driver nạp từ dữ liệu LuST, thay cho phân phối truncnorm tổng hợp. Việc này nằm
hoàn toàn ở khâu "đọc dữ liệu / môi trường", không nằm trong thuật toán.
"""
# (Trích từ notebook gốc, giữ nguyên 100%)


# ======================================================================
# ==== Notebook CODE CELL 0 (verbatim) ====
# ======================================================================

import numpy as np
import math as math
import pandas as pd
from numpy import random
from random import seed
from random import randint
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io  
from scipy.stats import expon
import random
try:
    from IPython.display import clear_output
except Exception:  # IPython không bắt buộc khi chạy script
    def clear_output(*args, **kwargs):
        pass
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import sys
import itertools 
from collections import deque

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from scipy.stats import truncnorm


# ======================================================================
# ==== Notebook CODE CELL 1 (verbatim) ====
# ======================================================================

def Loc_Fun(RSU_T,UAV_T,HAP_T,LEO_T,RSU_r,UAV_r,HAP_r,LEO_r,LEO_alt,HAP_alt,UAV_alt): 

    
    while True:
        RSU_loc=np.zeros((2, RSU_T))
        UAV_loc=np.zeros((3, UAV_T))
        HAP_loc=np.zeros((3, HAP_T))
        LEO_loc=np.zeros((3, LEO_T))
        
        R_UAV_asign= np.zeros((RSU_T,UAV_T))
        R_HAP_asign= np.zeros((RSU_T,HAP_T))
        R_LEO_asign= np.zeros((RSU_T,LEO_T))
        U_HAP_asign= np.zeros((UAV_T,HAP_T))
        U_LEO_asign= np.zeros((UAV_T,LEO_T))
        H_LEO_asign= np.zeros((HAP_T,LEO_T))
        
        RSU_loc[0][0]=RSU_r/2
        RSU_loc[1][0]=5
        for i in range(RSU_T-1):
            i=i+1
            rn=np.random.randint(int(RSU_r/2), 1.5*RSU_r, size=(1,1))


            RSU_loc[0][i]=RSU_loc[0][i-1]+rn
            RSU_loc[1][i]=5 


        UAV_loc[0][0]=UAV_r/2
        UAV_loc[2][0]=UAV_alt 
        for i in range(UAV_T-1):
            i=i+1
            rn=np.random.randint(int(UAV_r), 2*UAV_r, size=(1,1))
            UAV_loc[0][i]=UAV_loc[0][i-1]+rn
            UAV_loc[1][i]=0
            UAV_loc[2][i]=UAV_alt 


        HAP_loc[0][0]=HAP_r/2
        HAP_loc[2][0]=HAP_alt 
        for i in range(HAP_T-1):
            i=i+1
            rn=np.random.randint(int(HAP_r), 2*HAP_r, size=(1,1))
            HAP_loc[0][i]=HAP_loc[0][i-1]+rn
            HAP_loc[1][i]=0
            HAP_loc[2][i]=HAP_alt 
            
        LEO_loc[0][0]=LEO_r
        LEO_loc[2][0]=LEO_alt 
        for i in range(LEO_T-1):
            i=i+1
            LEO_loc[0][i]=LEO_r
            LEO_loc[1][i]=0
            LEO_loc[2][i]=LEO_alt 
#%%-----------------------------Loc Para End-------------------------------------------------------------------------%%

        for r in range(RSU_T):
            for l in range(UAV_T):
                if ((((UAV_loc[1][l] - RSU_loc[1][r])**2) + ((UAV_loc[0][l]-RSU_loc[0][r])**2)) <= (UAV_r)**2):
                    R_UAV_asign[r][l] = 1

        for r in range(RSU_T):
            for h in range(HAP_T):
                if ((((HAP_loc[1][h]- RSU_loc[1][r] )**2) + ((HAP_loc[0][h]-RSU_loc[0][r])**2)) <= (HAP_r)**2):
                    R_HAP_asign[r][h] = 1
                 
        for r in range(RSU_T):
            for s in range(LEO_T):
                if ((((LEO_loc[1][s]- RSU_loc[1][r] )**2) + ((LEO_loc[0][s]-RSU_loc[0][r])**2)) <= (LEO_r)**2):
                    R_LEO_asign[r][s] = 1
#%%--------------------------------------------------------------------------------------------%%

        for l in range(UAV_T):
            for h in range(HAP_T):
                if((HAP_loc[0][h]-HAP_r) < 0):
                    if (((UAV_loc[0][l]-UAV_r)> (HAP_loc[0][h]-HAP_r))and((UAV_loc[0][l]+UAV_r)**2< (HAP_loc[0][h]+HAP_r)**2)):
                        U_HAP_asign[l][h] = 1
                else:
                    if (((UAV_loc[0][l]-UAV_r)**2> (HAP_loc[0][h]-HAP_r)**2)and((UAV_loc[0][l]+UAV_r)**2< (HAP_loc[0][h]+HAP_r)**2)):
                        U_HAP_asign[l][h] = 1

        for l in range(UAV_T):
            for s in range(LEO_T):
                if((LEO_loc[0][s]-LEO_r) < 0):
                    if (((UAV_loc[0][l]-UAV_r)> (LEO_loc[0][s]-LEO_r))and((UAV_loc[0][l]+UAV_r)**2< (LEO_loc[0][s]+LEO_r)**2)):
                        U_LEO_asign[l][s] = 1
                else: 
                    if (((UAV_loc[0][l]-UAV_r)**2> (LEO_loc[0][s]-LEO_r)**2)and((UAV_loc[0][l]+UAV_r)**2< (LEO_loc[0][s]+LEO_r)**2)):
                        U_LEO_asign[l][s] = 1
#%%--------------------------------------------------------------------------------------------%%                    
        for h in range(HAP_T):
            for s in range(LEO_T):
                if((LEO_loc[0][s]-LEO_r) < 0):
                    if (((HAP_loc[0][h]-HAP_r)> (LEO_loc[0][s]-LEO_r))and((HAP_loc[0][h]+HAP_r)**2< (LEO_loc[0][s]+LEO_r)**2)):
                        H_LEO_asign[h][s] = 1
                else:
                    if (((HAP_loc[0][h]-HAP_r)**2> (LEO_loc[0][s]-LEO_r)**2)and((HAP_loc[0][h]+HAP_r)**2< (LEO_loc[0][s]+LEO_r)**2)):
                        H_LEO_asign[h][s] = 1
#%%--------------------------------------------------------------------------------------------%%               

        ru_sum=np.zeros((1, RSU_T))
        rh_sum=np.zeros((1, RSU_T))
        rs_sum=np.zeros((1, RSU_T))
        uh_sum=np.zeros((1, UAV_T))
        us_sum=np.zeros((1, UAV_T))
        hs_sum=np.zeros((1, HAP_T))
        for r in range(RSU_T):
            ru_sum[0][r]=sum(R_UAV_asign[r])
            rh_sum[0][r]=sum(R_HAP_asign[r])
            rs_sum[0][r]=sum(R_LEO_asign[r])
        for l in range(UAV_T):
            uh_sum[0][l]=sum(U_HAP_asign[l])
            us_sum[0][l]=sum(U_LEO_asign[l])
            
        for h in range(HAP_T):
            hs_sum[0][h]=sum(H_LEO_asign[h])
            
        if ((np.all(ru_sum >0)) and (np.all(rh_sum >0)) and (np.all(rs_sum >0)) and ((np.all(uh_sum >0))) and ((np.all(us_sum >0))) and ((np.all(hs_sum >0)))):
            break        
        
    return RSU_loc, UAV_loc, HAP_loc, LEO_loc


# ======================================================================
# ==== Notebook CODE CELL 2 (verbatim) ====
# ======================================================================

def VN_EN_Assign(IP,V_spd):
    out=[]
    V=IP[0]
    RSU_T=IP[1]
    RSU_loc=IP[2]
    RSU_r=IP[3]
    UAV_T=IP[4]
    UAV_loc=IP[5]
    UAV_r=IP[6]
    HAP_T=IP[7]
    HAP_loc=IP[8]
    HAP_r=IP[9]
    LEO_T=IP[10]
    LEO_loc=IP[11]
    LEO_r=IP[12]
    V_RSU_asign= np.zeros((V, RSU_T))
    V_RSU_Soj= np.zeros((V, RSU_T))
    V_RSU_dist= np.zeros((V, RSU_T))
    V_UAV_asign= np.zeros((V, UAV_T))
    V_UAV_Soj= np.zeros((V, UAV_T))
    V_UAV_dist= np.zeros((V, UAV_T))
    V_HAP_asign= np.zeros((V, HAP_T))
    V_HAP_Soj= np.zeros((V, HAP_T))
    V_HAP_dist= np.zeros((V, HAP_T))
    V_LEO_asign= np.zeros((V, LEO_T))
    V_LEO_Soj= np.zeros((V, LEO_T))
    V_LEO_dist= np.zeros((V, LEO_T))
    
    R_UAV_asign= np.zeros((RSU_T,UAV_T))
    R_HAP_asign= np.zeros((RSU_T,HAP_T))
    U_HAP_asign= np.zeros((UAV_T,HAP_T))
    R_UAV_dist= np.zeros((RSU_T, UAV_T))
    R_HAP_dist= np.zeros((RSU_T, HAP_T))
    U_HAP_dist= np.zeros((UAV_T, HAP_T))
    V_RSU_d= np.zeros((V, RSU_T))
    V_UAV_d= np.zeros((V, UAV_T))
    V_HAP_d= np.zeros((V,HAP_T))
    V_LEO_d= np.zeros((V,LEO_T))
    RSU_SC=np.zeros((1, V))
    VN_loc=np.zeros((3, V))
    while True:
        for i in range(V):
            VN_loc[2][i]=random.choice([1,2,3,4,5,6])
            while True:
                VN_loc[0][i]=np.random.randint(RSU_loc[0][0], RSU_loc[0][RSU_T-1], size=(1,1))
                VN_loc[1][i]=0
                for j in range(RSU_T):

                    if RSU_loc[0][j] >= VN_loc[0][i] and (RSU_r)**(2) > (RSU_loc[1][j] - VN_loc[1][i])**(2):

                        if ((RSU_loc[0][j]-(math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2)))) <= VN_loc[0][i] and VN_loc[0][i] <= (RSU_loc[0][j]+(math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2)))) ):
                            V_RSU_asign[i][j] = 1
                            V_RSU_d[i][j] = (math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2))+(RSU_loc[0][j]-VN_loc[0][i]))
                            V_RSU_Soj[i][j] = ((math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2))+(RSU_loc[0][j]-VN_loc[0][i]))/V_spd[i])
                            V_RSU_dist[i][j] = math.hypot(RSU_loc[0][j]  - VN_loc[0][i], RSU_loc[1][j]  - VN_loc[1][i])
                        else:
                            V_RSU_asign[i][j]=0
                            V_RSU_d[i][j] =0
                            V_RSU_Soj[i][j]=0
                            V_RSU_dist[i][j]=0
                    elif RSU_loc[0][j] < VN_loc[0][i] and (RSU_r)**(2) > (RSU_loc[1][j] - VN_loc[1][i])**(2):

                        if ((RSU_loc[0][j]-(math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2)))) <= VN_loc[0][i] and VN_loc[0][i] <= (RSU_loc[0][j]+(math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2))))  ):
                            V_RSU_asign[i][j] = 1
                            V_RSU_d[i][j] = (math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2))-(VN_loc[0][i] - RSU_loc[0][j]))
                            V_RSU_Soj[i][j] = ((math.sqrt((RSU_r)**(2) - (RSU_loc[1][j] - VN_loc[1][i])**(2))-(VN_loc[0][i] - RSU_loc[0][j]))/V_spd[i])
                            V_RSU_dist[i][j] =  math.hypot(RSU_loc[0][j]  - VN_loc[0][i], RSU_loc[1][j]  - VN_loc[1][i])
                        else:
                            V_RSU_asign[i][j]=0
                            V_RSU_d[i][j] =0
                            V_RSU_Soj[i][j]=0
                            V_RSU_dist[i][j]=0
                for j in range(RSU_T):
                    if (V_RSU_Soj[i][j]==0):
                        V_RSU_asign[i][j]=0
                        V_RSU_dist[i][j]=0
                if  ((sum(V_RSU_asign[i][:]) <4)): 
                    break  

            a_c = V_RSU_asign[i]
            ls = [ii for ii, e in enumerate(a_c) if e != 0]
            RSU_SC[0][i]=len(ls)



        for i in range(V):
            for j in range(UAV_T):
                if (((UAV_loc[0][j]+UAV_r) >= VN_loc[0][i]) and (VN_loc[0][i] > (UAV_loc[0][j]-UAV_r))):
                    V_UAV_asign[i][j] = 1
                    V_UAV_d[i][j] = (UAV_loc[0][j] + (UAV_r-VN_loc[0][i]))
                    V_UAV_Soj[i][j] = (V_UAV_d[i][j]/V_spd[i])
                    V_UAV_dist[i][j] =math.sqrt((UAV_loc[0][j] - VN_loc[0][i])**2 + (UAV_loc[1][j] - VN_loc[1][i])**2 + (UAV_loc[2][j] - 0)**2)


        for i in range(V):
            for j in range(HAP_T):
                if (((HAP_loc[0][j]+HAP_r) >= VN_loc[0][i]) and (VN_loc[0][i] > (HAP_loc[0][j]-HAP_r))):
                    V_HAP_asign[i][j] = 1
                    V_HAP_d[i][j] = (HAP_loc[0][j] + (HAP_r-VN_loc[0][i]))
                    V_HAP_Soj[i][j] = (V_HAP_d[i][j]/V_spd[i])  
                    V_HAP_dist[i][j] = math.sqrt((HAP_loc[0][j] - VN_loc[0][i])**2 + (HAP_loc[1][j] - VN_loc[1][i])**2 + (HAP_loc[2][j] - 0)**2)

        for i in range(V):
            for j in range(LEO_T):
                if (((LEO_loc[0][j]+LEO_r) >= VN_loc[0][i]) and (VN_loc[0][i] > (LEO_loc[0][j]-LEO_r))):
                    V_LEO_asign[i][j] = 1
                    V_LEO_d[i][j] = (LEO_loc[0][j] + (LEO_r-VN_loc[0][i]))
                    V_LEO_Soj[i][j] = (V_LEO_d[i][j]/V_spd[i])  
                    V_LEO_dist[i][j] = math.sqrt((LEO_loc[0][j] - VN_loc[0][i])**2 + (LEO_loc[1][j] - VN_loc[1][i])**2 + (LEO_loc[2][j] - 0)**2)
 
                    
                    
        vr_sum=np.zeros((1, V))
        vu_sum=np.zeros((1, V))
        vh_sum=np.zeros((1, V))
        vs_sum=np.zeros((1, V))
        for v in range(V):
            vr_sum[0][v]=sum(V_RSU_asign[v])
            vu_sum[0][v]=sum(V_UAV_asign[v])
            vh_sum[0][v]=sum(V_HAP_asign[v])
            vs_sum[0][v]=sum(V_LEO_asign[v])

        if ((np.all(vr_sum >0)) and (np.all(vu_sum >0)) and (np.all(vh_sum >0)) and (np.all(vs_sum >0))):
            break

    out.append(VN_loc)
    out.append(V_RSU_asign)
    out.append(V_RSU_Soj)
    out.append(V_RSU_dist)
    out.append(V_UAV_asign)
    out.append(V_UAV_Soj)
    out.append(V_UAV_dist)
    out.append(V_HAP_asign)
    out.append(V_HAP_Soj)
    out.append(V_HAP_dist)
    out.append(V_LEO_asign)
    out.append(V_LEO_Soj)
    out.append(V_LEO_dist)
    out.append(V_RSU_d)
    out.append(V_UAV_d)
    out.append(V_HAP_d)
    out.append(V_LEO_d)
    return out


# ======================================================================
# ==== Notebook CODE CELL 3 (verbatim) ====
# ======================================================================

def Chn_Capacity(N1N2_dist, BN2, PN1, b_0, theta):
    N0dbm = -45  # noise power in dBm
    N0 = 10 ** ((N0dbm - 30) / 10)  # noise power in Watts
    lam = (3 * (10 ** 8)) / BN2
    PL = 20 * math.log10(N1N2_dist / (10 ** 3)) + 92.45 + 20 * math.log10(BN2 / (10 ** 9))
    path_loss = 10 ** ((-PL) / 10)


    channel_gain = b_0 * (N1N2_dist ** theta) * path_loss

    # Calculate the Shannon channel capacity
    C = BN2 * math.log2(1 + (PN1 * channel_gain) / N0)

    return C


# ======================================================================
# ==== Notebook CODE CELL 4 (verbatim) ====
# ======================================================================

#-----------Edit-----------------------------
def Task_Proc_Main(IP, CR, DR_R_R, CU, DR_R_U, CH, DR_R_H, CL, DR_R_S,g1,g2, decisions):
    
#============Task Processing===============================================
    out=[]
    V=IP[0]
    RSU_T=IP[1]
    UAV_T=IP[4]
    HAP_T=IP[7]
    LEO_T=IP[10]
    TS=IP[31]
    TSD=IP[32]
    psi_dmp=IP[33]
    Pcomp_m=IP[34]
    Ptp_v=IP[35]
    Pcomp_r=IP[36]
    Ptp_r=IP[37]
    Pcomp_u=IP[38]
    Ptp_u=IP[39]
    Pcomp_h=IP[40]
    Ptp_h=IP[41]
    Pcomp_s=IP[42]
    Ptp_s=IP[43]
    Cm=IP[44]
    Soj_T_VR=IP[45]
    Soj_T_VU=IP[46]
    Soj_T_VH=IP[47]
    Soj_T_VS=IP[48]
    gamma_1=IP[49]
    gamma_2=IP[50]
    Ser_RSU=IP[51]
    Ser_UAV=IP[52]
    Ser_HAP=IP[53]
    Ser_LEO=IP[54]
    V_loc=IP[55]


    #--------Local Device Computation----------------------------

    TPcomp_v=np.zeros((1, (V)))
    EPcomp_v=np.zeros((1, (V)))
    for v in range(V):
        TPcomp_v[0][v]=((TS[0][v]*psi_dmp)/(Cm[0][v]))
        EPcomp_v[0][v]=(Pcomp_m[0][v]*TPcomp_v[0][v])

        
    #-----------Offloading Time and Energy------------------
    TPcomp_r=np.zeros((V, RSU_T))
    EPcomp_r=np.zeros((V, RSU_T))
    TPcomp_u=np.zeros((V, UAV_T))
    EPcomp_u=np.zeros((V, UAV_T))
    TPcomp_h=np.zeros((V, HAP_T))
    EPcomp_h=np.zeros((V, HAP_T))
    TPcomp_s=np.zeros((V, LEO_T))
    EPcomp_s=np.zeros((V, LEO_T))
    
    TPcomU_vr=np.zeros((V, RSU_T))
    EPcomU_vr=np.zeros((V, RSU_T))
    TPcomD_vr=np.zeros((V, RSU_T))
    EPcomD_vr=np.zeros((V, RSU_T))
    TPcomU_rv=np.zeros((V, RSU_T))
    EPcomU_rv=np.zeros((V, RSU_T))
    TPcomD_rv=np.zeros((V, RSU_T))
    EPcomD_rv=np.zeros((V, RSU_T))
    
    TPcomU_vu=np.zeros((V, UAV_T))
    EPcomU_vu=np.zeros((V, UAV_T))
    TPcomD_vu=np.zeros((V, UAV_T))
    EPcomD_vu=np.zeros((V, UAV_T))
    TPcomU_uv=np.zeros((V, UAV_T))
    EPcomU_uv=np.zeros((V, UAV_T))
    TPcomD_uv=np.zeros((V, UAV_T))
    EPcomD_uv=np.zeros((V, UAV_T))
    
    TPcomU_vh=np.zeros((V, HAP_T))
    EPcomU_vh=np.zeros((V, HAP_T))
    TPcomD_vh=np.zeros((V, HAP_T))
    EPcomD_vh=np.zeros((V, HAP_T))
    TPcomU_hv=np.zeros((V, HAP_T))
    EPcomU_hv=np.zeros((V, HAP_T))
    TPcomD_hv=np.zeros((V, HAP_T))
    EPcomD_hv=np.zeros((V, HAP_T))
    

    
    TPcomU_vs=np.zeros((V, LEO_T))
    EPcomU_vs=np.zeros((V, LEO_T))
    TPcomD_vs=np.zeros((V, LEO_T))
    EPcomD_vs=np.zeros((V, LEO_T))
    TPcomU_sv=np.zeros((V, LEO_T))
    EPcomU_sv=np.zeros((V, LEO_T))
    TPcomD_sv=np.zeros((V, LEO_T))
    EPcomD_sv=np.zeros((V, LEO_T))
    

    
    for v in range(V):
        if(decisions[v][0]==0):
            for r in range(RSU_T):
                if(decisions[v][1]==r):
                    TPcomp_r[v][r]=((TS[0][v]*psi_dmp)/(CR[v][r]))
                    EPcomp_r[v][r]=(Pcomp_r[0][r]*TPcomp_r[v][r])
                    TPcomU_vr[v][r]=((TS[0][v])/DR_R_R[v][r])
                    EPcomU_vr[v][r]=TPcomU_vr[v][r]*Ptp_v[0][v]
                    EPcomU_rv[v][r]=TPcomU_vr[v][r]*Ptp_r[0][r]
                    TPcomD_vr[v][r]=((TSD[0][v])/DR_R_R[v][r])
                    EPcomD_vr[v][r]=TPcomD_vr[v][r]*Ptp_v[0][v]
                    EPcomD_rv[v][r]=TPcomD_vr[v][r]*Ptp_r[0][r]
                    
                    
                    
    #------------Total Task Processing Time and Energy--------------\n    
    TP_offl_vr=np.zeros((V, RSU_T))
    EP_offl_vr=np.zeros((V, RSU_T))
    TP_offl_rv=np.zeros((V, RSU_T))
    EP_offl_rv=np.zeros((V, RSU_T))
    TP_offl_T_r=np.zeros((V, RSU_T))
    EP_offl_T_r=np.zeros((V, RSU_T))    

    TP_loc=np.zeros((1, (V)))
    EP_loc=np.zeros((1, (V)))
    for v in range(V):
        if(decisions[v][0]==0):
            for r in range(RSU_T):
                if(decisions[v][1]==r):
                    TP_offl_vr[v][r]=TPcomU_vr[v][r]  + TPcomD_vr[v][r]
                    EP_offl_vr[v][r]=EPcomU_vr[v][r]  + EPcomD_vr[v][r]
                    TP_offl_rv[v][r]=TPcomp_r[v][r]
                    EP_offl_rv[v][r]=EPcomp_r[v][r]
                    
                    TP_offl_T_r[v][r]=TP_offl_vr[v][r]  + TP_offl_rv[v][r]
                    EP_offl_T_r[v][r]=0.5*EP_offl_vr[v][r]  + 0.5*EP_offl_rv[v][r]

                    TP_loc[0][v]=TPcomp_v[0][v]
                    EP_loc[0][v]=0.5*EPcomp_v[0][v]                    

                
    for v in range(V):
        if(decisions[v][0]==1):
            for u in range(UAV_T):
                if(decisions[v][1]==u):
                    TPcomp_u[v][u]=((TS[0][v]*psi_dmp)/(CU[v][u]))
                    EPcomp_u[v][u]=(Pcomp_u[0][u]*TPcomp_u[v][u])
                    
                    TPcomU_vu[v][u]=((TS[0][v])/DR_R_U[v][u])
                    EPcomU_vu[v][u]=TPcomU_vu[v][u]*Ptp_v[0][v]
                    EPcomU_uv[v][u]=TPcomU_vu[v][u]*Ptp_u[0][u]
                    TPcomD_vu[v][u]=((TSD[0][v])/DR_R_U[v][u])
                    EPcomD_vu[v][u]=TPcomD_vu[v][u]*Ptp_v[0][v]
                    EPcomD_uv[v][u]=TPcomD_vu[v][u]*Ptp_u[0][u]
                    
                 
                    
    #------------Total Task Processing Time and Energy (UAV)--------------\n    
    TP_offl_vu=np.zeros((V, UAV_T))
    EP_offl_vu=np.zeros((V, UAV_T))
    TP_offl_uv=np.zeros((V, UAV_T))
    EP_offl_uv=np.zeros((V, UAV_T))
    TP_offl_T_u=np.zeros((V, UAV_T))
    EP_offl_T_u=np.zeros((V, UAV_T))    


    for v in range(V):
        if(decisions[v][0]==1):
            for u in range(UAV_T):
                if(decisions[v][1]==u):
                    TP_offl_vu[v][u]=TPcomU_vu[v][u]  + TPcomD_vu[v][u]
                    EP_offl_vu[v][u]=EPcomU_vu[v][u]  + EPcomD_vu[v][u]
                    TP_offl_uv[v][u]=TPcomp_u[v][u]
                    EP_offl_uv[v][u]=EPcomp_u[v][u]
                    
                    TP_offl_T_u[v][u]=TP_offl_vu[v][u]  + TP_offl_uv[v][u]
                    EP_offl_T_u[v][u]=0.5*EP_offl_vu[v][u]  + 0.5*EP_offl_uv[v][u]

                    TP_loc[0][v]=TPcomp_v[0][v]
                    EP_loc[0][v]=0.5*EPcomp_v[0][v]                    

                    
                
    for v in range(V):
        if(decisions[v][0]==2):
            for h in range(HAP_T):
                if(decisions[v][1]==h):
                    TPcomp_h[v][h]=((TS[0][v]*psi_dmp)/(CH[v][h]))
                    EPcomp_h[v][h]=(Pcomp_h[0][h]*TPcomp_h[v][h])
                    
                    TPcomU_vh[v][h]=((TS[0][v])/DR_R_H[v][h])
                    EPcomU_vh[v][h]=TPcomU_vh[v][h]*Ptp_v[0][v]
                    EPcomU_hv[v][h]=TPcomU_vh[v][h]*Ptp_h[0][h]
                    TPcomD_vh[v][h]=((TSD[0][v])/DR_R_H[v][h])
                    EPcomD_vh[v][h]=TPcomD_vh[v][h]*Ptp_v[0][v]
                    EPcomD_hv[v][h]=TPcomD_vh[v][h]*Ptp_h[0][h]
                    
    #------------Total Task Processing Time and Energy (HAP)--------------\n    
    TP_offl_vh=np.zeros((V, HAP_T))
    EP_offl_vh=np.zeros((V, HAP_T))
    TP_offl_hv=np.zeros((V, HAP_T))
    EP_offl_hv=np.zeros((V, HAP_T))
    TP_offl_T_h=np.zeros((V, HAP_T))
    EP_offl_T_h=np.zeros((V, HAP_T))    


    for v in range(V):
        if(decisions[v][0]==2):
            for h in range(HAP_T):
                if(decisions[v][1]==h):
                    TP_offl_vh[v][h]=TPcomU_vh[v][h]  + TPcomD_vh[v][h]
                    EP_offl_vh[v][h]=EPcomU_vh[v][h]  + EPcomD_vh[v][h]
                    TP_offl_hv[v][h]=TPcomp_h[v][h]
                    EP_offl_hv[v][h]=EPcomp_h[v][h]
                    
                    TP_offl_T_h[v][h]=TP_offl_vh[v][h]  + TP_offl_hv[v][h]
                    EP_offl_T_h[v][h]=0.5*EP_offl_vh[v][h]  + 0.5*EP_offl_hv[v][h]

                    TP_loc[0][v]=TPcomp_v[0][v]
                    EP_loc[0][v]=0.5*EPcomp_v[0][v]   
                    

    for v in range(V):
        if(decisions[v][0]==3):
            for s in range(LEO_T):
                if(decisions[v][1]==s):
                    TPcomp_s[v][s]=((TS[0][v]*psi_dmp)/(CL[v][s]))
                    EPcomp_s[v][s]=(Pcomp_s[0][s]*TPcomp_s[v][s])
                    
                    TPcomU_vs[v][s]=((TS[0][v])/DR_R_S[v][s])
                    EPcomU_vs[v][s]=TPcomU_vs[v][s]*Ptp_v[0][v]
                    EPcomU_sv[v][s]=TPcomU_vs[v][s]*Ptp_s[0][s]
                    TPcomD_vs[v][s]=((TSD[0][v])/DR_R_S[v][s])
                    EPcomD_vs[v][s]=TPcomD_vs[v][s]*Ptp_v[0][v]
                    EPcomD_sv[v][s]=TPcomD_vs[v][s]*Ptp_s[0][s]
                    
    
                    
    #------------Total Task Processing Time and Energy (HAP)--------------\n    
    TP_offl_vs=np.zeros((V, LEO_T))
    EP_offl_vs=np.zeros((V, LEO_T))
    TP_offl_sv=np.zeros((V, LEO_T))
    EP_offl_sv=np.zeros((V, LEO_T))
    TP_offl_T_s=np.zeros((V, LEO_T))
    EP_offl_T_s=np.zeros((V, LEO_T))    


    for v in range(V):
        if(decisions[v][0]==3):
            for s in range(LEO_T):
                if(decisions[v][1]==s):
                    TP_offl_vs[v][s]=TPcomU_vs[v][s]  + TPcomD_vs[v][s]
                    EP_offl_vs[v][s]=EPcomU_vs[v][s]  + EPcomD_vs[v][s]
                    TP_offl_sv[v][s]=TPcomp_s[v][s]
                    EP_offl_sv[v][s]=EPcomp_s[v][s]
                    
                    TP_offl_T_s[v][s]=TP_offl_vs[v][s]  + TP_offl_sv[v][s]
                    EP_offl_T_s[v][s]=0.5*EP_offl_vs[v][s]  + 0.5*EP_offl_sv[v][s]

                    TP_loc[0][v]=TPcomp_v[0][v]
                    EP_loc[0][v]=0.5*EPcomp_v[0][v]   
                    
                    
                    
                    
                    
    T_L=np.zeros((1,V))
    T_L_V=np.zeros((1,V))
    T_L_E=np.zeros((1,V))
    T_E=np.zeros((1,V))
    T_E_V=np.zeros((1,V))
    T_E_E=np.zeros((1,V))
    No_of_Soj_L_Fail=0
    No_of_Ser_H_Req=0
    
    for v in range(V):
        if(decisions[v][0]==0):
            for r in range(RSU_T):
                if(decisions[v][1]==r):
                    T_L[0][v]=max((decisions[v][2]*TP_offl_T_r[v][r]), ((1-decisions[v][2])*TP_loc[0][v]))
                    T_L_V[0][v]= (1-decisions[v][2])*TP_loc[0][v]
                    T_L_E[0][v]= (decisions[v][2]*TP_offl_T_r[v][r])
                    T_E[0][v]=((decisions[v][2]*EP_offl_T_r[v][r])+ ((1-decisions[v][2])*EP_loc[0][v]))
                    T_E_V[0][v]= (1-decisions[v][2])*EP_loc[0][v]
                    T_E_E[0][v]= (decisions[v][2]*EP_offl_T_r[v][r])
                    if(T_L_E[0][v]>Soj_T_VR[v][r]):
                        No_of_Soj_L_Fail=No_of_Soj_L_Fail+1
                    if V_loc[2][v] not in Ser_RSU[r]:
                        No_of_Ser_H_Req += 1
                        
    for v in range(V):
        if(decisions[v][0]==1):
            for u in range(UAV_T):
                if(decisions[v][1]==u):
                    T_L[0][v]=max((decisions[v][2]*TP_offl_T_u[v][u]), ((1-decisions[v][2])*TP_loc[0][v]))
                    T_E[0][v]=((decisions[v][2]*EP_offl_T_u[v][u])+ ((1-decisions[v][2])*EP_loc[0][v]))  
                    T_L_V[0][v]= (1-decisions[v][2])*TP_loc[0][v]
                    T_L_E[0][v]= (decisions[v][2]*TP_offl_T_u[v][u])
                    T_E_V[0][v]= (1-decisions[v][2])*EP_loc[0][v]
                    T_E_E[0][v]= (decisions[v][2]*EP_offl_T_u[v][u])
                    if(T_L_E[0][v]>Soj_T_VU[v][u]):
                        No_of_Soj_L_Fail=No_of_Soj_L_Fail+1
                    if V_loc[2][v] not in Ser_UAV[u]:
                        No_of_Ser_H_Req += 1
        
    for v in range(V):
        if(decisions[v][0]==2):
            for h in range(HAP_T):
                if(decisions[v][1]==h):
                    T_L[0][v]=max((decisions[v][2]*TP_offl_T_h[v][h]), ((1-decisions[v][2])*TP_loc[0][v]))
                    T_E[0][v]=((decisions[v][2]*EP_offl_T_h[v][h])+ ((1-decisions[v][2])*EP_loc[0][v]))  
                    T_L_V[0][v]= (1-decisions[v][2])*TP_loc[0][v]
                    T_L_E[0][v]= (decisions[v][2]*TP_offl_T_h[v][h])
                    T_E_V[0][v]= (1-decisions[v][2])*EP_loc[0][v]
                    T_E_E[0][v]= (decisions[v][2]*EP_offl_T_h[v][h])
                    if(T_L_E[0][v]>Soj_T_VH[v][h]):
                        No_of_Soj_L_Fail=No_of_Soj_L_Fail+1
                    if V_loc[2][v] not in Ser_HAP[h]:
                        No_of_Ser_H_Req += 1
                    
    for v in range(V):
        if(decisions[v][0]==3):
            for s in range(LEO_T):
                if(decisions[v][1]==s):
                    T_L[0][v]=max((decisions[v][2]*TP_offl_T_s[v][s]), ((1-decisions[v][2])*TP_loc[0][v]))
                    T_E[0][v]=((decisions[v][2]*EP_offl_T_s[v][s])+ ((1-decisions[v][2])*EP_loc[0][v]))  
                    T_L_V[0][v]= (1-decisions[v][2])*TP_loc[0][v]
                    T_L_E[0][v]= (decisions[v][2]*TP_offl_T_s[v][s])
                    T_E_V[0][v]= (1-decisions[v][2])*EP_loc[0][v]
                    T_E_E[0][v]= (decisions[v][2]*EP_offl_T_s[v][s])
                    if(T_L_E[0][v]>Soj_T_VS[v][s]):
                        No_of_Soj_L_Fail=No_of_Soj_L_Fail+1
                    if V_loc[2][v] not in Ser_LEO[s]:
                        No_of_Ser_H_Req += 1

    No_of_Ser_L_Fail=0
    No_of_Ser_H_Fail=0
    TC=np.zeros((1, V))
    
    for v in range(V):
        TC[0][v]= g1*T_L[0][v] + g2*T_E[0][v]
        if(T_L[0][v]>4):
            No_of_Ser_L_Fail=No_of_Ser_L_Fail+1


                    
    out.append(T_L)
    out.append(T_E)
    out.append(T_L_V)
    out.append(T_L_E)
    out.append(T_E_V)
    out.append(T_E_E)
    out.append(TP_loc)
    out.append(EP_loc)
    out.append(TC)
    out.append(No_of_Ser_L_Fail)
    out.append(No_of_Soj_L_Fail)
    out.append(No_of_Ser_H_Req)

    return out


# ======================================================================
# ==== Notebook CODE CELL 5 (verbatim) ====
# ======================================================================

def Data_Rate(IP, decisions, RSU_B, UAV_B, HAP_B, LEO_B):
    
    V=IP[0]
    RSU_T=IP[1]
    UAV_T=IP[4]
    HAP_T=IP[7]
    LEO_T=IP[10]
    
    PNR=IP[21]
    PNU=IP[22]
    PNH=IP[23]
    PNL=IP[24]
    b_0=IP[25]
    theta=IP[26]
    VN_RSU_dist=IP[27]
    VN_UAV_dist=IP[28]
    VN_HAP_dist=IP[29]
    VN_LEO_dist=IP[30]

    
    
    
    Rate_VR=np.zeros((V,RSU_T))
    Rate_VU=np.zeros((V,UAV_T))
    Rate_VH=np.zeros((V,HAP_T))
    Rate_VL=np.zeros((V,LEO_T))
    

    for v in range(V):
        if(decisions[v][0]==0):
            for r in range(RSU_T):
                if(decisions[v][1]==r):
                    Rate_VR[v][r]=Chn_Capacity(VN_RSU_dist[v][r], RSU_B[v][r], PNR[0][r], b_0, theta)

    for v in range(V):
        if(decisions[v][0]==1):
            for u in range(UAV_T):
                if(decisions[v][1]==u):
                    Rate_VU[v][u]=Chn_Capacity(VN_UAV_dist[v][u], UAV_B[v][u], PNU[0][u], b_0, theta)

    for v in range(V):
        if(decisions[v][0]==2):
            for h in range(HAP_T):
                if(decisions[v][1]==h):
                    Rate_VH[v][h]=Chn_Capacity(VN_HAP_dist[v][h], HAP_B[v][h], PNH[0][h], b_0, theta)

    for v in range(V):
        if(decisions[v][0]==3):
            for s in range(LEO_T):
                if(decisions[v][1]==s):
                    Rate_VL[v][s]=Chn_Capacity(VN_LEO_dist[v][s], LEO_B[v][s], PNL[0][s], b_0, theta)
                    
    return Rate_VR, Rate_VU, Rate_VH, Rate_VL


# ======================================================================
# ==== Notebook CODE CELL 6 (verbatim) ====
# ======================================================================


def Resource_Allocation( IP,decisions, NOdes_assign_R, NOdes_assign_U, NOdes_assign_H, NOdes_assign_L):
    out=[]
    
    V=IP[0]
    RSU_T=IP[1]
    UAV_T=IP[4]
    HAP_T=IP[7]
    LEO_T=IP[10]
    RSU_B=IP[13]
    UAV_B=IP[14]
    HAP_B=IP[15]
    LEO_B=IP[16]
    RSU_C=IP[17]
    UAV_C=IP[18]
    HAP_C=IP[19]
    LEO_C=IP[20]

    
    
    RSU_B_Assign=np.zeros((V,RSU_T))
    UAV_B_Assign=np.zeros((V,UAV_T))
    HAP_B_Assign=np.zeros((V,HAP_T))
    LEO_B_Assign=np.zeros((V,LEO_T))
    RSU_C_Assign=np.zeros((V,RSU_T))
    UAV_C_Assign=np.zeros((V,UAV_T))
    HAP_C_Assign=np.zeros((V,HAP_T))
    LEO_C_Assign=np.zeros((V,LEO_T))
    for v in range(V):
        if(decisions[v][0]==0):
            for r in range(RSU_T): 
                if(decisions[v][1]==r):
                    RSU_B_Assign[v][r]=RSU_B[0][r]/NOdes_assign_R[r]
                    RSU_C_Assign[v][r]=RSU_C[0][r]/NOdes_assign_R[r]

    for v in range(V):
        if(decisions[v][0]==1):
            for u in range(UAV_T): 
                if(decisions[v][1]==u):
                    UAV_B_Assign[v][u]=UAV_B[0][u]/NOdes_assign_U[u]
                    UAV_C_Assign[v][u]=UAV_C[0][u]/NOdes_assign_U[u]

    for v in range(V):
        if(decisions[v][0]==2):
            for h in range(HAP_T): 
                if(decisions[v][1]==h):
                    HAP_B_Assign[v][h]=HAP_B[0][h]/NOdes_assign_H[h]
                    HAP_C_Assign[v][h]=HAP_C[0][h]/NOdes_assign_H[h]
                    
    for v in range(V):
        if(decisions[v][0]==3):
            for s in range(LEO_T): 
                if(decisions[v][1]==s):
                    LEO_B_Assign[v][s]=LEO_B[0][s]/NOdes_assign_L[s]
                    LEO_C_Assign[v][s]=LEO_C[0][s]/NOdes_assign_L[s]
    out.append(RSU_B_Assign)
    out.append(UAV_B_Assign)
    out.append(HAP_B_Assign)
    out.append(LEO_B_Assign)
    out.append(RSU_C_Assign)
    out.append(UAV_C_Assign)
    out.append(HAP_C_Assign)
    out.append(LEO_C_Assign)
    
                    
    return out


# ======================================================================
# ==== Notebook CODE CELL 7 (verbatim) ====
# ======================================================================



class Agent:
    def __init__(self, optimizer, state_size, action_size, gamma, epsilon):
        self._state_size = state_size
        self._action_size = action_size
        self._optimizer = optimizer
        self.experience_replay = deque(maxlen=50000)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        
        # Build models with an input shape to ensure they are initialized
        self.q_network.build((None, 1))
        self.target_network.build((None, 1))
        
        # Initialize with dummy weights to avoid errors
        self._initialize_dummy_weights()
        
        self.alighn_target_model()

    def store(self, state, action_id, reward, next_state):
        self.experience_replay.append((state, action_id, reward, next_state))
        
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self._state_size, output_dim=10))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def _initialize_dummy_weights(self):
        dummy_weights = [np.zeros_like(weight) for weight in self.q_network.get_weights()]
        self.q_network.set_weights(dummy_weights)
        self.target_network.set_weights(dummy_weights)

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
    def act(self, state, q_values):
        if np.random.rand() <= self.epsilon:
            q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state in minibatch: 
            target = self.q_network.predict(state)  
            t = self.target_network.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=1, verbose=0)

    def load(self, name1, name2):
        self.q_network.load_weights(name1)
        self.target_network.load_weights(name2)

    def save(self, name1, name2):
        self.q_network.save_weights(name1)
        self.target_network.save_weights(name2)


# ======================================================================
# ==== Notebook CODE CELL 12 (verbatim) ====
# ======================================================================

State_Space_H={}
H_S_D=[0,1,2,3]
H_S_CV=[0, 1, 2]
M_S=[0, 1, 2, 3]
P_SL=[0, 1, 2, 3]
k=0

State_Space_H = np.zeros((len(H_S_D) * len(H_S_CV) * len(M_S) * len(P_SL), 4))
k = 0

for i1 in range(len(H_S_D)):
    for i2 in range(len(H_S_CV)):
        for i3 in range(len(M_S)):
            for i4 in range(len(P_SL)):
                state = np.zeros(4)
                state[0] = H_S_D[i1]
                state[1] = H_S_CV[i2]
                state[2] = M_S[i3]
                state[3] = P_SL[i4]
                State_Space_H[k] = state
                k += 1
        
Action_Space_H= [0, 1, 2, 3]


# ======================================================================
# ==== Notebook CODE CELL 13 (verbatim) ====
# ======================================================================


def Next_State_H( a_h, Req_Ser, RSU_T, NO_VUs_RSU, Max_U_RSU, UAV_T, NO_VUs_UAV, Max_U_UAV, HAP_T, NO_VUs_HAP, Max_U_HAP, LEO_T, NO_VUs_LEO, Max_U_LEO):

    S_N=np.zeros(4)

    if(a_h==0):
        S_N[0]=0

        H_S_CV_R=0
        H_S_CV_R_A=0
  
        for r in range(RSU_T):
     
            H_S_CV_R = H_S_CV_R + (NO_VUs_RSU[r]/Max_U_RSU)

        H_S_CV_R_A=(H_S_CV_R/RSU_T)

        if(H_S_CV_R_A <= 0.5):
            H_S_CV_R_s=0
        elif(0.5 < H_S_CV_R_A <= 1):
            H_S_CV_R_s=1
        else:
            H_S_CV_R_s=2
        S_N[1]=H_S_CV_R_s


        S_N[2]=0

        nodes_with_service=0
        service_layer_prob_R=0
        nodes_with_service = sum(1 for services in service_allocation_RSU.values() if Req_Ser in services)
        service_layer_prob_R=(nodes_with_service/RSU_T)
        if(0 <= service_layer_prob_R <  0.25 ):
            S_N[3]=3
        elif(0.25 <= service_layer_prob_R < 0.5):
            S_N[3]=2
        elif(0.5 <= service_layer_prob_R < 0.75):
            S_N[3]=1
        else:
            S_N[3]=0

    #------------------------------------------------------------
    if(a_h==1):
        S_N[0]=1

        H_S_CV_U=0
        H_S_CV_U_A=0

        for r in range(UAV_T):
            H_S_CV_U = H_S_CV_U + (NO_VUs_UAV[r]/Max_U_UAV)

        H_S_CV_U_A=(H_S_CV_U/UAV_T)

        if(H_S_CV_U_A <= 0.5):
            H_S_CV_U_s=0
        elif(0.5 < H_S_CV_U_A <= 1):
            H_S_CV_U_s=1
        else:
            H_S_CV_U_s=2
        S_N[1]=H_S_CV_U_s


        S_N[2]=2

        nodes_with_service=0
        service_layer_prob_U=0
        nodes_with_service = sum(1 for services in service_allocation_UAV.values() if Req_Ser in services)
        service_layer_prob_U=(nodes_with_service/UAV_T)
        if(0 <= service_layer_prob_U < 0.25 ):
            S_N[3]=3
        elif(0.25 <= service_layer_prob_U < 0.5):
            S_N[3]=2
        elif(0.5 <= service_layer_prob_U < 0.75):
            S_N[3]=1
        else:
            S_N[3]=0


    #------------------------------------------------------------
    if(a_h==2):
        S_N[0]=2

        H_S_CV_H=0
        H_S_CV_H_A=0

        for r in range(HAP_T):
            H_S_CV_H = H_S_CV_H + (NO_VUs_HAP[r]/Max_U_HAP)

        H_S_CV_H_A=(H_S_CV_H/HAP_T)

        if(H_S_CV_H_A <= 0.5):
            H_S_CV_H_s=0
        elif(0.5 < H_S_CV_H_A <= 1):
            H_S_CV_H_s=1
        else:
            H_S_CV_H_s=2
        S_N[1]=H_S_CV_H_s


        S_N[2]=1

        nodes_with_service=0
        service_layer_prob_H=0
        nodes_with_service = sum(1 for services in service_allocation_HAP.values() if Req_Ser in services)
        service_layer_prob_H=(nodes_with_service/HAP_T)
        if(0 <= service_layer_prob_H < 0.25 ):
            S_N[3]=3
        elif(0.25 <= service_layer_prob_H < 0.5):
            S_N[3]=2
        elif(0.5 <= service_layer_prob_H < 0.75):
            S_N[3]=1
        else:
            S_N[3]=0

     #------------------------------------------------------------
    if(a_h==3):
        S_N[0]=3

        H_S_CV_L=0
        H_S_CV_L_A=0

        for r in range(LEO_T):
            H_S_CV_L = H_S_CV_L + (NO_VUs_LEO[r]/Max_U_LEO)

        H_S_CV_L_A=(H_S_CV_L/LEO_T)

        if(H_S_CV_L_A <= 0.5):
            H_S_CV_L_s=0
        elif(0.5 < H_S_CV_L_A <= 1):
            H_S_CV_L_s=1
        else:
            H_S_CV_L_s=2
        S_N[1]=H_S_CV_L_s


        S_N[2]=3

        nodes_with_service=0
        service_layer_prob_L=0
        nodes_with_service = sum(1 for services in service_allocation_LEO.values() if Req_Ser in services)
        service_layer_prob_L=(nodes_with_service/LEO_T)
        if(0 <= service_layer_prob_L < 0.25 ):
            S_N[3]=3
        elif(0.25 <= service_layer_prob_L < 0.5):
            S_N[3]=2
        elif(0.5 <= service_layer_prob_L < 0.75):
            S_N[3]=1
        else:
            S_N[3]=0
    indices = np.where((State_Space_H == S_N).all(axis=1))[0]
    row_index = indices[0]
    return S_N, row_index


# ======================================================================
# ==== Notebook CODE CELL 14 (verbatim) ====
# ======================================================================

#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index, rewards):
    if rewards[current_row_index, current_column_index] != 0:
        return False
    else:
        return True

#define a function that will choose a random, non-terminal starting location
def get_starting_location(environment_rows):
  #get a random row and column index
    current_row_index = np.random.randint(environment_rows)
    #current_column_index = np.random.randint(environment_columns)
    # while is_terminal_state(current_row_index, current_column_index, rewards):
    #     current_row_index = np.random.randint(environment_rows)
    #     current_column_index = np.random.randint(environment_columns)
    return current_row_index


# ======================================================================
# ==== Notebook CODE CELL 15 (verbatim) ====
# ======================================================================

def get_next_action(current_row_index, epsilon, Action_Space, q_values):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index])
    else: #choose a random action
        return random.choice(Action_Space)


# ======================================================================
# ==== Notebook CODE CELL 16 (verbatim) ====
# ======================================================================

def get_next_action_M(current_row_index, Action_id, epsilon, Action_Space, q_values):
    if np.random.random() < epsilon:
        q_values_local= q_values[:, Action_id]
        a_id=np.argmax(q_values_local[current_row_index])
        return Action_Space[a_id]
    else: #choose a random action
        return random.choice(Action_Space)


# ======================================================================
# ==== Notebook CODE CELL 17 (verbatim) ====
# ======================================================================

State_Space_M={}
R_S_N=[0,1,2,3]
S_S_N=[0, 1]
ST_S_N=[0, 1]


State_Space_M = np.zeros((len(R_S_N) * len(S_S_N) * len(ST_S_N), 3))
k = 0
for i1 in range(len(R_S_N)):
    for i2 in range(len(S_S_N)):
        for i3 in range(len(ST_S_N)):
            state = np.zeros(3)
            state[0] = R_S_N[i1]
            state[1] = S_S_N[i2]
            state[2] = ST_S_N[i3]
            State_Space_M[k] = state
            k += 1


# ======================================================================
# ==== Notebook CODE CELL 18 (verbatim) ====
# ======================================================================

def M_Action_Space( a_h, v_id, VN_RSU_asign, VN_UAV_asign, VN_HAP_asign, VN_LEO_asign):
    
    if(a_h==0):
        VN_EN_asign_local=VN_RSU_asign[v_id]
        non_zero_indices = np.nonzero(VN_EN_asign_local)[0]
        M_A = np.zeros((len(non_zero_indices), len(VN_EN_asign_local)))


        for i, index in enumerate(non_zero_indices):
            row = np.zeros_like(VN_EN_asign_local)
            row[index] = 1
            M_A[i] = row

    elif (a_h==1):
        VN_EN_asign_local=VN_UAV_asign[v_id]
        non_zero_indices = np.nonzero(VN_EN_asign_local)[0]
        M_A = np.zeros((len(non_zero_indices), len(VN_EN_asign_local)))


        for i, index in enumerate(non_zero_indices):
            row = np.zeros_like(VN_EN_asign_local)
            row[index] = 1
            M_A[i] = row

    elif (a_h==2):
        VN_EN_asign_local=VN_HAP_asign[v_id]
        non_zero_indices = np.nonzero(VN_EN_asign_local)[0]
        M_A = np.zeros((len(non_zero_indices), len(VN_EN_asign_local)))
   
        for i, index in enumerate(non_zero_indices):
            row = np.zeros_like(VN_EN_asign_local)
            row[index] = 1
            M_A[i] = row

    else:
        VN_EN_asign_local=VN_LEO_asign[v_id]
        non_zero_indices = np.nonzero(VN_EN_asign_local)[0]
        M_A = np.zeros((len(non_zero_indices), len(VN_EN_asign_local)))


        for i, index in enumerate(non_zero_indices):
            row = np.zeros_like(VN_EN_asign_local)
            row[index] = 1
            M_A[i] = row
    return M_A


# ======================================================================
# ==== Notebook CODE CELL 19 (verbatim) ====
# ======================================================================



def Next_State_M(
    a_h, a_m, VU_speed, Req_Ser, NO_VUs_RSU, Max_U_RSU, service_allocation_RSU, VN_RSU_Soj, RSU_r,
    NO_VUs_UAV, Max_U_UAV, service_allocation_UAV, VN_UAV_Soj, UAV_r,
    NO_VUs_HAP, Max_U_HAP, service_allocation_HAP, VN_HAP_Soj, HAP_r,
    NO_VUs_LEO, Max_U_LEO, service_allocation_LEO, VN_LEO_Soj, LEO_r):


    N_S_M=np.zeros(3)
    non_zero_indices = np.nonzero(a_m)

    if(a_h==0):
 
        CV_EN=(NO_VUs_RSU[non_zero_indices]/Max_U_RSU)

        Service=service_allocation_RSU[non_zero_indices[0][0]]
        Soj_T=VN_RSU_Soj[non_zero_indices]
   
        cov=RSU_r
        spd=VU_speed

    elif(a_h==1):

        CV_EN=(NO_VUs_UAV[non_zero_indices]/Max_U_UAV)
        Service=service_allocation_UAV[non_zero_indices[0][0]]
        Soj_T=VN_UAV_Soj[non_zero_indices]
        cov=UAV_r
        spd=VU_speed

    elif(a_h==2):

        CV_EN=(NO_VUs_HAP[non_zero_indices]/Max_U_HAP)
        Service=service_allocation_HAP[non_zero_indices[0][0]]
        Soj_T=VN_HAP_Soj[non_zero_indices]
        cov=HAP_r
        spd=VU_speed

    elif(a_h==3):
   
        CV_EN=(NO_VUs_LEO[non_zero_indices]/Max_U_LEO)
        Service=service_allocation_LEO[non_zero_indices[0][0]]
        Soj_T=VN_LEO_Soj[non_zero_indices]
        cov=LEO_r
        spd=VU_speed
    #-------------------------------------------- 
  
    if(0 <= CV_EN < 0.25):
        N_S_M[0]=0
    elif(0.25 <= CV_EN < 0.5):
        N_S_M[0]=1

    elif(0.5 <= CV_EN < 0.75):
        N_S_M[0]=2

    else:
        N_S_M[0]=3

    #-----------------------------------------     
    if Req_Ser in Service:
        N_S_M[1]=1
    else:
        N_S_M[1]=0    
    #--------------------------------------------       
    if(0 <= Soj_T < (cov/VU_speed) ):
        N_S_M[2]=0
    else:
        N_S_M[2]=1

    #----------------------------------------------
    indices = np.where((State_Space_M == N_S_M).all(axis=1))[0]
    row_index_m = indices[0]
    col_index_m = 0
    return N_S_M, row_index_m, col_index_m, Soj_T


# ======================================================================
# ==== Notebook CODE CELL 20 (verbatim) ====
# ======================================================================

delta = 0.25
State_Space_L={}
F1_N=[0,1]
F2_N=[0, 1]
F3_N=[0, 1]


State_Space_L = np.zeros((len(F1_N) * len(F2_N) * len(F3_N), 3))
k = 0
for i1 in range(len(F1_N)):
    for i2 in range(len(F2_N)):
        for i3 in range(len(F3_N)):
            state = np.zeros(3)
            state[0] = R_S_N[i1]
            state[1] = S_S_N[i2]
            state[2] = ST_S_N[i3]
            State_Space_L[k] = state
            k += 1

#---------Action Space------------------------
upper_bound = 1 + delta 
values = np.arange(0, upper_bound, delta)

values[values > 1] = 1            
            
Action_Space_L= values


# ======================================================================
# ==== Notebook CODE CELL 21 (verbatim) ====
# ======================================================================

#-----------Edit-----------------------------
def Learning_Cost(IP, DL_Req, T_sog_me, CR, DR_R_R, CU, DR_R_U, CH, DR_R_H, CL, DR_R_S, a_h, a_m, a_l):
    
#============Task Processing===============================================
    out=[]
    V=IP[0]
    RSU_T=IP[1]
    UAV_T=IP[4]
    HAP_T=IP[7]
    LEO_T=IP[10]
    TS=IP[31]
    TSD=IP[32]
    psi_dmp=IP[33]
    Pcomp_m=IP[34]
    Ptp_v=IP[35]
    Pcomp_r=IP[36]
    Ptp_r=IP[37]
    Pcomp_u=IP[38]
    Ptp_u=IP[39]
    Pcomp_h=IP[40]
    Ptp_h=IP[41]
    Pcomp_s=IP[42]
    Ptp_s=IP[43]
    Cm=IP[44]

    #--------Local Device Computation----------------------------

    TPcomp_v=0
    EPcomp_v=0

    TPcomp_v=((TS[0][v_id]*psi_dmp)/(Cm[0][v_id]))
    EPcomp_v=(Pcomp_m[0][v_id]*TPcomp_v)

        
    #-----------Offloading Time and Energy------------------
    TPcomp_r=0
    EPcomp_r=0
    TPcomp_u=0
    EPcomp_u=0
    TPcomp_h=0
    EPcomp_h=0
    TPcomp_s=0
    EPcomp_s=0
    
    TPcomU_vr=0
    EPcomU_vr=0
    TPcomD_vr=0
    EPcomD_vr=0
    TPcomU_rv=0
    EPcomU_rv=0
    TPcomD_rv=0
    EPcomD_rv=0
    
    TPcomU_vu=0
    EPcomU_vu=0
    TPcomD_vu=0
    EPcomD_vu=0
    TPcomU_uv=0
    EPcomU_uv=0
    TPcomD_uv=0
    EPcomD_uv=0
    
    TPcomU_vh=0
    EPcomU_vh=0
    TPcomD_vh=0
    EPcomD_vh=0
    TPcomU_hv=0
    EPcomU_hv=0
    TPcomD_hv=0
    EPcomD_hv=0

    TPcomU_vs=0
    EPcomU_vs=0
    TPcomD_vs=0
    EPcomD_vs=0
    TPcomU_sv=0
    EPcomU_sv=0
    TPcomD_sv=0
    EPcomD_sv=0
    

    

    if(a_h==0):
        for r in range(RSU_T):
   
            if(np.nonzero(a_m)==r):
                TPcomp_r=((TS[0][v_id]*psi_dmp)/(CR[v_id][r]))
                EPcomp_r=(Pcomp_r[0][r]*TPcomp_r)
                TPcomU_vr=((TS[0][v_id])/DR_R_R)
                EPcomU_vr=TPcomU_vr*Ptp_v[0][v_id]
                EPcomU_rv=TPcomU_vr*Ptp_r[0][r]
                TPcomD_vr=((TSD[0][v_id])/DR_R_R)
                EPcomD_vr=TPcomD_vr*Ptp_v[0][v_id]
                EPcomD_rv=TPcomD_vr*Ptp_r[0][r]
                    
                    
                    
    #------------Total Task Processing Time and Energy--------------\n    
    TP_offl_vr=0
    EP_offl_vr=0
    TP_offl_rv=0
    EP_offl_rv=0
    TP_offl_T_r=0
    EP_offl_T_r=0   

    TP_loc=0
    EP_loc=0
 
    if(a_h==0):
        for r in range(RSU_T):
            if(np.nonzero(a_m)==r):
                TP_offl_vr=TPcomU_vr  + TPcomD_vr
                EP_offl_vr=EPcomU_vr  + EPcomD_vr
                TP_offl_rv=TPcomp_r
                EP_offl_rv=EPcomp_r

                TP_offl_T_r=TP_offl_vr  + TP_offl_rv
                EP_offl_T_r=0.5*EP_offl_vr  + 0.5*EP_offl_rv

                TP_loc=TPcomp_v
                EP_loc=0.5*EPcomp_v                    

                

    if(a_h==1):
        for u in range(UAV_T):
            if(np.nonzero(a_m)==u):
                TPcomp_u=((TS[0][v_id]*psi_dmp)/(CU[v_id][u]))
                EPcomp_u=(Pcomp_u[0][u]*TPcomp_u)

                TPcomU_vu=((TS[0][v_id])/DR_R_U)
                EPcomU_vu=TPcomU_vu*Ptp_v[0][v_id]
                EPcomU_uv=TPcomU_vu*Ptp_u[0][u]
                TPcomD_vu=((TSD[0][v_id])/DR_R_U)
                EPcomD_vu=TPcomD_vu*Ptp_v[0][v_id]
                EPcomD_uv=TPcomD_vu*Ptp_u[0][u]
                    
                 
                    
    #------------Total Task Processing Time and Energy (UAV)--------------\n    
    TP_offl_vu=0
    EP_offl_vu=0
    TP_offl_uv=0
    EP_offl_uv=0
    TP_offl_T_u=0
    EP_offl_T_u=0 


    for v in range(V):
        if(a_h==1):
            for u in range(UAV_T):
                if(np.nonzero(a_m)==u):
                    TP_offl_vu=TPcomU_vu  + TPcomD_vu
                    EP_offl_vu=EPcomU_vu  + EPcomD_vu
                    TP_offl_uv=TPcomp_u
                    EP_offl_uv=EPcomp_u
                    
                    TP_offl_T_u=TP_offl_vu  + TP_offl_uv
                    EP_offl_T_u=0.5*EP_offl_vu  + 0.5*EP_offl_uv

                    TP_loc=TPcomp_v
                    EP_loc=0.5*EPcomp_v                    

                    
                

    if(a_h==2):
        for h in range(HAP_T):
            if(np.nonzero(a_m)==h):
                TPcomp_h=((TS[0][v_id]*psi_dmp)/(CH[v_id][h]))
                EPcomp_h=(Pcomp_h[0][h]*TPcomp_h)

                TPcomU_vh=((TS[0][v_id])/DR_R_H)
                EPcomU_vh=TPcomU_vh*Ptp_v[0][v_id]
                EPcomU_hv=TPcomU_vh*Ptp_h[0][h]
                TPcomD_vh=((TSD[0][v_id])/DR_R_H)
                EPcomD_vh=TPcomD_vh*Ptp_v[0][v_id]
                EPcomD_hv=TPcomD_vh*Ptp_h[0][h]
                    
    #------------Total Task Processing Time and Energy (HAP)--------------\n    
    TP_offl_vh=0
    EP_offl_vh=0
    TP_offl_hv=0
    EP_offl_hv=0
    TP_offl_T_h=0
    EP_offl_T_h=0 


    for v in range(V):
        if(a_h==2):
            for h in range(HAP_T):
                if(np.nonzero(a_m)==h):
                    TP_offl_vh=TPcomU_vh  + TPcomD_vh
                    EP_offl_vh=EPcomU_vh  + EPcomD_vh
                    TP_offl_hv=TPcomp_h
                    EP_offl_hv=EPcomp_h
                    
                    TP_offl_T_h=TP_offl_vh  + TP_offl_hv
                    EP_offl_T_h=0.5*EP_offl_vh  + 0.5*EP_offl_hv

                    TP_loc=TPcomp_v
                    EP_loc=0.5*EPcomp_v  
                    
                    

                    
                    

    if(a_h==3):
        for s in range(LEO_T):
            if(np.nonzero(a_m)==s):
                TPcomp_s=((TS[0][v_id]*psi_dmp)/(CL[v_id][s]))
                EPcomp_s=(Pcomp_s[0][s]*TPcomp_s)

                TPcomU_vs=((TS[0][v_id])/DR_R_S)
                EPcomU_vs=TPcomU_vs*Ptp_v[0][v_id]
                EPcomU_sv=TPcomU_vs*Ptp_s[0][s]
                TPcomD_vs=((TSD[0][v_id])/DR_R_S)
                EPcomD_vs=TPcomD_vs*Ptp_v[0][v_id]
                EPcomD_sv=TPcomD_vs*Ptp_s[0][s]
                    
    
                    
    #------------Total Task Processing Time and Energy (HAP)--------------\n    
    TP_offl_vs=0
    EP_offl_vs=0
    TP_offl_sv=0
    EP_offl_sv=0
    TP_offl_T_s=0
    EP_offl_T_s=0  



    if(a_h==3):
        for s in range(LEO_T):
            if(np.nonzero(a_m)==s):
                TP_offl_vs=TPcomU_vs  + TPcomD_vs
                EP_offl_vs=EPcomU_vs  + EPcomD_vs
                TP_offl_sv=TPcomp_s
                EP_offl_sv=EPcomp_s

                TP_offl_T_s=TP_offl_vs  + TP_offl_sv
                EP_offl_T_s=0.5*EP_offl_vs  + 0.5*EP_offl_sv

                TP_loc=TPcomp_v
                EP_loc=0.5*EPcomp_v  
                    
                    
                    
                    
                    
    T_L=0
    T_L_V=0
    T_L_E=0
    T_E=0
    T_E_V=0
    T_E_E=0
    F1=0
    F2=0
    F3=0
    

    if(a_h==0):
        for r in range(RSU_T):
            if(np.nonzero(a_m)==r):
                T_L=max((a_l*TP_offl_T_r), ((1-a_l)*TP_loc))
                T_L_V= (1-a_l)*TP_loc
                T_L_E= (a_l*TP_offl_T_r)
                T_E=((a_l*EP_offl_T_r)+ ((1-a_l)*EP_loc))
                T_E_V= (1-a_l)*EP_loc
                T_E_E= (a_l*EP_offl_T_r)
                F1= T_L_V - T_sog_me
                F2= T_L - DL_Req
                F3= T_L_E - EP_loc
                
    if(a_h==1):
        for u in range(UAV_T):
            if(np.nonzero(a_m)==u):
                T_L=max((a_l*TP_offl_T_u), ((1-a_l)*TP_loc))
                T_E=((a_l*EP_offl_T_u)+ ((1-a_l)*EP_loc))  
                T_L_V= (1-a_l)*TP_loc
                T_L_E= (a_l*TP_offl_T_u)
                T_E_V= (1-a_l)*EP_loc
                T_E_E= (a_l*EP_offl_T_u)
                F1= T_L_V - T_sog_me
                F2= T_L - DL_Req
                F3= T_L_E - EP_loc        

    if(a_h==2):
        for h in range(HAP_T):
            if(np.nonzero(a_m)==h):
                T_L=max((a_l*TP_offl_T_h), ((1-a_l)*TP_loc))
                T_E=((a_l*EP_offl_T_h)+ ((1-a_l)*EP_loc))  
                T_L_V= (1-a_l)*TP_loc
                T_L_E= (a_l*TP_offl_T_h)
                T_E_V= (1-a_l)*EP_loc
                T_E_E= (a_l*EP_offl_T_h)
                F1= T_L_V - T_sog_me
                F2= T_L - DL_Req
                F3= T_L_E - EP_loc                    

    if(a_h==3):
        for s in range(LEO_T):
            if(np.nonzero(a_m)==s):
                T_L=max((a_l*TP_offl_T_s), ((1-a_l)*TP_loc))
                T_E=((a_l*EP_offl_T_s)+ ((1-a_l)*EP_loc))  
                T_L_V= (1-a_l)*TP_loc
                T_L_E= (a_l*TP_offl_T_s)
                T_E_V= (1-a_l)*EP_loc
                T_E_E= (a_l*EP_offl_T_s)
                F1= T_L_V - T_sog_me
                F2= T_L - DL_Req
                F3= T_L_E - EP_loc
                    
                    
    out.append(T_L)
    out.append(T_E)
    out.append(T_L_V)
    out.append(T_L_E)
    out.append(T_E_V)
    out.append(T_E_E)
    out.append(F1)
    out.append(F2)
    out.append(F3)
          
    return out


# ======================================================================
# ==== Notebook CODE CELL 9 — phần NETWORK CONSTANTS (verbatim, tới trước V=200) ====
# ======================================================================

gamma_1=0.5
gamma_2=0.5
RSU_r=50
UAV_r=300
HAP_r=1000
LEO_r=3000
RSU_T=int((1)*LEO_r/RSU_r) #Total RSU
UAV_T=int((1)*LEO_r/UAV_r) #Total UAV
HAP_T=int((1.5)*LEO_r/HAP_r) #Total HAP
LEO_T=1

B_RSU=(20*10**6)*np.ones((1, RSU_T))
B_UAV=(20*10**6)*np.ones((1, UAV_T))
B_HAP=(50*10**6)*np.ones((1, HAP_T))
B_LEO=(100*10**6)*np.ones((1, LEO_T))
C_RSU=(20*10**9)*np.ones((1, RSU_T))
C_UAV=(20*10**9)*np.ones((1, UAV_T))
C_HAP=(40*10**9)*np.ones((1, HAP_T))
C_LEO=(60*10**9)*np.ones((1, LEO_T))
UAV_alt=1000
HAP_alt=100000
LEO_alt=200000
services = [1, 2, 3, 4, 5 ,6]
services_per_RSU=3
services_per_UAV=3
services_per_HAP=4
services_per_LEO=6

service_allocation_RSU = {}
service_allocation_UAV = {}
service_allocation_HAP = {}
service_allocation_LEO = {}

for node_id in range(RSU_T):
    allocated_services = random.sample(services, services_per_RSU)
    service_allocation_RSU[node_id] = allocated_services
    
for node_id in range(UAV_T):
    allocated_services = random.sample(services, services_per_UAV)
    service_allocation_UAV[node_id] = allocated_services

for node_id in range(HAP_T):
    allocated_services = random.sample(services, services_per_HAP)
    service_allocation_HAP[node_id] = allocated_services

for node_id in range(LEO_T):
    allocated_services = random.sample(services, services_per_LEO)
    service_allocation_LEO[node_id] = allocated_services

RSU_loc, UAV_loc, HAP_loc, LEO_loc=Loc_Fun(RSU_T,UAV_T,HAP_T,LEO_T,RSU_r,UAV_r,HAP_r,LEO_r,LEO_alt,HAP_alt,UAV_alt) #---define locations---
Pcomp_r=(1.3)*np.ones((1, RSU_T))
Pcomp_u=(1.2)*np.ones((1, UAV_T))
Pcomp_h=(1.1)*np.ones((1, HAP_T))
Pcomp_s=(1.1)*np.ones((1, LEO_T))
P_tx_r=(1.3)*np.ones((1, RSU_T))
P_tx_u=(1.2)*np.ones((1, UAV_T))
P_tx_h=(1.1)*np.ones((1, HAP_T))
P_tx_s=(1.1)*np.ones((1, LEO_T))
P_rx_r=(1.1)*np.ones((1, RSU_T))
P_rx_u=(1)*np.ones((1, UAV_T))
P_rx_h=(0.9)*np.ones((1, HAP_T))
P_rx_s=(0.9)*np.ones((1, LEO_T))
b_0=1
theta=3
