# Plotting the location and type of pickup coil

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Location of Coils
pickup_coil_R = [6.5897,6.0102,5.5287,5.2195,5.13,5.131,5.131,5.131,5.131,5.13,5.13,5.13,5.13,5.13,5.13,5.13,5.13,5.13,5.13,5.13,5.129,5.1349,5.2819,5.5726,5.9859,6.4956,7.066,7.6583,8.2335,8.7838,9.3166,9.8271,10.3147,10.7765,11.2071,11.6005,11.9483,12.2497,12.5007,12.7012,12.8475,12.9404,12.9768,12.9326,12.8115,12.6252,12.452,12.1421,11.7929,11.4426,11.0946,10.7443,10.3963,10.048,9.632,9.2877,8.8784,8.3521,7.769,7.1807]
pickup_coil_Z = [-7.23,-7.0665,-6.7401,-6.2513,-5.669,-5.0601,-4.444,-3.829,-3.2129,-2.5969,-1.982,-1.366,-0.75,-0.135,0.481,1.097,1.712,2.328,2.944,3.5591,4.1747,4.7796,5.355,5.8717,6.2966,6.5988,6.757,6.759,6.6006,6.3476,6.0505,5.7175,5.3507,4.9514,4.5208,4.0561,3.5582,3.0296,2.4762,1.9021,1.3112,0.7105,0.1055,-0.497,-1.0907,-1.6682,-2.2596,-2.7932,-3.3004,-3.8065,-4.3145,-4.8203,-5.3283,-5.8354,-6.2968,-6.7994,-7.226,-7.4763,-7.5203,-7.387]

def Limiter_from_EQ(shotname: str) -> tuple[np.ndarray[float], np.ndarray[float]]:
   
    # Load the matlab file FILEPATH TO DATABASE you need to change this
    eq = loadmat('E:\\FUSION-EP\\M1 - Universitiet Gent\\Bayesian-current-tomography-CCP\\database\\' + shotname)

    # Return the R and Z coordinates of the limiter
    return (eq['limiter'][0][0][0][0], eq['limiter'][0][0][1][0])

wall_coords = Limiter_from_EQ(shotname = 'Equil_1_li_0d8_beta_0d1_CNL4E.mat')

# These are the index arrays for tangential and normal. Make sure they have commas!!!!!!!
Coil_T = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,34,35,42,43,49,54,55,56,57,58,59] # Array for Index, Tangential Coils
Coil_N = [31,32,36,37,38,39,40,41,44,45,46,47,48,50,51,52,53] # Array for Index, Normal Coils 

R_T = [pickup_coil_R[i] for i in Coil_T]
Z_T = [pickup_coil_Z[i] for i in Coil_T]

R_N = [pickup_coil_R[i] for i in Coil_N]
Z_N = [pickup_coil_Z[i] for i in Coil_N]

fig = plt.figure()
axes = plt.gca()
axes.plot(wall_coords[0], wall_coords[1], '-', c='black')
axes.scatter(pickup_coil_R, pickup_coil_Z, 3**2, marker='s', c='black', label='Available Coil Position')
axes.scatter(R_T, Z_T, s=3**2, c='red', marker='s', label='Tangential Coils')
axes.scatter(R_N, Z_N, s=3**2, c='green', marker='s', label='Normal Coils')
axes.set_title("Location and Type of Pickup Coil", fontsize = 12)
axes.set_xlabel('R [m]', fontsize = 12)
axes.set_ylabel('Z [m]', fontsize = 12)
axes.set_aspect('equal')
axes.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
fig.subplots_adjust(right=0.78)
plt.show()
