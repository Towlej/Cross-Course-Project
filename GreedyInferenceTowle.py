#!/usr/bin/env python
import os
import sys

# Add script directory to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Optional: verify
print("CWD before:", os.getcwd())

# Only change directory if you REALLY need relative file paths
os.chdir("/home/chocamo/projects/Cross_Course")

print("CWD after:", os.getcwd())
import numpy as np

# Meant as file cointaining different inferences/experiments that can be performed

from helpfunctions import MI_gain, Load_180_shots_dict, Meshgrids_from_EQ, Load_diag_pos, CalculatePlasmaParameters, PlasmaWallGapsErrors, PlasmaWallGapsFM, Downsample_meshgrid, XpointFluxFM, D_opt_covariance
from PickupCoil import DEMOPickupCoil
from FluxLoop import DEMOFluxLoop
from SaddleCoil import DEMOSaddleCoil
from ASEPosterior import ASEPosterior
from Structures import Structures
import time
import os


def CurrentTomInference(shotname: str, rng_number: int):
    """
        Code to perform one current tomography inference and present the results
        : params: The name of the EQuilibrium, an rng_number
    """

    # Load dict containing diagnostic positions
    dict_diag_pos = Load_diag_pos()
    saddle_coil_R, saddle_coil_Z = dict_diag_pos["saddle_coil_R"], dict_diag_pos["saddle_coil_Z"]

    # Create Structures class and set currents
    Structures_shot = Structures(EQ_shots_dict[shotname]["structs_current"])

    #-------------------------------------------------------------------

    # Calculate non centered covariance
    cov_matrix = (All_I_tom.T @ All_I_tom) / (All_I_tom.shape[0] - 1)

    # Add a small value to diagonal terms to ensure the positive definiteness
    cov_matrix[np.diag_indices_from(cov_matrix)] += 1e-1

    # Filter out points outside of limiter
    cov = cov_matrix[inner_points_bool][:, inner_points_bool]

    #-------------------------------------------------------------------
    # Create instance of tan pickupcoils
    TanPickupCoil = DEMOPickupCoil()

    # The stored data is for all 60 available pick-up coil positions, in case of current DEMO design, for tan coils these are all even locations (0, 2, 4...)
    Tan_mask = np.array([True, False]*30)

    # Load data with default uncertainties
    diag_tan_coils_current, diag_tan_coils_struct = EQ_shots_dict[shotname]["diag_tan_coils_current"][Tan_mask], EQ_shots_dict[shotname]["diag_tan_coils_struct"][Tan_mask]
    TanPickupCoil.load_data(diag_tan_coils_current + diag_tan_coils_struct)
    # Add noise
    TanPickupCoil.add_noise(rng_number)
    # Add forward model
    response_tan_matrix = np.load("Npy files/response_tan_matrix.npy", allow_pickle=True)
    TanPickupCoil.set_forward_model(response_tan_matrix)

    #-------------------------------------------------------------------
    # Create instance of norm pickupcoils
    NormPickupCoil = DEMOPickupCoil(Is_Tan = False)

    # The stored data is for all 60 available pick-up coil positions, in case of current DEMO design, for norm coils these are all uneven locations (1, 3, 5...)
    Norm_mask = np.array([False, True]*30)

    # Load data with default uncertainties
    diag_norm_coils_current, diag_norm_coils_struct = EQ_shots_dict[shotname]["diag_norm_coils_current"][Norm_mask], EQ_shots_dict[shotname]["diag_norm_coils_struct"][Norm_mask]
    NormPickupCoil.load_data(diag_norm_coils_current + diag_norm_coils_struct)
    # Add noise
    NormPickupCoil.add_noise(rng_number)
    # Add forward model
    response_norm_matrix = np.load("Npy files/response_norm_matrix.npy", allow_pickle=True)
    NormPickupCoil.set_forward_model(response_norm_matrix)

    #-------------------------------------------------------------------
    #Create instance of fluxloops
    FluxLoop = DEMOFluxLoop()
    # Load data with default uncertainties
    diag_flux_loops_current, diag_flux_loops_struct = EQ_shots_dict[shotname]["diag_flux_loops_current"], EQ_shots_dict[shotname]["diag_flux_loops_struct"]
    FluxLoop.load_data(diag_flux_loops_current + diag_flux_loops_struct)
    # Add noise
    FluxLoop.add_noise(rng_number)
    # Add forward model
    response_fluxloop_matrix = np.load("Npy files/response_fluxloop_matrix.npy", allow_pickle=True)
    FluxLoop.set_forward_model(response_fluxloop_matrix)

    #-------------------------------------------------------------------
    #Create instance of saddlecoils
    SaddleCoil = DEMOSaddleCoil()

    # Explicitely needs to know the locations in order to calculate the uncertainties
    SaddleCoil.set_positions(saddle_coil_R, saddle_coil_Z)

    # Load data with default uncertainties
    diag_saddle_coils_current, diag_saddle_coils_struct = EQ_shots_dict[shotname]["diag_saddle_coils_current"], EQ_shots_dict[shotname]["diag_saddle_coils_struct"]
    SaddleCoil.load_data(diag_saddle_coils_current + diag_saddle_coils_struct)
    # Add noise
    SaddleCoil.add_noise(rng_number)
    # Add forward model
    response_saddlecoil_matrix = np.load("Npy files/response_saddlecoil_matrix.npy", allow_pickle=True)
    SaddleCoil.set_forward_model(response_saddlecoil_matrix)

    #-------------------------------------------------------------------
    # Create instance of posterior class
    Post = ASEPosterior()

    # Set the mask
    Post.set_mask(inner_points_bool)

    # Set the diagnostic measurements comming from the structures
    Post.set_struct_measurements(diag_tan_coils_struct, diag_norm_coils_struct, diag_flux_loops_struct, diag_saddle_coils_struct)
    # Set the noisy diagnostic measurements (both struct and current)
    Post.set_diag_measurements(TanPickupCoil.data_noise, NormPickupCoil.data_noise, FluxLoop.data_noise, SaddleCoil.data_noise)
    # Set the uncertainties on these measurements
    Post.set_diag_measurements_unc(TanPickupCoil.unc, NormPickupCoil.unc, FluxLoop.unc, SaddleCoil.unc) 
    # Set the forward model on the diagnostics
    Post.set_forward_models(TanPickupCoil.forward_model_matrix, NormPickupCoil.forward_model_matrix, FluxLoop.forward_model_matrix, SaddleCoil.forward_model_matrix)

    # Set the prior
    Post.set_prior(cov)

    # Calculate the posterior
    Post.calculate_posterior()

    # Extract and reshape the mean posterior
    post_mean_tom = Post.expanded_mean.reshape((rVec.shape[0], rVec.shape[1]))
    # Extract the covariance
    expanded_cov = Post.expanded_cov

    # Print inverse determinant of expanded_cov (for BED)
    # Important to only use inner points, as otherwise you will get eigenvalues with value zero
    print(D_opt_covariance(expanded_cov[inner_points_bool][:, inner_points_bool]))
    print("MI GAIN")
    print(MI_gain(Post.expanded_cov[inner_points_bool][:, inner_points_bool], Post.prior_cov))
    # Calculate plasma parameters of interest
    Ip_inf, Zc_inf, Rc_inf, Ip_inf_std, Zc_inf_std, Rc_inf_std = CalculatePlasmaParameters(post_mean_tom, expanded_cov, rVec, zVec)

    # Give the infered plasma parameters and their std's
    print("Plasma current and std: {} ± {}".format(np.format_float_scientific(Ip_inf, precision = 3), np.format_float_scientific(Ip_inf_std, precision=3)))
    print("Z-centroid and std: {} ± {}".format(np.format_float_scientific(Zc_inf, precision = 3), np.format_float_scientific(Zc_inf_std, precision=3)))
    print("R-centroid and std: {} ± {}".format(np.format_float_scientific(Rc_inf, precision = 3), np.format_float_scientific(Rc_inf_std, precision=3)))

    # Give the relevant errors on the inferred plasma parameters
    Ip_, Zc_, Rc_ = EQ_shots_dict[shotname]['total_plasma_current'], EQ_shots_dict[shotname]['current_centroid_Z'], EQ_shots_dict[shotname]['current_centroid_R']

    # Give the relevant errors of the infered plasma parameters
    print("Current percentage error: ", (Ip_inf - Ip_)/Ip_*100)
    print("Z-centroid cm error: ", (Zc_inf - Zc_)*100)
    print("R-centroid cm error: ", (Rc_inf - Rc_)*100)

    # Calculate Flux at Xpoint
    XpointFlux, XpointFluxerror = XpointFluxFM(post_mean_tom, expanded_cov, rVec, zVec, XpointFMs, Structures_shot)
    print(XpointFlux, XpointFluxerror)
    
    # Calculate the six plasma-wall gap positions using precalculated forward models
    Gap_positions_FMs, Gap_positions_std_FMs = PlasmaWallGapsFM(post_mean_tom, expanded_cov, Structures_shot, XpointFlux, XpointFluxerror, plasmawallgapFMs)
    print(Gap_positions_FMs)
    print(Gap_positions_std_FMs)
    Gap_positions_FMs_error = PlasmaWallGapsErrors(Gap_positions_FMs, EQ_shots_dict[shotname]["gap_positions"])
    print(Gap_positions_FMs_error)

def CurrentTomInferenceZoom(shotname: str, rng_number: int):

    # =========================
    # 1. LOAD BASIC STRUCTURES
    # =========================

    dict_diag_pos = Load_diag_pos()
    saddle_coil_R, saddle_coil_Z = dict_diag_pos["saddle_coil_R"], dict_diag_pos["saddle_coil_Z"]

    Structures_shot = Structures(EQ_shots_dict[shotname]["structs_current"])

    # =========================
    # PRIOR COVARIANCE
    # =========================

    cov_matrix = (All_I_tom.T @ All_I_tom) / (All_I_tom.shape[0] - 1)
    cov_matrix[np.diag_indices_from(cov_matrix)] += 1e-1
    cov = cov_matrix[inner_points_bool][:, inner_points_bool]

    # =========================
    # DIAGNOSTIC DATA (NO MASKING)
    # =========================

    diag_tan_coils_current = EQ_shots_dict[shotname]["diag_tan_coils_current"]
    diag_tan_coils_struct  = EQ_shots_dict[shotname]["diag_tan_coils_struct"]

    diag_norm_coils_current = EQ_shots_dict[shotname]["diag_norm_coils_current"]
    diag_norm_coils_struct  = EQ_shots_dict[shotname]["diag_norm_coils_struct"]

    diag_flux_loops_current = EQ_shots_dict[shotname]["diag_flux_loops_current"]
    diag_flux_loops_struct  = EQ_shots_dict[shotname]["diag_flux_loops_struct"]

    diag_saddle_coils_current = EQ_shots_dict[shotname]["diag_saddle_coils_current"]
    diag_saddle_coils_struct  = EQ_shots_dict[shotname]["diag_saddle_coils_struct"]

    # =========================
    # FORWARD MODELS (NO MASKING HERE)
    # =========================
    
    response_tan_matrix = np.load("Npy files/response_tan_matrix_zoom_BED.npy", allow_pickle=True)
    response_norm_matrix = np.load("Npy files/response_norm_matrix_zoom_BED.npy", allow_pickle=True)
    response_fluxloop_matrix = np.load("Npy files/response_fluxloop_matrix_zoom_BED.npy", allow_pickle=True)
    response_saddlecoil_matrix = np.load("Npy files/response_saddlecoil_matrix_zoom_BED.npy", allow_pickle=True)


    # GREEDY START
    # 0 = tangential, 1 = normal


    config = np.zeros(60, dtype=int)
    flipped = np.zeros(60, dtype=bool)

    config_MI = -np.inf
    MI_threshold = 1e-6

    while True:

        # Track best candidate in this iteration
        best_trial_MI = config_MI
        best_trial_config = config.copy()
        best_trial_index = None

        # Try flipping each coil
        for i in range(60):

            if flipped[i]:
                continue

            # ----- create trial config -----
            trial_config = config.copy()
            trial_config[i] = 1 - trial_config[i]

            Tan_mask = (trial_config == 0)
            Norm_mask = (trial_config == 1)

            if np.sum(Tan_mask) == 0 or np.sum(Norm_mask) == 0:
                continue
            # ----- build diagnostics -----
            TanPickupCoil = DEMOPickupCoil()
            TanPickupCoil.load_data(
                diag_tan_coils_current[Tan_mask] +
                diag_tan_coils_struct[Tan_mask]
            )
            TanPickupCoil.add_noise(rng_number)
            TanPickupCoil.set_forward_model(response_tan_matrix[Tan_mask, :])

            NormPickupCoil = DEMOPickupCoil(Is_Tan=False)
            NormPickupCoil.load_data(
                diag_norm_coils_current[Norm_mask] +
                diag_norm_coils_struct[Norm_mask]
            )
            NormPickupCoil.add_noise(rng_number)
            NormPickupCoil.set_forward_model(response_norm_matrix[Norm_mask, :])

            FluxLoop = DEMOFluxLoop()
            FluxLoop.load_data(diag_flux_loops_current + diag_flux_loops_struct)
            FluxLoop.add_noise(rng_number)
            FluxLoop.set_forward_model(response_fluxloop_matrix)

            SaddleCoil = DEMOSaddleCoil()
            SaddleCoil.set_positions(saddle_coil_R, saddle_coil_Z)
            SaddleCoil.load_data(diag_saddle_coils_current + diag_saddle_coils_struct)
            SaddleCoil.add_noise(rng_number)
            SaddleCoil.set_forward_model(response_saddlecoil_matrix)

            # ----- posterior -----
            Post = ASEPosterior()
            Post.set_mask(inner_points_bool)

            Post.set_struct_measurements(
                diag_tan_coils_struct[Tan_mask],
                diag_norm_coils_struct[Norm_mask],
                diag_flux_loops_struct,
                diag_saddle_coils_struct
            )

            Post.set_diag_measurements(
                TanPickupCoil.data_noise,
                NormPickupCoil.data_noise,
                FluxLoop.data_noise,
                SaddleCoil.data_noise
            )

            Post.set_diag_measurements_unc(
                TanPickupCoil.unc,
                NormPickupCoil.unc,
                FluxLoop.unc,
                SaddleCoil.unc
            )

            Post.set_forward_models(
                TanPickupCoil.forward_model_matrix,
                NormPickupCoil.forward_model_matrix,
                FluxLoop.forward_model_matrix,
                SaddleCoil.forward_model_matrix
            )

            Post.set_prior(cov)



            Post.calculate_posterior()

            expanded_cov = Post.expanded_cov

            MI = MI_gain(
                expanded_cov[inner_points_bool][:, inner_points_bool], Post.prior_cov)
            

            print("| Best trial MI: {best_trial_MI}")
            print(f"Remaining candidates: {np.sum(~flipped)}")
            # ----- track best trial -----
            if MI > best_trial_MI:
                best_trial_MI = MI
                best_trial_config = trial_config
                best_trial_index = i

        # ----- accept or stop -----
        if best_trial_index is not None and best_trial_MI > config_MI + MI_threshold:

            config = best_trial_config
            config_MI = best_trial_MI

            flipped[best_trial_index] = True

        else:
            break

    
        
    # =========================
    # FINAL POSTERIOR (USING OPTIMIZED CONFIG)
    # =========================

    Tan_mask = (config == 0)
    Norm_mask = (config == 1)

    TanPickupCoil = DEMOPickupCoil()
    TanPickupCoil.load_data(
        diag_tan_coils_current[Tan_mask] +
        diag_tan_coils_struct[Tan_mask]
    )
    TanPickupCoil.add_noise(rng_number)
    TanPickupCoil.set_forward_model(response_tan_matrix[Tan_mask, :])

    NormPickupCoil = DEMOPickupCoil(Is_Tan=False)
    NormPickupCoil.load_data(
        diag_norm_coils_current[Norm_mask] +
        diag_norm_coils_struct[Norm_mask]
    )
    NormPickupCoil.add_noise(rng_number)
    NormPickupCoil.set_forward_model(response_norm_matrix[Norm_mask, :])

    FluxLoop = DEMOFluxLoop()
    FluxLoop.load_data(diag_flux_loops_current + diag_flux_loops_struct)
    FluxLoop.add_noise(rng_number)
    FluxLoop.set_forward_model(response_fluxloop_matrix)

    SaddleCoil = DEMOSaddleCoil()
    SaddleCoil.set_positions(saddle_coil_R, saddle_coil_Z)
    SaddleCoil.load_data(diag_saddle_coils_current + diag_saddle_coils_struct)
    SaddleCoil.add_noise(rng_number)
    SaddleCoil.set_forward_model(response_saddlecoil_matrix)

    Post = ASEPosterior()

    Post.set_mask(inner_points_bool)

    Post.set_struct_measurements(
        diag_tan_coils_struct[Tan_mask],
        diag_norm_coils_struct[Norm_mask],
        diag_flux_loops_struct,
        diag_saddle_coils_struct
    )

    Post.set_diag_measurements(
        TanPickupCoil.data_noise,
        NormPickupCoil.data_noise,
        FluxLoop.data_noise,
        SaddleCoil.data_noise
    )

    Post.set_diag_measurements_unc(
        TanPickupCoil.unc,
        NormPickupCoil.unc,
        FluxLoop.unc,
        SaddleCoil.unc
    )

    Post.set_forward_models(
        TanPickupCoil.forward_model_matrix,
        NormPickupCoil.forward_model_matrix,
        FluxLoop.forward_model_matrix,
        SaddleCoil.forward_model_matrix
    )

    Post.set_prior(cov)
    Post.calculate_posterior()

    post_mean_tom = Post.expanded_mean.reshape((rVec.shape[0], rVec.shape[1]))
    expanded_cov = Post.expanded_cov

    # =========================
    # FINAL COIL CONFIG SUMMARY
    # =========================

    Tan_mask = (config == 0)
    Norm_mask = (config == 1)

    num_tan = np.sum(Tan_mask)
    num_norm = np.sum(Norm_mask)

    print("\n===== FINAL COIL CONFIGURATION =====")
    print(f"Total coils: {len(config)}")
    print(f"Tangential coils: {num_tan}")
    print(f"Normal coils: {num_norm}")

    print("\nTangential coil indices:")
    print(np.where(Tan_mask)[0])

    print("\nNormal coil indices:")
    print(np.where(Norm_mask)[0])

    print(f"Accepted flip: coil {best_trial_index}")
    print(f"Flipped array: {flipped}")
    # =========================
    # PLASMA PARAMETERS
    # =========================

    Ip_inf, Zc_inf, Rc_inf, Ip_inf_std, Zc_inf_std, Rc_inf_std = CalculatePlasmaParameters(
        post_mean_tom, expanded_cov, rVec, zVec, zoom_factor, zoom_factor
    )

    print("Plasma current and std: {} ± {}".format(
        np.format_float_scientific(Ip_inf, precision=3),
        np.format_float_scientific(Ip_inf_std, precision=3)
    ))

    print("Z-centroid and std: {} ± {}".format(
        np.format_float_scientific(Zc_inf, precision=3),
        np.format_float_scientific(Zc_inf_std, precision=3)
    ))

    print("R-centroid and std: {} ± {}".format(
        np.format_float_scientific(Rc_inf, precision=3),
        np.format_float_scientific(Rc_inf_std, precision=3)
    ))

def Allinferencezoom():
    
    # Set output directory
    output_dir = "Results/alldiag_zoom_uncentcov_6fold_order/"

    # Define the content format
    content_template = """\
    Plasma current and std
    {plasma_current} pm {plasma_current_std}
    Z-centroid and std
    {z_centroid} pm {z_centroid_std}
    R-centroid and std
    {r_centroid} pm {r_centroid_std}
    Current percentage error
    {current_percentage_error}
    Z-centroid cm error
    {z_centroid_cm_error}
    R-centroid cm error
    {r_centroid_cm_error}
    Plasma gap positions and std
    {plasma_gap_positions}
    {plasma_gap_positions_std}
    Total plasma gap position errors (m)
    {total_plasma_gap_position_errors}
    """

    # Number of splits for kfold validation
    num_splits = 6
    
    # 20x20 cm inference
    # Load file containing data etc.
    EQ_shots_dict = Load_180_shots_dict()

    # Load the inner points mask
    inner_points_bool = np.load('Npy files/' + 'inner_points_bool_vacuum_vessel_zoom.npy', allow_pickle=True)

    # Load file containing FM's to calculate plasma-wall gap positions 
    plasmawallgapFMs = dict(np.load("Npy files/plasmawallgapFMszoom.npz"))

    # Load file containing FM's to calculate Xpoint
    XpointFMs = dict(np.load("Npy files/XpointFMszoom.npz"))    

    # Get meshes for coordinates
    rVec, zVec = Meshgrids_from_EQ("Equil_1_li_0d8_beta_0d1_CNL4E.mat")

    # Downscale the meshgrids
    zoom_factor: int = 10 
    rVec, zVec = Downsample_meshgrid(rVec, zoom_factor, zoom_factor), Downsample_meshgrid(zVec, zoom_factor, zoom_factor)

    # Load dict containing diagnostic positions
    dict_diag_pos = Load_diag_pos()
    saddle_coil_R, saddle_coil_Z = dict_diag_pos["saddle_coil_R"], dict_diag_pos["saddle_coil_Z"]    
    #-------------------------------------------------------------------

    # Using a "non-centered covariance" (so can use zero prior mean)
    All_I_tom = np.load('Npy files/' + '180_I_tom_zoom.npy', allow_pickle=True)

    total_elements = All_I_tom.shape[0]  # Total number of elements (e.g., 180)
    elements_per_fold = total_elements // num_splits  # Elements per fold (e.g., 30 if num_splits is 6)

    # Array to save all possible covariances
    cov_fold = np.empty((num_splits, All_I_tom.shape[1], All_I_tom.shape[1]))

    for fold_id in range(num_splits):
        # Start index for this fold
        start_idx = fold_id * elements_per_fold
        
        # Indices to keep (exclude the current fold)
        keep_indices = np.concatenate([np.arange(0, start_idx), np.arange(start_idx + elements_per_fold, total_elements)])
        
        # Filter the array
        filtered_array = All_I_tom[keep_indices]
        
        # Calculate non centered covariance
        cov_matrix = (filtered_array.T @ filtered_array) / (filtered_array.shape[0] - 1)

        # Add a small value to diagonal terms to ensure the positive definiteness
        cov_matrix[np.diag_indices_from(cov_matrix)] += 1e-1

        cov_fold[fold_id] = cov_matrix

    # Filter out points outside of limiter
    cov_fold = cov_fold[:, inner_points_bool, :][:, :, inner_points_bool]
    
    #-------------------------------------------------------------------

    # Create instance of tan pickupcoils
    TanPickupCoil = DEMOPickupCoil()

    # The stored data is for all 60 available pick-up coil positions, in case of current DEMO design, for tan coils these are all even locations (0, 2, 4...)
    Tan_mask = np.array([True, False]*30)

    # Add forward model
    response_tan_matrix = np.load("Npy files/response_tan_matrix_zoom.npy", allow_pickle=True)
    TanPickupCoil.set_forward_model(response_tan_matrix[Tan_mask, :])
    #-------------------------------------------------------------------

    # Create instance of norm pickupcoils
    NormPickupCoil = DEMOPickupCoil(Is_Tan = False)

    # The stored data is for all 60 available pick-up coil positions, in case of current DEMO design, for norm coils these are all uneven locations (1, 3, 5...)
    Norm_mask = np.array([False, True]*30)

    # Add forward model
    response_norm_matrix = np.load("Npy files/response_norm_matrix_zoom.npy", allow_pickle=True)
    NormPickupCoil.set_forward_model(response_norm_matrix[Norm_mask, :])
    #-------------------------------------------------------------------

    #Create instance of fluxloops
    FluxLoop = DEMOFluxLoop()

    # Add forward model
    response_fluxloop_matrix = np.load("Npy files/response_fluxloop_matrix_zoom.npy", allow_pickle=True)
    FluxLoop.set_forward_model(response_fluxloop_matrix)
    #-------------------------------------------------------------------

    #Create instance of saddlecoils
    SaddleCoil = DEMOSaddleCoil()

    # Explicitely needs to know the locations in order to calculate the uncertainties
    SaddleCoil.set_positions(saddle_coil_R, saddle_coil_Z)

    # Add forward model
    response_saddlecoil_matrix = np.load("Npy files/response_saddlecoil_matrix_zoom.npy", allow_pickle=True)
    SaddleCoil.set_forward_model(response_saddlecoil_matrix)
    #-------------------------------------------------------------------

    # Create instance of posterior class
    Post = ASEPosterior()
    # Set the mask
    Post.set_mask(inner_points_bool)
    # Set the forward model on the diagnostics
    Post.set_forward_models(TanPickupCoil.forward_model_matrix, NormPickupCoil.forward_model_matrix, FluxLoop.forward_model_matrix, SaddleCoil.forward_model_matrix)

    # Loop over all EQ's
    for i, shotname in enumerate(EQ_shots_dict):
        print(i)
        time_shot = time.time()

        # Create Structures class and set currents
        Structures_shot = Structures(EQ_shots_dict[shotname]["structs_current"])

        # Get fold id of this shot (either based on current centroids, or just order of names)
        #fold_id = EQ_shots_dict[shotname]["current_centroid_6foldcluster"]
        fold_id = i//30

        # Set the prior
        Post.set_prior(cov_fold[fold_id])
        #-------------------------------------------------------------------

        # Load data with default uncertainties
        diag_tan_coils_current, diag_tan_coils_struct = EQ_shots_dict[shotname]["diag_tan_coils_current"][Tan_mask], EQ_shots_dict[shotname]["diag_tan_coils_struct"][Tan_mask]
        TanPickupCoil.load_data(diag_tan_coils_current + diag_tan_coils_struct)
        #-------------------------------------------------------------------

        # Load data with default uncertainties
        diag_norm_coils_current, diag_norm_coils_struct = EQ_shots_dict[shotname]["diag_norm_coils_current"][Norm_mask], EQ_shots_dict[shotname]["diag_norm_coils_struct"][Norm_mask]
        NormPickupCoil.load_data(diag_norm_coils_current + diag_norm_coils_struct)
        #-------------------------------------------------------------------

        # Load data with default uncertainties
        diag_flux_loops_current, diag_flux_loops_struct = EQ_shots_dict[shotname]["diag_flux_loops_current"], EQ_shots_dict[shotname]["diag_flux_loops_struct"]
        FluxLoop.load_data(diag_flux_loops_current + diag_flux_loops_struct)
        #-------------------------------------------------------------------

        # Load data with default uncertainties
        diag_saddle_coils_current, diag_saddle_coils_struct = EQ_shots_dict[shotname]["diag_saddle_coils_current"], EQ_shots_dict[shotname]["diag_saddle_coils_struct"]
        SaddleCoil.load_data(diag_saddle_coils_current + diag_saddle_coils_struct)
        #-------------------------------------------------------------------

        # Set the diagnostic measurements comming from the structures
        Post.set_struct_measurements(diag_tan_coils_struct, diag_norm_coils_struct, diag_flux_loops_struct, diag_saddle_coils_struct)

        # For each EQ, do then inferences with random and one with clean diagnostics
        # Important to first do the inference without noise, so I do not need to reload the data (or explicitely access the clean data in the diagnostics)
        for rng_number in range(0,11):

            # Add noise
            TanPickupCoil.add_noise(rng_number)
            #-------------------------------------------------------------------
            # Add noise
            NormPickupCoil.add_noise(rng_number)
            #-------------------------------------------------------------------
            # Add noise
            FluxLoop.add_noise(rng_number)
            #-------------------------------------------------------------------
            # Add noise
            SaddleCoil.add_noise(rng_number)
            #-------------------------------------------------------------------

            # Set the noisy diagnostic measurements (both struct and current)
            Post.set_diag_measurements(TanPickupCoil.data_noise, NormPickupCoil.data_noise, FluxLoop.data_noise, SaddleCoil.data_noise)

            # TODO set uncertainties based on noisy measurements (as that is what we have irl)
            # Set the uncertainties on these measurements
            Post.set_diag_measurements_unc(TanPickupCoil.unc, NormPickupCoil.unc, FluxLoop.unc, SaddleCoil.unc) 

            # Calculate the posterior
            Post.calculate_posterior()

            # Extract and reshape the mean posterior
            post_mean_tom = Post.expanded_mean.reshape((rVec.shape[0], rVec.shape[1]))
            # Extract the covariance
            expanded_cov = Post.expanded_cov

            # Calculate plasma parameters of interest
            Ip_inf, Zc_inf, Rc_inf, Ip_inf_std, Zc_inf_std, Rc_inf_std = CalculatePlasmaParameters(post_mean_tom, expanded_cov, rVec, zVec, zoom_factor, zoom_factor)

            # Get the true values so we can calculate the errors
            Ip_, Zc_, Rc_ = EQ_shots_dict[shotname]['total_plasma_current'], EQ_shots_dict[shotname]['current_centroid_Z'], EQ_shots_dict[shotname]['current_centroid_R']

            # Calculate Flux at Xpoint
            XpointFlux, XpointFluxerror = XpointFluxFM(post_mean_tom, expanded_cov, rVec, zVec, XpointFMs, Structures_shot)
            
            # Calculate the six plasma-wall gap positions using precalculated forward models
            Gap_positions_FMs, Gap_positions_std_FMs = PlasmaWallGapsFM(post_mean_tom, expanded_cov, Structures_shot, XpointFlux, XpointFluxerror, plasmawallgapFMs)

            Gap_positions_FMs_error = PlasmaWallGapsErrors(Gap_positions_FMs, EQ_shots_dict[shotname]["gap_positions"])

            #-------------------------------------------------------------------

            # Create the filename
            filename = f"ITERlike-{rng_number}-{shotname}.txt"
            file_path = os.path.join(output_dir, filename)

            # Fill the content with actual values
            content = content_template.format(
                plasma_current=Ip_inf,
                plasma_current_std=abs(Ip_inf_std),
                z_centroid=Zc_inf,
                z_centroid_std=abs(Zc_inf_std),
                r_centroid=Rc_inf,
                r_centroid_std=abs(Rc_inf_std),
                current_percentage_error=(Ip_inf - Ip_)/Ip_*100,
                z_centroid_cm_error=(Zc_inf - Zc_)*100,
                r_centroid_cm_error=(Rc_inf - Rc_)*100,
                plasma_gap_positions=Gap_positions_FMs,
                plasma_gap_positions_std=Gap_positions_std_FMs,
                total_plasma_gap_position_errors=Gap_positions_FMs_error
            )

            # Write the content to the file
            with open(file_path, 'w') as file:
                file.write(content)

        print('running time: ', time.time() - time_shot, ' s.')

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    '''
    # 10x10 cm inference
    # Load file containing data etc.
    EQ_shots_dict = Load_180_shots_dict()

    # Load the inner points mask
    inner_points_bool = np.load('Npy files/' + 'inner_points_bool_vacuum_vessel.npy', allow_pickle=True)

    # Load file containing FM's to calculate plasma-wall gap positions
    plasmawallgapFMs = dict(np.load("Npy files/plasmawallgapFMs.npz"))

    # Load file containing FM's to calculate Xpoint
    XpointFMs = dict(np.load("Npy files/XpointFMs.npz"))  

    # Load all current tomograms, to be used for non-centered covariance
    All_I_tom = np.load('Npy files/' + '180_I_tom.npy', allow_pickle=True)

    # Choose an EQ from which we want to infer
    shotname = "Equil_8_li_1d4_beta_0d1_CNL4E.mat"

    # Get meshes for coordinates
    rVec, zVec = Meshgrids_from_EQ("Equil_1_li_0d8_beta_0d1_CNL4E.mat")

    # Choose an rng number
    rng_number = 42
    
    time_start = time.time()
    CurrentTomInference(shotname, rng_number)
    print('running time: ', time.time() - time_start, ' s.')
    '''
    
    # 20x20 cm inference
    # Load file containing data etc.
    EQ_shots_dict = Load_180_shots_dict()

    # Load the inner points mask
    inner_points_bool = np.load('Npy files/' + 'inner_points_bool_vacuum_vessel_zoom.npy', allow_pickle=True)

    # Load file containing FM's to calculate plasma-wall gap positions 
    plasmawallgapFMs = dict(np.load("Npy files/plasmawallgapFMszoomBED.npz"))

    # Load file containing FM's to calculate Xpoint
    XpointFMs = dict(np.load("Npy files/XpointFMszoomBED.npz"))    

    # Load all current tomograms, to be used for non-centered covariance
    All_I_tom = np.load('Npy files/' + '180_I_tom_zoom.npy', allow_pickle=True)

    # Choose an EQ from which we want to infer
    shotname = "Equil_8_li_1d4_beta_0d1_CNL4E.mat"

    # Get meshes for coordinates
    rVec, zVec = Meshgrids_from_EQ("Equil_1_li_0d8_beta_0d1_CNL4E.mat")

    # Downscale the meshgrids
    zoom_factor: int = 2  
    rVec, zVec = Downsample_meshgrid(rVec, zoom_factor, zoom_factor), Downsample_meshgrid(zVec, zoom_factor, zoom_factor)

    # Choose an rng number
    rng_number = 42
    
    time_start = time.time()
    CurrentTomInferenceZoom(shotname, rng_number)
    print('running time: ', time.time() - time_start, ' s.')

    '''
    time_start = time.time()
    Allinferencezoom()
    print('running time: ', time.time() - time_start, ' s.')
    '''
