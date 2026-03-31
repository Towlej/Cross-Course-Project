#!/usr/bin/env python
import os
import sys

# Add script directory to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Optional: verify
print("CWD before:", os.getcwd())

# Only change directory if you REALLY need relative file paths
os.chdir("/home/chocamo/projects/Cross_Course")
from pathlib import Path
print("CWD after:", os.getcwd())
import numpy as np

# Meant as file cointaining different inferences/experiments that can be performed
import random
from helpfunctions import MI_gain, Load_180_shots_dict, Meshgrids_from_EQ, Load_diag_pos, CalculatePlasmaParameters, PlasmaWallGapsErrors, PlasmaWallGapsFM, Downsample_meshgrid, XpointFluxFM, D_opt_covariance
from PickupCoil import DEMOPickupCoil
from FluxLoop import DEMOFluxLoop
from SaddleCoil import DEMOSaddleCoil
from ASEPosterior import ASEPosterior
from Structures import Structures
import time


#split shots into 150/30 for training / testing
def split_shots_with_tomography(eq_shots_dict, All_I_tom, num_folds=6, fold_id=0, seed=42):
    """
    Split BOTH:
    - EQ shots dict
    - All_I_tom array

    Ensures perfect alignment between shots and tomography data
    """

    assert 0 <= fold_id < num_folds, "fold_id must be between 0 and num_folds-1"

    keys = list(eq_shots_dict.keys())

    # deterministic shuffle
    random.seed(seed)
    random.shuffle(keys)

    # split into folds
    fold_size = len(keys) // num_folds
    folds = [keys[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

    # select test fold
    test_keys = folds[fold_id]

    # remaining folds = training
    train_keys = [k for i, fold in enumerate(folds) if i != fold_id for k in fold]

    # split dictionaries
    train_dict = {k: eq_shots_dict[k] for k in train_keys}
    test_dict = {k: eq_shots_dict[k] for k in test_keys}

    # build key → index map (CRITICAL for alignment)
    key_to_index = {k: i for i, k in enumerate(eq_shots_dict.keys())}

    train_indices = [key_to_index[k] for k in train_keys]
    test_indices = [key_to_index[k] for k in test_keys]

    # split tomography data
    All_I_tom_train = All_I_tom[train_indices]
    All_I_tom_test = All_I_tom[test_indices]

    # debug prints
    print("\n===== DATA SPLIT INFO =====")
    print(f"Fold used as TEST: {fold_id}")
    print(f"Train size: {len(train_keys)}")
    print(f"Test size: {len(test_keys)}")
    print("===========================\n")

    # sanity checks
    assert All_I_tom_train.shape[0] == len(train_keys)
    assert All_I_tom_test.shape[0] == len(test_keys)

    return (
        train_dict,
        test_dict,
        All_I_tom_train,
        All_I_tom_test,
        train_keys,
        test_keys
    )


def CurrentTomInferenceZoomAll(EQ_shots_dict, rng_number, All_I_tom, inner_points_bool, rVec, zVec, zoom_factor, fold_id=None):

    # =========================
    # 1. LOAD BASIC STRUCTURES
    # =========================
    shot_keys = list(EQ_shots_dict.keys())
    dict_diag_pos = Load_diag_pos()
    saddle_coil_R, saddle_coil_Z = dict_diag_pos["saddle_coil_R"], dict_diag_pos["saddle_coil_Z"]

    # =========================
    # PRIOR COVARIANCE
    # =========================
    cov_matrix = (All_I_tom.T @ All_I_tom) / (All_I_tom.shape[0] - 1)
    cov_matrix[np.diag_indices_from(cov_matrix)] += 1e-1
    cov = cov_matrix[inner_points_bool][:, inner_points_bool]

    # =========================
    # FORWARD MODELS (NO MASKING HERE)
    # =========================
    response_tan_matrix = np.load("Npy files/response_tan_matrix_zoom_BED.npy", allow_pickle=True)
    response_norm_matrix = np.load("Npy files/response_norm_matrix_zoom_BED.npy", allow_pickle=True)
    response_fluxloop_matrix = np.load("Npy files/response_fluxloop_matrix_zoom_BED.npy", allow_pickle=True)
    response_saddlecoil_matrix = np.load("Npy files/response_saddlecoil_matrix_zoom_BED.npy", allow_pickle=True)

    # =========================
    # GLOBAL GREEDY OPTIMIZATION
    # =========================

    config = np.zeros(60, dtype=int)
    flipped = np.zeros(60, dtype=bool)
    shot_results = []
    config_MI = -np.inf
    MI_threshold = 1e-6
    MI_list_final = []
    print(f"\n===== TRAINING FOLD {fold_id} =====")

    while True:

        best_trial_MI = config_MI
        best_trial_config = config.copy()
        best_trial_index = None

        # reset per-iteration MI storage
        

        for i in range(60):

            if flipped[i]:
                continue

            trial_config = config.copy()
            trial_config[i] = 1 - trial_config[i]

            Tan_mask = (trial_config == 0)
            Norm_mask = (trial_config == 1)

            if np.sum(Tan_mask) == 0 or np.sum(Norm_mask) == 0:
                continue

            MI_total_trial = 0

            # =========================
            # LOOP OVER ALL SHOTS
            # =========================
            for shotname in shot_keys:

                Structures_shot = Structures(EQ_shots_dict[shotname]["structs_current"])

                diag_tan_coils_current = EQ_shots_dict[shotname]["diag_tan_coils_current"]
                diag_tan_coils_struct  = EQ_shots_dict[shotname]["diag_tan_coils_struct"]

                diag_norm_coils_current = EQ_shots_dict[shotname]["diag_norm_coils_current"]
                diag_norm_coils_struct  = EQ_shots_dict[shotname]["diag_norm_coils_struct"]

                diag_flux_loops_current = EQ_shots_dict[shotname]["diag_flux_loops_current"]
                diag_flux_loops_struct  = EQ_shots_dict[shotname]["diag_flux_loops_struct"]

                diag_saddle_coils_current = EQ_shots_dict[shotname]["diag_saddle_coils_current"]
                diag_saddle_coils_struct  = EQ_shots_dict[shotname]["diag_saddle_coils_struct"]

                # ----- diagnostics -----
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

                expanded_cov_inner = expanded_cov[inner_points_bool][:, inner_points_bool]

                MI = MI_gain(expanded_cov_inner, Post.prior_cov)

                MI_total_trial += MI
                MI_list_final.append(MI)
                
            # ----- average MI over dataset -----
            MI_avg_trial = MI_total_trial / len(shot_keys)

            print(f"Coil {i} → Avg MI: {MI_avg_trial}")

            if MI_avg_trial > best_trial_MI:
                best_trial_MI = MI_avg_trial
                best_trial_config = trial_config
                best_trial_index = i
            else:
                continue

        # =========================
        # ACCEPT BEST CONFIG
        # =========================
        if best_trial_index is not None and best_trial_MI > config_MI + MI_threshold:
            config = best_trial_config
            config_MI = best_trial_MI
            flipped[best_trial_index] = True

            print(f"\nAccepted flip: coil {best_trial_index}")
            print(f"New average MI: {config_MI}")
        else:
            break

    # =========================
    # FINAL CONFIG SUMMARY
    # =========================
    Tan_mask = (config == 0)
    Norm_mask = (config == 1)

    # =========================
    # FINAL POSTERIOR PER SHOT
    # =========================
    for shotname in shot_keys:

        print(f"\n--- Evaluating shot: {shotname} ---")

        Structures_shot = Structures(EQ_shots_dict[shotname]["structs_current"])

        diag_tan_coils_current = EQ_shots_dict[shotname]["diag_tan_coils_current"]
        diag_tan_coils_struct  = EQ_shots_dict[shotname]["diag_tan_coils_struct"]

        diag_norm_coils_current = EQ_shots_dict[shotname]["diag_norm_coils_current"]
        diag_norm_coils_struct  = EQ_shots_dict[shotname]["diag_norm_coils_struct"]

        diag_flux_loops_current = EQ_shots_dict[shotname]["diag_flux_loops_current"]
        diag_flux_loops_struct  = EQ_shots_dict[shotname]["diag_flux_loops_struct"]

        diag_saddle_coils_current = EQ_shots_dict[shotname]["diag_saddle_coils_current"]
        diag_saddle_coils_struct  = EQ_shots_dict[shotname]["diag_saddle_coils_struct"]

        # ----- diagnostics -----
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

        Ip_inf, Zc_inf, Rc_inf, Ip_inf_std, Zc_inf_std, Rc_inf_std = CalculatePlasmaParameters(
            post_mean_tom, expanded_cov, rVec, zVec, zoom_factor, zoom_factor
        )

        print(f"Ip: {Ip_inf} ± {Ip_inf_std}")
        print(f"Zc: {Zc_inf} ± {Zc_inf_std}")
        print(f"Rc: {Rc_inf} ± {Rc_inf_std}")
        shot_results.append({
            "shot": shotname,
            "Ip": Ip_inf,
            "Ip_std": Ip_inf_std,
            "Zc": Zc_inf,
            "Zc_std": Zc_inf_std,
            "Rc": Rc_inf,
            "Rc_std": Rc_inf_std
        })
        
    MI_array = np.array(MI_list_final)

    # =========================
    # FINAL GLOBAL SUMMARY
    # =========================
    print("\n===== FINAL GLOBAL COIL CONFIGURATION =====")
    print(f"Tangential coils: {np.sum(Tan_mask)}")
    print(f"Normal coils: {np.sum(Norm_mask)}")
    print("Tangential indices:", np.where(Tan_mask)[0])
    print("Normal indices:", np.where(Norm_mask)[0])

    print("\n===== FINAL MI STATISTICS =====")
    print(f"Mean MI: {MI_array.mean()}")
    print(f"Std MI: {MI_array.std()}")
    print(f"Min MI: {MI_array.min()}")
    print(f"Max MI: {MI_array.max()}")

    Ip_vals = [r["Ip"] for r in shot_results]
    Zc_vals = [r["Zc"] for r in shot_results]
    Rc_vals = [r["Rc"] for r in shot_results]

    summary_stats = {
        "Ip_mean": np.mean(Ip_vals),
        "Ip_std": np.std(Ip_vals),
        "Zc_mean": np.mean(Zc_vals),
        "Zc_std": np.std(Zc_vals),
        "Rc_mean": np.mean(Rc_vals),
        "Rc_std": np.std(Rc_vals),
    }


    return config, summary_stats, Tan_mask, Norm_mask


    
def save_test_summary(
    summary_file_path,
    config,
    train_summary,
    test_results,
    fold_id=None  # include fold id
):
    summary_file_path = Path(summary_file_path)

    with open(summary_file_path, "a") as f:  # append mode
        f.write("\n" + "="*50 + "\n")
        f.write(f"===== FULL SUMMARY: FOLD {fold_id} =====\n\n" if fold_id is not None else "===== FULL SUMMARY =====\n\n")

        # FINAL CONFIG
        f.write("FINAL COIL CONFIGURATION\n")
        f.write("="*30 + "\n")
        f.write(f"Tangential coils: {np.sum(config==0)}\n")
        f.write(f"Normal coils: {np.sum(config==1)}\n")
        f.write(f"Tangential indices: {np.where(config==0)[0].tolist()}\n")
        f.write(f"Normal indices: {np.where(config==1)[0].tolist()}\n\n")

        # TRAINING SUMMARY
        f.write("TRAINING SUMMARY\n")
        f.write("="*30 + "\n")
        for key, value in train_summary.items():
            f.write(f"{key}: {value}\n")

        # TEST RESULTS
        f.write("\nTEST RESULTS PER SHOT\n")
        f.write("="*30 + "\n")
        for shot in test_results:
            f.write(f"\nShot: {shot['shot']}\n")
            for k, v in shot.items():
                if k != "shot":
                    f.write(f"{k}: {v}\n")

    print(f"[INFO] Fold {fold_id} summary appended to: {summary_file_path.resolve()}")

def TestInferenceOnDataset(
    EQ_shots_dict_test,
    config,
    rng_number,
    All_I_tom_train,
    inner_points_bool,
    rVec,
    zVec,
    XpointFMs,
    plasmawallgapFMs,
    zoom_factor   # <-- ADD THIS
):
    """
    Run full evaluation on test dataset (30 shots)
    """

    results = []

    # ---- PRIOR ----
    cov_matrix = (All_I_tom_train.T @ All_I_tom_train) / (All_I_tom_train.shape[0] - 1)
    cov_matrix[np.diag_indices_from(cov_matrix)] += 1e-1
    cov = cov_matrix[inner_points_bool][:, inner_points_bool]

    # ---- FORWARD MODELS ----
    response_tan_matrix = np.load("Npy files/response_tan_matrix_zoom_BED.npy", allow_pickle=True)
    response_norm_matrix = np.load("Npy files/response_norm_matrix_zoom_BED.npy", allow_pickle=True)
    response_fluxloop_matrix = np.load("Npy files/response_fluxloop_matrix_zoom_BED.npy", allow_pickle=True)
    response_saddlecoil_matrix = np.load("Npy files/response_saddlecoil_matrix_zoom_BED.npy", allow_pickle=True)

    dict_diag_pos = Load_diag_pos()
    saddle_coil_R, saddle_coil_Z = dict_diag_pos["saddle_coil_R"], dict_diag_pos["saddle_coil_Z"]

    Tan_mask = (config == 0)
    Norm_mask = (config == 1)

    print("\n===== RUNNING TEST SET =====\n")

    # =========================
    # LOOP OVER TEST SHOTS
    # =========================
    for shotname in EQ_shots_dict_test:

        print(f"\n===== SHOT: {shotname} =====")

        Structures_shot = Structures(EQ_shots_dict_test[shotname]["structs_current"])

        # ---- LOAD DATA ----
        diag_tan_coils_current = EQ_shots_dict_test[shotname]["diag_tan_coils_current"]
        diag_tan_coils_struct  = EQ_shots_dict_test[shotname]["diag_tan_coils_struct"]

        diag_norm_coils_current = EQ_shots_dict_test[shotname]["diag_norm_coils_current"]
        diag_norm_coils_struct  = EQ_shots_dict_test[shotname]["diag_norm_coils_struct"]

        diag_flux_loops_current = EQ_shots_dict_test[shotname]["diag_flux_loops_current"]
        diag_flux_loops_struct  = EQ_shots_dict_test[shotname]["diag_flux_loops_struct"]

        diag_saddle_coils_current = EQ_shots_dict_test[shotname]["diag_saddle_coils_current"]
        diag_saddle_coils_struct  = EQ_shots_dict_test[shotname]["diag_saddle_coils_struct"]

        # ---- DIAGNOSTICS ----
        TanPickupCoil = DEMOPickupCoil()
        TanPickupCoil.load_data(diag_tan_coils_current[Tan_mask] + diag_tan_coils_struct[Tan_mask])
        TanPickupCoil.add_noise(rng_number)
        TanPickupCoil.set_forward_model(response_tan_matrix[Tan_mask, :])

        NormPickupCoil = DEMOPickupCoil(Is_Tan=False)
        NormPickupCoil.load_data(diag_norm_coils_current[Norm_mask] + diag_norm_coils_struct[Norm_mask])
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

        # ---- POSTERIOR ----
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

        # ---- RESULTS ----
        expanded_cov = Post.expanded_cov
        post_mean_tom = Post.expanded_mean.reshape((rVec.shape[0], rVec.shape[1]))

        MI = MI_gain(expanded_cov[inner_points_bool][:, inner_points_bool], Post.prior_cov)
        Dopt = D_opt_covariance(expanded_cov[inner_points_bool][:, inner_points_bool])

        Ip_inf, Zc_inf, Rc_inf, Ip_inf_std, Zc_inf_std, Rc_inf_std = CalculatePlasmaParameters(
            post_mean_tom, expanded_cov, rVec, zVec, zoom_factor, zoom_factor
        )

        Ip_, Zc_, Rc_ = (
            EQ_shots_dict_test[shotname]['total_plasma_current'],
            EQ_shots_dict_test[shotname]['current_centroid_Z'],
            EQ_shots_dict_test[shotname]['current_centroid_R']
        )

        current_error = (Ip_inf - Ip_) / Ip_ * 100
        Z_error = (Zc_inf - Zc_) * 100
        R_error = (Rc_inf - Rc_) * 100

        XpointFlux, XpointFluxerror = XpointFluxFM(
            post_mean_tom, expanded_cov, rVec, zVec, XpointFMs, Structures_shot
        )

        Gap_positions, Gap_std = PlasmaWallGapsFM(
            post_mean_tom, expanded_cov, Structures_shot,
            XpointFlux, XpointFluxerror, plasmawallgapFMs
        )

        Gap_error = PlasmaWallGapsErrors(
            Gap_positions,
            EQ_shots_dict_test[shotname]["gap_positions"]
        )
        print(XpointFlux, XpointFluxerror)

        # --- Plasma-wall gaps ---
        print(Gap_positions)
        print(Gap_std)
        print(Gap_error)
        results.append({
            "shot": shotname,
            "MI": MI,
            "Dopt": Dopt,

            "Ip": Ip_inf,
            "Zc": Zc_inf,
            "Rc": Rc_inf,

            "current_error": current_error,
            "Z_error": Z_error,
            "R_error": R_error,

            "XpointFlux": XpointFlux,
            "XpointFlux_error": XpointFluxerror,

            "gap_positions": Gap_positions,
            "gap_std": Gap_std,
            "gap_error": Gap_error
        })

    # =========================
    # GLOBAL SUMMARY
    # =========================
    print("\n===== TEST SUMMARY =====")

    print("Mean MI:", np.mean([r["MI"] for r in results]))
    print("Mean current % error:", np.mean([r["current_error"] for r in results]))
    print("Mean Z error (cm):", np.mean([r["Z_error"] for r in results]))
    print("Mean R error (cm):", np.mean([r["R_error"] for r in results]))

    return results


if __name__ == "__main__":

    import time

    # =========================
    # SETTINGS
    # =========================
    fold_id = 2
    rng_number = 42
    zoom_factor = 2

    time_start = time.time()
    print("\n===== TESTING FILE WRITE =====")

    output_dir = Path("Results")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_file = output_dir / "test_write_check.txt"

    try:
        with open(test_file, "a") as f:
            f.write("Write test successful\n")
        print(f"[OK] File writing works: {test_file.resolve()}")
    except Exception as e:
        print(f"[ERROR] File writing failed: {e}")
        sys.exit(1)


    # =========================
    # LOAD DATA
    # =========================
    print("\n===== LOADING DATA =====")

    EQ_shots_dict_full = Load_180_shots_dict()

    All_I_tom = np.load("Npy files/180_I_tom_zoom.npy", allow_pickle=True)

    inner_points_bool = np.load(
        "Npy files/inner_points_bool_vacuum_vessel_zoom.npy",
        allow_pickle=True
    )

    plasmawallgapFMs = dict(np.load("Npy files/plasmawallgapFMszoomBED.npz"))
    XpointFMs = dict(np.load("Npy files/XpointFMszoomBED.npz"))

    # =========================
    # MESHGRID
    # =========================
    rVec, zVec = Meshgrids_from_EQ("Equil_1_li_0d8_beta_0d1_CNL4E.mat")

    rVec = Downsample_meshgrid(rVec, zoom_factor, zoom_factor)
    zVec = Downsample_meshgrid(zVec, zoom_factor, zoom_factor)

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    print("\n===== SPLITTING DATA =====")

    (
        EQ_shots_dict_train,
        EQ_shots_dict_test,
        All_I_tom_train,
        All_I_tom_test,
        train_keys,
        test_keys
    ) = split_shots_with_tomography(
        EQ_shots_dict_full,
        All_I_tom,
        num_folds=6,
        fold_id=fold_id,
        seed=42
    )

    # =========================
    # TRAIN (OPTIMIZE COILS)
    # =========================
    print("\n===== TRAINING (COIL OPTIMIZATION) =====")

    config, train_summary, Tan_mask, Norm_mask = CurrentTomInferenceZoomAll(
        EQ_shots_dict_train,
        rng_number,
        All_I_tom_train,
        inner_points_bool,
        rVec,
        zVec,
        zoom_factor,
        fold_id=fold_id  # <-- pass fold_id here
    )

    # =========================
    # TEST
    # =========================
    print("\n===== TESTING =====")

    results = TestInferenceOnDataset(
        EQ_shots_dict_test,
        config,
        rng_number,
        All_I_tom_train,
        inner_points_bool,
        rVec,
        zVec,
        XpointFMs,
        plasmawallgapFMs,
        zoom_factor   # <-- ADD THIS
    )

    print("\n===== TOTAL RUNTIME =====")
    print(time.time() - time_start, "seconds")

    summary_file_path = output_dir / f"test_{fold_id}_full_summary.txt"

    save_test_summary(
        summary_file_path=summary_file_path,
        config=config,
        train_summary=train_summary,
        test_results=results,
        fold_id=fold_id
    )

    '''
    time_start = time.time()
    Allinferencezoom()
    print('running time: ', time.time() - time_start, ' s.')
    '''
