import gc
import glob
import os

import numpy as np
from tabulate import tabulate
import scipy.io as sio
from tqdm import tqdm


def read_matrix(matrix_name):
    local_folder = "./matrices"
    file_path = os.path.join(local_folder, "**", f"{matrix_name}.mtx")
    file_mtx = glob.glob(file_path, recursive=True)

    A_shape, A_nnz, avg_nnz, row_var, regularity = None, None, 0.0, 0.0, 0.0

    if file_mtx:
        try:
            A = sio.mmread(file_mtx[0]).tocsr()
            A_shape = A.shape[0]
            A_nnz = A.nnz

            row_sizes = np.diff(A.indptr)
            row_var = float(np.var(row_sizes))

            avg_nnz_local = A_nnz / A_shape if A_shape > 0 else 1.0
            row_std = float(np.std(row_sizes))
            regularity = float(row_std / avg_nnz_local)

            del A
            del row_sizes
            gc.collect()
        except Exception:
            pass

    if avg_nnz == 0.0 and A_shape is not None and A_shape > 0:
        avg_nnz = A_nnz / A_shape

    return [A_shape, A_nnz, avg_nnz, row_var, float(regularity)]



def monte_carlo_analysis(matrix_list, target=100_000_000):
    print("\n--- Matrix Loading ---")
    valid_names = []
    features_list = []

    for nome in tqdm(matrix_list, desc="Matrix Loading", unit="mat"):
        feat = read_matrix(nome)
        if feat:
            valid_names.append(nome)
            features_list.append(feat)

    n_valid = len(valid_names)
    if n_valid < 10:
        raise ValueError("At least 10 matrices are needed.")

    features_array = np.array(features_list)
    valid_names = np.array(valid_names)

    CHUNK_SIZE = 5_000_000
    num_cycles = max(1, target // CHUNK_SIZE)
    actual_test_cases = num_cycles * CHUNK_SIZE

    print(f"\n--- Search on {actual_test_cases:,} groups ---")

    global_best_score = -1.0
    global_best = None

    for cycle in tqdm(range(num_cycles), desc="Processing chunks", unit="chunk(5M)"):
        random_indices = np.argsort(np.random.rand(CHUNK_SIZE, n_valid), axis=1)[:, :10]

        groups_data = features_array[random_indices]

        means = np.mean(groups_data, axis=1)
        vars_ = np.var(groups_data, axis=1)
        stds = np.std(groups_data, axis=1)

        cvs = np.zeros_like(means)
        mask = means != 0
        cvs[mask] = stds[mask] / means[mask]

        scores = np.sum(cvs, axis=1)

        best_idx_chunk = int(np.argmax(scores))
        best_score_chunk = scores[best_idx_chunk]

        if best_score_chunk > global_best_score:
            global_best_score = best_score_chunk
            best_group_names = valid_names[random_indices[best_idx_chunk]]

            global_best = {
                'group': best_group_names.tolist(),
                'score': global_best_score,
                'stats': {
                    'mean': means[best_idx_chunk],
                    'var': vars_[best_idx_chunk],
                    'std': stds[best_idx_chunk],
                    'cv': cvs[best_idx_chunk]
                }
            }

        del random_indices, groups_data, means, vars_, stds, cvs, scores

    return global_best, actual_test_cases


if __name__ == "__main__":
    input_list = [
        "BenElechi1", "CoupCons3D", "CurlCurl_4", "Emilia_923", "F1",
        "Fault_639", "FullChip", "GAP-road", "Geo_1438", "Hardesty3",
        "Hook_1498", "Long_Coup_dt6", "ML_Laplace", "PFlow_742", "RM07R",
        "Serena", "Si87H76", "Transport", "af_5_k101", "af_shell5",
        "audikw_1", "bmwcra_1", "boneS10", "coPapersCiteseer", "crankseg_2",
        "gsm_106857", "halfb", "inline_1", "ljournal-2008", "msdoor",
        "nd24k", "nlpkkt80", "pkustk14", "rgg_n_2_21_s0", "spal_004",
        "vas_stokes_2M", "Ge87H76", "Maragal_8", "Raj1",
        "TSOPF_RS_b678_c2", "gearbox", "m_t1", "ship_003", "shipsec5",
        "thermomech_dM", "x104", "Bump_2911", "Cube_Coup_dt6", "Flan_1565",
        "HV15R", "ML_Geer", "Queen_4147", "hollywood-2009", "kmer_U1a"
    ]

    try:
        best, n_comb = monte_carlo_analysis(input_list, target=100_000_000)
        print(f"Analyzed {n_comb} combinations")
        print(f"Best Combination: {', '.join(best['group'])}")
        print(f"Score: {best['score']:.4f}")

        metric_names = [
            'Size',
            ' NNZ',
            'Avg NNZ for Row',
            'Row Lenght Variance',
            'Structural Regularity'
        ]

        table_data = []
        for i, name in enumerate(metric_names):
            table_data.append([
                name,
                f"{best['stats']['mean'][i]:.4f}",
                f"{best['stats']['std'][i]:.4f}",
                f"{best['stats']['cv'][i] * 100:.2f}%"
            ])

        headers = ["Metric", "Mean", "Std Dev", "CV (%)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        print(f"Error: {e}")

#Best: kmer_U1a, af_shell5, shipsec5, BenElechi1, Raj1, Queen_4147, spal_004, af_5_k101, ML_Laplace, thermomech_dM