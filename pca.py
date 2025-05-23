from pathlib import Path
import numpy as np
import faiss # pca for many vectors
import argparse
import matplotlib.pyplot as plt
from typing import List
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("work_dir", help="Directory containing .descriptor.npy files.",
                    type=str)

def sample_from_mmap_chunks(descriptor_arrays: List[np.ndarray], global_indices: List[int]) -> np.ndarray:
    """
    Given a list of memory-mapped descriptor arrays and global indices into their concatenation,
    returns a single NumPy array containing the sampled rows.

    Args:
        descriptor_arrays (List[np.ndarray]): List of memory-mapped 2D arrays of shape (N_i, D)
        global_indices (List[int]): Indices into the virtual concatenation of these arrays

    Returns:
        np.ndarray: Sampled descriptors, shape (len(global_indices), D)
    """
    # Precompute chunk boundaries
    chunk_sizes = [arr.shape[0] for arr in descriptor_arrays]
    chunk_offsets = np.cumsum([0] + chunk_sizes)  # [0, size1, size1+size2, ...]

    # Prepare output buffer
    d = descriptor_arrays[0].shape[1]
    sampled = np.empty((len(global_indices), d), dtype=descriptor_arrays[0].dtype)

    # Map global indices to the correct array and local index
    insert_idx = 0
    for i in range(len(descriptor_arrays)):
        print(i)
        start, end = chunk_offsets[i], chunk_offsets[i + 1]
        print(f"Processing chunk {i}, range {start}-{end}")
        mask = (start <= np.array(global_indices)) & (np.array(global_indices) < end)
        local_indices = np.array(global_indices)[mask] - start
        print(f"Found {len(local_indices)} local indices in chunk {i}")

        if local_indices.size > 0:
            sampled[insert_idx:insert_idx + local_indices.size] = descriptor_arrays[i][local_indices]
            insert_idx += local_indices.size

    return sampled

def plot_descriptors(dataset_name, descriptors, pca_matrix, savefig_path=None, ax=None, plot_kwargs={}):
    assert pca_matrix.is_trained
    if not ax:
        fig, ax = plt.subplots()
        fig.suptitle(f"Model: {savefig_path.parents[1].stem}")

    transformed = pca_matrix.apply(descriptors)
    ax.hexbin(transformed[:, 0], transformed[:, 1], bins='log', label=dataset_name, **plot_kwargs)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(dataset_name)

    if savefig_path:
        plt.savefig(savefig_path)
    return ax

def main(args):
    work_dir = Path(args.work_dir)

    dataset_to_descriptors = {}
    for descriptor_path in work_dir.glob("*.descriptors.npy"):
        dataset_to_descriptors[descriptor_path.stem.split('.')[0]] = np.load(descriptor_path, mmap_mode='r')
    descriptors_list = list(dataset_to_descriptors.values())

    # prevent oom errors by sampling - I believe faiss does this anyways https://github.com/facebookresearch/faiss/issues/4262
    subset_size = 100_000 
    total_descriptors = sum(descriptors.shape[0] for descriptors in descriptors_list)
    indices = np.random.choice(total_descriptors, size=min(subset_size, total_descriptors), replace=False)
    print('sampling descriptors')
    sampled_descriptors = sample_from_mmap_chunks(descriptor_arrays=descriptors_list, global_indices=indices)
    
    print('PCA training')
    pca_matrix = faiss.PCAMatrix(sampled_descriptors.shape[1], 8)
    pca_matrix.train(sampled_descriptors)
    assert pca_matrix.is_trained
    print('training complete')
    faiss.write_VectorTransform(pca_matrix, f"{str(work_dir)}/pca.pca") # faiss c++ code doesn't like pathlib.Path
    print('saved pca matrix')
    
    fig_s, ax_s = plt.subplots(2, sharex=True, sharey=True)
    fig_c, ax_c = plt.subplots()
    # fig, ax = plt.subplots()
    cmaps = ['Reds', 'Blues', 'Greens']
    plot_dir = work_dir/'plots'; plot_dir.mkdir(exist_ok=True)
    for i, (dataset_name, descriptors) in enumerate(dataset_to_descriptors.items()):
        plot_descriptors(dataset_name, descriptors, pca_matrix, savefig_path=plot_dir/f'{dataset_name}.png')
        plot_descriptors(dataset_name, descriptors, pca_matrix, ax=ax_s[i],)
        plot_descriptors(dataset_name, descriptors, pca_matrix, ax=ax_c, plot_kwargs={'cmap':cmaps[i], 'alpha':0.5,})
        

    # leg.legend_handles[0].set_edgecolor('black')
    ax_c.set_title(None)
    ax_c.legend()
    leg = ax_c.get_legend()
    leg.legend_handles[0].set_facecolor('red')
    leg.legend_handles[1].set_facecolor('blue')
    fig_s.suptitle(f"Model: {work_dir.stem}")
    fig_s.savefig(plot_dir/'seperate.png')
    fig_c.suptitle(f"Model: {work_dir.stem}")
    fig_c.savefig(plot_dir/'combined.png')


if __name__ == "__main__":
    args = parser.parse_args()
    print(f'Starting PCA in: {args.work_dir}')
    main(args)
    print('SUCCESS')
