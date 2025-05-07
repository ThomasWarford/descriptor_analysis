from pathlib import Path
import numpy as np
import faiss # pca for many vectors
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("work_dir", help="Directory containing .descriptor.npy files.",
                    type=str)

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
    temp = np.memmap(
        'temp.array', mode='w+', dtype=descriptors_list[0].dtype,
    shape=(sum(descriptors.shape[0] for descriptors in descriptors_list), descriptors_list[0].shape[1]))

    start_idx = 0
    for descriptors in dataset_to_descriptors.values():
        temp[start_idx:start_idx+descriptors.shape[0]] = descriptors
        start_idx += descriptors.shape[0]
    
    pca_matrix = faiss.PCAMatrix(temp.shape[1], 8)
    pca_matrix.train(temp)
    assert pca_matrix.is_trained
    faiss.write_VectorTransform(pca_matrix, f"{str(work_dir)}/pca.pca") # faiss c++ code doesn't like pathlib.Path
    
    fig_s, ax_s = plt.subplots(2, sharex=True, sharey=True)
    fig_c, ax_c = plt.subplots()
    # fig, ax = plt.subplots()
    cmaps = ['Reds', 'Blues']
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
    main(args)