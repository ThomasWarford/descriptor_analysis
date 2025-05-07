import torch
from torch.utils.data import DataLoader
from pathlib import Path
from mace.data import AtomicData, LMDBDataset, config_from_atoms, KeySpecification
from mace.tools import torch_geometric, utils, torch_tools, DefaultKeys
from mace.calculators import mace_mp
import mace
from e3nn import o3
from mace.modules.utils import extract_invariant
import numpy as np
import ase
from npy_append_array import NpyAppendArray
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--work_dir", help="directory containing .model file. Directory name should correspond with model.",
                    type=str, required=True)
parser.add_argument("--site_info_dir", help="directory containing site information",
                    type=str, default=None, required=False,)
parser.add_argument("--dataset", help="choice of dataset",
                    choices=['mptrj', 'salex',], type=str, required=True)
parser.add_argument("--calculate_residuals", help="save the residual error for the config",
                    type=bool, required=False,)

DATASETS = {
    'mptrj': '/lustre/fswork/projects/rech/gax/ums98bp/gga-ggapu/mptrj-all.xyz',
    'salex': '/lustre/fsn1/projects/rech/gax/ums98bp/salex/train:/lustre/fsn1/projects/rech/gax/ums98bp/salex/val'
    }

class IdentifiedAtomicData(AtomicData):
    def __init__(
        self,
        identifier: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.identifier = identifier
        self.__dict__["identifier"] = identifier  # Ensures PyG sees it as a field
    
class IdentifiedLMDBDataset(LMDBDataset):
    def __getitem__(self, index):
        try:
            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
        except Exception as e:
            print(f"Error in index {index}")
            print(e)
            return None

        assert np.sum(atoms.get_cell() == atoms.cell) == 9

        if hasattr(atoms, "calc") and hasattr(atoms.calc, "results"):
            if "energy" in atoms.calc.results:
                atoms.info[DefaultKeys.ENERGY.value] = atoms.calc.results["energy"]
            if "forces" in atoms.calc.results:
                atoms.arrays[DefaultKeys.FORCES.value] = atoms.calc.results["forces"]
            if "stress" in atoms.calc.results:
                atoms.info[DefaultKeys.STRESS.value] = atoms.calc.results["stress"]

        config = config_from_atoms(
            atoms,
            key_specification=KeySpecification.from_defaults(),
        )

        if config.head == "Default":
            config.head = self.kwargs.get("head", "Default")

        try:
            atomic_data = IdentifiedAtomicData(
                identifier=atoms.info["sid"],
                **AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    heads=self.kwargs.get("heads", ["Default"]),
                ).to_dict()
        )
        except Exception as e:
            print(f"Error in index {index}")
            print(e)
            return None

        if self.transform:
            atomic_data = self.transform(atomic_data)
        return atomic_data


class LazyIdentifiedXYZDataset(torch.utils.data.IterableDataset):
    def __init__(self, xyz_file, z_table, cutoff, heads=None, start_idx=0, head=None, batch_size=65_536):
        """
        Dataset for loading XYZ configurations with batch loading for efficiency.
        
        Args:
            xyz_file: Path to the XYZ file
            z_table: Atomic number table
            cutoff: Cutoff radius for neighbor calculations
            heads: Model heads
            start_idx: Starting index for configurations
            head: Head label
            batch_size: Number of configurations to load into memory at once
        """
        self.xyz_file = xyz_file
        self.z_table = z_table
        self.cutoff = cutoff
        self.heads = heads
        self.head = head
        self.start_idx = start_idx
        self.current_idx = start_idx
        self.batch_size = batch_size
        
        # Cache for storing loaded configurations
        self.config_cache = []
        self.cache_index = 0
        
    def __iter__(self):
        self.current_idx = self.start_idx
        self.config_cache = []
        self.cache_index = 0
        return self
    
    def __len__(self):
        """
        Count the number of configurations in an XYZ file by counting lines that start with 'Lattice='.
        
        Args:
            xyz_file_path (str): Path to the XYZ file
            
        Returns:
            int: Number of configurations found
        """
        count = 0
        with open(self.xyz_file, 'r') as file:
            for line in file:
                if line.strip().startswith('Lattice='):
                    count += 1
            return count

        
    def _load_batch(self):
        """Load the next batch of configurations into memory."""
        self.config_cache = []
        self.cache_index = 0
        
        # Use ase.io.read with slice instead of iread for batch loading
        end_idx = self.current_idx + self.batch_size
        try:
            atoms_list = ase.io.read(self.xyz_file, index=f"{self.current_idx}:{end_idx}")
            if atoms_list:
                self.config_cache = atoms_list
                return True
            return False
        except Exception as e:
            # Handle end of file or reading errors
            print(f"Warning: Error loading batch: {e}")
            return False
        
    def __next__(self):
        # If cache is empty or we've used all cached configs, load next batch
        if not self.config_cache or self.cache_index >= len(self.config_cache):
            if not self._load_batch():
                raise StopIteration
                
        # Get the next configuration from the cache
        atoms = self.config_cache[self.cache_index]
        self.cache_index += 1
        
        # Process the configuration
        if self.head is not None:
            atoms.info["head"] = self.head
        
        config = config_from_atoms(atoms)
        atomic_data = AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.cutoff, heads=self.heads
        )
        
        # Add config index for tracking
        try:
            atomic_data.identifier = atoms.info['mp_id'] + '-' + str(atoms.info['calc_id']) + '-' + str(atoms.info['ionic_step'])
        except:
            atomic_data.identifier = self.current_idx
            
        # Add chemical formula for convenience
        atomic_data.chemical_formula = atoms.get_chemical_formula()

        self.current_idx += 1
        
        return atomic_data


def save_descriptors(
    model: torch.nn.Module,
    dataloader: DataLoader,
    descriptor_path: Path,
    device: str,
    site_info_path=None,
    site_residual_path=None
    ):
    torch_tools.set_default_dtype("float64")

    for batch_idx, batch in enumerate(dataloader):
        batch.to(device)
        output = model(batch.to_dict(),)
        descriptors_list = get_descriptors(batch, output, model)        
        atomic_numbers = torch.matmul(batch.node_attrs, torch.atleast_2d(model.atomic_numbers.double()).T)

                
        atomic_numbers_list = np.split(
                    torch_tools.to_numpy(atomic_numbers),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                    )[:-1] # drop last as its empty
        
        residuals = torch_tools.to_numpy(output["energy"]) - torch_tools.to_numpy(batch.energy)
        for idx, (identifier, residual, atomic_numbers, descriptors) in enumerate(zip(batch.identifier, residuals, atomic_numbers_list, descriptors_list)):
            with NpyAppendArray(descriptor_path) as npaa:
                npaa.append(descriptors.astype(np.float16))
            
            if site_info_path:
                N = len(atomic_numbers)
                dtype = np.dtype([
                        ('identifier', '<U30'),
                        ('dataset_index', np.uint32),
                        ('chemical_formula', '<U30'),
                        ('atomic_number', np.uint8),
                        ('indices', np.uint8),
                        ])
                config_info = np.empty(N, dtype=dtype)
                config_info['identifier'] = identifier
                config_info['dataset_index'] = batch_idx*dataloader.batch_size+idx
                config_info['chemical_formula'] = ase.formula.Formula.from_list([ase.data.chemical_symbols[int(z)] for z in atomic_numbers]).format('hill') 
                config_info['atomic_number'] = atomic_numbers.flatten()
                config_info['indices'] = np.arange(N)

                with NpyAppendArray(site_info_path) as npaa:
                    npaa.append(config_info)
            
            if site_residual_path:
                with NpyAppendArray(site_residual_path) as npaa:
                    npaa.append(residuals.astype('float16'))

                
            
def get_descriptors(batch, output, model, num_layers=-1, invariants_only=True):

        if num_layers == -1:
            num_layers = int(model.num_interactions)
        irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [
            irreps_out.dim for _ in range(int(model.num_interactions))
        ]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        descriptors = output["node_feats"]

        if invariants_only:
            descriptors = extract_invariant(
                descriptors,
                num_layers=num_layers,
                num_features=num_invariant_features,
                l_max=l_max,
            )

        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = descriptors[:, :to_keep].detach().cpu().numpy()

        descriptors = np.split(
            descriptors,
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        return descriptors[:-1]  # (drop last as its empty)

def main(args):
    device = 'cuda'
    work_dir = Path(args.work_dir)
    print(work_dir)
    models = list(work_dir.glob('*.model'))
    assert len(models) == 1
    model = mace_mp(models[0], device=device, return_raw_model=True)
    model = run_e3nn_to_cueq(model)
    model.to(device)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    dataset_path = DATASETS[args.dataset]
    if dataset_path[-3] == 'xyz':
        ds = LazyIdentifiedXYZDataset(dataset_path, z_table=z_table, cutoff=float(model.r_max))
    else:
        ds = IdentifiedLMDBDataset(
            dataset_path,
            float(model.r_max),
            z_table
        )

    dataloader = torch_geometric.dataloader.DataLoader(
        dataset=ds,
        batch_size=8,
        shuffle=False,
        drop_last=False
    )

    save_descriptors(
        model,
        dataloader,
        descriptor_path=work_dir/f'{args.dataset}.descriptors.npy',
        device=device,
        site_info_path=(f'{args.site_info_dir}/{args.dataset}.npy' if args.site_info_dir else None),
        site_residual_path=(work_dir/f'{args.dataset}.residuals.npy' if args.calculate_residuals else None),
    )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)