import os
import json

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.nn import radius_graph
from pymatgen.core import Structure
from tqdm import tqdm
import urllib.request
import zipfile
from torch_geometric.utils import subgraph
import torch
from pymatgen.core import Structure




class QMOF(InMemoryDataset):
    raw_file_names = ['qmof_structure_data.json']
    processed_file_names = ['data.pt']
    
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, cutoff: float = 5.0,target_column="bandgap"):
        """
        Args:
            root (str): Root directory. Expects
                root/raw/qmof_structure_data.json
            cutoff (float): radius (in Å) for connecting edges
        """
        self.cutoff = cutoff
        self.target_column = target_column
        self.ID2NAME = {}
        self.STRUCTURE_DATA = {}
        self.PROPERTY_DATA = {}
        self.types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        self.atm_nums =[1, 6, 7, 8, 9]
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'qmof_database')
    
    @property
    def raw_file_names(self):
        return ["qmof_structure_data.json","qmof.json"]
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    def download(self):
        # 1) Download ZIP into raw_dir
        download_url(
            url='https://figshare.com/ndownloader/files/51716795',
            folder="/scratch/saigum/MultiGraphFormer/data/",
            filename='qmof_download.zip',
        )
        # 2) Extract into raw_dir so that qmof_structure_data.json appears there
        extract_zip(
            path= '/scratch/saigum/MultiGraphFormer/data/qmof_download.zip',
            folder="/scratch/saigum/MultiGraphFormer/data/",
        )
    def process(self):
        # 1) Load your JSON
        path = os.path.join(self.raw_dir, self.raw_file_names[0])
        target_path = os.path.join(self.raw_dir,self.raw_file_names[1])
        
        with open(path) as f:
            struct_list = json.load(f)

        with open(target_path) as fp:
            properties = json.load(fp)
        
        # 2) Build lookup tables
        self.ID2NAME = {
            d['qmof_id']: d['name']
            for d in struct_list
        }
        self.STRUCTURE_DATA = {
            d['qmof_id']: d['structure']
            for d in struct_list
        }

        self.PROPERTY_DATA = {
            d["qmof_id"] : d["outputs"]["pbe"][self.target_column] 
            for d in properties
        }
        # 3) Convert each entry to a torch_geometric.data.Data
        data_list = []
        with tqdm(total=len(self.STRUCTURE_DATA.items())) as pbar:
            for mol_id, struct_dict in self.STRUCTURE_DATA.items():
                data = self.get_graph(mol_id)
                if self.pre_filter and not self.pre_filter(data):
                    continue
                if self.pre_transform:
                    data = self.pre_transform(data)
                data_list.append(data)
                pbar.update(1)

        # 4) Collate & save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    from torch_geometric.data import Data

    def get_graph(self, mol_id) -> Data:
        # Reconstruct the Pymatgen Structure
        struct = Structure.from_dict(self.STRUCTURE_DATA[mol_id])
        coords = torch.tensor(struct.cart_coords, dtype=torch.float)  # [N_total,3]

        bool_mask = [site.specie.Z in self.atm_nums for site in struct]
        mask = torch.tensor(bool_mask, dtype=torch.bool)

        # 2) Node features: only for selected atoms
        z = [ self.types[site.specie.symbol]
            for site,keep in zip(struct, bool_mask) if keep ]
        x = torch.tensor(z, dtype=torch.long).unsqueeze(1)  # [N_sel,1] or [N_sel,] as you like

        # 3) Build full-edge list (on all atoms)
        #    (you could also only loop over selected atoms, but subgraph is simpler)
        neghbrs = struct.get_all_neighbors(self.cutoff, include_index=True)
        edges = []
        for i, nbrs in enumerate(neghbrs):
            for site, dist, j, _ in nbrs:
                edges.append([i, j])
        edge_index_full = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E_full]

        # 4) Edge attributes = distance; we'll compute on full graph too
        row, col = edge_index_full
        edge_attr_full = (coords[row] - coords[col]).norm(dim=1, keepdim=True)

        # 5) Subgraph to selected nodes, relabeling them 0..N_sel-1
        sub_edge_index, sub_edge_attr = subgraph(
            mask,
            edge_index_full,
            edge_attr_full,
            relabel_nodes=True,
            num_nodes=coords.size(0)
        )

        # 6) Subset positions
        pos = coords[mask]  # [N_sel, 3]

        # 7) Target
        y = torch.tensor(self.PROPERTY_DATA[mol_id], dtype=torch.float).view(-1, 1)

        return Data(
            x=x,
            pos=pos,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=y,
            mol_id=mol_id,
            name=self.ID2NAME[mol_id],
        )


if __name__== '__main__':
    dataset = QMOF("../data/qmof_database")
    for i in range(len(dataset)):
        print(dataset[i])