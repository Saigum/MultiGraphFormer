import os
import json

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.nn import radius_graph
from pymatgen.core import Structure
from tqdm import tqdm
import urllib.request
import zipfile



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
        url = "https://figshare.com/ndownloader/files/51716795"
        zip_path = "/scratch/saigum/MultiGraphFormer/data/qmof_download.zip"
        extract_dir = "/scratch/saigum/MultiGraphFormer/data"

        os.makedirs(extract_dir, exist_ok=True)

        print(f"Downloading {url} to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
        
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

    def get_graph(self, mol_id) -> Data:
        # Reconstruct the Pymatgen Structure
        struct = Structure.from_dict(self.STRUCTURE_DATA[mol_id])
        coords = torch.tensor(struct.cart_coords, dtype=torch.float)  # [N,3]

        # Simple node feature: atomic number
        z = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
        x = z   

        # Build edges by radius_graph
        neghbrs = struct.get_all_neighbors(self.cutoff,include_index=True)
        edge_index=[]
        for i,atom in enumerate(neghbrs):
            for site,distance,index,image in atom:
                edge_index.append([i,index])

        edge_index = torch.tensor(edge_index,dtype=torch.int32).T
        row, col = edge_index
        edge_attr = (coords[row] - coords[col]).norm(dim=1, keepdim=True)


        y = torch.tensor(self.PROPERTY_DATA[mol_id], dtype=torch.float).view(-1, 1)

        data = Data(
            x=x,
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            mol_id=mol_id,
            name=self.ID2NAME[mol_id]
        )
        return data

if __name__== '__main__':
    dataset = QMOF("../data/qmof_database")
    for i in range(len(dataset)):
        print(dataset[i])