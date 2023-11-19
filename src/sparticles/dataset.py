import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
from .utils import make_tuple
import os
import shutil
import tarfile
import glob


# Random state for shuffling the dataset.
RANDOM_STATE = 42


# Names of the directories in the raw directory.
RAW_DIR_NAMES = ['signal', 'singletop', 'ttbar']


# Constant labels for noise and signal.
SIGNAL_LABEL = 1
BACKGROUND_LABEL = 0


# Match between directory and event type.
EVENT_LABELS = {
    'signal': SIGNAL_LABEL,
    'singletop': BACKGROUND_LABEL,
    'ttbar': BACKGROUND_LABEL
    }


# Number of events to keep for each event type.
# The total number of events in the dataset is the sum of the values in this dictionary.
# We can use these values to have a more balanced dataset.
DEFAULT_EVENT_SUBSETS = {
    'signal': 463056,
    'singletop': 242614,
    'ttbar': 6093298
}


# These are the columns we should keep from the raw pandas dataframe. 
# The nan columns are just a hack as we need to have the same number of columns for each row.
USEFUL_COLS = [
            # jet 1
            'pTj1',
            'etaj1',
            'phij1',
            'j1_quantile',
            'nan',
            'nan',
            # jet 2
            'pTj2',
            'etaj2',
            'phij2',
            'j2_quantile',
            'nan',
            'nan',
            # jet 3
            'pTj3',
            'etaj3',
            'phij3',
            'j3_quantile',
            'nan', 
            'nan',
            # b1 
            'pTb1',
            'etab1',
            'phib1',
            'b1_quantile',
            'b1m',
            'nan',
            # b2
            'pTb2',
            'etab2',
            'phib2',
            'b2_quantile',
            'b2m',
            'nan',
            # lepton
            'pTl1',
            'etal1',
            'phil1',
            'nan',
            'nan', 
            'nan',
            # energy
            'ETMiss',
            'nan',
            'ETMissPhi',
            'nan',
            'nan',
            'metsig_New',]


# A markdown table to display the structure of a single event.
EVENT_TABLE =   """
                Each event is a graph with 6/7 nodes. Each node is built from the raw file as follows:
                
                | Particle          | Feature 1 | Feature 2 | Feature 3   | Feature 4     | Feature 5 | Feature 6    |
                |-------------------|-----------|-----------|-------------|---------------|-----------|--------------|
                | jet1              |  'pTj1'   | 'etaj1'   |   'phij1'   | 'j1_quantile' |    nan    |     nan      |
                | jet2              |  'pTj2'   | 'etaj2'   |   'phij2'   | 'j2_quantile' |    nan    |     nan      |
                | jet3 (optional)   |  'pTj3'   | 'etaj3'   |   'phij3'   | 'j3_quantile' |    nan    |     nan      |
                | b1                |  'pTb1'   | 'etab1'   |   'phib1'   | 'b1_quantile' |   'b1m'   |     nan      |
                | b2                |  'pTb2'   | 'etab2'   |   'phib2'   | 'b2_quantile' |   'b2m'   |     nan      |
                | lepton            |  'pTl1'   | 'etal1'   |   'phil1'   |      nan      |    nan    |     nan      |
                | energy            | 'ETMiss'  |   nan     | 'ETMissPhi' |      nan      |    nan    | 'metsig_New' |
                """




class EventsDataset(InMemoryDataset):
    """
    Dataset of graphs representing collisions of particles. 
    There are three types of event:
        - signal, label 1
        - singletop, label 0
        - ttbar, label 0
    
    Each event is a graph with 6 or 7 nodes and 6 attributes. Graphs are fully connected.

    Args:
        root (str): Root directory where the dataset should be saved.
        url (str): URL to download the dataset from.
        event_subsets (dict, optional): Dictionary containing the number of events to keep for each event type. Defaults to {'signal': 463056, 'singletop': 242614, 'ttbar': 6093298}.
        add_edge_index (bool, optional): Whether to add the fully connected edge index to the data objects. Defaults to True.
        delete_raw_archive (bool, optional): Whether to delete the raw archive after extracting it. Defaults to False.
        transform (callable, optional): A function/transform that takes in a `torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in a `torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. Defaults to None.
        pre_filter (callable, optional): A function that takes in a `torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. Defaults to None.
    """

    def __init__(
            self,
            root, 
            url, 
            event_subsets : dict = DEFAULT_EVENT_SUBSETS,
            add_edge_index: bool = True,
            delete_raw_archive: bool = False, 
            transform=None, 
            pre_transform=None, 
            pre_filter=None):
        
        self.url = url
        self.delete_raw_archive = delete_raw_archive
        self.event_subsets = event_subsets
        self.add_edge_index = add_edge_index
        self.subset_string = '_'.join([f'{k}_{v}' for k,v in sorted(self.event_subsets.items())]) 
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        

    @property
    def raw_file_names(self):
        return RAW_DIR_NAMES


    @property
    def processed_file_names(self):
        # Notice the processed file names depend on the number of events we keep for each event type.
        return [f'events_{self.subset_string}.pt']
    

    @property
    def event_structure(self):
        """
        Returns the event structure of the dataset.
        The event structure is a table that describes the different types of events that can occur in the dataset.
        Returns:
            str: A string containing a markdown table representing the event structure of the dataset.
        """
        return EVENT_TABLE



    def download(self): 
        # Download raw directories to `self.raw_dir`.  
        # In version 1 the compressed file should contain three subdirectories: singletop, ttbar, signal.
        print(f'Downloading {self.url} to {self.raw_dir}...')
        print('This may take a while...')
        raw_archive = download_url(self.url, self.raw_dir, filename=f'events.tar', log=False)
        
        # Extract the tar files.
        print('Extracting files...')
        tar = tarfile.open(raw_archive)
        tar.extractall(self.raw_dir)
        tar.close()
        if self.delete_raw_archive:
            os.remove(raw_archive)

        # In case the compressed file contains a single directory, we move the files to the raw_dir.
        print('Moving files...')
        for dir in self.raw_file_names:
            # find the directory recursviely
            dirpath = glob.glob(f'{self.raw_dir}/**/{dir}', recursive=True)[0]
            shutil.move(dirpath, self.raw_dir)
            print(f'Moved {dirpath} to {self.raw_dir}')
        
        print('Cleaning up...')
        # Remove the directories which are not in self.raw_file_names.
        for f in os.listdir(self.raw_dir):
            if f not in self.raw_file_names + ['events.tar']:
                try:
                    shutil.rmtree(os.path.join(self.raw_dir, f))
                except NotADirectoryError:
                    os.remove(os.path.join(self.raw_dir, f))
        
        """
        At this stage, we should have the following directory structure.
        Notice h5 file names can change.

        root
        ├── processed
        └── raw
            ├── signal
            │   └── Wh_hbb_fullMix.h5
            ├── singletop
            │   └── singletop.h5
            └── ttbar
                └── ttbar.h5
        """
                
        

    def process(self):
        
        # Create a dictionary of h5 files, where keys are the event types and values are the path to the h5 file.
        # We don't know the .h5 file names, so we use glob to find them.
        h5_files = {d: glob.glob(f'{self.raw_dir}/{d}/*.h5', recursive=True)[0] for d in self.raw_file_names}
        

        # Read data into `Data` list.
        data_list = []

        for event_type, h5_file in h5_files.items():
            
            # Labels is the same for all events in the same directory.
            label = EVENT_LABELS[event_type]

            # Read data into pandas dataframe and filter out useless columns.
            graphs = pd.read_hdf(h5_file)
            graphs.drop(columns=list(set(graphs.columns) - set(USEFUL_COLS)), inplace=True)

            # Hackish way to have all rows with the same number of columns.
            graphs['nan'] = torch.nan

            # Rearrange columns to have the same order as USEFUL_COLS and create index column.
            graphs = graphs[USEFUL_COLS].reset_index()
            
            # Shuffle the dataframe and possibly keep only part of it.
            graphs = graphs.sample(n=self.event_subsets[event_type], random_state=RANDOM_STATE)
        
            for row in tqdm(graphs.values, total=graphs.shape[0], desc=f'Processing events in {h5_file}'): 
                
                event_id = int(row[0])
                graph_features = row[1:]

                # create tensor of node features
                x = torch.from_numpy(graph_features).reshape(7, -1)

                # some graphs have trash nodes with -99 values for the Pt column. We remove the nodes.
                x = x[x[:,0]>0]

                # graphs are all fully connected
                edge_index = None
                if self.add_edge_index:
                    directed_edge_index = torch.combinations(torch.arange(x.shape[0]), 2)
                    edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(1)], dim=0).T
    
                # TODO should we add the edge index here? Knowing it is fully connected, does it make sense to waste space for this ?
                # TODO make the event id a constant across multiple datasets
                data_list.append(Data(
                    x=x, 
                    event_id=f'{event_type}_{event_id}',
                    y=label,
                    edge_index=edge_index,
                    ))

                

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
