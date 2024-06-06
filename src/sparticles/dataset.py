import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
import os
from .utils import make_tuple
import shutil
import tarfile
import glob
from sparticles.transforms import MakeHomogeneous

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
DEFAULT_EVENT_SUBSETS = {
    'signal': 463056,
    'singletop': 242614,
    'ttbar': 6093298
}

# These are the columns we should keep from the raw pandas dataframe.
USEFUL_COLS = [
    # jet 1
    'pTj1', 'etaj1', 'phij1', 'j1_quantile', 'nan', 'nan',
    # jet 2
    'pTj2', 'etaj2', 'phij2', 'j2_quantile', 'nan', 'nan',
    # jet 3
    'pTj3', 'etaj3', 'phij3', 'j3_quantile', 'nan', 'nan',
    # b1
    'pTb1', 'etab1', 'phib1', 'b1_quantile', 'b1m', 'nan',
    # b2
    'pTb2', 'etab2', 'phib2', 'b2_quantile', 'b2m', 'nan',
    # lepton
    'pTl1', 'etal1', 'phil1', 'nan', 'nan', 'nan',
    # energy
    'ETMiss', 'nan', 'ETMissPhi', 'nan', 'nan', 'metsig_New',
]

# A markdown table to display the structure of a single event.
EVENT_TABLE = """
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
    def __init__(
            self,
            root,
            url,
            event_subsets: dict = DEFAULT_EVENT_SUBSETS,
            add_edge_index: bool = True,
            delete_raw_archive: bool = False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            download_type: int = 1):  # Added parameter `download_type`

        self.url = url
        self.delete_raw_archive = delete_raw_archive
        self.event_subsets = event_subsets
        self.add_edge_index = add_edge_index
        self.download_type = download_type  # Store download type
        self.subset_string = '_'.join([f'{k}_{v}' for k, v in sorted(self.event_subsets.items())])

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return RAW_DIR_NAMES

    @property
    def processed_file_names(self):
        return [f'events_{self.subset_string}.pt']

    @property
    def event_structure(self):
        return EVENT_TABLE

    def download(self):
        print(f'Downloading {self.url} to {self.raw_dir}...')
        print('This may take a while...')
        raw_archive = download_url(self.url, self.raw_dir, filename='events.tar', log=False)

        print('Extracting files...')
        with tarfile.open(raw_archive) as tar:
            if self.download_type == 1:
                tar.extractall(self.raw_dir)
            elif self.download_type == 2:
                members = tar.getmembers()
                for member in members:
                    if 'signal' in member.name and 'Wh_hbb_fullMix.h5' not in member.name:
                        continue
                    tar.extract(member, self.raw_dir)

        if self.delete_raw_archive:
            os.remove(raw_archive)

        print('Moving files...')
        for dir in self.raw_file_names:
            dirpath = glob.glob(f'{self.raw_dir}/**/{dir}', recursive=True)[0]
            shutil.move(dirpath, self.raw_dir)
            print(f'Moved {dirpath} to {self.raw_dir}')

        print('Cleaning up...')
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
        h5_files = {}

        for d in self.raw_file_names:
            dir_path = os.path.join(self.raw_dir, d)
            if d == 'signal':
                signal_file_path = os.path.join(dir_path, 'Wh_hbb_fullMix.h5')
                if os.path.exists(signal_file_path):
                    h5_files[d] = signal_file_path
            else:
                h5_files[d] = glob.glob(f'{dir_path}/*.h5', recursive=True)[0]

        data_list = []

        for event_type, h5_file in h5_files.items():
            label = EVENT_LABELS[event_type]
            graphs = pd.read_hdf(h5_file)
            graphs.drop(columns=list(set(graphs.columns) - set(USEFUL_COLS)), inplace=True)
            graphs['nan'] = torch.nan
            graphs = graphs[USEFUL_COLS].reset_index()
            graphs = graphs.sample(n=self.event_subsets[event_type], random_state=RANDOM_STATE)

            for row in tqdm(graphs.values, total=graphs.shape[0], desc=f'Processing events in {h5_file}'):
                event_id = int(row[0])
                graph_features = row[1:]
                x = torch.from_numpy(graph_features).reshape(7, -1)
                x = x[x[:, 0] > 0]

                edge_index = None
                if self.add_edge_index:
                    directed_edge_index = torch.combinations(torch.arange(x.shape[0]), 2)
                    edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(1)], dim=0).T

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



