import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io

from pathlib import Path
from typing import Sequence, Union, Optional, Callable, List, Tuple

from .utils import load_dataset_info
from .transforms import get_transforms

Pathlike = Union[str, Path]


def _lower_mode(mode: str):
    """modeを小文字化してassertion"""

    lower_mode = mode.lower().strip()
    assert lower_mode in ('train', 'val', 'test'),\
        f"`mode` must be 'train', 'val' or 'test', but `{mode}`"

    return lower_mode


class ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Pathlike], labels: Sequence[int],
                 transform: Optional[Callable],
                 device: torch.device = None):
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_paths = image_paths
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = io.read_image(str(img_path)).to(self.device)
        img = self.transform(img)
        label = self.labels[idx]

        return img, label


def read_csv(csv_path: Pathlike, sep: str = ',') -> Tuple[List[Path], List[int]]:
    """
    NNCのデータセットCSVから画像パスリストとラベルリストを取得して返す

    Parameters
    ----------
    csv_path : Pathlike
        CSVファイルパス
    sep : str, optional
        CSVの区切り文字, by default ','

    Returns
    -------
    Tuple[List[Path], List[int]]
        画像パスリスト、ラベルリスト
    """
    csv_path = Path(csv_path)
    assert csv_path.is_file(), f"{csv_path} doesn't exist. Confirm `csv_path`."

    root_path = csv_path.parent
    img_paths, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        next(f)  # skip the first row
        for line in f:
            img_path, label = line.strip().split(sep)
            img_paths.append(root_path / img_path)
            labels.append(int(label))

    return img_paths, labels


def build_dataset(datatype: str, args):
    """
    Build Dataset instance
    """
    json_path = Path(args.json_path)
    json_dict = load_dataset_info(json_path)
    data_info = json_dict["dataset_information"]

    color_mode = data_info["color_mode"].strip().lower()
    assert color_mode in ('rgb', 'gray'),\
        f"`color_mode` defined in JSON should be 'RGB' or 'Gray', but '{color_mode}'."

    datatype = _lower_mode(datatype)

    # Read CSV
    root_path = Path(data_info["root_path"])
    csv_path = root_path / data_info[datatype]
    image_paths, labels = read_csv(csv_path, sep=',')

    transform = get_transforms(datatype, json_dict)
    device = torch.device(args.device)

    return ImageDataset(image_paths, labels, transform, device=device)


def build_dataloader(datatype: str, args):
    datatype = _lower_mode(datatype)
    shuffle = (datatype == 'train')
    pin_memory = (args.device != 'cuda')  # 画像読み込み時点でGPUに乗せるため

    dataset = build_dataset(datatype, args)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             drop_last=False,
                             num_workers=0,
                             pin_memory=pin_memory)
    return data_loader
