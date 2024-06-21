from typing import Optional
from torch.utils import data
from torchvision import transforms

from .transformation import ToTensor
from .transformation import RandomIntensityShiftScale
from .transformation import RandomHorizontalFlip
from .transformation import NonRandomHorizontalFlip
from .transformation import RandomRotate
from .dataset import MICCAIBraTSDataset
from .dataset import MICCAIBraTSSubDataset


def get_data_loader(dataset_name: str,
                    modalities: list,
                    root_dir_paths: list,
                    use_augmentation: bool,
                    use_shuffle: bool,
                    batch_size: int,
                    num_workers: int,
                    drop_last: bool,
                    initial_randomize: bool = True,
                    patient_ids: Optional[list] = None,
                    dataset_class: Optional[list] = None,
                    window_width: Optional[int] = None,
                    window_center: Optional[int] = None,
                    window_scale: Optional[int] = None,
                    ) -> data.DataLoader:

    assert dataset_name in {'MICCAIBraTSDataset', 'MICCAIBraTSSubDataset'}

    if use_augmentation:
        transform = transforms.Compose([RandomHorizontalFlip(),
                                        ToTensor(),
                                        RandomRotate()])
    else:
        transform = transforms.Compose([ToTensor()])

    if dataset_name == 'MICCAIBraTSDataset':
        dataset = MICCAIBraTSDataset(
            root_dir_paths=root_dir_paths,
            transform=transform,
            modalities=modalities,
            patient_ids=patient_ids,
            initial_randomize=initial_randomize,
        )

    elif dataset_name == 'MICCAIBraTSSubDataset':
        dataset_class='normal'
        assert dataset_class in {'normal', 'abnormal'}
        
        dataset = MICCAIBraTSSubDataset(
            mode=dataset_class,
            root_dir_paths=root_dir_paths,
            transform=transform,
            modalities=modalities,
            patient_ids=patient_ids,
            initial_randomize=initial_randomize,
        )

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=use_shuffle,
                           num_workers=num_workers,
                           drop_last=drop_last,
                           pin_memory=False)
