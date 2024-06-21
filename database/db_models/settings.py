MICCAI_BraTS = 'MICCAI_BraTS'
SLICE_POSTFIX = '_Slices'


DATASET_TYPES = [MICCAI_BraTS]


DATASET_NAMES = {
    MICCAI_BraTS: [
        'MICCAI_BraTS_2019_Data_Training',
        'MICCAI_BraTS_2019_Data_Validation',
        'MICCAI_BraTS_2019_Data_Testing'
    ]
}


MODEL_NAMES = {
    MICCAI_BraTS: [
        'bottom2x2_margin-10-epoch=0299'
    ]
}


DB_TYPE_TO_VECTOR_LENGTH = {
    MICCAI_BraTS: 2048
}


def parse_dataset_type(dataset_name):
    if dataset_name in DATASET_NAMES[MICCAI_BraTS]:
        dataset_type = MICCAI_BraTS

    else:
        raise Exception('Dataset type was not assigned properly.')

    return dataset_type
