import os
import pathlib
import numpy as np
from tqdm import tqdm
import mongoengine as db
from mongoengine import PULL
from operator import itemgetter

from .retrieved_by_example import RetrievedByExample
from .retrieved_by_dice import RetrievedByDice
from .settings import SLICE_POSTFIX
from .settings import MODEL_NAMES
from .settings import parse_dataset_type


BATCH_SIZE = 10


class ImageBase(db.DynamicDocument):

    SLICE_ROOT_DIR_PATH = ""
    NIFTI_ROOT_DIR_PATH = ""
    CODE_ROOT_DIR_PATH = ""

    dataset_type = db.StringField(required=True)
    dataset_name = db.StringField(required=True)
    patient_id = db.StringField(required=True)
    slice_num = db.IntField(required=True)
    unique_key = db.StringField(required=True, unique=True)
    db_index = db.IntField(required=True, unique=True)

    is_abnormal = db.BooleanField(default=False)
    is_representative = db.BooleanField(default=False)

    retrieved_by_imagenet_feature = db.ListField(db.ReferenceField(
        'RetrievedByExample', reverse_delete_rule=PULL
    ), required=False)

    retrieved_by_finetuned_feature = db.ListField(db.ReferenceField(
        'RetrievedByExample', reverse_delete_rule=PULL
    ), required=False)

    retrieved_by_imagenet_feature_all = db.ListField(db.ReferenceField(
        'RetrievedByExample', reverse_delete_rule=PULL
    ), required=False)

    retrieved_by_finetuned_feature_all = db.ListField(db.ReferenceField(
        'RetrievedByExample', reverse_delete_rule=PULL
    ), required=False)

    retrieved_by_dice = db.ListField(db.ReferenceField(
        'RetrievedByDice', reverse_delete_rule=PULL
    ), required=False)

    _model_name = None
    _slice_dir_path = None
    _nifti_dir_path = None
    _code_dir_path = None

    meta = {
        'allow_inheritance': True,
        'strict': False,
        'db_alias': 'default',
    }

    @classmethod
    def init_paths(cls,
                   SLICE_ROOT_DIR_PATH,
                   NIFTI_ROOT_DIR_PATH,
                   CODE_ROOT_DIR_PATH):
        cls.SLICE_ROOT_DIR_PATH = pathlib.Path(SLICE_ROOT_DIR_PATH)
        cls.NIFTI_ROOT_DIR_PATH = pathlib.Path(NIFTI_ROOT_DIR_PATH)
        cls.CODE_ROOT_DIR_PATH = pathlib.Path(CODE_ROOT_DIR_PATH)

    @classmethod
    def create_record(cls,
                      dataset_name: str,
                      patient_id: str,
                      slice_num: int,
                      is_abnormal: bool,
                      db_index: int,
                      ):

        dataset_type = parse_dataset_type(dataset_name)

        unique_key = dataset_name + '_' + \
            patient_id + '_' + str(slice_num).zfill(4)

        record = cls(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            patient_id=patient_id,
            slice_num=slice_num,
            unique_key=unique_key,
            is_abnormal=is_abnormal,
            db_index=db_index,
        ).save()

        return record

    @classmethod
    def get_record_by_db_index(cls, db_index):
        return cls.objects.get(db_index=db_index)

    @classmethod
    def get_all_records_in_dataset(cls, dataset_name):
        return cls.objects(dataset_name=dataset_name).batch_size(BATCH_SIZE)

    @classmethod
    def is_empty_database(cls, dataset_name):
        try:
            records = cls.get_all_records_in_dataset(dataset_name)

            if len(records) > 0:
                return False
            else:
                return True

        except db.DoesNotExist:
            return True

    @classmethod
    def get_all_normal_records_in_dataset(cls, dataset_name):
        return [r for r in cls.get_all_records_in_dataset(dataset_name)
                if not r.is_abnormal]

    @classmethod
    def get_all_abnormal_records_in_dataset(cls, dataset_name):
        return [r for r in cls.get_all_records_in_dataset(dataset_name)
                if r.is_abnormal]

    @classmethod
    def get_all_representative_records_in_dataset(cls, dataset_name):
        return [r for r in cls.get_all_records_in_dataset(dataset_name)
                if r.is_representative]

    @classmethod
    def get_all_patient_ids(cls, dataset_name):
        return set(r.patient_id
                   for r in cls.get_all_records_in_dataset(dataset_name))

    def set_as_representative(self):
        self.update(set__is_representative=True)
        self.reload()

    def unset_as_representative(self):
        self.update(set__is_representative=False)
        self.reload()

    def set_as_abnormal(self):
        self.update(set__is_abnormal=True)
        self.reload()

    def unset_as_abnormal(self):
        self.update(set__is_abnormal=False)
        self.reload()

    def add_retrieved(self, retrieved, feature_type):
        assert feature_type in ['imagenet_feature',
                                'finetuned_feature',
                                'imagenet_feature_all',
                                'finetuned_feature_all',
                                'dice']
        self.update(**{'add_to_set__retrieved_by_' + feature_type: retrieved})
        self.reload()

    def remove_retrieved(self, retrieved, feature_type):
        assert feature_type in ['imagenet_feature',
                                'finetuned_feature',
                                'imagenet_feature_all',
                                'finetuned_feature_all',
                                'dice']
        self.update(**{'pull__retrieved_by_' + feature_type: retrieved})
        self.reload()

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        assert model_name in MODEL_NAMES[self.dataset_type]
        self._model_name = model_name

    @property
    def slice_dir_path(self):
        if self._slice_dir_path is None:
            self._slice_root_dir_path = self.SLICE_ROOT_DIR_PATH / \
                (self.dataset_name + SLICE_POSTFIX) / self.patient_id

        return self._slice_root_dir_path

    @property
    def code_dir_path(self):
        try:
            assert self.model_name is not None
        except Exception as e:
            print('Model name was not specified: ', e.args)

        if self._code_dir_path is None:
            self._code_dir_path = self.CODE_ROOT_DIR_PATH / \
                self.dataset_name / self.model_name / self.patient_id

        return self._code_dir_path

    @property
    def eac(self):
        code_path = self.code_dir_path / \
            ('eac_' + str(self.slice_num).zfill(4) + '.npy')
        return np.load(code_path).flatten().astype(np.float32)

    @property
    def nac(self):
        code_path = self.code_dir_path / \
            ('nac_' + str(self.slice_num).zfill(4) + '.npy')
        return np.load(code_path).flatten().astype(np.float32)

    @property
    def aac(self):
        code_path = self.code_dir_path / \
            ('aac_' + str(self.slice_num).zfill(4) + '.npy')
        return np.load(code_path).flatten().astype(np.float32)

    @property
    def imagenet_feature(self):
        code_dir_path = self.CODE_ROOT_DIR_PATH / self.dataset_name / \
            'resnet-not-finetuned' / self.patient_id
        code_path = code_dir_path / \
            ('feature_' + str(self.slice_num).zfill(4) + '.npy')
        return np.load(code_path).astype(np.float32)

    @property
    def finetuned_feature(self):
        code_dir_path = self.CODE_ROOT_DIR_PATH / self.dataset_name / \
            'resnet-finetuned' / self.patient_id
        code_path = code_dir_path / \
            ('feature_' + str(self.slice_num).zfill(4) + '.npy')
        return np.load(code_path).astype(np.float32)

    def is_same_patient(self, target):
        if self.patient_id == target.patient_id:
            return True

        return False

    def is_same_slice(self, target):
        if self.patient_id == target.patient_id:
            if self.slice_num == target.slice_num:
                return True

        return False

    def is_identical(self, record):
        if (self.dataset_name == record.dataset_name) and \
           (self.patient_id == record.patient_id) and \
           (self.slice_num == record.slice_num):
            return True

        else:
            return False


class BraTSImage(ImageBase):

    n_slices = 155
    modalities = ['t1', 't1ce', 'flair']

    @classmethod
    def build_dataset(cls, dataset_name: str):
        dataset_root_path = cls.SLICE_ROOT_DIR_PATH / \
            (dataset_name + SLICE_POSTFIX)

        db_index = 0

        print('Building BraTSImage database...')
        for patient_id in tqdm(os.listdir(dataset_root_path)):
            patient_dir_path = dataset_root_path / patient_id
            abnormal_areas = []

            for slice_num in range(cls.n_slices):
                label_path = patient_dir_path / \
                    (patient_id + '_seg_' + str(slice_num).zfill(4) + '.npy')
                label = np.load(label_path).astype(np.int32)

                abnormal_area = (label > 0).sum()

                if abnormal_area > 0:
                    is_abnormal = True
                else:
                    is_abnormal = False

                record = cls.create_record(
                    dataset_name=dataset_name,
                    patient_id=patient_id,
                    slice_num=slice_num,
                    is_abnormal=is_abnormal,
                    db_index=db_index,
                )

                abnormal_areas.append({
                    'record': record,
                    'area': abnormal_area,
                })

                db_index += 1

            abnormal_areas = sorted(abnormal_areas, key=itemgetter('area'))
            largest_record = abnormal_areas[-1]['record']
            largest_record.set_as_representative()

    @property
    def image(self):
        images = []
        for modality in self.modalities:
            file_name = self.patient_id + '_' + modality + \
                '_' + str(self.slice_num).zfill(4) + '.npy'
            image_path = self.slice_dir_path / file_name

            image = np.load(image_path)[np.newaxis, ...]
            images.append(image)

        images = np.concatenate(images, axis=0)

        return images

    @property
    def label(self):
        file_name = self.patient_id + '_seg_' + \
            str(self.slice_num).zfill(4) + '.npy'
        image_path = self.slice_dir_path / file_name

        label = np.load(image_path)
        label[label == 4] = 3

        return label

    @property
    def anatomy(self):
        file_name = self.patient_id + '_anatomy_' + \
            str(self.slice_num).zfill(4) + '.npy'
        image_path = self.slice_dir_path / file_name

        label = np.load(image_path)

        return label

    @property
    def nifti_dir_path(self):
        if self._nifti_dir_path is None:
            self._nifti_dir_path = self.NIFTI_ROOT_DIR_PATH \
                / self.dataset_name \
                / self.patient_id

        return self._nifti_dir_path

    def get_nifti_image_path(self, modality='t1ce'):
        assert modality in {'t1', 't1ce', 'flair'}
        file_name = self.patient_id + '_' + modality + '.nii.gz'
        image_path = self.nifti_dir_path / file_name
        return str(image_path)
