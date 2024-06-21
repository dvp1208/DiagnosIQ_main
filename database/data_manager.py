import os
import pathlib
import numpy as np
from tqdm import tqdm
import mongoengine as db
from annoy import AnnoyIndex
from operator import itemgetter
from mongoengine.queryset.visitor import Q
from mongoengine.connection import disconnect
from multiprocessing import Pool

from .db_models import BraTSImage
from .db_models import RetrievedByExample
from .db_models import RetrievedByDice
from .db_models import parse_dataset_type
from .db_models import MICCAI_BraTS
from .db_models import DB_TYPE_TO_VECTOR_LENGTH


RESNET_MODELS = {'imagenet_feature', 'finetuned_feature'}
RESNET_MODELS_ALL = {'imagenet_feature_all', 'finetuned_feature_all'}
TOPK = 30
N_PROCESSES = 8
ERROR_MESSAGE = 'Only the MICCAI BraTS 2019 dataset is available.'


def unwrap_self_f(arg, **kwarg):
    return DatabaseManager.f(*arg, **kwarg)


def unwrap_self_g(arg, **kwarg):
    return DatabaseManager.g(*arg, **kwarg)


def db_connect(project_name, dataset_type, alias, host):
    db_name = project_name + dataset_type.replace(' ', '-')

    return db.connect(db=db_name,
                      alias=alias,
                      host=host)


def db_drop(project_name, dataset_type):
    database = db_connect(dataset_type)
    db_name = project_name + dataset_type.replace(' ', '-')
    database.drop_database(db_name)


def calc_dice(label, output, n_classes, ignore_index=0, eps=1e-5):
    dice = []

    for i in range(n_classes):
        if i == ignore_index:
            continue

        ls = label == i
        os = output == i

        inter = np.sum(ls * os)
        union = np.sum(ls) + np.sum(os)
        score = 2.0 * inter / (union + eps)

        dice.append(score)

    mean_dice = np.mean(dice)

    return mean_dice


class DatabaseManager(object):

    distance_metric = 'euclidean'

    def __init__(self,
                 project_name,
                 dataset_name,
                 annoy_model_path,
                 slice_root_dir_path,
                 nifti_root_dir_path,
                 code_root_dir_path,
                 model_name=None,
                 n_trees=10,
                 vector_length=None,
                 init_nearest_neighbors=False,
                 alias='default',
                 host='localhost'):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_type = parse_dataset_type(self.dataset_name)
        self.model_name = model_name
        self.annoy_model_path = pathlib.Path(annoy_model_path)

        self.n_trees = n_trees

        if vector_length is None:
            self.vector_length = DB_TYPE_TO_VECTOR_LENGTH[self.dataset_type]

        else:
            self.vector_length = vector_length

        self.init_db(project_name, alias, host)

        self.init_image_model(slice_root_dir_path,
                              nifti_root_dir_path,
                              code_root_dir_path)

        if self.model_name is not None:
            if self.model_name in RESNET_MODELS:
                if init_nearest_neighbors:
                    self.init_nearest_neighbors()

            elif self.model_name in RESNET_MODELS_ALL:
                if init_nearest_neighbors:
                    self.init_nearest_neighbors_all()

            elif self.model_name == 'dice':
                if init_nearest_neighbors:
                    self.init_nearest_neighbors_dice()

            else:
                self.init_annoy_model()

    def init_db(self, project_name, alias, host):
        db_connect(project_name, self.dataset_name, alias, host)

    def init_image_model(self,
                         slice_root_dir_path,
                         nifti_root_dir_path,
                         code_root_dir_path):

        if self.dataset_type == MICCAI_BraTS:
            BraTSImage.init_paths(
                SLICE_ROOT_DIR_PATH=slice_root_dir_path,
                NIFTI_ROOT_DIR_PATH=nifti_root_dir_path,
                CODE_ROOT_DIR_PATH=code_root_dir_path,
            )
            self.image_model = BraTSImage

        else:
            raise NotImplementedError(ERROR_MESSAGE)

        self.image_model.model_name = self.model_name

        if self.image_model.is_empty_database(self.dataset_name):
            self.image_model.build_dataset(self.dataset_name)

    def get_image_model(self):
        return self.image_model

    def get_all_images(self):
        return self.image_model.get_all_records_in_dataset(self.dataset_name)

    def get_all_representative_images(self):
        return self.image_model.get_all_representative_records_in_dataset(
            self.dataset_name)

    def get_all_normal_images(self):
        return self.image_model.get_all_normal_records_in_dataset(
            self.dataset_name)

    def get_all_abnormal_images(self):
        return self.image_model.get_all_abnormal_records_in_dataset(
            self.dataset_name)

    def get_all_patient_ids(self):
        return self.image_model.get_all_patient_ids(self.dataset_name)

    def get_patient_images(self, patient_id):
        return self.image_model.objects((
            Q(dataset_name=self.dataset_name) & Q(patient_id=patient_id)
        ))

    def get_image(self, patient_id, slice_num):
        images = self.image_model.objects((
            Q(dataset_name=self.dataset_name) & Q(
                patient_id=patient_id) & Q(slice_num=slice_num)
        ))
        assert len(images) == 1
        return images[0]

    def get_representative_image(self, patient_id):
        images = self.image_model.objects((
            Q(dataset_name=self.dataset_name) & Q(
                patient_id=patient_id) & Q(is_representative=True)
        ))
        assert len(images) == 1
        return images[0]

    def init_annoy_model(self):
        try:
            assert self.model_name is not None
        except Exception:
            raise Exception('Model name was not specified.')

        save_dir_path = self.annoy_model_path \
            / self.dataset_type \
            / self.dataset_name
        annoy_model_name = self.model_name + '-n-' + str(self.n_trees) + '.ann'
        annoy_model_path = save_dir_path / annoy_model_name

        if not os.path.exists(annoy_model_path):
            annoy_model = self.build_annoy_model()
            os.makedirs(save_dir_path, exist_ok=True)
            annoy_model.save(str(annoy_model_path))

        else:
            annoy_model = AnnoyIndex(self.vector_length, self.distance_metric)
            annoy_model.load(str(annoy_model_path))

        self.annoy_model = annoy_model

    def get_annoy_model(self):
        return self.annoy_model

    def build_annoy_model(self):
        annoy_model = AnnoyIndex(self.vector_length, self.distance_metric)

        print('Building annoy model for {} in {}.'.format(
            self.model_name, self.dataset_name))
        for image_record in tqdm(self.image_model.get_all_records_in_dataset(
                self.dataset_name)):
            db_index = image_record.db_index

            if self.model_name == 'imagenet_feature_annoy':
                feature = image_record.imagenet_feature

            elif self.model_name == 'finetuned_feature_annoy':
                feature = image_record.finetuned_feature

            else:
                feature = image_record.eac

            annoy_model.add_item(db_index, feature)

        annoy_model.build(self.n_trees)

        return annoy_model

    def query(self, query, topk):
        results = []

        for db_index in self.annoy_model.get_nns_by_vector(query, topk):
            results.append(
                self.image_model.get_record_by_db_index(db_index)
            )

        return results

    def get_nearest_slices_with_patient_filter(self,
                                               query_record,
                                               topk_patient,
                                               topk_record=100):
        patient_ids = []
        filtered_results = []

        query = query_record.eac
        raw_results = self.query(query, topk=topk_record)

        for result in raw_results:
            if result.is_same_patient(query_record):
                continue

            if result.patient_id not in patient_ids:
                patient_ids.append(result.patient_id)
                filtered_results.append(result)

        filtered_results = filtered_results[:topk_patient]

        return filtered_results

    def get_nearest_slices_with_patient_filter_by_nac(self,
                                                      query_record,
                                                      topk_patient,
                                                      topk_record=100):
        patient_ids = []
        filtered_results = []

        query = query_record.nac
        raw_results = self.query(query, topk=topk_record)

        for result in raw_results:
            if result.is_same_patient(query_record):
                continue

            if result.patient_id not in patient_ids:
                patient_ids.append(result.patient_id)
                filtered_results.append(result)

        filtered_results = filtered_results[:topk_patient]

        return filtered_results

    def get_the_nearest_slice(self, query_record):
        query = query_record.eac
        raw_results = self.query(query, topk=2)

        top1_result = raw_results[0]
        top2_result = raw_results[1]

        if not query_record.is_same_slice(top1_result):
            nearest = top1_result
        else:
            nearest = top2_result

        return nearest

    def f(self, q_record):
        print(q_record.patient_id)

        all_records = self.get_all_images()
        q_patient_id = q_record.patient_id
        q_vector = getattr(q_record, self.model_name[:-4])

        results = []
        for r_record in tqdm(all_records):
            r_patient_id = r_record.patient_id

            if q_patient_id == r_patient_id:
                continue

            r_vector = getattr(r_record, self.model_name[:-4])

            distance = np.linalg.norm(q_vector - r_vector)

            results.append({
                'record': r_record,
                'distance': distance,
            })

        results = sorted(results, key=itemgetter('distance'))
        topk_nearest_records = results[:TOPK]

        for nearest in topk_nearest_records:
            nearest_record = nearest['record']

            retrieved = RetrievedByExample.create_record(
                patient_id=nearest_record.patient_id,
                dataset_name=self.dataset_name,
                slice_num=nearest_record.slice_num,
                model_name=self.model_name,
                distance_value=nearest['distance'],
            )

            q_record.add_retrieved(retrieved, self.model_name)

    def init_nearest_neighbors_all(self):
        pool = Pool(processes=N_PROCESSES)
        all_records = self.get_all_images()
        pool.map(unwrap_self_f, zip([self] * len(all_records), all_records))

    def init_nearest_neighbors_dice(self, topk=TOPK):
        all_rep_records = self.get_all_representative_images()
        all_records = self.get_all_images()

        if self.dataset_type == MICCAI_BraTS:
            n_classes_normal = 7
            n_classes_abnormal = 4

        else:
            raise NotImplementedError(ERROR_MESSAGE)

        for q_record in tqdm(all_rep_records):
            q_patient_id = q_record.patient_id
            q_normal = q_record.anatomy
            q_abnormal = q_record.label

            results = []
            for r_record in tqdm(all_records):
                r_patient_id = r_record.patient_id

                if q_patient_id == r_patient_id:
                    continue

                r_normal = r_record.anatomy
                r_abnormal = r_record.label

                dice_normal = calc_dice(
                    r_normal, q_normal, n_classes=n_classes_normal)
                dice_abnormal = calc_dice(
                    r_abnormal, q_abnormal, n_classes=n_classes_abnormal)
                dice_mean = (dice_normal + dice_abnormal) / 2.0

                results.append({
                    'record': r_record,
                    'dice_normal': dice_normal,
                    'dice_abnormal': dice_abnormal,
                    'dice_mean': dice_mean,
                })

            results = sorted(results, key=itemgetter(
                'dice_mean'), reverse=True)
            topk_nearest_records = results[:topk]

            for nearest in topk_nearest_records:
                nearest_record = nearest['record']

                retrieved = RetrievedByDice.create_record(
                    patient_id=nearest_record.patient_id,
                    dataset_name=self.dataset_name,
                    slice_num=nearest_record.slice_num,
                    model_name=self.model_name,
                    abnormal_dice=nearest['dice_abnormal'],
                    normal_dice=nearest['dice_normal'],
                    mean_dice=nearest['dice_mean'],
                )

                q_record.add_retrieved(retrieved, self.model_name)

    def init_nearest_neighbors(self, topk=TOPK):
        all_rep_records = self.get_all_representative_images()

        for q_record in tqdm(all_rep_records):
            q_patient_id = q_record.patient_id
            q_vector = getattr(q_record, self.model_name)

            results = []
            for r_record in all_rep_records:
                r_patient_id = r_record.patient_id

                if q_patient_id == r_patient_id:
                    continue

                r_vector = getattr(r_record, self.model_name)

                distance = np.linalg.norm(q_vector - r_vector)

                results.append({
                    'record': r_record,
                    'distance': distance,
                })

            results = sorted(results, key=itemgetter('distance'))
            topk_nearest_records = results[:topk]

            for nearest in topk_nearest_records:
                nearest_record = nearest['record']

                retrieved = RetrievedByExample.create_record(
                    patient_id=nearest_record.patient_id,
                    dataset_name=self.dataset_name,
                    slice_num=nearest_record.slice_num,
                    model_name=self.model_name,
                    distance_value=nearest['distance'],
                )

                q_record.add_retrieved(retrieved, self.model_name)

    def calc_isolated_samples(self, topk):
        all_rep_records = self.get_all_representative_images()

        isolated_samples = []
        for q_record in tqdm(all_rep_records):
            count = 0

            for r_record in all_rep_records:
                if r_record.patient_id == q_record.patient_id:
                    continue

                nearest_records = getattr(
                    r_record, 'retrieved_by_' + self.model_name)[:topk]

                for n_record in nearest_records:
                    if n_record.is_identical(q_record):
                        count += 1

            if count == 0:
                isolated_samples.append(q_record)

        return isolated_samples

    def check_if_retrieved_all(self):
        all_records = self.get_all_images()
        attr = 'retrieved_by_' + self.model_name

        count = 0
        for q_record in tqdm(all_records):
            if hasattr(q_record, attr):
                retrieved = getattr(q_record, attr)
                if len(retrieved) == 30:
                    count += 1

                elif len(retrieved) > 30:
                    raise Exception('Too long!!')

                else:
                    raise Exception('Too short!!')

        if count == len(all_records):
            return True
        else:
            return False

    def g(self, q_record):
        all_records = self.get_all_images()

        count = 0
        for r_record in all_records:
            if r_record.patient_id == q_record.patient_id:
                continue

            nearest_records = getattr(
                r_record, 'retrieved_by_' + self.model_name)[:TOPK]

            for n_record in nearest_records:
                if n_record.is_identical(q_record):
                    count += 1
                    break

        return q_record if count == 0 else None

    def _calc_isolated_samples_all(self, topk):
        pool = Pool(processes=N_PROCESSES)
        all_records = self.get_all_images()
        map_responses = pool.map(unwrap_self_g, zip(
            [self] * len(all_records), all_records))
        filtered_results = list(filter(lambda a: a is not None, map_responses))
        return filtered_results

    def calc_isolated_samples_all(self, topk):
        all_records = self.get_all_images()

        isolated_samples = []
        for q_record in tqdm(all_records):
            count = 0

            for r_record in all_records:
                if r_record.patient_id == q_record.patient_id:
                    continue

                nearest_records = getattr(
                    r_record, 'retrieved_by_' + self.model_name)[:topk]

                for n_record in nearest_records:
                    if n_record.is_identical(q_record):
                        count += 1
                        break

            if count == 0:
                isolated_samples.append(q_record)

        return isolated_samples
