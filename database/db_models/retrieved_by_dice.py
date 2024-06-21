from statistics import mean
import mongoengine as db
from torch import normal


class RetrievedByDice(db.Document):

    patient_id = db.StringField(required=True)
    dataset_name = db.StringField(required=True)
    slice_num = db.IntField(required=True)
    model_name = db.StringField(required=True)
    abnormal_dice = db.FloatField(required=True)
    normal_dice = db.FloatField(required=True)
    mean_dice = db.FloatField(required=True)

    @classmethod
    def create_record(cls,
                      patient_id: str,
                      dataset_name: str,
                      slice_num: int,
                      model_name: str,
                      abnormal_dice: float,
                      normal_dice: float, 
                      mean_dice: float,
                      ):

        record = cls(
            dataset_name=dataset_name,
            patient_id=patient_id,
            slice_num=slice_num,
            model_name=model_name,
            abnormal_dice=abnormal_dice,
            normal_dice=normal_dice,
            mean_dice=mean_dice,
        ).save()

        return record

    def is_identical(self, record):
        if (self.dataset_name == record.dataset_name) and \
           (self.patient_id == record.patient_id) and \
           (self.slice_num == record.slice_num):
            return True
        else:
            return False
