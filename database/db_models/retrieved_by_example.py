import mongoengine as db


class RetrievedByExample(db.Document):

    patient_id = db.StringField(required=True)
    dataset_name = db.StringField(required=True)
    slice_num = db.IntField(required=True)
    model_name = db.StringField(required=True)
    distance_value = db.FloatField(required=True)

    @classmethod
    def create_record(cls,
                      patient_id: str,
                      dataset_name: str,
                      slice_num: int,
                      model_name: str,
                      distance_value: float,
                      ):

        record = cls(
            dataset_name=dataset_name,
            patient_id=patient_id,
            slice_num=slice_num,
            model_name=model_name,
            distance_value=distance_value,
        ).save()

        return record

    def is_identical(self, record):
        if (self.dataset_name == record.dataset_name) and \
           (self.patient_id == record.patient_id) and \
           (self.slice_num == record.slice_num):
            return True
        else:
            return False
