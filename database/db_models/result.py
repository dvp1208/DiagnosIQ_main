import os
import datetime
import numpy as np
import mongoengine as db
from mongoengine import PULL
from datetime import datetime
import matplotlib.pyplot as plt


class RetrievedBySketch(db.Document):

    created_at = db.DateTimeField(default=datetime.now, required=True)
    retrieved_num = db.IntField(required=True)
    patient_id = db.StringField(required=True)
    slice_num = db.IntField(required=True)
    eval_point_1 = db.BooleanField(required=True)
    eval_point_2 = db.BooleanField(required=True)
    eval_point_3 = db.BooleanField(required=True)
    true_ratio = db.FloatField(required=True)
    is_same_patient = db.BooleanField(required=True)


class ResultSummary(db.Document):

    created_at = db.DateTimeField(default=datetime.now, required=True)
    user_unique_key = db.StringField(required=True)
    user_name = db.StringField(required=True)
    y_experience = db.IntField(required=True)

    stage_num = db.IntField(required=True)
    question_num = db.IntField(required=True)

    results = db.ListField(db.ReferenceField(RetrievedBySketch, reverse_delete_rule=PULL))

    template_image_path = db.StringField(required=True)
    sketch_image_path = db.StringField(required=True)
    matched_retrieved_num = db.IntField(required=True)

    @classmethod
    def save_summary(cls, user_unique_key, user_record, stage_num, question_num,
                     template_image_path, sketch_image_path, matched_retrieved_num,
                     result_summary):

        record = cls(
            user_unique_key=user_unique_key,
            user_name=user_record.user_name,
            y_experience=user_record.y_experience,
            stage_num=stage_num,
            question_num=question_num,
            template_image_path=template_image_path,
            sketch_image_path=sketch_image_path,
            matched_retrieved_num=matched_retrieved_num,
        ).save()

        for result_dict in result_summary:
            result_record = RetrievedBySketch(
                retrieved_num=result_dict['retrieved_num'],
                patient_id=result_dict['patient_id'],
                slice_num=result_dict['slice_num'],
                eval_point_1=result_dict['eval_point_1'],
                eval_point_2=result_dict['eval_point_2'],
                eval_point_3=result_dict['eval_point_3'] if 'eval_point_3' in result_dict.keys() else False,
                true_ratio=result_dict['true_ratio'],
                is_same_patient=result_dict['is_same_patient'],
            ).save()

            record.update(add_to_set__results=result_record)
            record.reload()

        user_record.update(add_to_set__results=record)
        user_record.reload()
