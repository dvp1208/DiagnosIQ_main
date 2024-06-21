import mongoengine as db
from mongoengine import PULL
from datetime import datetime

from .result import ResultSummary


class User(db.Document):

    created_at = db.DateTimeField(default=datetime.now, required=True)
    user_name = db.StringField(required=True)
    y_experience = db.IntField(required=True)

    results = db.ListField(db.ReferenceField(ResultSummary, reverse_delete_rule=PULL))

    @classmethod
    def create_record(cls, user_name, y_experience):
        record = cls(
            user_name=user_name,
            y_experience=y_experience,
        ).save()

        return record
