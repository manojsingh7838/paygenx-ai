
from django.db import models

class QA(models.Model):
    question = models.TextField()
    answer = models.TextField()
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:50]
