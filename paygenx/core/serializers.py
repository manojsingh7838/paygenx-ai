from rest_framework import serializers
from .models import QA

class QASerializer(serializers.ModelSerializer):
    class Meta:
        model = QA
        fields = '__all__'



# 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