import uuid
from django.db import models
from django.contrib.auth import get_user_model
from apps.hairstyles.models.presets import HairColorPreset, HairStylePreset
from apps.hairstyles.models.source_image import SourceImage

User = get_user_model()


def user_result_path(instance, filename: str) -> str:
    ext = filename.split('.')[-1]
    return f'users/user_{instance.user.id}/results/{uuid.uuid4()}.{ext}'


class GenerationResult(models.Model):
    class StatusChoices(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        SUCCESS = 'success', 'Success'
        FAILED = 'failed', 'Failed'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='generations')
    source_image = models.ForeignKey(SourceImage, on_delete=models.CASCADE, related_name='generations')
    color_preset = models.ForeignKey(HairColorPreset, on_delete=models.SET_NULL, null=True, blank=True)
    style_preset = models.ForeignKey(HairStylePreset, on_delete=models.SET_NULL, null=True, blank=True)
    is_ai_recommended = models.BooleanField(default=False)
    result_image = models.ImageField(upload_to=user_result_path, null=True, blank=True)
    status = models.CharField(max_length=15, choices=StatusChoices.choices, default=StatusChoices.PENDING)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Generation {self.id} ({self.status}) for User {self.user_id}"




