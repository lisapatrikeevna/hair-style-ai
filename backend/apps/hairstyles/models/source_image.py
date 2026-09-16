import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


def user_source_path(instance, filename: str) -> str:
    ext = filename.split('.')[-1]
    return f'users/user_{instance.user.id}/sources/{uuid.uuid4()}.{ext}'


def user_masked_path(instance, filename: str) -> str:
    ext = filename.split('.')[-1]
    return f'users/user_{instance.user.id}/masked/{uuid.uuid4()}.{ext}'


class SourceImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='source_images')
    original_image = models.ImageField(upload_to=user_source_path)
    masked_image = models.ImageField(upload_to=user_masked_path, null=True, blank=True)
    is_processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Source {self.id} for User {self.user_id}"


