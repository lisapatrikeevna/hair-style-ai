from django.db import models

import uuid
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from rest_framework_simplejwt.tokens import RefreshToken


class CustomUserManager(BaseUserManager):
    def create_user(self, email=None, password=None, **extra_fields):
        if email:
            email = self.normalize_email(email)

        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()

        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_guest', False)

        if not email:
            raise ValueError("Superuser must have an email address.")

        return self.create_user(email=email, password=password, **extra_fields)


class User(AbstractUser):
    username = models.CharField(max_length=55, unique=True, blank=True, null=True, verbose_name='Username')
    email = models.EmailField(unique=True, blank=True, null=True, verbose_name='Email')

    is_guest = models.BooleanField(default=False, verbose_name='Is Guest')
    guest_uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, verbose_name='Guest UUID')

    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Created At')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        if self.is_guest:
            return f"Guest-{self.guest_uuid}"
        return self.email or f"User-{self.id}"

    def get_tokens(self):
        """Generates access and refresh tokens for the user."""
        refresh = RefreshToken.for_user(self)
        return str(refresh.access_token), str(refresh)

    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']
