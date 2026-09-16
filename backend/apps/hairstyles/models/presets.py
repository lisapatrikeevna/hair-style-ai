from django.db import models


class HairColorPreset(models.Model):
    name = models.CharField(max_length=50)
    hex_code = models.CharField(max_length=7)
    prompt = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class HairStylePreset(models.Model):
    class LengthChoices(models.TextChoices):
        SHORT = 'short', 'Short'
        MEDIUM = 'medium', 'Medium'
        LONG = 'long', 'Long'

    name = models.CharField(max_length=50)
    length = models.CharField(max_length=10, choices=LengthChoices.choices)
    prompt = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.length})"



