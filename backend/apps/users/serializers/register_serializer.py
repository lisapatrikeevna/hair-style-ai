from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password

User = get_user_model()


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    password_confirm = serializers.CharField(
        write_only=True,
        style={'input_type': 'password'}
    )

    class Meta:
        model = User
        fields = ['email', 'username', 'password', 'password_confirm']

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({"password": "Passwords do not match."})

        email = attrs.get('email')
        if User.objects.filter(email=email).exists():
            raise serializers.ValidationError({"email": "User with this email already exists."})

        return attrs

    def create(self, validated_data):
        request = self.context.get('request')
        validated_data.pop('password_confirm', None)
        password = validated_data.pop('password')

        if request and request.user.is_authenticated and request.user.is_guest:
            user = request.user
            user.email = validated_data.get('email')
            user.username = validated_data.get('username')
            user.is_guest = False
            user.set_password(password)
            user.save()
            return user

        return User.objects.create_user(password=password, **validated_data)