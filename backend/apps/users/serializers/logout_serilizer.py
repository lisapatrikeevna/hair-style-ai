from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken, TokenError


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        # Если токен передали в тела запроса, берем его, иначе попробуем взять из контекста cookie
        token = attrs.get('refresh') or self.context.get('request').COOKIES.get('refresh_token')
        if not token:
            # Если токена нет вовсе, разрешаем очистку кук без ошибки
            attrs['token'] = None
            return attrs

        attrs['token'] = token
        return attrs

    def save(self, **kwargs):
        token = self.validated_data.get('token')
        if token:
            try:
                RefreshToken(token).blacklist()
            except TokenError:
                # Если токен уже просрочен или занесен в блеклист — игнорируем
                pass

            