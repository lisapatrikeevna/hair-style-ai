from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from apps.users.serializers.logout_serilizer import LogoutSerializer


class LogoutView(generics.GenericAPIView):
    permission_classes = [AllowAny]
    serializer_class = LogoutSerializer

    @extend_schema(
        summary="User Logout",
        description="Blacklists the refresh token and clears access_token and refresh_token httpOnly cookies.",
        responses={200: {"description": "Logout successful, cookies cleared"}}
    )
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        serializer.save()

        response = Response(
            {"detail": "Logout successful, cookies cleared"},
            status=status.HTTP_200_OK
        )

        is_production = not settings.DEBUG
        samesite_value = 'None' if is_production else 'Lax'

        # Сбрасываем httpOnly куки
        response.delete_cookie('access_token', path='/', samesite=samesite_value)
        response.delete_cookie('refresh_token', path='/', samesite=samesite_value)

        return response
