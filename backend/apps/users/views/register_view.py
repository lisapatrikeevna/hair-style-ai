import logging
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from apps.users.serializers.login_serializer import UserResponseSerializer
from apps.users.serializers.register_serializer import RegisterSerializer
from apps.users.views.login_view import set_auth_cookies

logger = logging.getLogger(__name__)


class RegistrationView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

    @extend_schema(
        summary="User Registration",
        description="Creates a new standard user account and returns authentication httpOnly cookies.",
        responses={201: UserResponseSerializer, 400: "Validation Errors"}
    )
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = serializer.save()
        logger.info("New user registered successfully: id=%s, email=%s", user.id, user.email)

        access_token, refresh_token = user.get_tokens()

        response = Response(
            UserResponseSerializer(user).data,
            status=status.HTTP_201_CREATED
        )

        return set_auth_cookies(response, access_token, refresh_token)