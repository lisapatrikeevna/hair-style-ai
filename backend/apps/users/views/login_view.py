import logging
from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from apps.users.serializers.login_serializer import LoginSerializer, UserResponseSerializer

User = get_user_model()
logger = logging.getLogger(__name__)


def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> Response:
    """Helper function to set JWT httpOnly cookies."""
    is_production = not settings.DEBUG
    cookie_settings = {
        "httponly": True,
        "secure": is_production,
        "samesite": 'None' if is_production else 'Lax',
        "path": '/',
    }
    response.set_cookie(key='access_token', value=access_token, **cookie_settings)
    response.set_cookie(key='refresh_token', value=refresh_token, **cookie_settings)
    return response


class LoginView(generics.GenericAPIView):
    serializer_class = LoginSerializer
    permission_classes = [AllowAny]

    @extend_schema(
        summary="User Login",
        description="Authenticates user, cleans up previous guest session if exists, and sets access/refresh httpOnly cookies.",
        responses={200: UserResponseSerializer, 400: "Invalid credentials"}
    )
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Store reference to current guest before replacing session
        current_guest = request.user if (request.user.is_authenticated and request.user.is_guest) else None

        user = serializer.validated_data['user']

        # Delete orphan guest record if user is logging into an existing account
        if current_guest and current_guest.id != user.id:
            current_guest.delete()

        access_token, refresh_token = user.get_tokens()

        response = Response(UserResponseSerializer(user).data, status=status.HTTP_200_OK)
        return set_auth_cookies(response, access_token, refresh_token)


class GuestAuthView(generics.GenericAPIView):
    permission_classes = [AllowAny]

    @extend_schema(
        summary="Guest Session Initialization",
        description="Creates a new guest user or retrieves existing guest session and returns httpOnly cookies.",
        responses={200: UserResponseSerializer}
    )
    def post(self, request, *args, **kwargs):
        guest_user = User.objects.create_user(is_guest=True)
        access_token, refresh_token = guest_user.get_tokens()

        response = Response(UserResponseSerializer(guest_user).data, status=status.HTTP_201_CREATED)
        return set_auth_cookies(response, access_token, refresh_token)