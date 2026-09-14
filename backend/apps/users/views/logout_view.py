import logging
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema
from django.contrib.auth import get_user_model

from apps.users.serializers.logout_serilizer import LogoutSerializer
from apps.users.serializers.login_serializer import UserResponseSerializer
from apps.users.views.login_view import set_auth_cookies

User = get_user_model()
logger = logging.getLogger(__name__)


class LogoutView(generics.GenericAPIView):
    serializer_class = LogoutSerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="User Logout",
        description="Blacklists current refresh token and initializes a new guest session.",
        responses={200: UserResponseSerializer}
    )
    def post(self, request, *args, **kwargs):
        # 1. Blacklist current token if passed in body/cookies
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            try:
                serializer.save()
            except Exception as e:
                logger.warning(f"Failed to blacklist token: {e}")

        # 2. Immediately create a new guest session
        guest_user = User.objects.create_user(is_guest=True)
        access_token, refresh_token = guest_user.get_tokens()

        # 3. Overwrite cookies with new guest tokens
        response = Response(UserResponseSerializer(guest_user).data, status=status.HTTP_200_OK)
        return set_auth_cookies(response, access_token, refresh_token)