from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from apps.users.serializers.login_serializer import UserResponseSerializer


class MeView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UserResponseSerializer

    @extend_schema(
        summary="Get Current User Profile",
        description="Returns profile information for the currently authenticated user (or guest).",
        responses={200: UserResponseSerializer, 401: "Unauthorized"}
    )
    def get(self, request, *args, **kwargs):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)