import logging
from django.conf import settings
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView as _Swagger,
    SpectacularRedocView as _Redoc
)
from drf_spectacular.renderers import OpenApiJsonRenderer

logger = logging.getLogger(__name__)


def check_schema_access(request):
    """Helper function to log and control schema permissions."""
    logger.info(
        f"Schema access: user={getattr(request.user, 'username', 'Anonymous')}, "
        f"auth={request.user.is_authenticated}, "
        f"super={getattr(request.user, 'is_superuser', False)}"
    )

    if settings.DEBUG:
        return

    if not request.user.is_authenticated:
        raise PermissionDenied("Authentication required for API schema.")
    if not request.user.is_superuser:
        raise PermissionDenied(
            f"Superuser access required. Current user: {request.user.username}"
        )


class SpectacularJSONView(SpectacularAPIView):
    renderer_classes = [OpenApiJsonRenderer]
    authentication_classes = [JWTAuthentication, SessionAuthentication, BasicAuthentication]

    def get_permissions(self):
        if settings.DEBUG:
            return [AllowAny()]
        return [IsAdminUser()]

    def check_permissions(self, request):
        super().check_permissions(request)
        check_schema_access(request)


class SwaggerUI(_Swagger):
    authentication_classes = [JWTAuthentication, SessionAuthentication, BasicAuthentication]

    def get_permissions(self):
        if settings.DEBUG:
            return [AllowAny()]
        return [IsAdminUser()]

    def check_permissions(self, request):
        super().check_permissions(request)
        check_schema_access(request)


class RedocUI(_Redoc):
    authentication_classes = [JWTAuthentication, SessionAuthentication, BasicAuthentication]

    def get_permissions(self):
        if settings.DEBUG:
            return [AllowAny()]
        return [IsAdminUser()]

    def check_permissions(self, request):
        super().check_permissions(request)
        check_schema_access(request)



