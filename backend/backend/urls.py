from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.permissions import AllowAny

from .api_schema import SpectacularJSONView, SwaggerUI, RedocUI

urlpatterns = [
    # Admin Panel
    path("admin/", admin.site.urls),

    # Main API Routing
    path("api/v1/", include("apps.routes")),

    # API Documentation (drf-spectacular)
    path("api/schema/", SpectacularJSONView.as_view(permission_classes=[AllowAny]), name="schema-json"),
    path("api/docs/", SwaggerUI.as_view(url_name="schema-json", permission_classes=[AllowAny]), name="swagger-ui"),
    path("api/redoc/", RedocUI.as_view(url_name="schema-json", permission_classes=[AllowAny]), name="redoc"),
]

# Static & Media handling in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
