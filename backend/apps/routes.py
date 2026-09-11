from django.urls import path, include

urlpatterns = [
    path("auth/", include("apps.users.urls")),
    # path("hairstyles/", include("apps.hairstyles.urls")),
]
