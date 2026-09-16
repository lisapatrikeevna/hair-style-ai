from django.urls import path

from apps.users.views.register_view import RegistrationView

app_name = 'users'

urlpatterns = [
    path('register/', RegistrationView.as_view(), name='register'),
]