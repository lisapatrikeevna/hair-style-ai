from django.urls import path

from apps.users.views.get_me import MeView
from apps.users.views.login_view import LoginView, GuestAuthView
from apps.users.views.logout_view import LogoutView
from apps.users.views.register_view import RegistrationView

app_name = 'users'

urlpatterns = [
    path('register/', RegistrationView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('guest/', GuestAuthView.as_view(), name='guest-auth'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('me/', MeView.as_view(), name='me'),
]