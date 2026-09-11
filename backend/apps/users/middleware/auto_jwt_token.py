from datetime import datetime
import threading
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken

_thread_locals = threading.local()


def get_current_user():
    return getattr(_thread_locals, 'user', None)


def set_current_user(user):
    _thread_locals.user = user


class JWTAuthenticationMiddleware(MiddlewareMixin):

    def process_request(self, request: Request, **kwargs):
        exempt_prefixes = []
        if getattr(settings, "STATIC_URL", None):
            exempt_prefixes.append(settings.STATIC_URL)
        if getattr(settings, "MEDIA_URL", None):
            exempt_prefixes.append(settings.MEDIA_URL)

        if request.path.startswith(tuple(exempt_prefixes)) or request.method == "OPTIONS":
            return None

        access_token = request.COOKIES.get('access_token')
        refresh_token = request.COOKIES.get('refresh_token')
        current_user = None

        if access_token:
            try:
                token = AccessToken(access_token)
                current_user = token['user_id']
                if datetime.fromtimestamp(token['exp']) < datetime.now():
                    raise TokenError('Token is expired')
                request.META['HTTP_AUTHORIZATION'] = f"Bearer {access_token}"

            except TokenError:
                new_access_token = self.refresh_access_token(refresh_token)

                if new_access_token:
                    request.META['HTTP_AUTHORIZATION'] = f"Bearer {new_access_token}"
                    request._new_access_token = new_access_token
                    current_user = AccessToken(new_access_token)['user_id']
                else:
                    self.clear_jwt_cookies(request)

        elif refresh_token:
            new_access_token = self.refresh_access_token(refresh_token)
            if new_access_token:
                request.META["HTTP_AUTHORIZATION"] = f"Bearer {new_access_token}"
                request._new_access_token = new_access_token
                current_user = AccessToken(new_access_token)['user_id']
            else:
                self.clear_jwt_cookies(request)

        if current_user is not None:
            set_current_user(current_user)

    def process_response(self, request: Request, response: Response, **kwargs):
        new_access_token = getattr(request, "_new_access_token", None)
        if new_access_token:
            access_expiry = AccessToken(new_access_token)["exp"]

            is_local = getattr(settings, "IS_LOCAL", True)
            secure_flag = not is_local
            samesite_val = "Lax" if is_local else "None"

            response.set_cookie(
                key="access_token",
                value=new_access_token,
                httponly=True,
                secure=secure_flag,
                samesite=samesite_val,
                expires=datetime.fromtimestamp(access_expiry),
                path='/',
            )
        return response

    def refresh_access_token(self, refresh_token):
        try:
            refresh = RefreshToken(refresh_token)
            new_access_token = str(refresh.access_token)
            return new_access_token
        except TokenError:
            return None

    def clear_jwt_cookies(self, request: Request):
        request.COOKIES.pop("access_token", None)
        request.COOKIES.pop("refresh_token", None)