from django.urls import path
#from predictor.views import predict, result
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('result/', views.result, name='result'),
    #path('', home,  name='home'),
    #path('predict/', predict, name='predict'),
]