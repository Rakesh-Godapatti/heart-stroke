from django.urls import path
from . import views 


urlpatterns = [
    
    path('', views.index,name="index"),
    path('about',views.about,name="about"),
    path('training',views.training,name="training"),
    path('prediction',views.prediction,name="prediction"),
    path('chart',views.chart,name="chart"),
]