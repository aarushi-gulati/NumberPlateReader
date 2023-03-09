from .predict import predict
import os
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        result = predict(image_path)
        return render(request, 'result.html', {'result': result})
    return render(request, 'upload.html')
