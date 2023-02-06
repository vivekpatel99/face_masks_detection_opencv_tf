# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter
FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip

COPY requirements.txt /opt/app/
RUN pip install -r /opt/app/requirements.txt

EXPOSE 8888

ENTRYPOINT ["--ip=0.0.0.0","--allow-root","--no-browser"]
