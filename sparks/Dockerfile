FROM tensorflow/tensorflow

WORKDIR /opt/deepcardio
COPY requirements4tensort.txt .
RUN pip install -r requirements4tensort.txt

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ENV PYTHONPATH "/opt/deepcardio"
ENV DASH_HOST "0.0.0.0"

EXPOSE 8050

CMD ["python", "/opt/deepcardio/dash_app/deepcardio_app.py"]
