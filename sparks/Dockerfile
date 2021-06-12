FROM python:3.7.3

WORKDIR /opt/deepcardio
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH "/opt/deepcardio"
ENV DASH_HOST "0.0.0.0"

EXPOSE 8050

CMD ["python", "dash_app/deepcardio_app.py"]
