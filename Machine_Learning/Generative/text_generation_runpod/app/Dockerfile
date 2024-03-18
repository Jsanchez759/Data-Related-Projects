FROM python:3.11-bullseye

WORKDIR /app

# Copy everything inside app folder app
COPY . /app

RUN pip install -r requirements.txt

CMD python server_model.py