FROM python:3.11.5

WORKDIR /app

RUN apt-get update -y
RUN apt install pkg-config
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

# FROM python:3.11.5-slim
# WORKDIR /app
# COPY --from=stage-one /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
# COPY --from=stage-one /app /app
EXPOSE 5000

CMD python3 flaskapp.py