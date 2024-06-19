FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "starter.py", "./" ]

RUN pipenv install --system --deploy

EXPOSE 9696

CMD [ "python", "starter.py", "2023", "5" ]

# docker build -t homework04:v1 .
# docker run -it --rm -p 9696:9696 homework04:v1