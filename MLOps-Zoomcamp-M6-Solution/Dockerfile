FROM python:3.11-slim

RUN pip install -U pip pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "integraton-test/", "integraton-test/" ]
COPY [ "model/", "model/" ]
COPY [ "pycode/", "pycode/" ]
COPY [ "tests/", "tests/" ]

ENTRYPOINT [ "python" ]
CMD [ "batch.py 2023 1" ]