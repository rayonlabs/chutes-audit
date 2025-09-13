FROM python:3.12
RUN apt update && apt-get -y install libportaudio2 build-essential curl postgresql-client
RUN curl -sSL https://install.python-poetry.org | python3 -
ADD pyproject.toml /audit/
ADD poetry.lock /audit/
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.local/bin:/root/.cargo/bin:$PATH
WORKDIR /audit
RUN poetry install --no-root
ADD audit.py /audit/audit.py
