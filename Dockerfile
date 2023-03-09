FROM registry-1.docker.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as base

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

RUN apt update -y \
  && apt install -y \
  python3 \
  && rm -rf /var/lib/apt/lists/*

FROM base AS python-deps

RUN apt update -y \
  && apt install -y \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Install pipenv and compilation dependencies
RUN pip3 install pipenv

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

FROM base AS runtime

RUN apt update -y \
  && apt install -y \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv

WORKDIR /app

# Install application into container
COPY . .

# Run the application
ENTRYPOINT ["/.venv/bin/python3", "main.py"]
