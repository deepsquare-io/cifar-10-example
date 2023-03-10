FROM ghcr.io/deepsquare-io/openmpi:devel-cuda11.8.0-cudnn8-rockylinux8 as base

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

RUN dnf install -y \
  python39 \
  && rm -rf /var/lib/apt/lists/*

FROM base AS python-deps

RUN dnf install -y \
  python39-pip \
  python39-devel \
  cmake \
  && rm -rf /var/lib/apt/lists/*

# Install pipenv and compilation dependencies
RUN /usr/bin/python3.9 -m pip install pipenv

ENV HOROVOD_WITH_PYTORCH=1
ENV HOROVOD_WITH_MPI=1
ENV HOROVOD_CUDA_HOME=/usr/local/cuda
ENV HOROVOD_WITHOUT_GLOO=1
ENV HOROVOD_GPU=CUDA

FROM python-deps AS builder

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv --python /usr/bin/python3.9 install --deploy

FROM base AS runtime

RUN dnf install -y \
  gcc-c++ \
  python39-devel \
  && dnf clean all

COPY --from=builder /.venv /.venv

WORKDIR /app

# Install application into container
COPY main.py model.py ./

# Run the application
ENTRYPOINT ["/.venv/bin/python3", "main.py"]
