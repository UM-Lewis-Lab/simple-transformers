FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as base
ENV LC_ALL C.UTF-8

# pin poetry version
ENV POETRY_VERSION 1.1.13

# reduce size of image
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# traceback on segfault
ENV PYTHONFAULTHANDLER 1
# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

ENV WORKDIR /src
ENV VIRTUAL_ENV /opt/venv

WORKDIR ${WORKDIR}

RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # git-state
      git \
      # primary interpreter
      python3.9 \
      # required by accelerate
      python3-distutils \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


FROM base AS build
# build dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # required by poetry
      python3-pip \
      python3.9-venv \
      # for deepspeed models
      python3.9-dev \
 && apt-get clean

RUN python3.9 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:${PATH}"

COPY pyproject.toml poetry.lock ./
RUN pip3 install "poetry==${POETRY_VERSION}" && poetry install --no-root

FROM base AS runtime
ENV PATH "${VIRTUAL_ENV}/bin:${PATH}"
COPY --from=build $VIRTUAL_ENV $VIRTUAL_ENV
