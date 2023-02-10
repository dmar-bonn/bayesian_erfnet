FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
	apt-get install -y nano git python3-tk python3-pip python3-opencv wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    rm -r ~/.cache/pip

ENV PYTHONPATH="${PYTHONPATH}:/bayesian_erfnet"

WORKDIR "/bayesian_erfnet"
