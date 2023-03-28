FROM ubuntu:20.04

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update
RUN apt-get -y install curl
RUN apt-get -y install poppler-utils
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip
RUN apt-get -y install tesseract-ocr
RUN apt-get -y install tesseract-ocr-por

RUN pip3 install fuzzywuzzy
RUN pip3 install opencv-python-headless
RUN pip3 install pdf2image
RUN pip3 install pytesseract
RUN pip3 install python-Levenshtein

COPY . /app
WORKDIR /app

CMD python3 src/nfe_ocr.py

