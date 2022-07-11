
FROM rocm/pytorch:latest

RUN \
    pip install click && \
    pip install deepdiff && \
    pip install fastapi && \
    pip install lightning-cloud && \
    pip install -U Jinja2 && \
    pip install starsessions && \ 
    pip uninstall fsspec && \ 
    pip install fsspec==2022.1.0 && \ 
    pip install jsonargparse && \
    pip install -U jsonargparse[signatures]

#for flash 
RUN python -m pip install urllib3

#for bolts and transformers
RUN pip install protobuf==3.9.2

RUN \
    git clone https://github.com/Lightning-AI/lightning.git && \
    cd lightning && \ 
    python setup.py install
    