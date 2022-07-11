
FROM rocm/pytorch:latest

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

RUN \
git clone https://github.com/Lightning-AI/lightning.git && \
git clone https://github.com/pallets/click.git && \
cd click && \
python setup.py install && \ 
cd .. && \
pip install deepdiff && \
pip install fastapi && \
pip install lightning-cloud && \
pip install -U Jinja2 && \
pip install starsessions && \ 
pip uninstall fsspec && \ 
pip install fsspec==2022.1.0 && \ 
pip install jsonargparse && \
#needs docstring_parser-0.14.1
pip install -U jsonargparse[signatures] && \
#for flash 
python -m pip install urllib3 && \ 
#for bolts and transformers
#pip uninstall protobuf && \
pip install protobuf==3.9.2 && \
cd lightning && \ 
python setup.py install 
