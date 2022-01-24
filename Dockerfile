FROM continuumio/miniconda3:4.9.2 AS build
RUN conda create -n callc-streamlit numpy pandas pip python=3.8 rdkit r-base r-rjava r-mgcv r-doparallel -c conda-forge -c bioconda -c r && \
    conda install conda-pack -c conda-forge && \
    conda-pack -n callc-streamlit -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack
FROM debian:buster-slim AS runtime
WORKDIR /callc
COPY --from=build /venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY rt /callc
RUN chmod 777 /callc/temp
RUN pip install joblib==1.0 mordred==1.2 networkx==2.6 scikit-learn==0.24 scipy==1.7 streamlit>=0.88 threadpoolctl==2.2 \
    xgboost==1.4 plotly==5.3 pygam \
    PyQt5>=5 typing-extensions==4 statsmodels
EXPOSE 8502
CMD ["streamlit", "run", "callc_streamlit.py", "--server.port", "8502"]