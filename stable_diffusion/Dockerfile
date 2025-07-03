FROM stable_diffusion_xl

COPY ./ ./

WORKDIR /

# RUN pip install -r ./requirements.txt --default-timeout=10000

CMD ["uvicorn", "stable_diffusion_xl:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

# FROM python:3.10

# COPY ./ ./

# WORKDIR /

# RUN pip install -r ./requirements.txt --default-timeout=10000

# CMD ["uvicorn", "stable_diffusion_xl:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

