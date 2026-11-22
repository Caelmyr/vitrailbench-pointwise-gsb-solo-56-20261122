FROM python:3.11-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 VITRAIL_DATA=/var/lib/vitrail OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /opt/vitrail
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY vitrail vitrail
COPY static static
RUN useradd -U -u 11509 vitrail && install -d -o vitrail -g vitrail /var/lib/vitrail
USER vitrail
EXPOSE 8000
CMD ["python", "-m", "vitrail"]
