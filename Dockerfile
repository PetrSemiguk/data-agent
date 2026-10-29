# Same Python version as the local .venv
FROM python:3.14.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes.
# --only-binary: every package must come as a ready wheel, nothing is compiled.
COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

# A user without root rights; it may write only to the app's data folders.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data uploads outputs logs memory \
    && chown app:app uploads outputs logs memory

# Code is owned by root, so the app (and the kernel) cannot change it.
COPY . .

# The root file system is read-only at run time: Jupyter, IPython and
# matplotlib keep their runtime and cache files in /tmp instead of $HOME.
ENV JUPYTER_RUNTIME_DIR=/tmp/jupyter/runtime \
    JUPYTER_DATA_DIR=/tmp/jupyter/data \
    JUPYTER_CONFIG_DIR=/tmp/jupyter/config \
    IPYTHONDIR=/tmp/ipython \
    MPLCONFIGDIR=/tmp/matplotlib

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
