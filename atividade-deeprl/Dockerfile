# Ambiente da atividade "Boas Práticas em DeepRL".
# Só CPU: os ambientes da atividade (Acrobot, CartPole) são pequenos e não ganham com GPU.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# git: registrar o commit em metadata.json (opcional, mas recomendado)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU: ~200 MB, contra vários GB da versão com CUDA.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Teste de fumaça durante o build: se falhar, a imagem não é criada.
RUN pytest -q tests

EXPOSE 6006
CMD ["bash"]
