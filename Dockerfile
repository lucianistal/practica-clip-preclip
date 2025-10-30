FROM python:3.9-slim

# Copiar requirements
COPY requirements.txt /opt/requirements.txt

# Crear virtualenv e instalar dependencias
RUN python -m venv /opt/.venv \
    && /opt/.venv/bin/pip install --upgrade pip \
    && /opt/.venv/bin/pip install -r /opt/requirements.txt

# Aseguramos que el virtualenv esté en el PATH
ENV PATH="/opt/.venv/bin:$PATH"

# Directorio de trabajo
WORKDIR /opt/project

# Copiar código
COPY src/ src/

# Comando por defecto
CMD ["bash"]

