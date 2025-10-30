# Análisis de Similitud Multimodal: Pre-CLIP y CLIP

Proyecto de práctica sobre similitud multimodal entre imágenes y captions. Incluye un enfoque **Pre-CLIP** con modelos estándar de imagen y texto, y un enfoque **CLIP** con modelo multimodal pre-entrenado.

**Autor:** Lucía Nistal Palacios  
**Asignatura:** Sistemas Interactivos Inteligentes  

---

## Descripción del Proyecto

Este proyecto compara dos enfoques para medir la similitud entre imágenes y sus descripciones textuales:

1. **Pre-CLIP**: Usa modelos independientes para procesar imágenes (ResNet18) y texto (DistilUSE multilingual), calculando similitudes entre embeddings de espacios diferentes.

2. **CLIP**: Utiliza el modelo multimodal CLIP de OpenAI, que proyecta imágenes y texto en un mismo espacio vectorial, permitiendo comparaciones directas.

El objetivo es demostrar cómo CLIP mejora significativamente la alineación semántica entre modalidades visuales y textuales.

---

## Estructura del Proyecto

```
project/
│
├── dataset/                  # Dataset de imágenes organizadas por categorías
│   ├── animales/            # 5 imágenes de animales
│   ├── colores/             # 5 imágenes de colores
│   ├── ropa/                # 5 imágenes de ropa
│   ├── colegio/             # 5 imágenes de material escolar
│   └── captions.csv         # Descripciones de las imágenes
│
├── src/                      # Código 
│   ├── extract_image_embeddings_preclip.py
│   ├── extract_text_embeddings_preclip.py
│   ├── compute_preclip_similarity.py
│   ├── extract_image_embeddings.py
│   ├── extract_text_embeddings.py
│   ├── compute_clip_similarity.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/                  # Resultados generados
│   ├── image_embeddings_preclip.pt
│   ├── text_embeddings_preclip.pt
│   ├── preclip_similarities.csv
│   ├── image_embeddings_clip.pt
│   ├── text_embeddings_clip.pt
│   ├── clip_similarities.csv
│   └── similarity_comparison.png
│
├── requirements.txt
├── Dockerfile
├── Makefile
└── README.md
```

---

## Requisitos

- Docker
- Python 3.9+
- Make (opcional)

---

## Instalación y Ejecución

### 1. Construir la imagen Docker

```bash
make build
```

O manualmente:

```bash
docker build -t clip-practice .
```

### 2. Ejecutar el análisis completo

#### Enfoque Pre-CLIP

```bash
# Extraer embeddings de imágenes con ResNet18
docker run --rm -v $(pwd):/opt/project clip-practice python src/extract_image_embeddings_preclip.py

# Extraer embeddings de texto con DistilUSE
docker run --rm -v $(pwd):/opt/project clip-practice python src/extract_text_embeddings_preclip.py

# Calcular matriz de similitud
docker run --rm -v $(pwd):/opt/project clip-practice python src/compute_preclip_similarity.py
```

#### Enfoque CLIP

```bash
# Extraer embeddings de imágenes con CLIP
docker run --rm -v $(pwd):/opt/project clip-practice python src/extract_image_embeddings.py

# Extraer embeddings de texto con CLIP
docker run --rm -v $(pwd):/opt/project clip-practice python src/extract_text_embeddings.py

# Calcular matriz de similitud
docker run --rm -v $(pwd):/opt/project clip-practice python src/compute_clip_similarity.py
```

#### Evaluación y comparación

```bash
# Generar gráficos comparativos
docker run --rm -v $(pwd):/opt/project clip-practice python src/evaluate.py
```


---

## Dataset

El dataset incluye 20 imágenes organizadas en 4 categorías:

- **Animales** (5): pájaro, gato, perro, serpiente, conejo
- **Colores** (5): amarillo, rojo, rosa, verde, azul
- **Ropa** (5): chaqueta, zapatos, camiseta, sombrero, pantalones
- **Colegio** (5): calculadora, mochila, lápiz, reloj, goma de borrar

Cada imagen tiene su descripción en inglés en `dataset/captions.csv`.

---

## Resultados

El script `evaluate.py` genera:

- **Matrices de similitud**: CSV con similitudes coseno entre todas las parejas imagen-caption
- **Gráfico comparativo**: Muestra la similitud promedio diagonal (imagen-caption correcta) para ambos enfoques


## Modelos Utilizados

### Pre-CLIP
- **Imágenes**: ResNet18 pre-entrenado (torchvision)
- **Texto**: `sentence-transformers/distiluse-base-multilingual-cased`

### CLIP
- **Modelo**: `openai/clip-vit-base-patch32` (Transformers de HuggingFace)

---


