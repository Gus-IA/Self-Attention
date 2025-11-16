# Self-Attention & MLP on MNIST (PyTorch Lightning)

Este proyecto implementa desde cero un pipeline completo para entrenar modelos de *Deep Learning* sobre el dataset MNIST usando **PyTorch** y **PyTorch Lightning**, explorando dos enfoques:

1. **MLP clásico** (baseline)
2. **Self-Attention escalar** aplicado a imágenes mediante parches (patch embeddings)

Incluye:
- Carga y preprocesamiento de MNIST desde *OpenML*
- Creación de datasets personalizados con Pandas y PyTorch
- DataModules de Lightning
- Entrenamiento y validación con Lightning Trainer
- Visualización de imágenes, parches y predicciones
- Implementación propia de *Scaled Dot-Product Self-Attention*


---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
