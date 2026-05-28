# Predicción académica con Random Forest

Este proyecto fue adaptado para trabajar con el archivo `dataset.csv`. Aunque el repositorio original estaba orientado a predicción de préstamos, el dataset cargado corresponde a información académica de estudiantes y su variable objetivo es `Target`, con tres posibles clases:

- `Dropout`: estudiante en riesgo de deserción o retiro.
- `Enrolled`: estudiante aún matriculado.
- `Graduate`: estudiante graduado.

## Objetivo del proyecto

Construir una aplicación en Streamlit que permita explorar el dataset, entrenar un modelo de Random Forest y realizar predicciones individuales sobre el resultado académico de un estudiante.

## Estructura

```text
proyecto_random_forest_estudiantes/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv
└── models/
```

## Variables del dataset

El dataset contiene 4.424 registros y 35 columnas. La columna objetivo es `Target`. Las demás columnas se usan como variables predictoras, por ejemplo:

- Estado civil.
- Modo de aplicación.
- Curso.
- Nacionalidad.
- Deudor.
- Pago de matrícula al día.
- Género.
- Becario.
- Edad al momento de matrícula.
- Unidades curriculares inscritas, evaluadas y aprobadas.
- Promedio académico del primer y segundo semestre.
- Tasa de desempleo, inflación y PIB.

## Resultados aproximados del modelo

Con una partición 80/20 y Random Forest con 200 árboles, el modelo obtiene resultados aproximados de:

- Accuracy: 78.08 %
- F1 macro: 70.50 %
- Recall macro: 69.36 %

Estos valores pueden cambiar ligeramente si se modifican los parámetros o la partición de entrenamiento.

## Instalación

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Entrenar el modelo:

```bash
python train_model.py
```

3. Ejecutar la aplicación:

```bash
streamlit run app.py
```

## Explicación del modelo

Random Forest es un algoritmo de aprendizaje supervisado basado en múltiples árboles de decisión. Cada árbol realiza una predicción y el bosque combina esos resultados para producir una clasificación más estable. En este caso, el modelo aprende patrones entre variables académicas, personales y económicas para clasificar el estado final del estudiante.

## Nota importante

Este modelo no debe utilizarse como una decisión definitiva sobre un estudiante. Su función es servir como apoyo analítico para identificar tendencias, posibles riesgos académicos y oportunidades de acompañamiento institucional.
