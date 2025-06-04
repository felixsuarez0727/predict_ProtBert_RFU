# BERT-RFU: Predicción de Propiedades de Péptidos usando ProtBERT

Este proyecto implementa un modelo de regresión para predecir valores RFU de péptidos utilizando embeddings generados por ProtBERT. El núcleo del modelo es una arquitectura de red neuronal definida en `model/network.py` que combina:

1. **Embeddings de ProtBERT**: Para representación semántica de secuencias peptídicas
2. **Capas lineales con activación ReLU**: Para transformación de características
3. **Regularización Dropout**: Para prevenir sobreajuste
4. **Capa de salida lineal**: Para predicción de propiedades continuas

El modelo fue entrenado y evaluado en un conjunto de datos. El flujo completo de procesamiento de datos, entrenamiento y evaluación está documentado en el cuaderno principal `protbert_peptidos.ipynb`.

## Tabla de Contenidos

- [Requisitos Previos](#requisitos-previos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Resultados](#resultados)

## Requisitos Previos

- Python 3.8+
- PyTorch 1.10+
- Transformers
- scikit-learn
- pandas

## Estructura del Proyecto

```
BERT-rfu/
├── protbert_peptidos.ipynb    # Cuaderno principal de flujo de trabajo
├── data/
│   ├── data_rfu.csv           # Conjunto de datos principal
│   └── dataset.py             # Cargador de dataset personalizado
├── model/
│   └── network.py             # Arquitectura del modelo
└── saved_model/
    ├── predictions_plot.png   # Visualización de predicciones
    ├── scaler.pkl             # Escalador de características
    └── test_results.csv       # Métricas en conjunto de evaluación
```

## Uso

El flujo completo de procesamiento de datos, entrenamiento y evaluación está implementado en el cuaderno Jupyter `protbert_peptidos.ipynb`.

Para ejecutar el cuaderno:

1. Abra el cuaderno en un entorno con las dependencias instaladas (Google Colab o Jupyter local)
2. Monte su Google Drive si está usando Colab y actualice la ruta de trabajo:
   ```python
   %cd /content/drive/MyDrive/...  # Editar esta línea con la ruta a su directorio del proyecto
   ```
3. Ejecute las celdas secuencialmente

## Resultados

El modelo alcanzó las siguientes métricas de rendimiento en el conjunto de evaluación:

| Métrica  | Valor  |
| -------- | ------ |
| **RMSE** | 0.8251 |
| **MAE**  | 0.5573 |
| **R²**   | 0.5808 |

Los resultados completos de las predicciones y métricas pueden consultarse en [`saved_model/test_results.csv`](saved_model/test_results.csv).

**Interpretación**:

- **RMSE (Error Cuadrático Medio)**: 0.8251 indica la magnitud promedio del error de predicción
- **MAE (Error Absoluto Medio)**: 0.5573 muestra el error absoluto promedio
- **R² (Coeficiente de Determinación)**: 0.5808 sugiere que el modelo explica aproximadamente el 58% de la varianza
