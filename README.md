# 🏠 Predicción de Precios en Airbnb (Madrid)

Proyecto de Machine Learning aplicado para predecir el precio nocturno de alojamientos en Airbnb utilizando datos reales. El objetivo no es solo maximizar métricas, sino aplicar un pipeline riguroso, reproducible y bien justificado metodológicamente.

## 📊 Dataset
- **Origen:** Datos reales extraídos de Airbnb mediante scraping.
- **Enfoque:** Alojamientos en Madrid.
- **Características:** Propiedades físicas, información del host, reviews, disponibilidad y ubicación.
- **Desafío principal:** Datos crudos con alta asimetría en el target, valores nulos dispersos y categorías duplicadas por codificación UTF-8.

## 🛠️ Metodología y Pipeline

El flujo de trabajo sigue las buenas prácticas de ML para evitar *data leakage* y garantizar generalización:

1. **Limpieza Inicial:** Eliminación de columnas con >40% de nulos y unificación de categorías duplicadas (ej. `City_Madrid` vs `City_马德里`).
2. **División:** `train_test_split` (80/20, `random_state=42`) para mantener un conjunto de test completamente ciego.
3. **Transformación del Target:** Aplicación de `np.log1p(Price)` para corregir la alta asimetría original (skewness: 4.37 → 0.39).
4. **Imputación y Escalado:** Relleno de nulos con mediana/moda (calculados solo en train) y estandarización con `StandardScaler`.
5. **Selección de Features (Lasso):** Regularización L1 con `GridSearchCV` redujo el espacio de variables de ~50 a **6 features clave**, eliminando ruido y multicolinealidad.
6. **Modelado y Optimización:** `GradientBoostingRegressor` ajustado mediante validación cruzada. Configuración óptima: `learning_rate=0.2`, `n_estimators=50`, `max_depth=3`.
