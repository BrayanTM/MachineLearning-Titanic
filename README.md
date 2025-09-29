# Predicción de Supervivencia del Titanic con Árboles de Decisión

## Descripción del Proyecto

Este proyecto implementa un modelo de machine learning utilizando **árboles de decisión** para predecir la supervivencia de los pasajeros del RMS Titanic. El modelo analiza características demográficas y sociales de los pasajeros para determinar la probabilidad de supervivencia durante el naufragio histórico del 15 de abril de 1912.

## Objetivo

Desarrollar un modelo predictivo que pueda determinar si un pasajero del Titanic habría sobrevivido al desastre basándose en sus características personales y de viaje, utilizando algoritmos de árboles de decisión.

## Dataset

El proyecto utiliza el dataset `DataSet_Titanic.csv` que contiene las siguientes características:

### Variables Predictoras (Features)
- **Clase**: Clase del boleto (1ª, 2ª, 3ª clase)
- **Género**: Sexo del pasajero (0: Masculino, 1: Femenino)
- **Edad**: Edad del pasajero en años
- **HermEsp**: Número de hermanos/cónyuges a bordo
- **PadHij**: Número de padres/hijos a bordo

### Variable Objetivo (Target)
- **Sobreviviente**: Supervivencia (0: No sobrevivió, 1: Sobrevivió)

## Tecnologías Utilizadas

- **Python 3.x**
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Scikit-learn**: Implementación del algoritmo de árboles de decisión
- **Matplotlib**: Visualización de datos y resultados
- **Seaborn**: Gráficos estadísticos avanzados

## Estructura del Proyecto

```
proyecto-titanic/
│
├── main.py                 # Script principal con el modelo
├── README.md              # Documentación del proyecto
├── requeriments.txt       # Dependencias del proyecto
└── data/
    └── DataSet_Titanic.csv # Dataset del Titanic
```

## Funcionalidades Principales

### 1. Carga y Exploración de Datos
- Lectura del dataset CSV
- Visualización de las primeras filas
- Separación de variables predictoras y objetivo

### 2. Modelo de Árbol de Decisión
- **Algoritmo**: DecisionTreeClassifier de Scikit-learn
- **Parámetros**: 
  - `max_depth=4`: Profundidad máxima del árbol para evitar sobreajuste
  - `random_state=42`: Semilla para reproducibilidad

### 3. Evaluación del Modelo
- **Precisión (Accuracy)**: Porcentaje de predicciones correctas
- **Matriz de Confusión**: Análisis detallado de aciertos y errores
- **Matriz de Confusión Normalizada**: Porcentajes de clasificación

### 4. Visualizaciones
- **Árbol de Decisión**: Representación gráfica del modelo entrenado
- **Matriz de Confusión**: Visualización de resultados de clasificación
- **Importancia de Variables**: Gráfico de barras mostrando qué características son más importantes para la predicción

## Instalación y Uso

### Prerrequisitos
Asegúrate de tener Python 3.x instalado en tu sistema.

### Instalación de Dependencias
```bash
pip install -r requeriments.txt
```

### Ejecución del Proyecto
```bash
python main.py
```

## Resultados del Modelo

El modelo genera:

1. **Métricas de Rendimiento**: Precisión del modelo en el conjunto de datos
2. **Matriz de Confusión**: Tabla que muestra:
   - Verdaderos Positivos (TP): Supervivientes predichos correctamente
   - Verdaderos Negativos (TN): No supervivientes predichos correctamente
   - Falsos Positivos (FP): Predichos como supervivientes pero no sobrevivieron
   - Falsos Negativos (FN): Predichos como no supervivientes pero sí sobrevivieron

3. **Visualización del Árbol**: Muestra las reglas de decisión aprendidas por el modelo
4. **Importancia de Variables**: Ranking de qué características son más influyentes en la predicción

## Interpretación de Resultados

El árbol de decisión revela patrones importantes:
- **Género**: Históricamente, las mujeres tuvieron mayor probabilidad de supervivencia
- **Clase**: Los pasajeros de primera clase tuvieron ventajas de supervivencia
- **Edad**: Los niños pequeños tuvieron prioridad en los botes salvavidas
- **Familia**: El número de familiares a bordo puede influir en las decisiones

## Limitaciones del Modelo

- **Datos históricos**: El modelo refleja las circunstancias específicas del Titanic
- **Sobreajuste**: Con `max_depth=4` se limita la complejidad para generalizar mejor
- **Variables categóricas**: Algunas variables están codificadas numéricamente

## Posibles Mejoras

1. **Validación cruzada**: Implementar k-fold cross-validation
2. **Optimización de hiperparámetros**: Usar GridSearch o RandomSearch
3. **Ingeniería de características**: Crear nuevas variables derivadas
4. **Ensemble methods**: Combinar múltiples árboles (Random Forest)
5. **Datos de prueba separados**: División train/test para evaluación más robusta

## Contribuciones

Este proyecto es educativo y está abierto a mejoras. Áreas de contribución:
- Optimización del modelo
- Nuevas visualizaciones
- Análisis exploratorio más profundo
- Documentación adicional

## Licencia

Proyecto educativo - uso libre para fines académicos.

---

**Nota**: Este proyecto utiliza datos históricos del Titanic para fines educativos en machine learning y no debe usarse para decisiones críticas reales.
