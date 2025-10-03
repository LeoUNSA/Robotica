# BMSSP - Bounded Multi-Source Shortest Path

## Descripción General

Este proyecto implementa el algoritmo BMSSP (Bounded Multi-Source Shortest Path), una extensión del algoritmo de Dijkstra que calcula los caminos más cortos desde múltiples vértices fuente hacia todos los demás vértices en un grafo dirigido ponderado, con la restricción de que las distancias totales no excedan un límite máximo especificado.

## Características del Algoritmo

### Funcionalidad Principal
- **Múltiples fuentes**: Calcula caminos desde varios vértices de origen simultáneamente
- **Restricción de límite**: Solo considera caminos cuya distancia total no exceda el bound especificado
- **Optimización**: Utiliza una cola de prioridad para garantizar eficiencia en la búsqueda
- **Reconstrucción de caminos**: Permite obtener la secuencia completa de vértices en el camino óptimo
- **Medición de rendimiento**: Incluye medición precisa del tiempo de ejecución del algoritmo

### Complejidad Computacional
- **Temporal**: O((V + E) log V × S)
  - V: número de vértices
  - E: número de aristas
  - S: número de fuentes
- **Espacial**: O(V × S) para almacenar distancias desde cada fuente

### Métricas de Rendimiento
- **Precisión temporal**: Medición con resolución de microsegundos
- **Reportes duales**: Tiempo mostrado en segundos y milisegundos
- **Exclusión de E/O**: Solo mide el tiempo del algoritmo central, excluyendo lectura/escritura de archivos

## Estructura del Proyecto

```
/
├── README.md                 # Documentación principal
├── BMSSP_README.md          # Documentación técnica detallada
├── bmssp.cpp                # Implementación del algoritmo
├── Makefile                 # Archivo de compilación
├── input.txt                # Archivo de entrada básico
├── input_complex.txt        # Ejemplo con grafo complejo
└── output.txt               # Archivo de salida (generado)
```

## Instalación y Compilación

### Requisitos del Sistema
- Compilador C++ compatible con C++17 o superior
- Make (opcional, para usar el Makefile)
- Sistema operativo: Linux, macOS, Windows (con MinGW)

### Compilación

#### Usando Makefile (recomendado):
```bash
# Compilar el programa
make

# Compilar y ejecutar
make run

# Limpiar archivos generados
make clean

# Recompilar desde cero
make rebuild

# Mostrar ayuda
make help
```

#### Compilación manual:
```bash
g++ -std=c++17 -Wall -Wextra -O2 -o bmssp bmssp.cpp
```

## Uso del Programa

### Formato del Archivo de Entrada

El programa lee un archivo `input.txt` con el siguiente formato:

```
<número_vértices> <número_aristas>
<origen1> <destino1> <peso1>
<origen2> <destino2> <peso2>
...
<número_fuentes>
<fuente1> <fuente2> ... <fuenteN>
<límite_máximo>
```

### Ejemplo de Entrada

```
6 9
0 1 4
0 2 2
1 2 1
1 3 5
2 3 8
2 4 10
3 4 2
3 5 6
4 5 3
3
0 1 2
15
```

**Interpretación:**
- Grafo con 6 vértices (numerados 0-5) y 9 aristas
- Aristas dirigidas con sus respectivos pesos
- Tres vértices fuente: 0, 1, y 2
- Límite máximo de distancia: 15

### Ejecución

```bash
./bmssp
```

El programa generará:
1. **Salida en consola**: Visualización del grafo y resultados en tiempo real
2. **Archivo output.txt**: Resultados detallados en formato estructurado

## Interpretación de Resultados

### Salida en Consola
```
=== ALGORITMO BMSSP ===
Grafo cargado exitosamente desde input.txt
Fuentes: 0 1 2 
Límite (bound): 15

Grafo:
Vértice 0: (1, peso: 4) (2, peso: 2) 
...

=== EJECUTANDO BMSSP ===
Algoritmo ejecutado en: 0.000023 segundos
Algoritmo ejecutado en: 0.023 milisegundos

=== RESULTADOS ===
Desde fuente 0:
Vértices alcanzables dentro del límite 15:
  Vértice 0 -> Distancia: 0
  Vértice 1 -> Distancia: 4
  ...
```

### Archivo de Salida (output.txt)
```
=== RESULTADOS DEL ALGORITMO BMSSP ===
Límite (bound): 15
Fuentes: 0 1 2 
Tiempo de ejecución: 0.000023 segundos
Tiempo de ejecución: 0.023 milisegundos

--- DESDE FUENTE 0 ---
Vértices alcanzables:
  Vértice 0 -> Distancia: 0
  Vértice 1 -> Distancia: 4
  ...
```

## Implementación Técnica

### Estructuras de Datos Principales

1. **Clase Graph**: Representa el grafo mediante lista de adyacencia
2. **Struct Edge**: Almacena destino y peso de cada arista
3. **Priority Queue**: Optimiza la selección del siguiente vértice a procesar
4. **Matrices de distancias**: Almacenan distancias mínimas desde cada fuente
5. **Cronómetro de alta precisión**: Utiliza `std::chrono::high_resolution_clock` para medición temporal

### Algoritmo Principal

1. **Inicialización**: Configurar distancias infinitas y agregar fuentes a la cola
2. **Medición de inicio**: Captura timestamp antes de ejecutar el algoritmo
3. **Procesamiento**: Extraer vértice con menor distancia y actualizar vecinos
4. **Restricción de límite**: Descartar caminos que excedan el bound
5. **Optimización**: Evitar reprocesamiento de vértices ya optimizados
6. **Medición de finalización**: Calcular tiempo transcurrido con precisión de microsegundos

## Casos de Uso

### Aplicaciones Prácticas
- **Logística**: Optimización de rutas de distribución desde múltiples centros
- **Redes de comunicación**: Análisis de conectividad con restricciones de latencia
- **Robótica**: Planificación de trayectorias con limitaciones energéticas
- **Análisis de grafos sociales**: Medición de influencia limitada en redes

### Ventajas del Enfoque Multi-fuente
- Eficiencia computacional al procesar múltiples orígenes simultáneamente
- Comparación directa de accesibilidad desde diferentes puntos de partida
- Identificación de regiones óptimas de cobertura

## Limitaciones y Consideraciones

### Restricciones del Algoritmo
- Requiere pesos de aristas no negativos
- Diseñado para grafos dirigidos (puede adaptarse a no dirigidos)
- El límite debe ser apropiado para el tamaño y estructura del grafo

### Optimizaciones Implementadas
- Terminación temprana cuando se excede el límite
- Uso de structured bindings (C++17) para mayor legibilidad
- Gestión eficiente de memoria con contenedores STL
- **Medición de rendimiento de alta precisión** con `chrono::high_resolution_clock`
- **Reporte dual de tiempo** en segundos y milisegundos para diferentes escalas
- **Separación de timing**: Solo mide el algoritmo core, excluyendo I/O

### Archivos de Ejemplo

### Ejemplo Básico (input.txt)
Grafo de 6 vértices diseñado para demostrar funcionalidad básica
- **Tiempo típico**: ~0.02-0.05 milisegundos
- **Complejidad**: Ideal para pruebas rápidas y validación

### Ejemplo Complejo (input_complex.txt)
Grafo de 10 vértices con 4 fuentes para pruebas de rendimiento
- **Tiempo típico**: ~0.03-0.08 milisegundos  
- **Complejidad**: Mayor densidad de aristas para análisis de escalabilidad

### Análisis de Rendimiento
- **Grafos pequeños (≤10 vértices)**: Ejecución en microsegundos
- **Escalabilidad**: El tiempo crece logarítmicamente con el tamaño del grafo
- **Comparación**: Permite evaluar el impacto del número de fuentes y densidad de aristas

## Contribución y Desarrollo

### Estructura del Código
- Código modular con separación clara de responsabilidades
- Documentación inline para funciones principales
- Manejo robusto de errores de entrada/salida

### Posibles Extensiones
- Soporte para grafos no dirigidos
- Implementación de algoritmos relacionados (A*, Bellman-Ford acotado)
- Interfaz gráfica para visualización de resultados
- Paralelización para grafos de gran escala
- **Benchmarking avanzado** con múltiples iteraciones y análisis estadístico
- **Profiling detallado** por fases del algoritmo
- **Comparación de rendimiento** con otras implementaciones de caminos más cortos

## Información del Curso

Este proyecto forma parte del repositorio del curso de Robótica, implementado como ejemplo práctico de algoritmos de búsqueda en grafos con aplicaciones en navegación y planificación de rutas.
