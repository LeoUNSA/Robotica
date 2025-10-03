# BMSSP - Bounded Multi-Source Shortest Path

## Descripción del Algoritmo

El algoritmo BMSSP (Bounded Multi-Source Shortest Path) es una variante del algoritmo de Dijkstra que encuentra los caminos más cortos desde múltiples fuentes hacia todos los demás vértices en un grafo dirigido ponderado, con la restricción de que las distancias no excedan un límite especificado (bound).

### Características principales:

1. **Multi-fuente**: Permite calcular caminos desde múltiples nodos fuente simultáneamente
2. **Acotado**: Solo considera caminos cuya distancia total no exceda el límite especificado
3. **Eficiente**: Utiliza una cola de prioridad para optimizar la búsqueda
4. **Completo**: Proporciona tanto las distancias como la posibilidad de reconstruir los caminos

## Formato del Archivo de Entrada (input.txt)

```
<número_de_vértices> <número_de_aristas>
<fuente1> <destino1> <peso1>
<fuente2> <destino2> <peso2>
...
<número_de_fuentes>
<fuente1> <fuente2> ... <fuenteN>
<límite_máximo>
```

### Ejemplo:
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

Este ejemplo representa:
- Un grafo con 6 vértices (0-5) y 9 aristas
- Las aristas con sus pesos respectivos
- 3 fuentes: vértices 0, 1 y 2
- Límite máximo de distancia: 15

## Compilación y Ejecución

### Usando Makefile:
```bash
# Compilar
make

# Compilar y ejecutar
make run

# Limpiar archivos generados
make clean

# Ver ayuda
make help
```

### Compilación manual:
```bash
g++ -std=c++17 -Wall -Wextra -O2 -o bmssp bmssp.cpp
./bmssp
```

## Archivos Generados

### output.txt
Contiene los resultados del algoritmo incluyendo:
- Las fuentes utilizadas
- El límite máximo aplicado
- Para cada fuente, los vértices alcanzables y sus distancias mínimas
- Ejemplos de caminos reconstruidos

## Complejidad

- **Temporal**: O((V + E) log V * S), donde:
  - V = número de vértices
  - E = número de aristas  
  - S = número de fuentes
- **Espacial**: O(V * S) para almacenar las distancias desde cada fuente

## Casos de Uso

1. **Análisis de redes**: Encontrar puntos accesibles desde múltiples centros de distribución
2. **Planificación de rutas**: Calcular rutas desde múltiples puntos de origen con restricciones de distancia
3. **Análisis de grafos sociales**: Determinar la influencia limitada desde múltiples nodos fuente
4. **Robótica**: Planificación de caminos desde múltiples posiciones iniciales con limitaciones de energía

## Notas Importantes

- Los vértices se numeran desde 0
- El algoritmo maneja grafos dirigidos
- Los pesos de las aristas deben ser no negativos
- Si no existe camino dentro del límite, el vértice no aparecerá en los resultados
