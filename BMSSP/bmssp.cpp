#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;

struct Edge {
    int destination;
    int weight;
    
    Edge(int dest, int w) : destination(dest), weight(w) {}
};

class Graph {
private:
    int vertices;
    vector<vector<Edge>> adjacencyList;
    
public:
    Graph(int v) : vertices(v) {
        adjacencyList.resize(v);
    }
    
    void addEdge(int source, int destination, int weight) {
        adjacencyList[source].push_back(Edge(destination, weight));
    }
    
    // Implementación del algoritmo BMSSP (Bounded Multi-Source Shortest Path)
    map<int, vector<int>> bmssp(const vector<int>& sources, int bound) {
        // Inicializar distancias con infinito
        vector<vector<int>> dist(sources.size(), vector<int>(vertices, INT_MAX));
        vector<vector<int>> parent(sources.size(), vector<int>(vertices, -1));
        map<int, vector<int>> shortestPaths;
        
        // Priority queue: (distancia, vértice, índice_fuente)
        priority_queue<tuple<int, int, int>, 
                      vector<tuple<int, int, int>>, 
                      greater<tuple<int, int, int>>> pq;
        
        // Inicializar fuentes
        for (int i = 0; i < sources.size(); i++) {
            int source = sources[i];
            dist[i][source] = 0;
            pq.push({0, source, i});
        }
        
        while (!pq.empty()) {
            auto [currentDist, currentVertex, sourceIndex] = pq.top();
            pq.pop();
            
            // Si la distancia actual es mayor que la almacenada, saltar
            if (currentDist > dist[sourceIndex][currentVertex]) {
                continue;
            }
            
            // Si excede el límite, no procesar más
            if (currentDist > bound) {
                continue;
            }
            
            // Procesar todos los vecinos
            for (const Edge& edge : adjacencyList[currentVertex]) {
                int nextVertex = edge.destination;
                int newDist = currentDist + edge.weight;
                
                // Si encontramos un camino más corto y dentro del límite
                if (newDist <= bound && newDist < dist[sourceIndex][nextVertex]) {
                    dist[sourceIndex][nextVertex] = newDist;
                    parent[sourceIndex][nextVertex] = currentVertex;
                    pq.push({newDist, nextVertex, sourceIndex});
                }
            }
        }
        
        // Construir los caminos más cortos para cada fuente
        for (int i = 0; i < sources.size(); i++) {
            vector<int> sourcePaths;
            for (int v = 0; v < vertices; v++) {
                if (dist[i][v] != INT_MAX && dist[i][v] <= bound) {
                    sourcePaths.push_back(v);
                }
            }
            shortestPaths[sources[i]] = sourcePaths;
        }
        
        return shortestPaths;
    }
    
    // Función para obtener distancias desde múltiples fuentes
    map<int, vector<pair<int, int>>> getDistancesFromSources(const vector<int>& sources, int bound) {
        vector<vector<int>> dist(sources.size(), vector<int>(vertices, INT_MAX));
        map<int, vector<pair<int, int>>> distances;
        
        priority_queue<tuple<int, int, int>, 
                      vector<tuple<int, int, int>>, 
                      greater<tuple<int, int, int>>> pq;
        
        // Inicializar fuentes
        for (int i = 0; i < sources.size(); i++) {
            int source = sources[i];
            dist[i][source] = 0;
            pq.push({0, source, i});
        }
        
        while (!pq.empty()) {
            auto [currentDist, currentVertex, sourceIndex] = pq.top();
            pq.pop();
            
            if (currentDist > dist[sourceIndex][currentVertex]) {
                continue;
            }
            
            if (currentDist > bound) {
                continue;
            }
            
            for (const Edge& edge : adjacencyList[currentVertex]) {
                int nextVertex = edge.destination;
                int newDist = currentDist + edge.weight;
                
                if (newDist <= bound && newDist < dist[sourceIndex][nextVertex]) {
                    dist[sourceIndex][nextVertex] = newDist;
                    pq.push({newDist, nextVertex, sourceIndex});
                }
            }
        }
        
        // Construir el mapa de distancias
        for (int i = 0; i < sources.size(); i++) {
            vector<pair<int, int>> sourceDistances;
            for (int v = 0; v < vertices; v++) {
                if (dist[i][v] != INT_MAX && dist[i][v] <= bound) {
                    sourceDistances.push_back({v, dist[i][v]});
                }
            }
            distances[sources[i]] = sourceDistances;
        }
        
        return distances;
    }
    
    // Función para reconstruir el camino desde una fuente a un destino
    vector<int> reconstructPath(int source, int destination, const vector<int>& sources, int bound) {
        vector<vector<int>> dist(sources.size(), vector<int>(vertices, INT_MAX));
        vector<vector<int>> parent(sources.size(), vector<int>(vertices, -1));
        
        priority_queue<tuple<int, int, int>, 
                      vector<tuple<int, int, int>>, 
                      greater<tuple<int, int, int>>> pq;
        
        int sourceIndex = -1;
        for (int i = 0; i < sources.size(); i++) {
            if (sources[i] == source) {
                sourceIndex = i;
                break;
            }
        }
        
        if (sourceIndex == -1) {
            return {}; // Fuente no encontrada
        }
        
        dist[sourceIndex][source] = 0;
        pq.push({0, source, sourceIndex});
        
        while (!pq.empty()) {
            auto [currentDist, currentVertex, srcIdx] = pq.top();
            pq.pop();
            
            if (currentDist > dist[srcIdx][currentVertex]) {
                continue;
            }
            
            if (currentDist > bound) {
                continue;
            }
            
            for (const Edge& edge : adjacencyList[currentVertex]) {
                int nextVertex = edge.destination;
                int newDist = currentDist + edge.weight;
                
                if (newDist <= bound && newDist < dist[srcIdx][nextVertex]) {
                    dist[srcIdx][nextVertex] = newDist;
                    parent[srcIdx][nextVertex] = currentVertex;
                    pq.push({newDist, nextVertex, srcIdx});
                }
            }
        }
        
        // Reconstruir el camino
        vector<int> path;
        if (dist[sourceIndex][destination] == INT_MAX) {
            return path; // No hay camino
        }
        
        int current = destination;
        while (current != -1) {
            path.push_back(current);
            current = parent[sourceIndex][current];
        }
        
        reverse(path.begin(), path.end());
        return path;
    }
    
    void printGraph() {
        cout << "\nGrafo:\n";
        for (int i = 0; i < vertices; i++) {
            cout << "Vértice " << i << ": ";
            for (const Edge& edge : adjacencyList[i]) {
                cout << "(" << edge.destination << ", peso: " << edge.weight << ") ";
            }
            cout << endl;
        }
    }
};

// Función para leer el grafo desde un archivo
Graph readGraphFromFile(const string& filename, vector<int>& sources, int& bound) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        exit(1);
    }
    
    int vertices, edges;
    file >> vertices >> edges;
    
    Graph graph(vertices);
    
    for (int i = 0; i < edges; i++) {
        int source, destination, weight;
        file >> source >> destination >> weight;
        graph.addEdge(source, destination, weight);
    }
    
    int numSources;
    file >> numSources;
    sources.resize(numSources);
    
    for (int i = 0; i < numSources; i++) {
        file >> sources[i];
    }
    
    file >> bound;
    file.close();
    
    return graph;
}

// Función para escribir resultados a un archivo
void writeResultsToFile(const string& filename, 
                       const map<int, vector<pair<int, int>>>& distances,
                       const map<int, vector<int>>& reachableVertices,
                       const vector<int>& sources,
                       int bound,
                       double executionTime) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: No se pudo crear el archivo " << filename << endl;
        return;
    }
    
    file << "=== RESULTADOS DEL ALGORITMO BMSSP ===\n";
    file << "Límite (bound): " << bound << "\n";
    file << "Fuentes: ";
    for (int source : sources) {
        file << source << " ";
    }
    file << "\n";
    file << "Tiempo de ejecución: " << fixed << setprecision(6) << executionTime << " segundos\n";
    file << "Tiempo de ejecución: " << fixed << setprecision(3) << executionTime * 1000 << " milisegundos\n\n";
    
    for (int source : sources) {
        file << "--- DESDE FUENTE " << source << " ---\n";
        file << "Vértices alcanzables:\n";
        
        if (distances.find(source) != distances.end()) {
            for (const auto& pair : distances.at(source)) {
                file << "  Vértice " << pair.first << " -> Distancia: " << pair.second << "\n";
            }
        }
        file << "\n";
    }
    
    file.close();
    cout << "Resultados guardados en: " << filename << endl;
}

int main() {
    string inputFile = "input.txt";
    string outputFile = "output.txt";
    
    vector<int> sources;
    int bound;
    
    cout << "=== ALGORITMO BMSSP (Bounded Multi-Source Shortest Path) ===\n\n";
    
    // Leer el grafo desde el archivo
    Graph graph = readGraphFromFile(inputFile, sources, bound);
    
    cout << "Grafo cargado exitosamente desde " << inputFile << endl;
    cout << "Fuentes: ";
    for (int source : sources) {
        cout << source << " ";
    }
    cout << "\nLímite (bound): " << bound << endl;
    
    // Mostrar el grafo
    graph.printGraph();
    
    // Ejecutar el algoritmo BMSSP
    cout << "\n=== EJECUTANDO BMSSP ===\n";
    
    // Medir tiempo de ejecución
    auto startTime = chrono::high_resolution_clock::now();
    
    auto reachableVertices = graph.bmssp(sources, bound);
    auto distances = graph.getDistancesFromSources(sources, bound);
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    double executionTime = duration.count() / 1000000.0; // Convertir a segundos
    
    cout << "Algoritmo ejecutado en: " << fixed << setprecision(6) << executionTime << " segundos" << endl;
    cout << "Algoritmo ejecutado en: " << fixed << setprecision(3) << executionTime * 1000 << " milisegundos" << endl;
    
    // Mostrar resultados en consola
    cout << "\n=== RESULTADOS ===\n";
    for (int source : sources) {
        cout << "\nDesde fuente " << source << ":\n";
        cout << "Vértices alcanzables dentro del límite " << bound << ":\n";
        
        if (distances.find(source) != distances.end()) {
            for (const auto& pair : distances[source]) {
                cout << "  Vértice " << pair.first << " -> Distancia: " << pair.second << endl;
            }
        }
        
        // Mostrar algunos caminos de ejemplo
        if (reachableVertices.find(source) != reachableVertices.end() && 
            !reachableVertices[source].empty()) {
            cout << "  Ejemplo de camino a vértice " << reachableVertices[source][0] << ": ";
            vector<int> path = graph.reconstructPath(source, reachableVertices[source][0], sources, bound);
            for (int i = 0; i < path.size(); i++) {
                cout << path[i];
                if (i < path.size() - 1) cout << " -> ";
            }
            cout << endl;
        }
    }
    
    // Escribir resultados al archivo de salida
    writeResultsToFile(outputFile, distances, reachableVertices, sources, bound, executionTime);
    
    return 0;
}
