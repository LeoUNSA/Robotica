# Makefile para el proyecto BMSSP

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET = bmssp
SOURCE = bmssp.cpp

# Regla principal
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

# Regla para ejecutar el programa
run: $(TARGET)
	./$(TARGET)

# Regla para limpiar archivos generados
clean:
	rm -f $(TARGET) output.txt

# Regla para recompilar desde cero
rebuild: clean $(TARGET)

# Mostrar ayuda
help:
	@echo "Makefile para BMSSP (Bounded Multi-Source Shortest Path)"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make        - Compila el programa"
	@echo "  make run    - Compila y ejecuta el programa"
	@echo "  make clean  - Elimina archivos generados"
	@echo "  make rebuild- Recompila desde cero"
	@echo "  make help   - Muestra esta ayuda"

.PHONY: run clean rebuild help
