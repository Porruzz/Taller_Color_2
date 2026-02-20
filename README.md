# Taller Color OpenCV

Este proyecto implementa varios algoritmos de procesamiento de color utilizando OpenCV. 
Se incluyen dos modos de ejecución para cada ejercicio:

1.  **Modo Manual**: Implementación de los algoritmos "desde cero" utilizando bucles anidados y fórmulas matemáticas directas, sin depender de las funciones de alto nivel de OpenCV (como `cvtColor`, `kmeans`, etc.), tal como se solicita en el taller.
2.  **Modo Nativo**: Implementación utilizando las funciones optimizadas de la librería OpenCV.

## Ejercicios Incluidos

-   **BGR a HSV**: Conversión de espacios de color.
-   **Modificación de Saturación**: Aumento directo de la intensidad del color.
-   **K-Means**: Segmentación por color (cuantización).
-   **Gray World**: Algoritmo de balance de blancos para constancia de color.
-   **Corrección Gamma**: Ajuste radiométrico de brillo.
-   **Corrección de Viñeteo**: Compensación de la caída de luz en los bordes.

## Requisitos

-   OpenCV 4.x
-   OpenCV Contrib (para el módulo `xphoto` en el modo nativo de Gray World)
-   CMake 3.10+
-   Compilador C++11 o superior

## Compilación y Ejecución

```bash
mkdir build && cd build
cmake ..
make
./taller
```

## Uso

Al ejecutar el programa, se presentará un menú interactivo. Puedes cambiar entre el modo **Manual** y **Nativo** presionando la opción `8` en el menú.
