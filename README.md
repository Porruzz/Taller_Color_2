# Taller de Procesamiento de Color (Sin OpenCV)

Este proyecto implementa diversos algoritmos de procesamiento de imágenes "desde cero" en C++, utilizando el formato de imagen PPM para evitar dependencias de librerías externas como OpenCV.

## Algoritmos Implementados
1. **Conversión BGR a HSV**: Transformación manual de espacios de color.
2. **Modificación de Saturación**: Aumento manual de la intensidad del color.
3. **K-Means Clustering**: Segmentación de colores y cuantización manual.
4. **Gray World**: Balance de blancos y constancia de color.
5. **Corrección Gamma**: Ajuste de luminancia mediante tablas de búsqueda (LUT).
6. **Corrección de Viñeteo**: Compensación de la caída de luz radial.

## Requisitos
*   Compilador C++ (g++)
*   ImageMagick (opcional, para convertir imágenes a .ppm)

## Cómo Ejecutar

1.  **Preparar la imagen**: El programa busca un archivo llamado `imagen.ppm`. Si tienes un JPG, conviértelo:
    ```bash
    convert imagen.jpg -compress none imagen.ppm
    ```

2.  **Compilar**:
    ```bash
    g++ main.cpp -o taller -O3
    ```

3.  **Correr**:
    ```bash
    ./taller
    ```

4.  **Ver Resultados**: Los resultados se guardarán como archivos `.ppm` en la carpeta raíz (ej. `ejercicio3_kmeans.ppm`). Puedes abrirlos con cualquier visor de imágenes.

---
Desarrollado para el Taller de Color.
