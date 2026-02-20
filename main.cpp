#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

// --- ESTRUCTURAS Y FUNCIONES AUXILIARES ---

struct Pixel {
  double r, g, b;
  Pixel() : r(0), g(0), b(0) {}
  Pixel(double r_, double g_, double b_) : r(r_), g(g_), b(b_) {}
};

double distancia_euclidiana(const Pixel &p1, const Pixel &p2) {
  return sqrt(pow(p1.r - p2.r, 2) + pow(p1.g - p2.g, 2) + pow(p1.b - p2.b, 2));
}

// Conversión manual BGR -> HSV
void bgr2hsv(const Vec3b &bgr, Vec3b &hsv) {
  double b = bgr[0] / 255.0;
  double g = bgr[1] / 255.0;
  double r = bgr[2] / 255.0;

  double cmax = max({r, g, b});
  double cmin = min({r, g, b});
  double delta = cmax - cmin;

  double h = 0;
  if (delta == 0)
    h = 0;
  else if (cmax == r)
    h = 60 * fmod(((g - b) / delta), 6);
  else if (cmax == g)
    h = 60 * (((b - r) / delta) + 2);
  else if (cmax == b)
    h = 60 * (((r - g) / delta) + 4);

  if (h < 0)
    h += 360;

  double s = (cmax == 0) ? 0 : (delta / cmax);
  double v = cmax;

  hsv[0] = static_cast<uchar>(h / 2);   // 0-179
  hsv[1] = static_cast<uchar>(s * 255); // 0-255
  hsv[2] = static_cast<uchar>(v * 255); // 0-255
}

// Conversión manual HSV -> BGR
void hsv2bgr(const Vec3b &hsv, Vec3b &bgr) {
  double h = hsv[0] * 2.0;
  double s = hsv[1] / 255.0;
  double v = hsv[2] / 255.0;

  double c = v * s;
  double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
  double m = v - c;

  double r_prime, g_prime, b_prime;
  if (h >= 0 && h < 60) {
    r_prime = c;
    g_prime = x;
    b_prime = 0;
  } else if (h >= 60 && h < 120) {
    r_prime = x;
    g_prime = c;
    b_prime = 0;
  } else if (h >= 120 && h < 180) {
    r_prime = 0;
    g_prime = c;
    b_prime = x;
  } else if (h >= 180 && h < 240) {
    r_prime = 0;
    g_prime = x;
    b_prime = c;
  } else if (h >= 240 && h < 300) {
    r_prime = x;
    g_prime = 0;
    b_prime = c;
  } else {
    r_prime = c;
    g_prime = 0;
    b_prime = x;
  }

  bgr[0] = static_cast<uchar>(saturate_cast<uchar>((b_prime + m) * 255));
  bgr[1] = static_cast<uchar>(saturate_cast<uchar>((g_prime + m) * 255));
  bgr[2] = static_cast<uchar>(saturate_cast<uchar>((r_prime + m) * 255));
}

// --- EJERCICIOS ---

void ejercicio1_bgr_to_hsv() {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty()) {
    cout << "No se encontró imagen.jpg" << endl;
    return;
  }

  int rows = img_bgr.rows;
  int cols = img_bgr.cols;
  Mat img_hsv(rows, cols, CV_8UC3);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      bgr2hsv(img_bgr.at<Vec3b>(i, j), img_hsv.at<Vec3b>(i, j));
    }
  }

  imshow("Original BGR", img_bgr);
  imshow("HSV (Manual)", img_hsv);
  waitKey(0);
  destroyAllWindows();
}

void ejercicio2_modificar_saturacion() {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty())
    return;

  int rows = img_bgr.rows;
  int cols = img_bgr.cols;
  Mat img_hsv(rows, cols, CV_8UC3);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      bgr2hsv(img_bgr.at<Vec3b>(i, j), img_hsv.at<Vec3b>(i, j));
    }
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b &pixel_hsv = img_hsv.at<Vec3b>(i, j);
      int new_s = static_cast<int>(pixel_hsv[1] * 1.5);
      pixel_hsv[1] = saturate_cast<uchar>(new_s);
    }
  }

  Mat img_resultado(rows, cols, CV_8UC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      hsv2bgr(img_hsv.at<Vec3b>(i, j), img_resultado.at<Vec3b>(i, j));
    }
  }

  imshow("Original", img_bgr);
  imshow("Saturacion Aumentada", img_resultado);
  waitKey(0);
  destroyAllWindows();
}

void ejercicio3_kmeans_manual(int K = 5) {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty())
    return;

  Mat img_small;
  resize(img_bgr, img_small, Size(160, 120));

  int rows = img_small.rows;
  int cols = img_small.cols;
  int total_pixels = rows * cols;

  cout << "Procesando " << total_pixels << " pixeles con K=" << K << endl;

  vector<Pixel> data;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b p = img_small.at<Vec3b>(i, j);
      data.push_back(Pixel(p[2], p[1], p[0]));
    }
  }

  vector<Pixel> centroides(K);
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(0, total_pixels - 1);
  for (int k = 0; k < K; k++) {
    centroides[k] = data[dis(gen)];
  }

  vector<int> asignaciones(total_pixels);
  int max_iteraciones = 15;

  for (int iter = 0; iter < max_iteraciones; iter++) {
    for (int i = 0; i < total_pixels; i++) {
      double dist_min = 1e18;
      int k_best = 0;
      for (int k = 0; k < K; k++) {
        double d = distancia_euclidiana(data[i], centroides[k]);
        if (d < dist_min) {
          dist_min = d;
          k_best = k;
        }
      }
      asignaciones[i] = k_best;
    }

    vector<Pixel> nuevos_centroides(K, Pixel(0, 0, 0));
    vector<int> counts(K, 0);
    for (int i = 0; i < total_pixels; i++) {
      int k = asignaciones[i];
      nuevos_centroides[k].r += data[i].r;
      nuevos_centroides[k].g += data[i].g;
      nuevos_centroides[k].b += data[i].b;
      counts[k]++;
    }

    for (int k = 0; k < K; k++) {
      if (counts[k] > 0) {
        centroides[k].r = nuevos_centroides[k].r / counts[k];
        centroides[k].g = nuevos_centroides[k].g / counts[k];
        centroides[k].b = nuevos_centroides[k].b / counts[k];
      }
    }
  }

  Mat img_quantized(rows, cols, CV_8UC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      int k = asignaciones[idx];
      img_quantized.at<Vec3b>(i, j) =
          Vec3b(saturate_cast<uchar>(centroides[k].b),
                saturate_cast<uchar>(centroides[k].g),
                saturate_cast<uchar>(centroides[k].r));
    }
  }

  imshow("Original", img_small);
  imshow("K-Means Manual K=" + to_string(K), img_quantized);
  waitKey(0);
  destroyAllWindows();
}

void ejercicio4_gray_world() {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty())
    return;

  int rows = img_bgr.rows;
  int cols = img_bgr.cols;
  double sum_b = 0, sum_g = 0, sum_r = 0;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b p = img_bgr.at<Vec3b>(i, j);
      sum_b += p[0];
      sum_g += p[1];
      sum_r += p[2];
    }
  }

  int n = rows * cols;
  double avg_b = sum_b / n, avg_g = sum_g / n, avg_r = sum_r / n;
  double gray_avg = (avg_b + avg_g + avg_r) / 3.0;

  double kb = gray_avg / avg_b, kg = gray_avg / avg_g, kr = gray_avg / avg_r;

  Mat img_res(rows, cols, CV_8UC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b p = img_bgr.at<Vec3b>(i, j);
      img_res.at<Vec3b>(i, j) = Vec3b(saturate_cast<uchar>(p[0] * kb),
                                      saturate_cast<uchar>(p[1] * kg),
                                      saturate_cast<uchar>(p[2] * kr));
    }
  }

  imshow("Original", img_bgr);
  imshow("Gray World", img_res);
  waitKey(0);
  destroyAllWindows();
}

void ejercicio6_gamma(double gamma = 1.5) {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty())
    return;

  uchar lut[256];
  for (int i = 0; i < 256; i++)
    lut[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

  Mat img_res = img_bgr.clone();
  for (int i = 0; i < img_res.rows; i++) {
    for (int j = 0; j < img_res.cols; j++) {
      Vec3b &p = img_res.at<Vec3b>(i, j);
      p[0] = lut[p[0]];
      p[1] = lut[p[1]];
      p[2] = lut[p[2]];
    }
  }

  imshow("Original", img_bgr);
  imshow("Gamma Correccion", img_res);
  waitKey(0);
  destroyAllWindows();
}

void ejercicio7_vignette(double k = 0.4) {
  Mat img_bgr = imread("imagen.jpg");
  if (img_bgr.empty())
    return;

  int rows = img_bgr.rows, cols = img_bgr.cols;
  double cx = cols / 2.0, cy = rows / 2.0;
  double d_max = sqrt(cx * cx + cy * cy);

  Mat img_res(rows, cols, CV_8UC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double d = sqrt(pow(j - cx, 2) + pow(i - cy, 2));
      double f = 1.0 / (1.0 - k * pow(d / d_max, 2));
      Vec3b p = img_bgr.at<Vec3b>(i, j);
      img_res.at<Vec3b>(i, j) =
          Vec3b(saturate_cast<uchar>(p[0] * f), saturate_cast<uchar>(p[1] * f),
                saturate_cast<uchar>(p[2] * f));
    }
  }

  imshow("Original", img_bgr);
  imshow("Vignette Corregido", img_res);
  waitKey(0);
  destroyAllWindows();
}

int main() {
  int opcion = -1;
  while (opcion != 0) {
    cout << "\n--- TALLER OPENCV COLOR ---" << endl;
    cout << "1. BGR -> HSV (Manual)" << endl;
    cout << "2. Saturacion (+50%)" << endl;
    cout << "3. K-Means Manual" << endl;
    cout << "4. Gray World" << endl;
    cout << "6. Gamma" << endl;
    cout << "7. Vignette" << endl;
    cout << "0. Salir" << endl;
    cin >> opcion;

    if (opcion == 1)
      ejercicio1_bgr_to_hsv();
    else if (opcion == 2)
      ejercicio2_modificar_saturacion();
    else if (opcion == 3)
      ejercicio3_kmeans_manual();
    else if (opcion == 4)
      ejercicio4_gray_world();
    else if (opcion == 6)
      ejercicio6_gamma();
    else if (opcion == 7)
      ejercicio7_vignette();
  }
  return 0;
}
