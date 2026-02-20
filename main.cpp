#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

// --- EXPLICACIÓN DEL TALLER ---
// El taller pide implementar los algoritmos "desde cero", lo cual significa
// NO usar funciones como cvtColor(), kmeans(), inRange(), etc.
// Sin embargo, usaremos las funciones "Nativas" de OpenCV para E/S
// (lectura/escritura): imread(), imshow(), waitKey() y la estructura Mat de
// datos.

// --- ESTRUCTURAS Y FUNCIONES AUXILIARES ---

struct PixelDouble {
  double r, g, b;
  PixelDouble() : r(0), g(0), b(0) {}
  PixelDouble(double r_, double g_, double b_) : r(r_), g(g_), b(b_) {}
};

double distancia_euclidiana(const PixelDouble &p1, const PixelDouble &p2) {
  return sqrt(pow(p1.r - p2.r, 2) + pow(p1.g - p2.g, 2) + pow(p1.b - p2.b, 2));
}

// 1. CONVERSIÓN BGR -> HSV (MANUAL)
// Prohibido: cvtColor(img, hsv, COLOR_BGR2HSV)
void bgr2hsv_manual(const Vec3b &bgr, Vec3b &hsv) {
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

  hsv[0] = static_cast<uchar>(h / 2);   // Rango OpenCV: 0-179
  hsv[1] = static_cast<uchar>(s * 255); // 0-255
  hsv[2] = static_cast<uchar>(v * 255); // 0-255
}

// CONVERSIÓN HSV -> BGR (MANUAL)
void hsv2bgr_manual(const Vec3b &hsv, Vec3b &bgr) {
  double h = hsv[0] * 2.0;
  double s = hsv[1] / 255.0;
  double v = hsv[2] / 255.0;

  double c = v * s;
  double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
  double m = v - c;

  double r_p, g_p, b_p;
  if (h < 60) {
    r_p = c;
    g_p = x;
    b_p = 0;
  } else if (h < 120) {
    r_p = x;
    g_p = c;
    b_p = 0;
  } else if (h < 180) {
    r_p = 0;
    g_p = c;
    b_p = x;
  } else if (h < 240) {
    r_p = 0;
    g_p = x;
    b_p = c;
  } else if (h < 300) {
    r_p = x;
    g_p = 0;
    b_p = c;
  } else {
    r_p = c;
    g_p = 0;
    b_p = x;
  }

  bgr[0] = saturate_cast<uchar>((b_p + m) * 255);
  bgr[1] = saturate_cast<uchar>((g_p + m) * 255);
  bgr[2] = saturate_cast<uchar>((r_p + m) * 255);
}

// --- EJERCICIOS ---

void ejercicio1_hsv() {
  Mat img = imread("imagen.jpg");
  if (img.empty())
    return;
  Mat hsv(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < img.rows; i++)
    for (int j = 0; j < img.cols; j++)
      bgr2hsv_manual(img.at<Vec3b>(i, j), hsv.at<Vec3b>(i, j));
  imshow("1. BGR a HSV (MANUAL)", hsv);
  waitKey(0);
}

void ejercicio2_saturacion() {
  Mat img = imread("imagen.jpg");
  if (img.empty())
    return;
  Mat hsv(img.rows, img.cols, CV_8UC3);
  Mat res(img.rows, img.cols, CV_8UC3);

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      bgr2hsv_manual(img.at<Vec3b>(i, j), hsv.at<Vec3b>(i, j));
      Vec3b p = hsv.at<Vec3b>(i, j);
      p[1] = saturate_cast<uchar>(p[1] * 1.5); // Aumentar saturación
      hsv2bgr_manual(p, res.at<Vec3b>(i, j));
    }
  }
  imshow("2. Saturacion +50% (MANUAL)", res);
  waitKey(0);
}

void ejercicio3_kmeans(int K = 5) {
  Mat img_orig = imread("imagen.jpg");
  if (img_orig.empty())
    return;
  Mat img;
  resize(img_orig, img, Size(160, 120)); // Resize nativo para velocidad

  int total = img.rows * img.cols;
  vector<PixelDouble> centroids(K);
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(0, total - 1);

  for (int k = 0; k < K; k++) {
    Vec3b p = img.at<Vec3b>(dis(gen) / img.cols, dis(gen) % img.cols);
    centroids[k] = PixelDouble(p[2], p[1], p[0]);
  }

  vector<int> labels(total);
  for (int iter = 0; iter < 15; iter++) {
    for (int i = 0; i < total; i++) {
      Vec3b p_vec = img.at<Vec3b>(i / img.cols, i % img.cols);
      PixelDouble p(p_vec[2], p_vec[1], p_vec[0]);
      double min_d = 1e18;
      int best_k = 0;
      for (int k = 0; k < K; k++) {
        double d = distancia_euclidiana(p, centroids[k]);
        if (d < min_d) {
          min_d = d;
          best_k = k;
        }
      }
      labels[i] = best_k;
    }
    vector<PixelDouble> sums(K, PixelDouble(0, 0, 0));
    vector<int> counts(K, 0);
    for (int i = 0; i < total; i++) {
      int k = labels[i];
      Vec3b p = img.at<Vec3b>(i / img.cols, i % img.cols);
      sums[k].r += p[2];
      sums[k].g += p[1];
      sums[k].b += p[0];
      counts[k]++;
    }
    for (int k = 0; k < K; k++) {
      if (counts[k] > 0) {
        centroids[k].r = sums[k].r / counts[k];
        centroids[k].g = sums[k].g / counts[k];
        centroids[k].b = sums[k].b / counts[k];
      }
    }
  }

  Mat out(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < total; i++) {
    int k = labels[i];
    out.at<Vec3b>(i / img.cols, i % img.cols) =
        Vec3b(centroids[k].b, centroids[k].g, centroids[k].r);
  }
  imshow("3. K-Means (MANUAL)", out);
  waitKey(0);
}

void ejercicio4_grayworld() {
  Mat img = imread("imagen.jpg");
  if (img.empty())
    return;
  double b = 0, g = 0, r = 0;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      Vec3b p = img.at<Vec3b>(i, j);
      b += p[0];
      g += p[1];
      r += p[2];
    }
  }
  int n = img.rows * img.cols;
  double ab = b / n, ag = g / n, ar = r / n;
  double gray = (ab + ag + ar) / 3.0;
  Mat res(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      Vec3b p = img.at<Vec3b>(i, j);
      res.at<Vec3b>(i, j) = Vec3b(saturate_cast<uchar>(p[0] * (gray / ab)),
                                  saturate_cast<uchar>(p[1] * (gray / ag)),
                                  saturate_cast<uchar>(p[2] * (gray / ar)));
    }
  }
  imshow("4. Gray World (MANUAL)", res);
  waitKey(0);
}

void ejercicio6_gamma(double g = 1.5) {
  Mat img = imread("imagen.jpg");
  uchar lut[256];
  for (int i = 0; i < 256; i++)
    lut[i] = saturate_cast<uchar>(pow(i / 255.0, g) * 255.0);
  Mat res = img.clone();
  for (int i = 0; i < res.rows; i++)
    for (int j = 0; j < res.cols; j++)
      for (int c = 0; c < 3; c++)
        res.at<Vec3b>(i, j)[c] = lut[res.at<Vec3b>(i, j)[c]];
  imshow("6. Gamma (MANUAL)", res);
  waitKey(0);
}

void ejercicio7_vignette(double k = 0.4) {
  Mat img = imread("imagen.jpg");
  double cx = img.cols / 2.0, cy = img.rows / 2.0;
  double dmax = sqrt(cx * cx + cy * cy);
  Mat res(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      double d = sqrt(pow(j - cx, 2) + pow(i - cy, 2));
      double f = 1.0 / (1.0 - k * pow(d / dmax, 2));
      Vec3b p = img.at<Vec3b>(i, j);
      res.at<Vec3b>(i, j) =
          Vec3b(saturate_cast<uchar>(p[0] * f), saturate_cast<uchar>(p[1] * f),
                saturate_cast<uchar>(p[2] * f));
    }
  }
  imshow("7. Vignette (MANUAL)", res);
  waitKey(0);
}

// COMPARATIVA: FUNCIÓN NATIVA DE OPENCV (PARA TU REFERENCIA)
void comparativa_nativa() {
  Mat img = imread("imagen.jpg");
  if (img.empty())
    return;
  Mat hsv_nativa;
  // Esto es lo que el taller NO quiere que uses para los ejercicios, pero es
  // bueno saberlo:
  cvtColor(img, hsv_nativa, COLOR_BGR2HSV);
  imshow("NATIVA: cvtColor (Referencia)", hsv_nativa);
  waitKey(0);
}

int main() {
  int op;
  while (true) {
    cout << "\n--- TALLER COLOR (ALGORITMOS MANUALES + I/O NATIVO) ---" << endl;
    cout << "1. HSV Manual\n2. Saturacion Manual\n3. K-Means Manual\n4. Gray "
            "World Manual\n6. Gamma Manual\n7. Vignette Manual\n8. VER NATIVA "
            "(Comparación)\n0. Salir\nOpcion: ";
    cin >> op;
    if (op == 0)
      break;
    switch (op) {
    case 1:
      ejercicio1_hsv();
      break;
    case 2:
      ejercicio2_saturacion();
      break;
    case 3:
      ejercicio3_kmeans();
      break;
    case 4:
      ejercicio4_grayworld();
      break;
    case 6:
      ejercicio6_gamma();
      break;
    case 7:
      ejercicio7_vignette();
      break;
    case 8:
      comparativa_nativa();
      break;
    }
  }
  destroyAllWindows();
  return 0;
}
