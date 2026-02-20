#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

// --- ESTRUCTURAS Y FUNCIONES AUXILIARES ---

struct Pixel {
    double r, g, b;
    Pixel() : r(0), g(0), b(0) {}
    Pixel(double r_, double g_, double b_) : r(r_), g(g_), b(b_) {}
};

double distancia_euclidiana(const Pixel& p1, const Pixel& p2) {
    return sqrt(pow(p1.r - p2.r, 2) + pow(p1.g - p2.g, 2) + pow(p1.b - p2.b, 2));
}

// Conversión manual BGR -> HSV
void bgr2hsv(const Vec3b& bgr, Vec3b& hsv) {
    double b = bgr[0] / 255.0;
    double g = bgr[1] / 255.0;
    double r = bgr[2] / 255.0;

    double cmax = max({r, g, b});
    double cmin = min({r, g, b});
    double delta = cmax - cmin;

    double h = 0;
    if (delta == 0) h = 0;
    else if (cmax == r) h = 60 * fmod(((g - b) / delta), 6);
    else if (cmax == g) h = 60 * (((b - r) / delta) + 2);
    else if (cmax == b) h = 60 * (((r - g) / delta) + 4);

    if (h < 0) h += 360;

    double s = (cmax == 0) ? 0 : (delta / cmax);
    double v = cmax;

    hsv[0] = static_cast<uchar>(h / 2); // 0-179
    hsv[1] = static_cast<uchar>(s * 255); // 0-255
    hsv[2] = static_cast<uchar>(v * 255); // 0-255
}

// Conversión manual HSV -> BGR
void hsv2bgr(const Vec3b& hsv, Vec3b& bgr) {
    double h = hsv[0] * 2.0;
    double s = hsv[1] / 255.0;
    double v = hsv[2] / 255.0;

    double c = v * s;
    double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
    double m = v - c;

    double r_prime, g_prime, b_prime;
    if (h >= 0 && h < 60) { r_prime = c; g_prime = x; b_prime = 0; }
    else if (h >= 60 && h < 120) { r_prime = x; g_prime = c; b_prime = 0; }
    else if (h >= 120 && h < 180) { r_prime = 0; g_prime = c; b_prime = x; }
    else if (h >= 180 && h < 240) { r_prime = 0; g_prime = x; b_prime = c; }
    else if (h >= 240 && h < 300) { r_prime = x; g_prime = 0; b_prime = c; }
    else { r_prime = c; g_prime = 0; b_prime = x; }

    bgr[0] = static_cast<uchar>((b_prime + m) * 255);
    bgr[1] = static_cast<uchar>((g_prime + m) * 255);
    bgr[2] = static_cast<uchar>((r_prime + m) * 255);
}

// --- EJERCICIOS ---

void ejercicio1_bgr_to_hsv(bool native) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;

    Mat img_hsv;
    if (native) {
        cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV);
    } else {
        img_hsv.create(img_bgr.rows, img_bgr.cols, CV_8UC3);
        for (int i = 0; i < img_bgr.rows; i++) {
            for (int j = 0; j < img_bgr.cols; j++) {
                bgr2hsv(img_bgr.at<Vec3b>(i, j), img_hsv.at<Vec3b>(i, j));
            }
        }
    }
    imshow(native ? "1. HSV (Nativo)" : "1. HSV (Manual)", img_hsv);
    waitKey(0);
}

void ejercicio2_modificar_saturacion(bool native) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;

    Mat img_resultado;
    if (native) {
        Mat hsv;
        cvtColor(img_bgr, hsv, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(hsv, channels);
        channels[1] *= 1.5;
        merge(channels, hsv);
        cvtColor(hsv, img_resultado, COLOR_HSV2BGR);
    } else {
        Mat hsv(img_bgr.rows, img_bgr.cols, CV_8UC3);
        img_resultado.create(img_bgr.rows, img_bgr.cols, CV_8UC3);
        for (int i = 0; i < img_bgr.rows; i++) {
            for (int j = 0; j < img_bgr.cols; j++) {
                bgr2hsv(img_bgr.at<Vec3b>(i, j), hsv.at<Vec3b>(i, j));
                Vec3b& p = hsv.at<Vec3b>(i, j);
                p[1] = saturate_cast<uchar>(p[1] * 1.5);
                hsv2bgr(p, img_resultado.at<Vec3b>(i, j));
            }
        }
    }
    imshow(native ? "2. Saturacion +50% (Nativo)" : "2. Saturacion +50% (Manual)", img_resultado);
    waitKey(0);
}

void ejercicio3_kmeans(bool native, int K = 5) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;
    Mat img_small;
    resize(img_bgr, img_small, Size(160, 120));
    Mat img_quantized;

    if (native) {
        Mat samples = img_small.reshape(1, img_small.total());
        samples.convertTo(samples, CV_32F);
        Mat labels, centers;
        kmeans(samples, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
        centers.convertTo(centers, CV_8U);
        img_quantized.create(img_small.size(), img_small.type());
        for (int i = 0; i < img_small.total(); i++) {
            img_quantized.at<Vec3b>(i / img_small.cols, i % img_small.cols) = centers.at<Vec3b>(labels.at<int>(i));
        }
    } else {
        vector<Pixel> data;
        for (int i = 0; i < img_small.rows; i++) {
            for (int j = 0; j < img_small.cols; j++) {
                Vec3b p = img_small.at<Vec3b>(i, j);
                data.push_back(Pixel(p[2], p[1], p[0]));
            }
        }
        vector<Pixel> centroides(K);
        random_device rd; mt19937 gen(rd());
        uniform_int_distribution<> dis(0, (int)data.size() - 1);
        for (int k = 0; k < K; k++) centroides[k] = data[dis(gen)];

        vector<int> asignaciones(data.size());
        for (int iter = 0; iter < 10; iter++) {
            for (int i = 0; i < (int)data.size(); i++) {
                double dmin = 1e18; int kbest = 0;
                for (int k = 0; k < K; k++) {
                    double d = distancia_euclidiana(data[i], centroides[k]);
                    if (d < dmin) { dmin = d; kbest = k; }
                }
                asignaciones[i] = kbest;
            }
            vector<Pixel> sum(K, Pixel(0,0,0)); vector<int> count(K, 0);
            for (int i = 0; i < (int)data.size(); i++) {
                int k = asignaciones[i];
                sum[k].r += data[i].r; sum[k].g += data[i].g; sum[k].b += data[i].b;
                count[k]++;
            }
            for (int k = 0; k < K; k++) if (count[k] > 0) centroides[k] = Pixel(sum[k].r/count[k], sum[k].g/count[k], sum[k].b/count[k]);
        }
        img_quantized.create(img_small.size(), img_small.type());
        for (int i = 0; i < img_small.rows; i++) {
            for (int j = 0; j < img_small.cols; j++) {
                int k = asignaciones[i * img_small.cols + j];
                img_quantized.at<Vec3b>(i, j) = Vec3b(saturate_cast<uchar>(centroides[k].b), saturate_cast<uchar>(centroides[k].g), saturate_cast<uchar>(centroides[k].r));
            }
        }
    }
    imshow(native ? "3. K-Means (Nativo)" : "3. K-Means (Manual)", img_quantized);
    waitKey(0);
}

void ejercicio4_gray_world(bool native) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;
    Mat img_res;
    if (native) {
        Ptr<xphoto::WhiteBalancer> wb = xphoto::createGrayworldWB();
        wb->balanceWhite(img_bgr, img_res);
    } else {
        Scalar avg = mean(img_bgr);
        double gavg = (avg[0] + avg[1] + avg[2]) / 3.0;
        double sb = gavg / avg[0], sg = gavg / avg[1], sr = gavg / avg[2];
        img_res.create(img_bgr.size(), img_bgr.type());
        for (int i = 0; i < img_bgr.rows; i++) {
            for (int j = 0; j < img_bgr.cols; j++) {
                Vec3b p = img_bgr.at<Vec3b>(i, j);
                img_res.at<Vec3b>(i, j) = Vec3b(saturate_cast<uchar>(p[0]*sb), saturate_cast<uchar>(p[1]*sg), saturate_cast<uchar>(p[2]*sr));
            }
        }
    }
    imshow(native ? "4. Gray World (Nativo)" : "4. Gray World (Manual)", img_res);
    waitKey(0);
}

void ejercicio6_gamma(bool native, double gamma = 1.5) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;
    Mat img_res;
    if (native) {
        Mat lut(1, 256, CV_8U);
        uchar* p = lut.ptr();
        for (int i = 0; i < 256; i++) p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        LUT(img_bgr, lut, img_res);
    } else {
        uchar lut[256];
        for (int i = 0; i < 256; i++) lut[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        img_res.create(img_bgr.size(), img_bgr.type());
        for (int i = 0; i < img_bgr.rows; i++) {
            for (int j = 0; j < img_bgr.cols; j++) {
                Vec3b p = img_bgr.at<Vec3b>(i, j);
                img_res.at<Vec3b>(i, j) = Vec3b(lut[p[0]], lut[p[1]], lut[p[2]]);
            }
        }
    }
    imshow(native ? "6. Gamma (Nativo)" : "6. Gamma (Manual)", img_res);
    waitKey(0);
}

void ejercicio7_vignette(bool native, double k = 0.4) {
    Mat img_bgr = imread("imagen.jpg");
    if (img_bgr.empty()) return;
    int rows = img_bgr.rows, cols = img_bgr.cols;
    double cx = cols / 2.0, cy = rows / 2.0;
    double dmax = sqrt(cx*cx + cy*cy);
    Mat img_res = img_bgr.clone();

    if (native) {
        Mat mask(rows, cols, CV_32F);
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                mask.at<float>(i, j) = 1.0 / (1.0 - k * pow(sqrt(pow(j - cx, 2) + pow(i - cy, 2)) / dmax, 2));
        Mat ch[3]; split(img_bgr, ch);
        for(int i=0; i<3; i++) {
            ch[i].convertTo(ch[i], CV_32F);
            multiply(ch[i], mask, ch[i]);
            ch[i].convertTo(ch[i], CV_8U);
        }
        merge(ch, 3, img_res);
    } else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double f = 1.0 / (1.0 - k * pow(sqrt(pow(j - cx, 2) + pow(i - cy, 2)) / dmax, 2));
                Vec3b p = img_bgr.at<Vec3b>(i, j);
                img_res.at<Vec3b>(i, j) = Vec3b(saturate_cast<uchar>(p[0]*f), saturate_cast<uchar>(p[1]*f), saturate_cast<uchar>(p[2]*f));
            }
        }
    }
    imshow(native ? "7. Vignette (Nativo)" : "7. Vignette (Manual)", img_res);
    waitKey(0);
}

int main() {
    int opt = -1; bool native = false;
    while (opt != 0) {
        cout << "\n--- TALLER OPENCV COLOR [" << (native ? "NATIVO" : "MANUAL") << "] ---" << endl;
        cout << "1. BGR -> HSV\n2. Saturacion +50%\n3. K-Means\n4. Gray World\n6. Gamma\n7. Vignette\n8. CAMBIAR MODO\n0. Salir\n>> ";
        cin >> opt;
        if (opt == 8) { native = !native; continue; }
        switch (opt) {
            case 1: ejercicio1_bgr_to_hsv(native); break;
            case 2: ejercicio2_modificar_saturacion(native); break;
            case 3: ejercicio3_kmeans(native); break;
            case 4: ejercicio4_gray_world(native); break;
            case 6: ejercicio6_gamma(native); break;
            case 7: ejercicio7_vignette(native); break;
        }
        destroyAllWindows();
    }
    return 0;
}
