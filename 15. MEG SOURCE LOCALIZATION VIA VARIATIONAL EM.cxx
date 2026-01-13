#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

/*
    Sparse Bayesian Learning for EEG/MEG Inverse Problem
    ---------------------------------------------------
    m = L j + noise
    j ~ N(0, Alpha^-1)
    noise ~ N(0, sigma^2 I)

    Variational Bayesian EM Algorithm
*/

typedef vector<double> Vec;
typedef vector<Vec> Mat;

// ---------- Linear Algebra Utilities ----------

Vec matVecMul(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < x.size(); j++)
            y[i] += A[i][j] * x[j];
    return y;
}

double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        s += a[i] * b[i];
    return s;
}

// ---------- Core SBL-EM Algorithm ----------

void sparseBayesianInverse(
        const Mat& L,
        const Vec& m,
        Vec& j_mean,
        int maxIter = 100,
        double tol = 1e-6
) {
    int M = m.size();       // sensors
    int N = L[0].size();    // sources

    Vec alpha(N, 1.0);
    double sigma2 = 1.0;

    Vec j_old(N, 0.0);

    for (int iter = 0; iter < maxIter; iter++) {

        // ---- E-step (posterior mean approximation) ----
        for (int i = 0; i < N; i++) {
            double numerator = 0.0;
            double denom = alpha[i];

            for (int k = 0; k < M; k++) {
                numerator += L[k][i] * m[k];
                denom += (L[k][i] * L[k][i]) / sigma2;
            }

            j_mean[i] = numerator / denom;
        }

        // ---- M-step (update hyperparameters) ----
        for (int i = 0; i < N; i++) {
            alpha[i] = 1.0 / (j_mean[i] * j_mean[i] + 1e-12);
        }

        // ---- Update noise variance ----
        Vec Lj = matVecMul(L, j_mean);
        double err = 0.0;
        for (int k = 0; k < M; k++)
            err += (m[k] - Lj[k]) * (m[k] - Lj[k]);

        sigma2 = err / M;

        // ---- Convergence check ----
        double diff = 0.0;
        for (int i = 0; i < N; i++)
            diff += fabs(j_mean[i] - j_old[i]);

        if (diff < tol) break;
        j_old = j_mean;
    }
}

// ---------- Data Generation (Simulation) ----------

void generateSyntheticEEG(Mat& L, Vec& true_j, Vec& m) {
    int M = L.size();
    int N = L[0].size();

    default_random_engine gen;
    normal_distribution<double> noise(0.0, 0.05);

    // Sparse true source
    for (int i = 0; i < N; i++)
        true_j[i] = (i == N / 3 || i == 2 * N / 3) ? 1.0 : 0.0;

    m = matVecMul(L, true_j);

    for (int i = 0; i < M; i++)
        m[i] += noise(gen);
}

// ---------- Main CLI Program ----------

int main() {
    cout << "\n==============================================\n";
    cout << " EEG/MEG Inverse Problem via Sparse Bayesian EM\n";
    cout << " Author Level : MIT / Global Research Standard\n";
    cout << "==============================================\n\n";

    bool repeat = true;

    while (repeat) {
        int M = 10;   // sensors
        int N = 20;   // sources

        Mat L(M, Vec(N));
        Vec m(M), j_est(N), j_true(N);

        // Random lead-field matrix
        default_random_engine gen;
        uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                L[i][j] = dist(gen);

        generateSyntheticEEG(L, j_true, m);

        sparseBayesianInverse(L, m, j_est);

        cout << "\nEstimated Source Activity (Sparse):\n";
        for (int i = 0; i < N; i++) {
            cout << "Source " << setw(2) << i
                 << " : " << fixed << setprecision(6)
                 << j_est[i] << "\n";
        }

        cout << "\nRun another computation? (1 = Yes, 0 = No): ";
        cin >> repeat;
    }

    cout << "\nProgram finished. Bayesian inference complete.\n";
    return 0;
}
