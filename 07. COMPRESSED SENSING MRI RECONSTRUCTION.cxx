#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>

using namespace std;

// ================= Utility Functions =================

double l2_norm(const vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return sqrt(sum);
}

vector<double> matvec(const vector<vector<double>>& A, const vector<double>& x) {
    vector<double> result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

vector<double> transpose_matvec(const vector<vector<double>>& A, const vector<double>& x) {
    vector<double> result(A[0].size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            result[j] += A[i][j] * x[i];
    return result;
}

// ================= Soft Thresholding =================

vector<double> soft_threshold(const vector<double>& x, double lambda) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] > lambda)
            result[i] = x[i] - lambda;
        else if (x[i] < -lambda)
            result[i] = x[i] + lambda;
        else
            result[i] = 0.0;
    }
    return result;
}

// ================= FISTA Algorithm =================

vector<double> FISTA(
    const vector<vector<double>>& A,
    const vector<double>& b,
    double lambda,
    int maxIter
) {
    size_t n = A[0].size();
    vector<double> x(n, 0.0), y(n, 0.0), x_prev(n, 0.0);

    double t = 1.0;
    double L = 1.0;  // Adaptive Lipschitz estimate

    for (int iter = 0; iter < maxIter; ++iter) {

        vector<double> Ay = matvec(A, y);
        for (size_t i = 0; i < Ay.size(); ++i)
            Ay[i] -= b[i];

        vector<double> grad = transpose_matvec(A, Ay);

        // Adaptive step size
        double grad_norm = l2_norm(grad);
        if (grad_norm > 1e-9)
            L = min(1000.0, max(1.0, grad_norm));

        vector<double> temp(n);
        for (size_t i = 0; i < n; ++i)
            temp[i] = y[i] - (1.0 / L) * grad[i];

        x = soft_threshold(temp, lambda / L);

        double t_next = (1 + sqrt(1 + 4 * t * t)) / 2;

        for (size_t i = 0; i < n; ++i)
            y[i] = x[i] + ((t - 1) / t_next) * (x[i] - x_prev[i]);

        x_prev = x;
        t = t_next;

        if (iter % 10 == 0) {
            cout << "Iteration " << iter
                 << " | ||x||₂ = " << l2_norm(x) << endl;
        }
    }
    return x;
}

// ================= Main Program Loop =================

int main() {
    cout << "==============================================\n";
    cout << " Compressed Sensing MRI Reconstruction (C++)\n";
    cout << " Adaptive FISTA | MIT-Level Scientific CLI\n";
    cout << "==============================================\n\n";

    while (true) {
        int m, n, iter;
        double lambda;

        cout << "Enter number of measurements (m): ";
        cin >> m;
        cout << "Enter signal dimension (n): ";
        cin >> n;

        vector<vector<double>> A(m, vector<double>(n));
        vector<double> b(m);

        cout << "\nEnter matrix A (" << m << "x" << n << "):\n";
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                cin >> A[i][j];

        cout << "\nEnter measurement vector b:\n";
        for (int i = 0; i < m; ++i)
            cin >> b[i];

        cout << "\nEnter lambda (regularization strength): ";
        cin >> lambda;

        cout << "Enter number of FISTA iterations: ";
        cin >> iter;

        vector<double> x = FISTA(A, b, lambda, iter);

        cout << "\n===== Reconstructed Signal =====\n";
        cout << fixed << setprecision(6);
        for (double xi : x)
            cout << xi << " ";
        cout << "\n================================\n";

        char choice;
        cout << "\nRun another reconstruction? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y')
            break;
    }

    cout << "\nProgram finished. Stay legendary.\n";
    return 0;
}
