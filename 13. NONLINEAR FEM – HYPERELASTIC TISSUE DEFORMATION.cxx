#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

/* ===============================
   Matrix utilities (2x2 FEM demo)
   =============================== */

struct Matrix2 {
    double a11, a12, a21, a22;
};

Matrix2 transpose(const Matrix2& A) {
    return {A.a11, A.a21, A.a12, A.a22};
}

Matrix2 multiply(const Matrix2& A, const Matrix2& B) {
    return {
        A.a11 * B.a11 + A.a12 * B.a21,
        A.a11 * B.a12 + A.a12 * B.a22,
        A.a21 * B.a11 + A.a22 * B.a21,
        A.a21 * B.a12 + A.a22 * B.a22
    };
}

double determinant(const Matrix2& A) {
    return A.a11 * A.a22 - A.a12 * A.a21;
}

double trace(const Matrix2& A) {
    return A.a11 + A.a22;
}

/* ===============================
   Hyperelastic Material Model
   =============================== */

struct Material {
    double mu;   // shear modulus
    double K;    // bulk modulus
};

double strainEnergy(const Matrix2& F, const Material& mat) {
    Matrix2 Ft = transpose(F);
    Matrix2 C = multiply(Ft, F);

    double Ic = trace(C);
    double J = determinant(F);

    return 0.5 * mat.mu * (Ic - 3.0)
         + 0.5 * mat.K  * pow(J - 1.0, 2.0);
}

/* ===============================
   First Piola-Kirchhoff Stress
   =============================== */

Matrix2 computeStress(const Matrix2& F, const Material& mat) {
    double J = determinant(F);

    return {
        mat.mu * F.a11 + mat.K * (J - 1) * F.a22,
        mat.mu * F.a12 - mat.K * (J - 1) * F.a21,
        mat.mu * F.a21 - mat.K * (J - 1) * F.a12,
        mat.mu * F.a22 + mat.K * (J - 1) * F.a11
    };
}

/* ===============================
   Newton-Raphson Solver
   =============================== */

void newtonSolver(Matrix2& F, const Material& mat, int maxIter) {
    double tol = 1e-6;

    for (int iter = 0; iter < maxIter; ++iter) {
        Matrix2 P = computeStress(F, mat);
        double residual =
            fabs(P.a11) + fabs(P.a12) +
            fabs(P.a21) + fabs(P.a22);

        cout << "Iteration " << iter + 1
             << " | Residual = " << residual << endl;

        if (residual < tol) {
            cout << "Converged successfully.\n";
            return;
        }

        // Line search step
        double alpha = 0.1;
        F.a11 -= alpha * P.a11;
        F.a12 -= alpha * P.a12;
        F.a21 -= alpha * P.a21;
        F.a22 -= alpha * P.a22;
    }

    cout << "Warning: Maximum iterations reached.\n";
}

/* ===============================
   MAIN CLI PROGRAM
   =============================== */

int main() {
    cout << "=============================================\n";
    cout << " OPTIMAL SURGICAL PLANNING FEM SIMULATOR\n";
    cout << " Nonlinear Biomechanical Tissue Deformation\n";
    cout << "=============================================\n\n";

    char repeat;

    do {
        Material mat;
        Matrix2 F;

        cout << "Enter material parameters:\n";
        cout << "Shear modulus (mu): ";
        cin >> mat.mu;
        cout << "Bulk modulus (K): ";
        cin >> mat.K;

        cout << "\nEnter initial deformation gradient F:\n";
        cout << "F11 F12: ";
        cin >> F.a11 >> F.a12;
        cout << "F21 F22: ";
        cin >> F.a21 >> F.a22;

        cout << "\nRunning nonlinear FEM solver...\n\n";
        newtonSolver(F, mat, 50);

        cout << "\nFinal deformation gradient:\n";
        cout << fixed << setprecision(6);
        cout << "[ " << F.a11 << "  " << F.a12 << " ]\n";
        cout << "[ " << F.a21 << "  " << F.a22 << " ]\n";

        double energy = strainEnergy(F, mat);
        cout << "\nFinal strain energy: " << energy << "\n";

        cout << "\nCompute another surgical scenario? (y/n): ";
        cin >> repeat;

    } while (repeat == 'y' || repeat == 'Y');

    cout << "\nSimulation terminated. Stay legendary.\n";
    return 0;
}
