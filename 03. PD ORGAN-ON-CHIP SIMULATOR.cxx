#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

/* =========================
   GLOBAL MODEL PARAMETERS
   ========================= */
constexpr int N = 50;           // Number of compartments
constexpr double GAMMA = 1.0;   // Rosenbrock-W parameter

struct PKPDModel {
    vector<vector<double>> Vmax;
    vector<vector<double>> Km;
    vector<double> kmet;

    PKPDModel() {
        Vmax.resize(N, vector<double>(N, 0.0));
        Km.resize(N, vector<double>(N, 1.0));
        kmet.resize(N, 0.05);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    Vmax[i][j] = 0.1 * (i + 1) / (j + 1);
                    Km[i][j] = 0.5 + 0.01 * abs(i - j);
                }
            }
        }
    }
};

/* =========================
   RHS FUNCTION
   ========================= */
void compute_rhs(
    const vector<double>& C,
    vector<double>& dCdt,
    const PKPDModel& model
) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            sum += model.Vmax[i][j] * C[j] / (model.Km[i][j] + C[j])
                 - model.Vmax[j][i] * C[i] / (model.Km[j][i] + C[i]);
        }

        dCdt[i] = sum - model.kmet[i] * C[i];
    }
}

/* =========================
   ANALYTIC JACOBIAN
   ========================= */
void compute_jacobian(
    const vector<double>& C,
    vector<vector<double>>& J,
    const PKPDModel& model
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            J[i][j] = 0.0;

            if (i == j) {
                double diag = -model.kmet[i];
                for (int k = 0; k < N; k++) {
                    if (k == i) continue;
                    double Km = model.Km[k][i];
                    double V = model.Vmax[k][i];
                    diag -= V * Km / pow(Km + C[i], 2);
                }
                J[i][j] = diag;
            } else {
                double Km = model.Km[i][j];
                double V = model.Vmax[i][j];
                J[i][j] = V * Km / pow(Km + C[j], 2);
            }
        }
    }
}

/* =========================
   LINEAR SOLVER (GAUSS)
   ========================= */
void solve_linear_system(
    vector<vector<double>>& A,
    vector<double>& b
) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        double pivot = A[i][i];
        for (int j = i; j < n; j++) A[i][j] /= pivot;
        b[i] /= pivot;

        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = A[k][i];
            for (int j = i; j < n; j++)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }
}

/* =========================
   ROSENBROCK-W STEP
   ========================= */
void rosenbrock_step(
    vector<double>& C,
    double dt,
    const PKPDModel& model
) {
    vector<double> f(N);
    vector<vector<double>> J(N, vector<double>(N));
    vector<double> k(N);

    compute_rhs(C, f, model);
    compute_jacobian(C, J, model);

    for (int i = 0; i < N; i++) {
        J[i][i] = 1.0 - dt * GAMMA * J[i][i];
        for (int j = 0; j < N; j++) {
            if (i != j)
                J[i][j] *= -dt * GAMMA;
        }
        k[i] = dt * f[i];
    }

    solve_linear_system(J, k);

    for (int i = 0; i < N; i++)
        C[i] += k[i];
}

/* =========================
   MAIN CLI PROGRAM
   ========================= */
int main() {
    PKPDModel model;

    cout << "=========================================\n";
    cout << " Nonlinear PK/PD Organ-on-Chip Simulator\n";
    cout << " Rosenbrock-W | Stiff ODE | C++ CLI\n";
    cout << "=========================================\n";

    char repeat;
    do {
        double T, dt;
        cout << "\nEnter simulation time (hours): ";
        cin >> T;
        cout << "Enter timestep dt: ";
        cin >> dt;

        vector<double> C(N, 0.0);
        C[0] = 10.0; // Initial dose

        int steps = static_cast<int>(T / dt);

        for (int s = 0; s < steps; s++)
            rosenbrock_step(C, dt, model);

        cout << "\nFinal Concentrations:\n";
        cout << fixed << setprecision(6);
        for (int i = 0; i < N; i++) {
            cout << "Compartment " << setw(2) << i
                 << " : " << C[i] << "\n";
        }

        cout << "\nRun another simulation? (y/n): ";
        cin >> repeat;

    } while (repeat == 'y' || repeat == 'Y');

    cout << "\nSimulation finished. Scientific credibility achieved.\n";
    return 0;
}
