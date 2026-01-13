#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

/* ============================================================
   1D LYMPHATIC TRANSPORT SIMULATOR
   Numerical Method : MacCormack Scheme
   Physics          : Continuity + Momentum + Valve Dynamics
   Author           : High-End Scientific Computing Demo
   ============================================================ */

const double PI = 3.141592653589793;

/* ----------- Physical Parameters ----------- */
struct Parameters {
    double rho;     // Density (kg/m^3)
    double mu;      // Viscosity (Pa·s)
    double A0;      // Reference area (m^2)
    double k;       // Wall stiffness
    double R_open;  // Valve open resistance
    double R_closed;// Valve closed resistance
    double omega;   // Valve frequency
};

/* ----------- Valve Resistance ----------- */
double valveResistance(double t, const Parameters& p) {
    return p.R_open +
           (p.R_closed - p.R_open) * pow(sin(p.omega * t), 2.0);
}

/* ----------- Pressure Law ----------- */
double pressure(double A, const Parameters& p) {
    return p.k * (A - p.A0);
}

/* ----------- Main Simulation ----------- */
void simulate(int Nx, int Nt, double L, double T, const Parameters& p) {

    double dx = L / (Nx - 1);
    double dt = T / Nt;

    vector<double> A(Nx, p.A0);
    vector<double> Q(Nx, 0.0);

    vector<double> A_pred(Nx), Q_pred(Nx);

    for (int n = 0; n < Nt; n++) {
        double t = n * dt;
        double Rv = valveResistance(t, p);

        /* ---------- Predictor Step ---------- */
        for (int i = 1; i < Nx - 1; i++) {
            double pL = pressure(A[i], p);
            double pR = pressure(A[i+1], p);

            A_pred[i] = A[i]
                - dt/dx * (Q[i+1] - Q[i]);

            Q_pred[i] = Q[i]
                - dt/dx * ((Q[i+1]*Q[i+1]/A[i+1]) - (Q[i]*Q[i]/A[i]))
                - dt * (A[i]/p.rho) * ((pR - pL) / dx)
                - dt * (8.0 * PI * p.mu * Q[i] / A[i]);
        }

        /* ---------- Corrector Step ---------- */
        for (int i = 1; i < Nx - 1; i++) {
            double pL = pressure(A_pred[i-1], p);
            double pR = pressure(A_pred[i], p);

            A[i] = 0.5 * (A[i] + A_pred[i]
                - dt/dx * (Q_pred[i] - Q_pred[i-1]));

            Q[i] = 0.5 * (Q[i] + Q_pred[i]
                - dt/dx * ((Q_pred[i]*Q_pred[i]/A_pred[i]) -
                           (Q_pred[i-1]*Q_pred[i-1]/A_pred[i-1]))
                - dt * (A_pred[i]/p.rho) * ((pR - pL) / dx)
                - dt * (8.0 * PI * p.mu * Q_pred[i] / A_pred[i]));
        }

        /* ---------- Boundary Conditions ---------- */
        Q[0] = Q[1];
        A[0] = p.A0;

        Q[Nx-1] = -A[Nx-1] / Rv;
        A[Nx-1] = A[Nx-2];
    }

    /* ---------- Output ---------- */
    cout << "\nPosition\tArea\t\tFlow\n";
    for (int i = 0; i < Nx; i += Nx / 10) {
        cout << fixed << setprecision(6)
             << i * dx << "\t"
             << A[i] << "\t"
             << Q[i] << "\n";
    }
}

/* ===================== MAIN ===================== */
int main() {

    cout << "\n=== 1D LYMPHATIC TRANSPORT SIMULATOR ===\n";
    cout << "MacCormack Scheme | Valve Dynamics | CLI Program\n";

    char repeat;

    do {
        int Nx, Nt;
        double L, T;

        cout << "\nEnter number of spatial points: ";
        cin >> Nx;

        cout << "Enter number of time steps: ";
        cin >> Nt;

        cout << "Enter vessel length (m): ";
        cin >> L;

        cout << "Enter total simulation time (s): ";
        cin >> T;

        Parameters p;
        p.rho = 1000.0;
        p.mu = 0.004;
        p.A0 = 1e-6;
        p.k  = 2e6;
        p.R_open   = 1e7;
        p.R_closed = 1e9;
        p.omega    = 2 * PI;

        simulate(Nx, Nt, L, T, p);

        cout << "\nRun another simulation? (y/n): ";
        cin >> repeat;

    } while (repeat == 'y' || repeat == 'Y');

    cout << "\nSimulation completed. Program terminated.\n";
    return 0;
}
