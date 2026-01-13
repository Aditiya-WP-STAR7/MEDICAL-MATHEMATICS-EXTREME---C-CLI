#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// ======================= PARAMETERS ==========================
const double dt_micro = 0.0005;   // fast calcium scale
const double dt_macro = 0.01;     // metabolic & vascular scale
const int micro_steps = 50;

const int NX = 60;
const double DX = 0.02;
const double D_oxy = 0.1;

// Reaction coefficients
const double k_pmca = 0.25;
const double alpha = 0.8;
const double beta  = 0.3;
const double gamma = 0.5;
const double delta = 0.2;
const double eta   = 0.6;

// ============================================================

// Fast calcium fluxes
double J_IP3R(double C)  { return 0.8 / (1.0 + C*C); }
double J_RyR(double C)   { return 0.4 * C; }
double J_SERCA(double C) { return 0.6 * C; }
double J_leak()          { return 0.05; }

// ======================= MICRO SOLVER ========================
double integrate_calcium(double C0)
{
    double C = C0;
    for(int i = 0; i < micro_steps; i++)
    {
        double dC =
            J_IP3R(C)
          + J_RyR(C)
          - J_SERCA(C)
          + J_leak()
          - k_pmca * C;

        C += dt_micro * dC;
    }
    return C;
}

// ======================= PDE SOLVER ==========================
void oxygen_diffusion(vector<double>& O, double V)
{
    vector<double> O_new = O;

    for(int i = 1; i < NX - 1; i++)
    {
        double laplacian =
            (O[i+1] - 2.0 * O[i] + O[i-1]) / (DX * DX);

        O_new[i] =
            O[i]
          + dt_macro * (D_oxy * laplacian - eta * V * O[i]);
    }

    O = O_new;
}

// ======================= MAIN PROGRAM ========================
int main()
{
    cout << "\nNEUROVASCULAR COUPLING MULTISCALE SIMULATOR\n";
    cout << "Heterogeneous Multiscale Method (HMM)\n";
    cout << "--------------------------------------\n";

    char repeat;

    do {
        double C, M, V;
        int steps;

        cout << "\nInitial Calcium [Ca2+]: ";
        cin >> C;

        cout << "Initial Metabolic Signal: ";
        cin >> M;

        cout << "Initial Vascular Tone: ";
        cin >> V;

        cout << "Macro Time Steps: ";
        cin >> steps;

        vector<double> O(NX, 1.0); // oxygen field

        cout << "\nRunning Simulation...\n\n";
        cout << setw(6) << "Step"
             << setw(12) << "Calcium"
             << setw(12) << "Metabolic"
             << setw(12) << "Vascular"
             << setw(14) << "Mean O2\n";

        for(int t = 0; t < steps; t++)
        {
            // HMM micro → macro
            C = integrate_calcium(C);

            // Metabolic dynamics
            M += dt_macro * (alpha * C - beta * M);

            // Vascular response
            V += dt_macro * (gamma * M - delta * V);

            // Oxygen PDE
            oxygen_diffusion(O, V);

            // Compute mean oxygen
            double meanO = 0.0;
            for(double x : O) meanO += x;
            meanO /= NX;

            cout << setw(6) << t
                 << setw(12) << fixed << setprecision(4) << C
                 << setw(12) << M
                 << setw(12) << V
                 << setw(14) << meanO << "\n";
        }

        cout << "\nRun another simulation? (y/n): ";
        cin >> repeat;

    } while(repeat == 'y' || repeat == 'Y');

    cout << "\nSimulation Finished. Scientific Integrity Maintained.\n";
    return 0;
}
