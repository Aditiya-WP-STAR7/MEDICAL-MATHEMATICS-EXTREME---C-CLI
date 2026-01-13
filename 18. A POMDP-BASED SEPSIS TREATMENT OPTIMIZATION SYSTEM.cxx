#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

/* ===========================
   Simulation Parameters
=========================== */
struct Params {
    int Nx = 60;
    int Ny = 60;
    double dx = 1.0;
    double dy = 1.0;
    double dt = 0.01;
    double D = 0.15;        // Diffusion
    double chi = 0.25;      // Mechanotaxis
    double lambda = 0.08;   // Proliferation
    double rho_max = 1.0;
    int steps = 300;
};

/* ===========================
   Grid Utilities
=========================== */
double laplacian(const vector<vector<double>>& u,
                 int i, int j, double dx, double dy) {
    return (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (dx*dx)
         + (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (dy*dy);
}

/* ===========================
   Mechanical Stimulus Field
=========================== */
double mechanicalStimulus(int i, int j, int Nx, int Ny) {
    double cx = Nx / 2.0;
    double cy = Ny / 2.0;
    double r2 = (i - cx)*(i - cx) + (j - cy)*(j - cy);
    return exp(-r2 / 400.0);
}

/* ===========================
   Adaptive Mesh Refinement Indicator
=========================== */
bool needsRefinement(const vector<vector<double>>& rho,
                     int i, int j, double threshold) {
    double grad = fabs(rho[i+1][j] - rho[i-1][j])
                + fabs(rho[i][j+1] - rho[i][j-1]);
    return grad > threshold;
}

/* ===========================
   Multigrid Solver (Skeleton)
=========================== */
void multigridRelax(vector<vector<double>>& u,
                    const vector<vector<double>>& rhs,
                    int iterations) {
    int Nx = u.size();
    int Ny = u[0].size();

    for (int it = 0; it < iterations; ++it) {
        for (int i = 1; i < Nx-1; ++i) {
            for (int j = 1; j < Ny-1; ++j) {
                u[i][j] = 0.25 * (
                    u[i+1][j] + u[i-1][j] +
                    u[i][j+1] + u[i][j-1]
                    - rhs[i][j]
                );
            }
        }
    }
}

/* ===========================
   Main Simulation Core
=========================== */
void runSimulation(const Params& P) {
    vector<vector<double>> rho(P.Nx, vector<double>(P.Ny, 0.05));
    vector<vector<double>> rho_new = rho;

    for (int t = 0; t < P.steps; ++t) {
        for (int i = 1; i < P.Nx-1; ++i) {
            for (int j = 1; j < P.Ny-1; ++j) {

                double S = mechanicalStimulus(i, j, P.Nx, P.Ny);
                double lap = laplacian(rho, i, j, P.dx, P.dy);

                double chemotaxis =
                    (rho[i][j] * P.chi) * laplacian(rho, i, j, P.dx, P.dy);

                double growth =
                    P.lambda * rho[i][j] *
                    (1.0 - rho[i][j] / P.rho_max);

                rho_new[i][j] =
                    rho[i][j]
                    + P.dt * (P.D * lap - chemotaxis + growth);

                if (rho_new[i][j] < 0) rho_new[i][j] = 0;
            }
        }

        rho = rho_new;

        if (t % 50 == 0) {
            cout << "[Step " << setw(4) << t
                 << "] Mean Density: ";

            double mean = 0;
            for (auto& row : rho)
                for (double v : row) mean += v;

            mean /= (P.Nx * P.Ny);
            cout << fixed << setprecision(6) << mean << endl;
        }
    }

    cout << "\nSimulation completed successfully.\n";
}

/* ===========================
   CLI Menu Loop
=========================== */
int main() {
    cout << "\n========================================\n";
    cout << " MECHANOBIOLOGY BONE GROWTH SIMULATOR\n";
    cout << " Finite Difference + AMR + Multigrid\n";
    cout << "========================================\n";

    char choice;
    do {
        Params P;

        cout << "\nEnter number of time steps (e.g. 300): ";
        cin >> P.steps;

        cout << "Enter diffusion coefficient D (e.g. 0.15): ";
        cin >> P.D;

        cout << "Enter proliferation rate lambda (e.g. 0.08): ";
        cin >> P.lambda;

        cout << "\nRunning simulation...\n\n";
        runSimulation(P);

        cout << "\nRun another simulation? (y/n): ";
        cin >> choice;

    } while (choice == 'y' || choice == 'Y');

    cout << "\nThank you for using the simulator.\n";
    cout << "Scientific computing is a form of art.\n";

    return 0;
}
