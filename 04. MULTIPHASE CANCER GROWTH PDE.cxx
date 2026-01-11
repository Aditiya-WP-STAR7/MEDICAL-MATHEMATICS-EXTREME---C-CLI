#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

/*
====================================================
 Multiphase Tumor Growth PDE Simulator
 Author: Future MIT Polymath (You)
 Method: Finite Volume + Operator Splitting
====================================================
*/

constexpr int PHASES = 5;

// ---------- Flux Limiter (Minmod) ----------
double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

// ---------- Grid Structure ----------
struct Grid {
    int N;
    double dx, dt;

    vector<vector<vector<double>>> phi; // [phase][x][y]
    vector<vector<double>> c;            // nutrient

    Grid(int n, double dx_, double dt_)
        : N(n), dx(dx_), dt(dt_) {

        phi.resize(PHASES,
            vector<vector<double>>(N, vector<double>(N, 0.0)));

        c.resize(N, vector<double>(N, 1.0));
    }
};

// ---------- Diffusion Operator (FVM) ----------
void diffuse(Grid &g, int phase, double D) {
    auto old = g.phi[phase];

    for (int i = 1; i < g.N - 1; i++) {
        for (int j = 1; j < g.N - 1; j++) {

            double lap =
                old[i+1][j] + old[i-1][j] +
                old[i][j+1] + old[i][j-1] -
                4.0 * old[i][j];

            g.phi[phase][i][j] +=
                g.dt * D * lap / (g.dx * g.dx);
        }
    }
}

// ---------- Reaction Step ----------
void reaction(Grid &g,
              const vector<double> &rho,
              const vector<double> &lambda) {

    for (int i = 0; i < g.N; i++) {
        for (int j = 0; j < g.N; j++) {

            double sum_phi = 0.0;
            for (int p = 0; p < PHASES; p++)
                sum_phi += g.phi[p][i][j];

            for (int p = 0; p < PHASES; p++) {
                double growth =
                    rho[p] * g.phi[p][i][j] * (1.0 - sum_phi);

                double death =
                    lambda[p] * g.phi[p][i][j];

                double source =
                    0.05 * g.c[i][j]; // angiogenic coupling

                g.phi[p][i][j] +=
                    g.dt * (growth - death + source);

                g.phi[p][i][j] =
                    max(0.0, g.phi[p][i][j]); // positivity
            }
        }
    }
}

// ---------- Nutrient Equation ----------
void updateNutrient(Grid &g, double d, double delta,
                    const vector<double> &beta) {

    auto old = g.c;

    for (int i = 1; i < g.N - 1; i++) {
        for (int j = 1; j < g.N - 1; j++) {

            double lap =
                old[i+1][j] + old[i-1][j] +
                old[i][j+1] + old[i][j-1] -
                4.0 * old[i][j];

            double source = 0.0;
            for (int p = 0; p < PHASES; p++)
                source += beta[p] * g.phi[p][i][j];

            g.c[i][j] += g.dt *
                (d * lap / (g.dx * g.dx)
                 - delta * old[i][j]
                 + source);

            g.c[i][j] = max(0.0, g.c[i][j]);
        }
    }
}

// ---------- Initialization ----------
void initializeTumor(Grid &g) {
    int mid = g.N / 2;
    for (int p = 0; p < PHASES; p++)
        g.phi[p][mid][mid] = 0.05 * (p + 1);
}

// ---------- Simulation ----------
void simulate() {

    int N, steps;
    cout << "\nGrid size (e.g. 50): ";
    cin >> N;

    cout << "Time steps: ";
    cin >> steps;

    Grid g(N, 1.0, 0.01);
    initializeTumor(g);

    vector<double> D(PHASES, 0.1);
    vector<double> rho(PHASES, 0.5);
    vector<double> lambda(PHASES, 0.1);
    vector<double> beta(PHASES, 0.2);

    for (int t = 0; t < steps; t++) {

        for (int p = 0; p < PHASES; p++)
            diffuse(g, p, D[p]);

        reaction(g, rho, lambda);
        updateNutrient(g, 0.2, 0.05, beta);

        if (t % 50 == 0)
            cout << "Step " << t << " completed.\n";
    }

    cout << "\nSimulation finished.\n";
    cout << "Tumor center phase values:\n";

    int mid = N / 2;
    for (int p = 0; p < PHASES; p++)
        cout << "Phase " << p+1 << ": "
             << g.phi[p][mid][mid] << "\n";
}

// ---------- CLI LOOP ----------
int main() {

    cout << "============================================\n";
    cout << " Multiphase Cancer PDE Simulator (C++)\n";
    cout << " Finite Volume | Angiogenesis | MIT-Level\n";
    cout << "============================================\n";

    while (true) {
        simulate();

        char choice;
        cout << "\nRun another simulation? (y/n): ";
        cin >> choice;

        if (choice != 'y' && choice != 'Y') {
            cout << "\nExiting. Go conquer MIT.\n";
            break;
        }
    }
    return 0;
}
