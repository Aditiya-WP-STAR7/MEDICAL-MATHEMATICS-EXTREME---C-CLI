#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

using namespace std;

/* ===================== GLOBAL PARAMETERS ===================== */
const int GRID_SIZE = 20;          // FEM-like spatial resolution
const int IMMUNE_DIM = 500;        // Immune ODE dimension
const double RHO = 1060.0;         // Blood density (kg/m^3)
const double MU = 0.004;           // Blood viscosity (Pa·s)

/* ===================== DATA STRUCTURES ===================== */
struct Velocity {
    double u, v, w;
};

struct Cell {
    Velocity vel;
    double pressure;
};

/* ===================== INITIALIZATION ===================== */
void initializeGrid(vector<Cell>& grid) {
    for (auto& c : grid) {
        c.vel = {0.01, 0.0, 0.0};
        c.pressure = 80.0;
    }
}

void initializeImmune(vector<double>& y) {
    for (int i = 0; i < IMMUNE_DIM; i++)
        y[i] = 0.01 * sin(i * 0.01);
}

/* ===================== IMMUNE DYNAMICS ===================== */
void immuneODE(
    const vector<double>& y,
    vector<double>& dydt,
    double flowMagnitude
) {
    for (int i = 0; i < IMMUNE_DIM; i++) {
        double linear = -0.02 * y[i];
        double flowCoupling = 0.001 * flowMagnitude * y[i];
        double nonlinear = 0.0;

        if (i < IMMUNE_DIM - 1)
            nonlinear = 0.00001 * y[i] * y[i + 1];

        dydt[i] = linear + flowCoupling + nonlinear;
    }
}

/* ===================== NAVIER–STOKES STEP ===================== */
double navierStokesStep(
    vector<Cell>& grid,
    double dt
) {
    double totalVelocity = 0.0;

    for (auto& c : grid) {
        double laplacian = -0.1 * c.vel.u;
        double immuneForce = 0.0005;

        c.vel.u += dt * (
            (-c.vel.u * c.vel.u)
            + (MU / RHO) * laplacian
            + immuneForce
        );

        totalVelocity += fabs(c.vel.u);
    }

    return totalVelocity / grid.size();
}

/* ===================== IMMUNE INTEGRATION ===================== */
void integrateImmune(
    vector<double>& y,
    double flowMag,
    double dt
) {
    vector<double> dydt(IMMUNE_DIM);
    immuneODE(y, dydt, flowMag);

    for (int i = 0; i < IMMUNE_DIM; i++)
        y[i] += dt * dydt[i];
}

/* ===================== ADAPTIVE TIME STEP ===================== */
double adaptiveTimeStep(double flowMag) {
    if (flowMag > 0.1) return 0.0005;
    if (flowMag > 0.05) return 0.001;
    return 0.002;
}

/* ===================== MAIN SIMULATION ===================== */
void runSimulation() {
    vector<Cell> grid(GRID_SIZE * GRID_SIZE * GRID_SIZE);
    vector<double> immune(IMMUNE_DIM);

    initializeGrid(grid);
    initializeImmune(immune);

    double time = 0.0;
    double T_END = 0.1;

    while (time < T_END) {
        double flowMag = navierStokesStep(grid, 0.001);
        double dt = adaptiveTimeStep(flowMag);

        integrateImmune(immune, flowMag, dt);
        time += dt;
    }

    cout << "\nSimulation completed.\n";
    cout << "Final average flow magnitude: "
         << fixed << setprecision(6)
         << navierStokesStep(grid, 0.0) << endl;

    cout << "Sample immune variables:\n";
    for (int i = 0; i < 5; i++)
        cout << "y[" << i << "] = " << immune[i] << endl;
}

/* ===================== CLI LOOP ===================== */
int main() {
    cout << "============================================\n";
    cout << " MULTISCALE CARDIOVASCULAR–IMMUNE SIMULATOR\n";
    cout << " Hybrid FEM Navier–Stokes + 500D Immune ODE\n";
    cout << "============================================\n";

    char choice;
    do {
        runSimulation();

        cout << "\nRun another simulation? (y/n): ";
        cin >> choice;

    } while (choice == 'y' || choice == 'Y');

    cout << "\nProgram terminated. Stay scientific.\n";
    return 0;
}
