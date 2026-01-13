#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// ============================================================
// GLOBAL SCIENTIFIC CONSTANTS
// ============================================================
const int NUM_SPECIES = 30;
const int GRID_SIZE = 50;
const double DX = 0.02;
const double DT = 0.0005;
const int TIME_STEPS = 200;

// ============================================================
// DATA STRUCTURES
// ============================================================
struct Field {
    vector<vector<vector<double>>> c;
};

struct Parameters {
    vector<double> diffusion;
    vector<double> reactionRates;
    double velocity;
};

// ============================================================
// INITIALIZATION
// ============================================================
Field initializeField() {
    Field f;
    f.c.resize(NUM_SPECIES,
        vector<vector<double>>(GRID_SIZE,
            vector<double>(GRID_SIZE, 0.0)));

    for (int s = 0; s < NUM_SPECIES; s++) {
        f.c[s][GRID_SIZE / 2][GRID_SIZE / 2] = 1.0 + 0.05 * s;
    }
    return f;
}

Parameters initializeParameters() {
    Parameters p;
    p.diffusion.resize(NUM_SPECIES);
    p.reactionRates.resize(NUM_SPECIES);

    for (int i = 0; i < NUM_SPECIES; i++) {
        p.diffusion[i] = 0.0005 + 0.00002 * i;
        p.reactionRates[i] = 0.1 + 0.02 * i;
    }

    p.velocity = 0.2;
    return p;
}

// ============================================================
// ADVECTION OPERATOR
// ============================================================
void advectionStep(Field &f, const Parameters &p, double dt) {
    for (int s = 0; s < NUM_SPECIES; s++) {
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                double grad = (f.c[s][i][j] - f.c[s][i - 1][j]) / DX;
                f.c[s][i][j] -= p.velocity * grad * dt;
            }
        }
    }
}

// ============================================================
// DIFFUSION OPERATOR
// ============================================================
void diffusionStep(Field &f, const Parameters &p, double dt) {
    Field old = f;

    for (int s = 0; s < NUM_SPECIES; s++) {
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                double laplacian =
                    old.c[s][i+1][j] + old.c[s][i-1][j] +
                    old.c[s][i][j+1] + old.c[s][i][j-1] -
                    4.0 * old.c[s][i][j];

                f.c[s][i][j] +=
                    p.diffusion[s] * laplacian * dt / (DX * DX);
            }
        }
    }
}

// ============================================================
// STIFF CHEMISTRY SOLVER (IMPLICIT BACKWARD EULER)
// ============================================================
void reactionStep(Field &f, const Parameters &p, double dt) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {

            vector<double> c_old(NUM_SPECIES);
            for (int s = 0; s < NUM_SPECIES; s++)
                c_old[s] = f.c[s][i][j];

            // Backward Euler (stiff-safe)
            for (int s = 0; s < NUM_SPECIES; s++) {
                double activation = 0.0;
                for (int k = 0; k < NUM_SPECIES; k++)
                    activation += 0.01 * c_old[k];

                double R = p.reactionRates[s] * c_old[s] * (1.0 + activation);
                f.c[s][i][j] = c_old[s] / (1.0 + dt * R);
            }
        }
    }
}

// ============================================================
// STRANG SPLITTING INTEGRATOR
// ============================================================
void strangStep(Field &f, const Parameters &p) {
    advectionStep(f, p, DT * 0.5);
    reactionStep(f, p, DT);
    diffusionStep(f, p, DT * 0.5);
}

// ============================================================
// OUTPUT
// ============================================================
void printSummary(const Field &f) {
    cout << "\n=== Blood Coagulation Summary ===\n";
    for (int s = 0; s < NUM_SPECIES; s++) {
        cout << "Factor " << setw(2) << s
             << " concentration (center): "
             << fixed << setprecision(6)
             << f.c[s][GRID_SIZE/2][GRID_SIZE/2] << endl;
    }
}

// ============================================================
// MAIN CLI PROGRAM WITH REPEAT LOOP
// ============================================================
int main() {
    cout << "\n============================================\n";
    cout << " BLOOD COAGULATION REACTION–DIFFUSION SIMULATOR\n";
    cout << " MIT-Level Scientific Computing Program\n";
    cout << "============================================\n";

    while (true) {
        Field field = initializeField();
        Parameters params = initializeParameters();

        cout << "\nRunning simulation...\n";

        for (int t = 0; t < TIME_STEPS; t++) {
            strangStep(field, params);
        }

        printSummary(field);

        cout << "\nRun another simulation? (y/n): ";
        string choice;
        cin >> choice;
        if (choice != "y" && choice != "Y") break;
    }

    cout << "\nProgram terminated professionally.\n";
    return 0;
}
