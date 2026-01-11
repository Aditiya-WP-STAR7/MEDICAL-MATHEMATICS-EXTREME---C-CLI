#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace std;

/*
 ============================================================
  HIGH-FIDELITY OPTIMAL CONTROL FOR DRUG DELIVERY
  Direct Collocation + Interior Point Method
  Author: Aditiya WP
  Target: MIT / Top Global Universities
 ============================================================
*/

// ----------------------------
// Problem Parameters
// ----------------------------
const int MAX_ITER = 500;
const double TOL = 1e-6;

// ----------------------------
// System Dynamics
// x1' = -k * x1 + u
// ----------------------------
double dynamics(double x, double u, double k) {
    return -k * x + u;
}

// ----------------------------
// Objective Integrand
// ----------------------------
double runningCost(double x, double u) {
    return x * x + u * u;
}

// ----------------------------
// Optimization Solver
// ----------------------------
void solveOptimalControl(
    int N,
    double T,
    double k,
    double xmax
) {
    double dt = T / (N - 1);
    vector<double> x(N, 0.1);
    vector<double> u(N, 0.05);

    double mu = 1.0;           // Barrier parameter
    double alpha = 0.01;       // Step size

    cout << "\n[INFO] Solving with " << N << " collocation nodes...\n";

    for (int iter = 0; iter < MAX_ITER; iter++) {
        double cost = 0.0;

        vector<double> grad_x(N, 0.0);
        vector<double> grad_u(N, 0.0);

        for (int i = 0; i < N - 1; i++) {
            double x_mid = 0.5 * (x[i] + x[i + 1]);
            double u_mid = 0.5 * (u[i] + u[i + 1]);

            cost += runningCost(x_mid, u_mid) * dt;

            // State constraint barrier
            cost -= mu * log(xmax - x_mid);

            // Control constraint barrier
            cost -= mu * log(u_mid);

            // Gradient (simplified)
            grad_x[i] += 2 * x_mid * dt + mu / (xmax - x_mid);
            grad_u[i] += 2 * u_mid * dt + mu / u_mid;

            // Dynamics penalty
            double defect = x[i + 1] - x[i] - dt * dynamics(x_mid, u_mid, k);
            cost += defect * defect * 100.0;

            grad_x[i] -= 200.0 * defect;
            grad_x[i + 1] += 200.0 * defect;
            grad_u[i] -= 200.0 * defect * dt;
        }

        // Gradient descent step
        double max_update = 0.0;
        for (int i = 1; i < N - 1; i++) {
            x[i] -= alpha * grad_x[i];
            u[i] -= alpha * grad_u[i];

            // Projection for constraints
            if (x[i] >= xmax) x[i] = xmax - 1e-6;
            if (u[i] <= 0.0) u[i] = 1e-6;

            max_update = max(max_update, fabs(grad_x[i]) + fabs(grad_u[i]));
        }

        mu *= 0.99;

        if (iter % 50 == 0) {
            cout << "Iter " << setw(4) << iter
                 << " | Cost = " << scientific << cost
                 << " | MaxGrad = " << max_update << endl;
        }

        if (max_update < TOL) {
            cout << "[CONVERGED] Optimization successful.\n";
            break;
        }
    }

    // Output final trajectory
    cout << "\nTime\tState(x)\tControl(u)\n";
    for (int i = 0; i < N; i += N / 10) {
        cout << fixed << setprecision(5)
             << i * dt << "\t" << x[i] << "\t" << u[i] << endl;
    }
}

// ----------------------------
// Main CLI Loop
// ----------------------------
int main() {
    cout << "=============================================\n";
    cout << " OPTIMAL CONTROL: DRUG DELIVERY SYSTEM (CLI)\n";
    cout << " Direct Collocation + Interior Point Method\n";
    cout << "=============================================\n";

    while (true) {
        int N;
        double T, k, xmax;

        cout << "\nEnter number of nodes (>=1000): ";
        cin >> N;
        if (N < 1000) N = 1000;

        cout << "Enter time horizon T: ";
        cin >> T;

        cout << "Enter decay constant k: ";
        cin >> k;

        cout << "Enter maximum state xmax: ";
        cin >> xmax;

        solveOptimalControl(N, T, k, xmax);

        char choice;
        cout << "\nRun another simulation? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y') break;
    }

    cout << "\n[EXIT] Program finished.\n";
    return 0;
}
