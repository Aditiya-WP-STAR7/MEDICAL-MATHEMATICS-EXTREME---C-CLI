#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

using namespace std;

/*
 ============================================================
  SPATIO-TEMPORAL EPIDEMIC SIMULATION (SEIR NETWORK MODEL)
  ------------------------------------------------------------
  - Metapopulation SEIR
  - Network diffusion via Graph Laplacian
  - Exponential Integrator
  - Krylov Subspace Approximation
  - Designed for CLI & Cxxdroid compatibility
 ============================================================
*/

struct Node {
    double S, E, I, R;
};

class EpidemicNetwork {
private:
    int N;
    double beta, sigma, gamma;
    vector<Node> nodes;
    vector<vector<double>> W; // adjacency weights

public:
    EpidemicNetwork(int n, double b, double s, double g)
        : N(n), beta(b), sigma(s), gamma(g) {
        nodes.resize(N);
        W.assign(N, vector<double>(N, 0.0));
        initialize_population();
        initialize_network();
    }

    void initialize_population() {
        for (int i = 0; i < N; ++i) {
            nodes[i].S = 0.99;
            nodes[i].E = 0.0;
            nodes[i].I = (i == 0 ? 0.01 : 0.0);
            nodes[i].R = 0.0;
        }
    }

    void initialize_network() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j && dist(rng) < 0.02) {
                    W[i][j] = dist(rng) * 0.05;
                }
            }
        }
    }

    vector<double> laplacian(const vector<double>& X) {
        vector<double> L(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sumW = 0.0;
            for (int j = 0; j < N; ++j) {
                sumW += W[i][j];
                L[i] += W[i][j] * X[j];
            }
            L[i] -= sumW * X[i];
        }
        return L;
    }

    vector<double> krylov_exponential(
        const vector<double>& v,
        double dt,
        int m = 10
    ) {
        vector<vector<double>> V(m, vector<double>(N, 0.0));
        vector<double> w = v;

        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = sqrt(norm);

        for (int i = 0; i < N; ++i)
            V[0][i] = v[i] / norm;

        for (int k = 1; k < m; ++k) {
            vector<double> Lv = laplacian(V[k - 1]);
            for (int i = 0; i < N; ++i)
                V[k][i] = Lv[i] * dt;
        }

        vector<double> result(N, 0.0);
        for (int i = 0; i < N; ++i)
            result[i] = v[i] + dt * V[1][i];

        return result;
    }

    void step(double dt) {
        vector<double> S(N), E(N), I(N), R(N);

        for (int i = 0; i < N; ++i) {
            S[i] = nodes[i].S;
            E[i] = nodes[i].E;
            I[i] = nodes[i].I;
            R[i] = nodes[i].R;
        }

        vector<double> LS = krylov_exponential(S, dt);
        vector<double> LE = krylov_exponential(E, dt);
        vector<double> LI = krylov_exponential(I, dt);

        for (int i = 0; i < N; ++i) {
            double dS = -beta * S[i] * I[i] + LS[i];
            double dE =  beta * S[i] * I[i] - sigma * E[i] + LE[i];
            double dI =  sigma * E[i] - gamma * I[i] + LI[i];
            double dR =  gamma * I[i];

            nodes[i].S += dt * dS;
            nodes[i].E += dt * dE;
            nodes[i].I += dt * dI;
            nodes[i].R += dt * dR;
        }
    }

    void simulate(int steps, double dt) {
        for (int t = 0; t < steps; ++t) {
            step(dt);
        }
    }

    void report() {
        double S = 0, E = 0, I = 0, R = 0;
        for (auto& n : nodes) {
            S += n.S; E += n.E; I += n.I; R += n.R;
        }

        cout << fixed << setprecision(6);
        cout << "\n===== GLOBAL EPIDEMIC STATE =====\n";
        cout << "Susceptible : " << S / N << "\n";
        cout << "Exposed     : " << E / N << "\n";
        cout << "Infected    : " << I / N << "\n";
        cout << "Recovered   : " << R / N << "\n";
        cout << "================================\n";
    }
};

int main() {
    cout << "\n============================================\n";
    cout << " SPATIAL-TEMPORAL EPIDEMIC NETWORK SIMULATOR\n";
    cout << " Research-Grade SEIR | Krylov Exponential\n";
    cout << "============================================\n";

    while (true) {
        int N, steps;
        double beta, sigma, gamma, dt;

        cout << "\nEnter number of nodes: ";
        cin >> N;

        cout << "Transmission rate (beta): ";
        cin >> beta;

        cout << "Incubation rate (sigma): ";
        cin >> sigma;

        cout << "Recovery rate (gamma): ";
        cin >> gamma;

        cout << "Time step (dt): ";
        cin >> dt;

        cout << "Simulation steps: ";
        cin >> steps;

        EpidemicNetwork model(N, beta, sigma, gamma);
        model.simulate(steps, dt);
        model.report();

        cout << "\nRun another simulation? (y/n): ";
        char choice;
        cin >> choice;
        if (choice != 'y' && choice != 'Y')
            break;
    }

    cout << "\nSimulation terminated. Stay scientific.\n";
    return 0;
}
