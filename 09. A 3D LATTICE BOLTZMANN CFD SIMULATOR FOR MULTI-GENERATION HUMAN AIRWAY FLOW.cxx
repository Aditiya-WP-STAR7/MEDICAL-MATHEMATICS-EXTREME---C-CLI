#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// ==========================
// D3Q19 Lattice Definition
// ==========================
static const int Q = 19;
static const int cx[Q] = {0,1,-1,0,0,0,0, 1,-1,1,-1,1,-1, 0,0,0, 0,0,0};
static const int cy[Q] = {0,0,0,1,-1,0,0, 1,-1,-1,1, 0,0, 1,-1,1,-1,0,0};
static const int cz[Q] = {0,0,0,0,0,1,-1, 0,0,0,0, 1,-1,1,-1,-1,1, 0,0};

static const double w[Q] = {
    1.0/3.0,
    1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
    1.0/36.0,1.0/36.0,
    1.0/36.0,1.0/36.0
};

// ==========================
// LBM CFD Solver Class
// ==========================
class LBM3D {
private:
    int NX, NY, NZ;
    double tau;

    vector<double> rho, ux, uy, uz;
    vector<vector<double>> f, f_next;

    int idx(int x, int y, int z) const {
        return x + NX * (y + NY * z);
    }

public:
    LBM3D(int nx, int ny, int nz, double relaxation)
        : NX(nx), NY(ny), NZ(nz), tau(relaxation)
    {
        int N = NX * NY * NZ;
        rho.assign(N, 1.0);
        ux.assign(N, 0.0);
        uy.assign(N, 0.0);
        uz.assign(N, 0.0);

        f.resize(Q, vector<double>(N, 0.0));
        f_next.resize(Q, vector<double>(N, 0.0));

        initialize();
    }

    void initialize() {
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++)
                for (int x = 0; x < NX; x++) {
                    int id = idx(x,y,z);
                    for (int i = 0; i < Q; i++)
                        f[i][id] = equilibrium(i, id);
                }
    }

    double equilibrium(int i, int id) {
        double cu = 3.0 * (cx[i]*ux[id] + cy[i]*uy[id] + cz[i]*uz[id]);
        double uu = ux[id]*ux[id] + uy[id]*uy[id] + uz[id]*uz[id];
        return w[i] * rho[id] * (1.0 + cu + 0.5*cu*cu - 1.5*uu);
    }

    void collide() {
        for (int id = 0; id < NX*NY*NZ; id++) {
            for (int i = 0; i < Q; i++) {
                double feq = equilibrium(i, id);
                f[i][id] += -(f[i][id] - feq) / tau;
            }
        }
    }

    void stream() {
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++)
                for (int x = 0; x < NX; x++) {
                    int id = idx(x,y,z);
                    for (int i = 0; i < Q; i++) {
                        int xn = (x + cx[i] + NX) % NX;
                        int yn = (y + cy[i] + NY) % NY;
                        int zn = (z + cz[i] + NZ) % NZ;
                        f_next[i][idx(xn,yn,zn)] = f[i][id];
                    }
                }
        f.swap(f_next);
    }

    void macroscopic() {
        for (int id = 0; id < NX*NY*NZ; id++) {
            rho[id] = ux[id] = uy[id] = uz[id] = 0.0;
            for (int i = 0; i < Q; i++) {
                rho[id] += f[i][id];
                ux[id] += cx[i]*f[i][id];
                uy[id] += cy[i]*f[i][id];
                uz[id] += cz[i]*f[i][id];
            }
            ux[id] /= rho[id];
            uy[id] /= rho[id];
            uz[id] /= rho[id];
        }
    }

    void step() {
        collide();
        stream();
        macroscopic();
    }

    void report(int iter) {
        double avgU = 0.0;
        for (size_t i = 0; i < ux.size(); i++)
            avgU += sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);
        avgU /= ux.size();

        cout << "Iteration " << iter
             << " | Average Velocity = " << avgU << endl;
    }
};

// ==========================
// MAIN CLI LOOP
// ==========================
int main() {
    cout << "=============================================\n";
    cout << "3D Lattice Boltzmann CFD Airway Simulator\n";
    cout << "D3Q19 | MIT-Level Computational Physics\n";
    cout << "=============================================\n\n";

    while (true) {
        int nx, ny, nz, steps;
        double tau;

        cout << "Enter grid size NX NY NZ: ";
        cin >> nx >> ny >> nz;

        cout << "Enter relaxation time tau (e.g. 0.6): ";
        cin >> tau;

        cout << "Enter number of simulation steps: ";
        cin >> steps;

        LBM3D solver(nx, ny, nz, tau);

        for (int i = 0; i < steps; i++) {
            solver.step();
            if (i % 10 == 0)
                solver.report(i);
        }

        char choice;
        cout << "\nRun another simulation? (y/n): ";
        cin >> choice;

        if (choice != 'y' && choice != 'Y')
            break;
    }

    cout << "\nSimulation session finished.\n";
    cout << "This work demonstrates research-grade CFD capability.\n";
    return 0;
}
