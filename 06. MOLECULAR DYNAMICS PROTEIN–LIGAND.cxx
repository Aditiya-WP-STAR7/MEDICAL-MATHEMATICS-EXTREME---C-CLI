#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

// ================== Physical Constants ==================
constexpr double kB = 1.380649e-23;     // Boltzmann constant (J/K)
constexpr double fs = 1e-15;            // femtosecond
constexpr double dt = 2.0 * fs;         // 2 fs timestep

// ================== Vector Math ==================
struct Vec3 {
    double x, y, z;

    Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
};

// ================== Atom ==================
struct Atom {
    Vec3 r;       // position
    Vec3 v;       // velocity
    Vec3 f;       // force
    double m;     // mass
};

// ================== Force Field (AMBER/CHARMM Abstraction) ==================
class ForceField {
public:
    virtual void compute(vector<Atom>& atoms) = 0;
};

// Simple Lennard-Jones placeholder (extendable)
class LennardJonesFF : public ForceField {
public:
    void compute(vector<Atom>& atoms) override {
        for (auto& a : atoms) a.f = {0,0,0};

        for (size_t i=0;i<atoms.size();i++) {
            for (size_t j=i+1;j<atoms.size();j++) {
                Vec3 dr = atoms[i].r - atoms[j].r;
                double r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + 1e-9;
                double inv6 = pow(1.0/r2,3);
                double fmag = 48.0 * inv6 * (inv6 - 0.5) / r2;
                Vec3 f = dr * fmag;
                atoms[i].f += f;
                atoms[j].f -= f;
            }
        }
    }
};

// ================== SHAKE Constraint (Bond Fixing) ==================
void applySHAKE(vector<Atom>& atoms, double bondLength) {
    for (size_t i=0;i+1<atoms.size();i+=2) {
        Vec3 d = atoms[i+1].r - atoms[i].r;
        double dist = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
        double corr = (dist - bondLength) / dist * 0.5;
        atoms[i].r += d * corr;
        atoms[i+1].r -= d * corr;
    }
}

// ================== BAOAB Langevin Integrator ==================
class LangevinBAOAB {
    double gamma;
    double temperature;
    mt19937 rng;
    normal_distribution<double> normal;

public:
    LangevinBAOAB(double g, double T)
        : gamma(g), temperature(T),
          rng(random_device{}()), normal(0.0,1.0) {}

    void step(vector<Atom>& atoms, ForceField& ff) {
        // B
        for (auto& a : atoms)
            a.v += a.f * (0.5 * dt / a.m);

        // A
        for (auto& a : atoms)
            a.r += a.v * (0.5 * dt);

        // O (thermostat)
        for (auto& a : atoms) {
            double c1 = exp(-gamma * dt);
            double c2 = sqrt((1 - c1*c1) * kB * temperature / a.m);
            a.v = a.v * c1 + Vec3{
                c2 * normal(rng),
                c2 * normal(rng),
                c2 * normal(rng)
            };
        }

        // A
        for (auto& a : atoms)
            a.r += a.v * (0.5 * dt);

        // Recompute forces
        ff.compute(atoms);

        // B
        for (auto& a : atoms)
            a.v += a.f * (0.5 * dt / a.m);
    }
};

// ================== Main CLI Program ==================
int main() {
    cout << "=============================================\n";
    cout << " Molecular Dynamics Protein-Ligand Simulator\n";
    cout << " BAOAB Langevin | SHAKE | AMBER/CHARMM Style\n";
    cout << "=============================================\n";

    while (true) {
        int N;
        double T, gamma;
        long steps;

        cout << "\nNumber of atoms: ";
        cin >> N;

        cout << "Temperature (K): ";
        cin >> T;

        cout << "Friction gamma (1/s): ";
        cin >> gamma;

        cout << "Simulation steps: ";
        cin >> steps;

        vector<Atom> atoms(N);
        for (auto& a : atoms) {
            a.r = {drand48(), drand48(), drand48()};
            a.v = {0,0,0};
            a.m = 1.66054e-27;
        }

        LennardJonesFF ff;
        ff.compute(atoms);

        LangevinBAOAB integrator(gamma, T);

        auto start = chrono::high_resolution_clock::now();

        for (long i=0;i<steps;i++) {
            integrator.step(atoms, ff);
            applySHAKE(atoms, 1.0e-10);

            if (i % (steps/10 + 1) == 0)
                cout << "Progress: " << (100.0*i/steps) << "%\n";
        }

        auto end = chrono::high_resolution_clock::now();
        cout << "Simulation completed in "
             << chrono::duration<double>(end-start).count()
             << " seconds.\n";

        char again;
        cout << "\nRun another simulation? (y/n): ";
        cin >> again;
        if (again != 'y' && again != 'Y') break;
    }

    return 0;
}
