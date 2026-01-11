#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <iomanip>

using namespace std;

/*
 MIT-Level Computational Neuroscience Simulator
 Spatio-Temporal Hodgkin-Huxley Neural Field (3D)
 Author: Aditiya WP
*/

constexpr int NX = 10;
constexpr int NY = 10;
constexpr int NZ = 10;
constexpr int NEURONS = NX * NY * NZ;

constexpr double Cm = 1.0;
constexpr double sigma = 0.25;
constexpr double dt = 0.01;
constexpr double dx = 1.0;
constexpr int SYNAPTIC_DELAY_STEPS = 10;

struct Neuron {
    double V;
    double m, h, n;
};

vector<Neuron> field(NEURONS);
deque<vector<double>> delayBuffer;

/* Hodgkin-Huxley rate functions */
double alpha_m(double V) { return (0.1 * (25.0 - V)) / (exp((25.0 - V) / 10.0) - 1.0); }
double beta_m(double V)  { return 4.0 * exp(-V / 18.0); }

double alpha_h(double V) { return 0.07 * exp(-V / 20.0); }
double beta_h(double V)  { return 1.0 / (exp((30.0 - V) / 10.0) + 1.0); }

double alpha_n(double V) { return (0.01 * (10.0 - V)) / (exp((10.0 - V) / 10.0) - 1.0); }
double beta_n(double V)  { return 0.125 * exp(-V / 80.0); }

/* Ionic currents */
double INa(double V, double m, double h) {
    return 120.0 * pow(m,3) * h * (V - 115.0);
}

double IK(double V, double n) {
    return 36.0 * pow(n,4) * (V - 12.0);
}

double IL(double V) {
    return 0.3 * (V - 10.6);
}

int idx(int x, int y, int z) {
    return x + NX * (y + NY * z);
}

/* Diffusion operator */
double laplacian(int x, int y, int z) {
    int id = idx(x,y,z);
    double center = field[id].V;
    double sum = -6.0 * center;

    if(x>0) sum += field[idx(x-1,y,z)].V;
    if(x<NX-1) sum += field[idx(x+1,y,z)].V;
    if(y>0) sum += field[idx(x,y-1,z)].V;
    if(y<NY-1) sum += field[idx(x,y+1,z)].V;
    if(z>0) sum += field[idx(x,y,z-1)].V;
    if(z<NZ-1) sum += field[idx(x,y,z+1)].V;

    return sum / (dx*dx);
}

/* Initialization */
void initialize() {
    for(auto &n : field) {
        n.V = -65.0;
        n.m = 0.05;
        n.h = 0.6;
        n.n = 0.32;
    }
    delayBuffer.clear();
}

/* Simulation step */
void step() {
    vector<double> previousV(NEURONS);
    for(int i=0;i<NEURONS;i++) previousV[i] = field[i].V;

    delayBuffer.push_back(previousV);
    if(delayBuffer.size() > SYNAPTIC_DELAY_STEPS)
        delayBuffer.pop_front();

    vector<Neuron> next = field;

    for(int z=0; z<NZ; z++)
    for(int y=0; y<NY; y++)
    for(int x=0; x<NX; x++) {
        int id = idx(x,y,z);
        Neuron &n = field[id];

        double Isyn = 0.0;
        if(delayBuffer.size() == SYNAPTIC_DELAY_STEPS) {
            Isyn = 0.05 * (delayBuffer.front()[id] - n.V);
        }

        double Iion = INa(n.V,n.m,n.h) + IK(n.V,n.n) + IL(n.V);
        double diffusion = sigma * laplacian(x,y,z);

        next[id].V += dt * (diffusion - Iion + Isyn) / Cm;

        next[id].m += dt * (alpha_m(n.V)*(1-n.m) - beta_m(n.V)*n.m);
        next[id].h += dt * (alpha_h(n.V)*(1-n.h) - beta_h(n.V)*n.h);
        next[id].n += dt * (alpha_n(n.V)*(1-n.n) - beta_n(n.V)*n.n);
    }

    field = next;
}

int main() {
    cout << "=== 3D Hodgkin-Huxley Neural Field Simulator ===\n";
    cout << "Domain: " << NX << "x" << NY << "x" << NZ << " neurons\n";

    while(true) {
        initialize();

        int steps;
        cout << "\nEnter simulation steps (0 to exit): ";
        cin >> steps;
        if(steps <= 0) break;

        for(int t=0; t<steps; t++) step();

        double avgV = 0.0;
        for(auto &n : field) avgV += n.V;
        avgV /= NEURONS;

        cout << fixed << setprecision(4);
        cout << "Simulation complete.\n";
        cout << "Average membrane potential: " << avgV << " mV\n";
    }

    cout << "\nProgram terminated. Thank you.\n";
    return 0;
}
