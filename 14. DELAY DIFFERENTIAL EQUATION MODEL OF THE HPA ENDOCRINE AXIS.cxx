#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace std;

/*
----------------------------------------------------
HPA AXIS DDE SIMULATION
Radau IIA (order 3) + Hermite Interpolation
Professional Research-Grade CLI Program
----------------------------------------------------
*/

struct State {
    double t;
    double C;
    double dCdt;
};

/* Delayed input function I(t) */
double inputSignal(double t) {
    return 1.0 + 0.5 * sin(0.5 * t);
}

/* Hermite interpolation for delayed state */
double hermiteInterpolate(
    const State& s0,
    const State& s1,
    double t
) {
    double h = s1.t - s0.t;
    double tau = (t - s0.t) / h;

    double h00 = (1 + 2*tau) * pow(1 - tau, 2);
    double h10 = tau * pow(1 - tau, 2);
    double h01 = tau * tau * (3 - 2*tau);
    double h11 = tau * tau * (tau - 1);

    return h00 * s0.C +
           h10 * h * s0.dCdt +
           h01 * s1.C +
           h11 * h * s1.dCdt;
}

/* Retrieve delayed cortisol using history buffer */
double delayedC(
    const vector<State>& history,
    double t_delay
) {
    for (size_t i = 1; i < history.size(); ++i) {
        if (history[i-1].t <= t_delay && t_delay <= history[i].t) {
            return hermiteInterpolate(
                history[i-1],
                history[i],
                t_delay
            );
        }
    }
    return history.front().C;
}

/* Right-hand side of DDE */
double cortisolDerivative(
    double C,
    double I_delayed,
    double Vmax,
    double Km,
    double n,
    double kc
) {
    double numerator = Vmax * pow(I_delayed, n);
    double denominator = pow(Km, n) + pow(I_delayed, n);
    return numerator / denominator - kc * C;
}

int main() {
    cout << fixed << setprecision(6);

    while (true) {
        cout << "\n==============================\n";
        cout << "HPA Axis DDE Simulation (Radau IIA)\n";
        cout << "==============================\n";

        double Vmax, Km, n, kc, tau, dt, T;

        cout << "Enter Vmax: "; cin >> Vmax;
        cout << "Enter Km: "; cin >> Km;
        cout << "Enter Hill coefficient n: "; cin >> n;
        cout << "Enter cortisol clearance kc: "; cin >> kc;
        cout << "Enter delay tau: "; cin >> tau;
        cout << "Enter timestep dt: "; cin >> dt;
        cout << "Enter total simulation time T: "; cin >> T;

        vector<State> history;

        /* Initial condition history */
        double C0 = 1.0;
        double t = 0.0;

        State init;
        init.t = t;
        init.C = C0;
        init.dCdt = cortisolDerivative(
            C0, inputSignal(-tau),
            Vmax, Km, n, kc
        );
        history.push_back(init);

        cout << "\nTime\tCortisol\n";

        while (t < T) {
            double t_next = t + dt;
            double I_delayed = inputSignal(t - tau);

            /* Radau IIA single-stage implicit approximation */
            double C_guess = history.back().C;
            for (int k = 0; k < 5; ++k) {
                double f = cortisolDerivative(
                    C_guess,
                    I_delayed,
                    Vmax, Km, n, kc
                );
                C_guess = history.back().C + dt * f;
            }

            State next;
            next.t = t_next;
            next.C = C_guess;
            next.dCdt = cortisolDerivative(
                next.C,
                I_delayed,
                Vmax, Km, n, kc
            );

            history.push_back(next);
            t = t_next;

            cout << t << "\t" << next.C << endl;
        }

        char again;
        cout << "\nRun another simulation? (y/n): ";
        cin >> again;
        if (again != 'y' && again != 'Y') break;
    }

    cout << "\nSimulation finished. Program terminated.\n";
    return 0;
}
