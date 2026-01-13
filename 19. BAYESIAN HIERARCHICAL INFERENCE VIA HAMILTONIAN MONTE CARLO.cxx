#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

/* ============================================
   Reverse-Mode Automatic Differentiation
   ============================================ */
struct Var {
    double val;
    double grad;
    vector<pair<Var*, double>> deps;

    Var(double v = 0.0) : val(v), grad(0.0) {}

    void backward(double g = 1.0) {
        grad += g;
        for (auto &d : deps)
            d.first->backward(g * d.second);
    }
};

Var operator+(Var &a, Var &b) {
    Var out(a.val + b.val);
    out.deps = {{&a, 1.0}, {&b, 1.0}};
    return out;
}

Var operator-(Var &a, Var &b) {
    Var out(a.val - b.val);
    out.deps = {{&a, 1.0}, {&b, -1.0}};
    return out;
}

Var operator*(Var &a, Var &b) {
    Var out(a.val * b.val);
    out.deps = {{&a, b.val}, {&b, a.val}};
    return out;
}

Var exp(Var &a) {
    Var out(std::exp(a.val));
    out.deps = {{&a, out.val}};
    return out;
}

/* ============================================
   Bayesian Hierarchical Model
   ============================================ */
struct BayesianModel {
    vector<double> y;
    vector<double> t;
    double sigma;

    BayesianModel(vector<double> y_, vector<double> t_, double s)
        : y(y_), t(t_), sigma(s) {}

    Var logPosterior(Var &theta, Var &mu, Var &omega) {
        Var logp(0.0);

        // Likelihood
        for (size_t i = 0; i < y.size(); i++) {
            Var pred = theta * *(new Var(t[i]));
            Var diff = *(new Var(y[i])) - pred;
            logp = logp - (diff * diff) * *(new Var(0.5 / (sigma * sigma)));
        }

        // Prior: theta ~ N(mu, omega)
        Var d = theta - mu;
        logp = logp - (d * d) * *(new Var(0.5 / (omega.val * omega.val)));

        return logp;
    }
};

/* ============================================
   Hamiltonian Monte Carlo with NUTS-like Stop
   ============================================ */
struct HMC {
    double step;
    int maxSteps;

    HMC(double e = 0.01, int m = 50) : step(e), maxSteps(m) {}

    double sample(BayesianModel &model, double initTheta) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> normal(0.0, 1.0);

        double theta = initTheta;
        double momentum = normal(gen);

        double theta0 = theta;
        double momentum0 = momentum;

        Var vtheta(theta);
        Var mu(0.0);
        Var omega(1.0);

        Var lp = model.logPosterior(vtheta, mu, omega);
        lp.backward();

        double grad = vtheta.grad;

        for (int i = 0; i < maxSteps; i++) {
            momentum += 0.5 * step * grad;
            theta += step * momentum;

            Var vt(theta);
            vt.grad = 0.0;
            Var lp_new = model.logPosterior(vt, mu, omega);
            lp_new.backward();
            grad = vt.grad;

            momentum += 0.5 * step * grad;

            // NUTS-like U-turn condition
            if ((theta - theta0) * momentum0 < 0)
                break;
        }

        return theta;
    }
};

/* ============================================
   CLI Interface
   ============================================ */
int main() {
    cout << fixed << setprecision(6);

    while (true) {
        int n;
        cout << "\nNumber of observations: ";
        cin >> n;

        vector<double> y(n), t(n);
        for (int i = 0; i < n; i++) {
            cout << "y[" << i << "], t[" << i << "]: ";
            cin >> y[i] >> t[i];
        }

        double sigma;
        cout << "Noise standard deviation: ";
        cin >> sigma;

        BayesianModel model(y, t, sigma);
        HMC sampler(0.01, 100);

        double theta = 0.0;
        cout << "\nRunning Bayesian HMC Sampling...\n";

        for (int i = 0; i < 20; i++) {
            theta = sampler.sample(model, theta);
            cout << "Iteration " << i + 1 << " | Theta Estimate: " << theta << endl;
        }

        char choice;
        cout << "\nRun another experiment? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y')
            break;
    }

    cout << "\nProgram terminated. Bayesian inference complete.\n";
    return 0;
}
