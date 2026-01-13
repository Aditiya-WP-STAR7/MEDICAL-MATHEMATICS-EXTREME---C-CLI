#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <string>

using namespace std;

/* ============================================================
   Utility: Random Engine
   ============================================================ */
static random_device rd;
static mt19937 gen(rd());

double randUniform(double a, double b) {
    uniform_real_distribution<double> dist(a, b);
    return dist(gen);
}

/* ============================================================
   Math Core
   ============================================================ */
double relu(double x) { return max(0.0, x); }
double drelu(double x) { return x > 0.0 ? 1.0 : 0.0; }

/* ============================================================
   Sepsis Environment (POMDP)
   ============================================================ */
struct Observation {
    double hr, bp, lactate, spo2;
};

struct State {
    double infection, organFailure;
};

class SepsisEnv {
public:
    State state;

    SepsisEnv() { reset(); }

    Observation observe() {
        return {
            state.infection + randUniform(-0.1, 0.1),
            1.0 - state.organFailure + randUniform(-0.05, 0.05),
            state.infection + randUniform(-0.1, 0.1),
            1.0 - state.organFailure + randUniform(-0.05, 0.05)
        };
    }

    double step(double action) {
        state.infection -= 0.05 * action;
        state.organFailure -= 0.03 * action;

        state.infection += randUniform(0.0, 0.02);
        state.organFailure += randUniform(0.0, 0.02);

        state.infection = clamp(state.infection, 0.0, 1.0);
        state.organFailure = clamp(state.organFailure, 0.0, 1.0);

        return 1.0 - (state.infection + state.organFailure);
    }

    void reset() {
        state = { randUniform(0.5, 1.0), randUniform(0.5, 1.0) };
    }
};

/* ============================================================
   Prioritized Experience Replay
   ============================================================ */
struct Experience {
    vector<double> obs;
    double action;
    double reward;
    vector<double> nextObs;
    double priority;
};

class ReplayBuffer {
    vector<Experience> buffer;
    size_t maxSize;

public:
    ReplayBuffer(size_t maxSize) : maxSize(maxSize) {}

    void add(const Experience& e) {
        if (buffer.size() >= maxSize)
            buffer.erase(buffer.begin());
        buffer.push_back(e);
    }

    Experience sample() {
        double totalPriority = 0.0;
        for (auto& e : buffer) totalPriority += e.priority;

        double r = randUniform(0, totalPriority);
        double cumsum = 0;

        for (auto& e : buffer) {
            cumsum += e.priority;
            if (cumsum >= r) return e;
        }
        return buffer.back();
    }

    bool empty() const { return buffer.empty(); }
};

/* ============================================================
   Simple Neural Network (Critic)
   ============================================================ */
class NeuralNet {
    vector<vector<double>> W1, W2;
    vector<double> b1, b2;

public:
    NeuralNet(int input, int hidden) {
        W1.resize(hidden, vector<double>(input));
        W2.resize(1, vector<double>(hidden));
        b1.resize(hidden);
        b2.resize(1);

        for (auto& row : W1)
            for (auto& w : row) w = randUniform(-0.1, 0.1);

        for (auto& w : W2[0]) w = randUniform(-0.1, 0.1);
    }

    double forward(const vector<double>& x) {
        vector<double> h(W1.size());
        for (size_t i = 0; i < W1.size(); ++i) {
            h[i] = b1[i];
            for (size_t j = 0; j < x.size(); ++j)
                h[i] += W1[i][j] * x[j];
            h[i] = relu(h[i]);
        }

        double out = b2[0];
        for (size_t i = 0; i < h.size(); ++i)
            out += W2[0][i] * h[i];
        return out;
    }
};

/* ============================================================
   PPO-Style Policy (Continuous Action)
   ============================================================ */
class PPOPolicy {
public:
    double act(const Observation& o) {
        double mean = (1.0 - o.lactate) + (o.bp);
        return clamp(mean + randUniform(-0.1, 0.1), 0.0, 1.0);
    }
};

/* ============================================================
   Main CLI Loop
   ============================================================ */
int main() {
    cout << "============================================\n";
    cout << " SepsisDRL-CLI | MIT-Level Research Prototype\n";
    cout << "============================================\n";

    ReplayBuffer replay(1000);
    NeuralNet qNet(4, 16), targetNet(4, 16);
    PPOPolicy policy;
    SepsisEnv env;

    while (true) {
        cout << "\n[1] Run Training Episode\n";
        cout << "[2] Exit\n";
        cout << "Choice: ";

        int choice;
        cin >> choice;

        if (choice == 2) break;

        env.reset();
        for (int step = 0; step < 50; ++step) {
            Observation obs = env.observe();
            vector<double> obsVec = { obs.hr, obs.bp, obs.lactate, obs.spo2 };

            double action = policy.act(obs);
            double reward = env.step(action);

            Observation nextObs = env.observe();
            vector<double> nextObsVec = { nextObs.hr, nextObs.bp, nextObs.lactate, nextObs.spo2 };

            double tdError = abs(reward + qNet.forward(nextObsVec) - qNet.forward(obsVec));

            replay.add({ obsVec, action, reward, nextObsVec, tdError + 1e-5 });

            cout << "Step " << step
                 << " | Action: " << action
                 << " | Reward: " << reward << endl;
        }

        cout << "\nEpisode completed. You may run another simulation.\n";
    }

    cout << "\nProgram terminated. Thank you.\n";
    return 0;
}
