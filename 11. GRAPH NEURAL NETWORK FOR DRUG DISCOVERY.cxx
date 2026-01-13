#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>

using namespace std;

/* ----------------- Utility Functions ----------------- */

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double random_weight() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

/* ----------------- Atom Node ----------------- */

struct Atom {
    vector<double> features;
    vector<int> neighbors;
};

/* ----------------- Graph Neural Network ----------------- */

class GraphNeuralNetwork {
private:
    int feature_size;
    vector<vector<double>> W;
    vector<vector<double>> U;
    vector<double> attention;

    double learning_rate;

public:
    GraphNeuralNetwork(int f, double lr) {
        feature_size = f;
        learning_rate = lr;

        W.resize(f, vector<double>(f));
        U.resize(f, vector<double>(f));
        attention.resize(2 * f);

        for (int i = 0; i < f; i++) {
            for (int j = 0; j < f; j++) {
                W[i][j] = random_weight();
                U[i][j] = random_weight();
            }
        }

        for (double &a : attention)
            a = random_weight();
    }

    vector<double> matvec(const vector<vector<double>>& M, const vector<double>& v) {
        vector<double> result(feature_size, 0);
        for (int i = 0; i < feature_size; i++)
            for (int j = 0; j < feature_size; j++)
                result[i] += M[i][j] * v[j];
        return result;
    }

    double compute_attention(const vector<double>& hv, const vector<double>& hu) {
        double score = 0;
        for (int i = 0; i < feature_size; i++) {
            score += attention[i] * hv[i];
            score += attention[i + feature_size] * hu[i];
        }
        return exp(score);
    }

    vector<vector<double>> forward(vector<Atom>& graph) {
        vector<vector<double>> new_features(graph.size(),
                                             vector<double>(feature_size, 0));

        for (int v = 0; v < graph.size(); v++) {
            vector<double> self = matvec(W, graph[v].features);
            vector<double> message(feature_size, 0);

            double attention_sum = 0;
            for (int u : graph[v].neighbors)
                attention_sum += compute_attention(graph[v].features,
                                                    graph[u].features);

            for (int u : graph[v].neighbors) {
                double alpha = compute_attention(graph[v].features,
                                                  graph[u].features) / attention_sum;
                vector<double> msg = matvec(U, graph[u].features);
                for (int i = 0; i < feature_size; i++)
                    message[i] += alpha * msg[i];
            }

            for (int i = 0; i < feature_size; i++)
                new_features[v][i] = relu(self[i] + message[i]);
        }

        return new_features;
    }

    double readout(const vector<vector<double>>& features) {
        double sum = 0;
        for (auto& f : features)
            for (double x : f)
                sum += x;
        return sum / features.size();
    }

    void train(vector<Atom>& graph, double target, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto output = forward(graph);
            double prediction = readout(output);
            double loss = pow(prediction - target, 2);

            double grad = 2 * (prediction - target);

            for (int i = 0; i < feature_size; i++)
                for (int j = 0; j < feature_size; j++)
                    W[i][j] -= learning_rate * grad;

            if (epoch % 10 == 0)
                cout << "Epoch " << epoch
                     << " | Loss: " << loss
                     << " | Prediction: " << prediction << endl;
        }
    }
};

/* ----------------- CLI Interface ----------------- */

int main() {
    srand(42);

    while (true) {
        cout << "\n=====================================\n";
        cout << "GNN Drug Discovery Simulator (CLI)\n";
        cout << "=====================================\n";

        int atoms;
        cout << "Enter number of atoms in molecule: ";
        cin >> atoms;

        vector<Atom> molecule(atoms);

        for (int i = 0; i < atoms; i++) {
            molecule[i].features.resize(3);
            cout << "Atom " << i << " features (3 values): ";
            for (double &x : molecule[i].features)
                cin >> x;
        }

        int bonds;
        cout << "Enter number of bonds: ";
        cin >> bonds;

        for (int i = 0; i < bonds; i++) {
            int a, b;
            cin >> a >> b;
            molecule[a].neighbors.push_back(b);
            molecule[b].neighbors.push_back(a);
        }

        double target;
        cout << "Target biological activity: ";
        cin >> target;

        GraphNeuralNetwork gnn(3, 0.01);
        gnn.train(molecule, target, 100);

        char again;
        cout << "\nRun another experiment? (y/n): ";
        cin >> again;
        if (again != 'y' && again != 'Y')
            break;
    }

    cout << "\nProgram terminated.\n";
    return 0;
}
