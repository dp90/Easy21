#ifndef EASY21_H
#define EASY21_H

#include <tuple>
using namespace std;

class Easy21 {
    private:
        int N[21][10][2] = {0};
        double Vmax[21][10] = {0.0};
        double weights[36] = {0.0};
        float features[36] = {0.0};
    public:
        double Q[21][10][2] = {0.0}; // 21 player states, 10 dealer states, 2 actions: hit & stick
        Easy21();
        int draw();
        int step(array<int, 2>* ptS, int* ptA);
        int policy(array<int, 2>* ptS);
        int policy_LA(array<int, 2>* ptS);
        void MC_episode();
        void SARSA_episode(float labda);
        void assign_features(array<int, 2>* ptS, int *ptA);
        double approx_value();
        void SARSA_LA_episode(float labda);
        void compute_Q_LA();
        void calc_Vmax(bool save);
        double MSE(Easy21 extGame);
};

#endif