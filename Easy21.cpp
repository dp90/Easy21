#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tuple>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include "Easy21.h"

using namespace std;

Easy21::Easy21() {
}

int Easy21::draw() {
    int redCard = rand() % 3;
    int cardValue = rand() % 10 + 1;
    int score = (redCard == 1) ? -cardValue : cardValue;
    return score; 
}

int Easy21::step(array<int, 2>* ptS, int* ptA) {
    int sp = (*ptS)[0];
    int sd = (*ptS)[1];
    int a = *ptA;
    int r;
    if (a == 1) { // player's action is "hit"
        sp += draw(); // player draws card
        if (sp > 21 || sp < 1) { // player loses
            r = -1;
            sd = 99; // set state to terminal values
            sp = 99;
        } else { // player is still in the game
            r = 0;
        }
    } else { // player's action is "stick" and dealer begins to play
        while (sd < 17 && sd > 0) {sd += draw();} // dealer draws until loss or score above 16
        if (sd < 1 || sd > 21) {r = 1;} // dealer loses automatically, so reward is 1
        else {
            if (sd > sp) {
                r = -1;
            } else if (sd == sp) {
                r = 0;
            } else if (sd < sp) {
                r = 1;
            }
        }
        sp = 99;
        sd = 99;
    }
    
    (*ptS)[0] = sp;
    (*ptS)[1] = sd;
    return r;
}

int Easy21::policy(array<int, 2>* ptS) {
    int a;
    float eps;
    double N_0 = 200.0;
    int ixP = (*ptS)[0] - 1;
    int ixD = (*ptS)[1] - 1;
    double N_S = N[ixP][ixD][0] + N[ixP][ixD][1];
    eps = N_0 / (N_0 + N_S);
    if ((double) rand() / RAND_MAX > eps) { // Is random number between 0-1 larger than eps?
        a = (Q[ixP][ixD][1] > Q[ixP][ixD][0] ) ? 1 : 0; // select action with highest value
    } else { // Is random number between 0-1 smaller than eps?
        a = rand() % 2; // select action randomly
    }
    return a;
}

int Easy21::policy_LA(array<int, 2>* ptS) {
    float eps = 0.05;
    int action;
    int hit = 1;
    int stick = 0;
    int *pHit = &hit;
    int *pStick = &stick;
    double qHit, qStick;

    if ((double) rand() / RAND_MAX > eps) { // Is random number between 0-1 larger than eps? Act greedily.
        assign_features(ptS,pHit);
        qHit = approx_value();
        assign_features(ptS,pStick);
        qStick = approx_value();
        action = (qHit >= qStick) ? hit : stick; // select action with highest value
    } else { // Is random number between 0-1 smaller than eps?
        action = rand() % 2; // select action randomly
    }

    return action;
}

void Easy21::MC_episode() {
    vector<array<int, 2>> states;
    vector<int> actions, rewards;
    tuple<array<int, 2>, int> stepResult;

    array<int, 2> s = {rand() % 10 + 1, rand() % 10 + 1}; // State s0: draw cards for player and dealer
    array<int, 2>* pS = &s; // declare pointer to state
    int a = policy(pS); // find action from policy
    int* pA = &a; // declare pointer to action
    int r;
    states.push_back(s);
    actions.push_back(a);

    while (s[0] != 99) {
        r = step(pS, pA);
        states.push_back(s);
        rewards.push_back(r);
        if (s[0] != 99) {
            a = policy(pS);
            actions.push_back(a);
        } else { break; }
    }

    // Update the value function
    double G = (double)r; // return for each t is the final return, because gamma = 1
    int ixP = 0;
    int ixD = 0;
    int ixA = 0;
    int *ptixP = &ixP;
    int *ptixD = &ixD;
    int *ptixA = &ixA;
    double alpha;
    for (int i = 0; i < states.size()-1; i++) { // loop over all states, except the terminal state
        *ptixP = states[i][0] - 1;
        *ptixD = states[i][1] - 1;
        *ptixA = actions[i];
        N[ixP][ixD][ixA] += 1; // increment the number of times visited
        alpha = 1.0 / (double)N[ixP][ixD][ixA]; // compute alpha
        Q[ixP][ixD][ixA] += alpha*(G - Q[ixP][ixD][ixA]); // update action value function
    }
}

void Easy21::SARSA_episode(float labda) {
    double E[21][10][2] = {0}; // To store the eligibility trace
    array<int, 2> s = {rand() % 10 + 1, rand() % 10 + 1}; // State s0: draw cards for player and dealer
    array<int, 2>* pS = &s; // declare pointer to state
    int a = policy(pS); // find action from policy
    int* pA = &a; // declare pointer to action
    int r, a_new;
    double alpha, delta;
    double *qPtr, *ePtr;
    vector<double*> qPtrs, ePtrs;
    int ixPO, ixDO, ixAO, ixPN, ixDN, ixAN;

    while (s[0] != 99) {
        ixPO = s[0] - 1;
        ixDO = s[1] - 1;
        ixAO = a;
        E[ixPO][ixDO][ixAO] += 1.0;
        ePtr = &E[ixPO][ixDO][ixAO];
        ePtrs.push_back(ePtr);
        N[ixPO][ixDO][ixAO] += 1;
        alpha = 1.0 / (double)N[ixPO][ixDO][ixAO];
        r = step(pS, pA);
        qPtr = &Q[ixPO][ixDO][ixAO]; // Assign addres of current Q-value to pointer
        qPtrs.push_back(qPtr); // put pointer in pointer-vector for comp. eligb. trace

        if (s[0] != 99) {
            a_new = policy(pS);
            ixPN = s[0] - 1;
            ixDN = s[1] - 1;
            ixAN = a_new;
            delta = r + Q[ixPN][ixDN][ixAN] - *qPtr;
            for (int i = 0; i<qPtrs.size(); i++) {
                *qPtrs[i] += alpha * delta * (*ePtrs[i]);
                *ePtrs[i] *= labda;
            }
            *pA = a_new;
        } else {
            delta = r - *qPtr;
            for (int i = 0; i<qPtrs.size(); i++) {
                *qPtrs[i] += alpha * delta * (*ePtrs[i]);
                *ePtrs[i] *= labda;
            }
        }
    }
}

void Easy21::assign_features(array<int, 2>* ptS, int* ptA) {
    int sp = (*ptS)[0];
    int sd = (*ptS)[1];
    int sa = *ptA;
    float parr[6] = {0.0};
    float darr[3] = {0.0};
    float pf[36] = {0.0};
    float df[36] = {0.0};
    float af[36] = {0.0};

    if (sp == 1 || sp == 2 || sp == 3) {parr[0] = 1.0;} 
    else if (sp == 4 || sp == 5 || sp == 6) {parr[0] = 1.0; parr[1] = 1.0;} 
    else if (sp == 7 || sp == 8 || sp == 9) {parr[1] = 1.0; parr[2] = 1.0;} 
    else if (sp == 10 || sp == 11 || sp == 12) {parr[2] = 1.0; parr[3] = 1.0;} 
    else if (sp == 13 || sp == 14 || sp == 15) {parr[3] = 1.0; parr[4] = 1.0;} 
    else if (sp == 16 || sp == 17 || sp == 18) {parr[4] = 1.0; parr[5] = 1.0;} 
    else {parr[5] = 1.0;}

    if (sd == 1 || sd == 2 || sd == 3) {darr[0] = 1.0;}
    else if (sd == 4) {darr[0] = 1.0; darr[1] = 1.0;}
    else if (sd == 5 || sd == 6) {darr[1] = 1.0;}
    else if (sd == 7) {darr[1] = 1.0; darr[2] = 1.0;}
    else {darr[2] = 1.0;}

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            pf[i*6 + j] = parr[j];
        }
    }
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 6; j++) {
                df[k*18 + i*6 + j] = darr[i];
            }
        }
    }
    for (int i = 0; i < 18; i++) {
        af[18*sa + i] = 1.0;
    }
    for (int i = 0; i < 36; i++) {
        features[i] = pf[i]*df[i]*af[i];
    }

    return;
}

double Easy21::approx_value() {
    double q = 0.0;
    int nW = sizeof( weights ) / sizeof( weights[0] );
    for (int i = 0; i < nW; i++) {
        q += weights[i] * (double)features[i];
    }
    return q;
}

void Easy21::SARSA_LA_episode(float labda) {
    array<int, 2> s = {rand() % 10 + 1, rand() % 10 + 1}; // State s0: draw cards for player and dealer
    array<int, 2>* pS = &s; // declare pointer to state
    int a = policy_LA(pS); // find action from policy
    int* pA = &a; // declare pointer to action
    int r, a_new;
    double delta, qOld, qNew;
    double alpha = 0.01;
    double E[36] = {0.0};
    
    assign_features(pS, pA);
    qOld = approx_value();

    while (s[0] != 99) {
        r = step(pS, pA); // Also updates the state

        if (s[0] != 99) {
            for (int i = 0; i < 36; i++) {
                E[i] = labda*E[i] + (double)features[i];
            }
            a_new = policy_LA(pS); // Get new action
            *pA = a_new;
            assign_features(pS, pA); // Assign values to features based on new state and action
            qNew = approx_value();
            delta = (double)r + qNew - qOld;
            for (int i = 0; i < 36; i++) {
                weights[i] += alpha*delta*E[i];
            }
            qOld = qNew;
        } else {
            delta = (double)r - qOld;
            for (int i = 0; i < 36; i++) {
                E[i] = labda*E[i] + (double)features[i];
                weights[i] += alpha*delta*E[i];
            }
        }
    }
}

void Easy21::compute_Q_LA() {
    array<int, 2> s; 
    array<int, 2>* pS = &s; 
    int a;
    int* pA = &a;
    for (int i = 1; i < 22; i++) {
        for (int j = 1; j < 11; j++) {
            for (int k = 0; k < 2; k++) {
                s[0] = i;
                s[1] = j;
                a = k;
                assign_features(pS, pA);
                Q[i-1][j-1][k] = approx_value();
            }
        }
    }
}

void Easy21::calc_Vmax(bool save) {
    // Compute Vmax
    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 10; j++) {
            Vmax[i][j] = max(Q[i][j][0], Q[i][j][1]);
        }
    }
    
    if (save == true) { // save data to Vmax.dat for plotting
        ofstream file("Vmax.dat");
        file << "#pScore dScore V" << endl;
        for (int j = 1; j < 11; j++){
            for (int i = 1; i < 22; i++) {
                file << j << " " << i << " " << Vmax[i-1][j-1] << endl;
            }
            file << endl;
        }
        file.close();
    }
}

double Easy21::MSE(Easy21 extGame) {
    double MSE = 0;
    double q;
    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 2; k++) {
                q = (Q[i][j][k]) - (extGame.Q[i][j][k]);
                MSE += (q*q);
            }
        }
    }
    MSE /= (21.0 * 10.0 * 2.0);
    return MSE;
}