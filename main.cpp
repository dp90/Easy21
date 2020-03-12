#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tuple>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include "Easy21.h"

using namespace std::chrono;
using namespace std;

int main () {
    srand(time(NULL)); // Set seed to rand() equal to time for actually random numbers.
    
    int learningType, nEpisodes;
    cout << "Enter the number of the desired learning type: " << endl;
    cout << "1 for Monte Carlo" << endl;
    cout << "2 for SARSA(lambda) with table look-up" << endl;
    cout << "3 for SARSA(lambda) with linear action value function approximation" << endl;
    cout << "4 for MSE of SARSA(lambda) with table look-up Vs. Monte Carlo" << endl;
    cout << "5 for MSE of SARSA(lambda) with lin. action value func. approx. Vs. Monte Carlo" << endl;
    cin >> learningType;
    while (learningType != 1 && learningType != 2 && learningType != 3 && learningType != 4 && learningType != 5) {
        cout << "Please select 1, 2, 3, 4 or 5" << endl;
        cin >> learningType;
    }

    cout << "Enter the number of episodes the model should be trained: " << endl;
    cin >> nEpisodes;
    
    bool saveVmax = true;
    time_point<system_clock> start, end; 
    start = system_clock::now(); // Start timing

    if (learningType == 1) { // Monte Carlo
        Easy21 game;
        for (int i = 0; i < nEpisodes; i++) {
            game.MC_episode(); // each episode updates the Q lookup table
        }
        game.calc_Vmax(saveVmax); // calculates the Vmax(s) from Q(s,a)

    } else if (learningType == 2) { // SARSA(lambda) with table lookup - eligibility trace
        Easy21 game;
        float labda;
        cout << "Enter value for lambda" << endl;
        cin >> labda;
        for (int i = 0; i < nEpisodes; i++) {
            game.SARSA_episode(labda); // each step in each episode updates Q
        }
        game.calc_Vmax(saveVmax); // calculates the Vmax(s) from Q(s,a)

    } else if (learningType == 3) { // SARSA(lambda) with linear function approximation
        Easy21 game;
        float labda;
        cout << "Enter value for lambda" << endl;
        cin >> labda;
        for (int i = 0; i < nEpisodes; i++) {
            game.SARSA_LA_episode(labda); // each step in each episode updates weights
        }
        game.compute_Q_LA(); // use weights to compute Q(s,a)
        game.calc_Vmax(saveVmax); // calculates the Vmax(s) from Q(s,a)

    } else if (learningType == 4) { // Compares the MSEs of Q(s,a) calculated with SARSA(lambda) with table 
        Easy21 gameMC;              // lookup for various lambdas with 'real' Q(s,a) computed with MC.
        for (int i = 0; i < 2000000; i++) { // Train Monte Carlo game to find 'real' Q(s,a)
            gameMC.MC_episode();
        }
        float labdas[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        double MSEs[11];
        for (int j = 0; j < 11; j++) { // Compute Q(s,a) with SARSA(lambda) for various lambdas
            Easy21 game;
            float labda = labdas[j];
            for (int i = 0; i < nEpisodes; i++) {
                game.SARSA_episode(labda);
            }
            MSEs[j] = game.MSE(gameMC); // Compute MSE between SARSA(lambda) and 'real' Q(s,a)
        }
        // Save lambda vs MSE values to file
        ofstream file("SarsaLambdaTableMSEs.dat");
        file << "#lambda MSE" << endl;
        for (int j = 0; j < 11; j++){
            file << labdas[j] << " " << MSEs[j] << endl;
        }
        file.close();

        // Create data for plots of episode vs MSE for lambda = 0.0 and lambda = 1.0
        double MSE0[nEpisodes], MSE1[nEpisodes];
        Easy21 game0, game1;
        for (int i = 0; i < nEpisodes; i++) {
            game0.SARSA_episode(0.0);
            MSE0[i] = game0.MSE(gameMC);
            game1.SARSA_episode(1.0);
            MSE1[i] = game1.MSE(gameMC);
        }
        // Save MSE values at the end of each
        ofstream file2("SarsaLambdaTableEpVsMse.dat");
        file2 << "#episode lambda0 lambda1" << endl;
        for (int j = 0; j < nEpisodes; j++){
            file2 << j+1 << " " << MSE0[j] << " " << MSE1[j] << endl;
        }
        file2.close();

    } else if (learningType == 5) { // Compares the MSEs of Q(s,a) calculated with SARSA(lambda) with lin. func.
        Easy21 gameMC;              // approximation for various lambdas with 'real' Q(s,a) computed with MC.
        for (int i = 0; i < 2000000; i++) { // Train Monte Carlo game to find 'real' Q(s,a)
            gameMC.MC_episode();
        }
        
        float labdas[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        double MSEs[11];
        for (int j = 0; j < 11; j++) { // Compute Q(s,a) with SARSA(lambda) for various lambdas
            Easy21 game;
            float labda = labdas[j];
            for (int i = 0; i < nEpisodes; i++) { // Update weights for n Episodes
                game.SARSA_LA_episode(labda);
            }
            game.compute_Q_LA(); // Compute the resulting Q(s,a)
            MSEs[j] = game.MSE(gameMC); // Compute the MSE compared to 'real' Q(s,a)
        }

        // Save lambda vs MSE values
        ofstream file("SarsaLambdaLfaMSEs.dat");
        file << "#lambda MSE" << endl;
        for (int j = 0; j < 11; j++){
            file << labdas[j] << " " << MSEs[j] << endl;
        }
        file.close();
        
        // Create data for plots of episode vs MSE for lambda = 0.0 and lambda = 1.0
        double MSE0[nEpisodes], MSE1[nEpisodes];
        Easy21 game0, game1;
        for (int i = 0; i < nEpisodes; i++) {
            game0.SARSA_LA_episode(0.0);
            game0.compute_Q_LA();
            MSE0[i] = game0.MSE(gameMC);
            game1.SARSA_LA_episode(1.0);
            game1.compute_Q_LA();
            MSE1[i] = game1.MSE(gameMC);
        }
        // Save MSE values
        ofstream file2("SarsaLambdaLfaEpVsMse.dat");
        file2 << "#episode lambda0 lambda1" << endl;
        for (int j = 0; j < nEpisodes; j++){
            file2 << j+1 << " " << MSE0[j] << " " << MSE1[j] << endl;
        }
        file2.close(); 
    }
    
    end = system_clock::now();
    duration<double> elapsed_seconds = end - start; 
    cout << "Elapsed time: " << elapsed_seconds.count() << "s" << endl; 

    return 0;
}