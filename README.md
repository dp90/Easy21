# Easy21

The Easy21 game is part of the course Reinforcement Learning as taught by David Silver at UC London in 2015. The course contents, including the assignemt to which solutions are provided in this repository, can be found 
[here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html). The applied algorithms are based on the contents of the book [Reinforcement Learning - An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) by R. S. Sutton & A. G. Barto.  
I wrote the code for this assignment in C++ (v17) and compiled it on Linux Mint 19.3 with the g++ compiler. The results are saved in various ".dat" files and accessed by gnuplot to create the corresponding figures.  
The files in this repository are:
- Easy21.h: header file of the Easy21 class, declaring its variables and functions.
- Easy21.cpp: defines the Easy21 game, the Monte Carlo method of updating its action value function Q(s,a), the SARSA(lambda) method of updating Q(s,a) with table look-up, and the SARSA(lambda) method of updating the weights of a linear action value function approximation method. 
- main.cpp: uses the Easy21 class to compute results with the different methods. 

## Monte-Carlo
The optimal value function as computed with the Monte-Carlo method (5,000,000 episodes in 6.24 seconds) is shown below (Sutton & Barto, p. 99). 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/Vmaxplot.jpeg "Monte-Carlo optimal value function")

## SARSA(\\lambda) - Table lookup
The SARSA(\\lambda) method with table lookup (Sutton & Barto, p. 303 and [here](http://incompleteideas.net/book/first/ebook/node77.html)) was compared to the Monte-Carlo method by computing the mean squared error (MSE) between the action value function Q(s,a) computed in the previous section, and Q(s,a) computed with SARSA(lambda) over 1000 episodes at values of lambda = 0.0, 0.1, ..., 1.0. The results can be found below 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/SARSA_table_lambda_vs_MSE.jpeg "SARSA table lookup: lambda vs MSE")

For the values of lambda = 0.0 and lambda = 1.0, the MSE is also computed and plotted at the end of each episode, as shown below. 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/SARSA_table_episode_vs_MSE.jpeg "SARSA table lookup: episode vs MSE")

## SARSA(\\lambda) - Table lookup
The SARSA(\\lambda) method with a linear action value function approximation (Sutton & Barto, p. 307) was compared to the Monte-Carlo method by computing the mean squared error (MSE) between the action value function Q(s,a) computed with the Monte-Carlo method, and Q(s,a) computed with SARSA(lambda) over 1000 episodes at values of lambda = 0.0, 0.1, ..., 1.0. The results can be found below 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/SARSA_LFA_lambda_vs_MSE.jpeg "SARSA LFA: lambda vs MSE")

For the values of lambda = 0.0 and lambda = 1.0, the MSE is also computed and plotted at the end of each episode, as shown below. 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/SARSA_LFA_episode_vs_MSE.jpeg "SARSA LFA: episode vs MSE")
