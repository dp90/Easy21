# Easy21

The Easy21 game is part of the course Reinforcement Learning as taught by David Silver at UC London in 2015. The course contents, including the assignemt to which solutions are provided in this repository, can be found 
[here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).  
I wrote the code for this assignment in C++ (v17) and compiled it on Linux Mint 19.3 with the g++ compiler. The results are saved in various ".dat" files and accessed by gnuplot to create the corresponding figures.  
The files in this repository are:
- Easy21.h: header file of the Easy21 class, declaring its variables and functions.
- Easy21.cpp: defines the Easy21 game, the Monte Carlo method of updating its action value function Q(s,a), the SARSA(lambda) method of updating Q(s,a) with table look-up, and the SARSA(lambda) method of updating the weights of a linear action value function approximation method. 
- main.cpp: uses the Easy21 class to compute results with the different methods. 

## Monte-Carlo
The optimal value function as computed with the Monte-Carlo method (5,000,000 episodes in 6.24 seconds) is shown below. 

![alt text](https://github.com/dp90/Easy21/blob/master/Images/Vmaxplot.jpeg "Monte-Carlo optimal value function")

## SARSA(\\lambda)
The SARSA(\\lambda) method was implemented according to
