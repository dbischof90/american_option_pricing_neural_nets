# Pricing American options with neural networks
![Alt text](scatter_cv.png?raw=true)

This project implements a regression pricer with a neural network as a functional regressor space and follows "Kohler, M., Krzyżak, A., & Todorovic, N. (2010). Pricing of High‐Dimensional American Options by Neural Networks. Mathematical Finance: An International Journal of Mathematics, Statistics and Financial Economics, 20(3), 383-410.". 
The advantage of this approach over other regression-based approaches such as Longstaff-Schwartz is a better approximation of complex payoff profiles and a better scale to higher dimensions in basket option profile. 

The network is calibrated via a recursive gradient descent procedure to a given data generation process. Exemplary experiments can be found in `main.py`, which contain code to create boxplots that analyse the dispersion of the price estimate and a visual inspection to the network layers.
