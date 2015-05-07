//#include "doublependulum_dynamics.h"
#include "doublependulum_dynamics_by_hand.h"

#include <iostream>
#include <stdlib.h>

double uniform(double low, double high) {
	return (high - low)*(rand() / double(RAND_MAX)) + low;
}

int main() {

	srand(0);

	double g = 9.82;
	double new_formulation_weights[10] = {1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0};
	//double* weights = new_formulation_weights;
	double weights[10] = { 6.77162349e-04,   1.09713969e-02,   1.17480109e-01, 1.33821462e+00,  -1.51396982e-02,   8.05025355e-02, 4.90277003e-02,  -2.37042973e-01,  -7.72811944e-01, -9.07072501e-03 };
	double delta = 0.01;

	for (int i = 0; i < 2; i++) {

		Vector4d x;
		//x << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);
		x << 1.03797335,  1.02902607,  1.36454759,  3.37006453;

		VectorXd u(2+4);
		//u << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);
		//u << uniform(-1,1), uniform(-1,1), 0, .1, 0, 0;
		u << 0.0        ,    0.0        , -1.79733488, -1.90260658, 75.22704557,    5.04912089;

		// VectorXd a1 = doublependulum::continuous_dynamics(x, u, true_weights);
		// VectorXd a2 = doublependulum_by_hand::continuous_dynamics(x, u, new_formulation_weights);

		// if ((a1 - a2).norm() > 1e-3) {
		// 	std::cout << "Doesn't match... Norm is: " << (a1-a2).norm() << "\n";
		// 	std::cout << "a1:\n" << a1 << "\na2:\n" << a2 << "\n";
		// }

		std::cout << "X:\n" << x << "\n";
		std::cout << "U:\n" << u << "\n";

		VectorXd x_next = doublependulum::rk45_DP(doublependulum::continuous_dynamics, x, u, delta, weights);
		std::cout << "Simulated next state:\n" << x_next << "\n";

	}

	std::cout << "Done.\n";

}