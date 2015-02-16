// #include "cartpole_dynamics.h"
#include "cartpole_dynamics_by_hand.h"

#include <iostream>
#include <stdlib.h>

double uniform(double low, double high) {
	return (high - low)*(rand() / double(RAND_MAX)) + low;
}

int main(int argc, char* argv[]) {

	srand(0);

	double new_formulation_weights[6] = {1.0, 1.0/8, -1.0/8, 1.0/10, -1.0, -1.0/3};

	// for (int i = 0; i < 1000; i++) {

		Vector4d x;
		// x << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);
		x << atof(argv[1]), atof(argv[2]), atof(argv[3]), atof(argv[4]);
		// std::cout << "X:\n" << x << "\n";

		VectorXd u(1+4);
		// u << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);
		// u << uniform(-1,1), 0.0, 0.0, 0.0, 0.0;
		u << atof(argv[5]), 0, 0, 0, 0;
		// std::cout << "U:\n" << u << "\n";

		double delta = atof(argv[6]);
		// std::cout << "Delta:\n" << delta << "\n";

		// VectorXd a1 = cartpole::continuous_dynamics(x, u, true_weights);
		VectorXd a = cartpole::rk45_DP(cartpole::continuous_dynamics, x, u, delta, new_formulation_weights);

		std::cout << "Integrated state:\n" << a << "\n";

	// }

}