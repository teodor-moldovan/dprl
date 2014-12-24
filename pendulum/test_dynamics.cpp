#include "pendulum_dynamics.h"
#include "pendulum_dynamics_by_hand.h"

#include <iostream>
#include <stdlib.h>

using namespace pendulum;

double uniform(double low, double high) {
	return (high - low)*rand() + low;
}

int main() {

	srand(0);

	double new_formulation_weights[3] = {1, 0.05, 9.82};

	for (int i = 0; i < 1000; i++) {

		Vector2d x;
		x << uniform(-2,2), uniform(-2,2);

		VectorXd u(1+2);
		u << uniform(-1,1), uniform(-1,1), uniform(-1,1);

		VectorXd a1 = continuous_dynamics(x, u, true_weights);
		VectorXd a2 = pendulum_by_hand::continuous_dynamics(x, u, new_formulation_weights);

		if ((a1 - a2).norm() > 1e-3) {
			std::cout << "Doesn't match...\n";
			std::cout << "a1:\n" << a1 << "\na2:\n" << a2 << "\n";
		}

	}

}