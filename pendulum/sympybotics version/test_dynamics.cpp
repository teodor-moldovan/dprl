#include "pendulum_dynamics.h"
#include "pendulum_dynamics_by_hand.h"

#include <iostream>
#include <stdlib.h>

double uniform(double low, double high) {
	return (high - low)*(rand() / double(RAND_MAX)) + low;
}

int main() {

	srand(0);

	double new_formulation_weights[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0/3, 0.0, -1.0/2, 0.0, 1.0};

	for (int i = 0; i < 10; i++) {

		Vector2d x;
		x << uniform(-2,2), uniform(-2,2);

		Vector3d u;
		u << uniform(-1,1), 0,0;//uniform(-1,1), uniform(-1,1);

		Vector2d a1 = pendulum::continuous_dynamics(x, u, true_weights);
		Vector2d a2 = pendulum_by_hand::continuous_dynamics(x, u, new_formulation_weights);

		std::cout << "x:\n" << x << "\nu:\n" << u << "\ndx:\n" << a2 << "\n";

		if ((a1 - a2).norm() > 1e-3) {
			std::cout << "Doesn't match...\n";
			std::cout << "a1:\n" << a1 << "\na2:\n" << a2 << "\n";
		}

	}

}