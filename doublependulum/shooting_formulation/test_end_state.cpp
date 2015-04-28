#include "end_state.h"

#include <iostream>
#include <stdlib.h>

double uniform(double low, double high) {
	return (high - low)*(rand() / double(RAND_MAX)) + low;
}

int main() {

	srand(0);

	double g = 9.82;
	double true_weights[10] = {1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0};

	for (int i = 0; i < 1; i++) {

		VectorX x_start;
		x_start << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);

		int T = 10;
		StdVectorU U(T);
		for(int t = 0; t < T; ++t) {
			U[t] << uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1);
		}

		double delta = uniform(0, .2);

		VectorXd end = end_state(U, x_start, delta, true_weights);

		std::cout << "Controls and Virtual Controls:\n";
		std::cout << "[";
		for(int t = 0; t < T; ++t){
			std::cout << "[";
			for(int i = 0; i < NU+NV; ++i) {
				std::cout << U[t][i] << ", ";
			}
			std::cout << "],\n";
		}
		std::cout << "]\n";

		std::cout << "Propagated state:\n" << end << "\n";

		MatrixXd jac = end_state_numerical_jacobian(x_start, U, delta, true_weights);
		std::cout << "Jacobian:\n" << jac.transpose() << std::endl;

	}

	std::cout << "Done.\n";

}