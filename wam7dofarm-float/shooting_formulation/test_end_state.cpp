#include "end_state.h"

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>

double L_1xx= 0.2948635027286643;
double L_1xy= 0.007950224642941463;
double L_1xz= 9.310396523921739e-05;
double L_1yy= 0.11350016774623556;
double L_1yz= 0.00018710326437239235;
double L_1zz= 0.25065343014390495;
double l_1x= -0.04774629602006741;
double l_1y= 1.3124776494944914;
double l_1z= -0.0071593278549063;
double m_1= 10.76768767;
double L_2xx= 0.026068399567247713;
double L_2xy= 1.3466772047875293e-05;
double L_2xz= 0.0001170014538992409;
double L_2yy= 0.014722016956250276;
double L_2yz= -3.658929201603989e-05;
double L_2zz= 0.019348136639933563;
double l_2x= -0.0091829432778148;
double l_2y= 0.1203406033546184;
double l_2z= 0.0597559546040184;
double m_2= 3.87493756;
double L_3xx= 0.13671601022982002;
double L_3xy= 0.016804341893703847;
double L_3xz= -5.098354246793442e-06;
double L_3yy= 0.005883535414949282;
double L_3yz= 5.29476123100595e-06;
double L_3zz= 0.13951370169796007;
double l_3x= -0.0689527275069978;
double l_3y= 0.37398727014185695;
double l_3z= 5.963749185690001e-05;
double m_3= 1.80228141;
double L_4xx= 0.05719268907382418;
double L_4xy= -1.4664960862005844e-05;
double L_4xz= -8.193104072137586e-05;
double L_4yy= 0.057164710331919694;
double L_4yz= 9.416994916116324e-05;
double L_4zz= 0.003004403916759314;
double l_4x= 0.0119651256995648;
double l_4y= -0.0005506465517368;
double l_4z= 0.3185421897008248;
double m_4= 2.40016804;
double L_5xx= 5.5875116438106664e-05;
double L_5xy= -2.5644166207300225e-07;
double L_5xz= 1.8822125803638124e-09;
double L_5yy= 7.817171765487431e-05;
double L_5yz= 8.326150732760728e-07;
double L_5zz= 6.593536865538978e-05;
double l_5x= 1.10406465499e-05;
double l_5y= 0.0006326831305123001;
double l_5z= 0.0005393766104656001;
double m_5= 0.12376019;
double L_6xx= 0.0009310667837560834;
double L_6xy= -1.482919758097513e-06;
double L_6xz= 2.0050697830214196e-06;
double L_6yy= 0.0004983343572518748;
double L_6yz= 0.00022161841983553936;
double L_6zz= 0.0005748349955269667;
double l_6x= -5.1251927736799995e-05;
double l_6y= -0.007118901958061599;
double l_6z= 0.0103169938266304;
double m_6= 0.41797364;
double L_7xx= 3.8450469630123506e-05;
double L_7xy= 1.9089296612290408e-07;
double L_7xz= -1.7711087782618944e-08;
double L_7yy= 3.8779079316551796e-05;
double L_7yz= 3.623287873060732e-08;
double L_7zz= 7.408226329976235e-05;
double l_7x= -5.4739540422e-06;
double l_7y= 1.11984715689e-05;
double l_7z= -0.00022211045626559998;
double m_7= 0.06864753;

double true_weights[70] = {L_1xx,
 L_1xy,
 L_1xz,
 L_1yy,
 L_1yz,
 L_1zz,
 l_1x,
 l_1y,
 l_1z,
 m_1,
 L_2xx,
 L_2xy,
 L_2xz,
 L_2yy,
 L_2yz,
 L_2zz,
 l_2x,
 l_2y,
 l_2z,
 m_2,
 L_3xx,
 L_3xy,
 L_3xz,
 L_3yy,
 L_3yz,
 L_3zz,
 l_3x,
 l_3y,
 l_3z,
 m_3,
 L_4xx,
 L_4xy,
 L_4xz,
 L_4yy,
 L_4yz,
 L_4zz,
 l_4x,
 l_4y,
 l_4z,
 m_4,
 L_5xx,
 L_5xy,
 L_5xz,
 L_5yy,
 L_5yz,
 L_5zz,
 l_5x,
 l_5y,
 l_5z,
 m_5,
 L_6xx,
 L_6xy,
 L_6xz,
 L_6yy,
 L_6yz,
 L_6zz,
 l_6x,
 l_6y,
 l_6z,
 m_6,
 L_7xx,
 L_7xy,
 L_7xz,
 L_7yy,
 L_7yz,
 L_7zz,
 l_7x,
 l_7y,
 l_7z,
 m_7};

double uniform(double low, double high) {
	return (high - low)*(rand() / double(RAND_MAX)) + low;
}

int main() {

	srand(0);

	for (int i = 0; i < 1; i++) {

		VectorX x_start; x_start.setZero();
		// for(int i = 0; i < NX; ++i) {
		// 	x_start[i] = uniform(-.1,.1);
		// }

		int T = 10;
		StdVectorU U(T);
		for(int t = 0; t < T; ++t) {
			for(int i = 0; i < NU+NV; ++i) {
				U[t][i] = uniform(-1,1);
			}
		}

		double delta = uniform(0, .02);

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

		std::clock_t start;
		start = std::clock();

		MatrixXd jac = end_state_numerical_jacobian(x_start, U, delta, true_weights);

		double solvetime = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		std::cout << "Jacobian:\n" << jac.transpose() << std::endl;

		printf("Solve time: %5.3f s\n", solvetime);

	}

	std::cout << "Done.\n";

}