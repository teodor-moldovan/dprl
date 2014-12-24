/*
FORCES - Fast interior point code generation for multistage problems.
Copyright (C) 2011-14 Alexander Domahidi [domahidi@control.ee.ethz.ch],
Automatic Control Laboratory, ETH Zurich.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "../include/pendulum_QP_solver.h"

/* for square root */
#include <math.h> 

/* SAFE DIVISION ------------------------------------------------------- */
#define MAX(X,Y)  ((X) < (Y) ? (Y) : (X))
#define MIN(X,Y)  ((X) < (Y) ? (X) : (Y))
/*#define SAFEDIV_POS(X,Y)  ( (Y) < EPS ? ((X)/EPS) : (X)/(Y) ) 
#define EPS (1.0000E-013) */
#define BIGM (1E30)
#define BIGMM (1E60)

/* includes for parallel computation if necessary */


/* SYSTEM INCLUDES FOR PRINTING ---------------------------------------- */




/* LINEAR ALGEBRA LIBRARY ---------------------------------------------- */
/*
 * Initializes a vector of length 143 with a value.
 */
void pendulum_QP_solver_LA_INITIALIZEVECTOR_143(pendulum_QP_solver_FLOAT* vec, pendulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<143; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 44 with a value.
 */
void pendulum_QP_solver_LA_INITIALIZEVECTOR_44(pendulum_QP_solver_FLOAT* vec, pendulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<44; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 230 with a value.
 */
void pendulum_QP_solver_LA_INITIALIZEVECTOR_230(pendulum_QP_solver_FLOAT* vec, pendulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<230; i++ )
	{
		vec[i] = value;
	}
}


/* 
 * Calculates a dot product and adds it to a variable: z += x'*y; 
 * This function is for vectors of length 230.
 */
void pendulum_QP_solver_LA_DOTACC_230(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<230; i++ ){
		*z += x[i]*y[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [10 x 10]
 *             f  - column vector of size 10
 *             z  - column vector of size 10
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 10
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_FLOAT* H, pendulum_QP_solver_FLOAT* f, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* grad, pendulum_QP_solver_FLOAT* value)
{
	int i;
	pendulum_QP_solver_FLOAT hz;	
	for( i=0; i<10; i++){
		hz = H[i]*z[i];
		grad[i] = hz + f[i];
		*value += 0.5*hz*z[i] + f[i]*z[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [3 x 3]
 *             f  - column vector of size 3
 *             z  - column vector of size 3
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 3
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void pendulum_QP_solver_LA_DIAG_QUADFCN_3(pendulum_QP_solver_FLOAT* H, pendulum_QP_solver_FLOAT* f, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* grad, pendulum_QP_solver_FLOAT* value)
{
	int i;
	pendulum_QP_solver_FLOAT hz;	
	for( i=0; i<3; i++){
		hz = H[i]*z[i];
		grad[i] = hz + f[i];
		*value += 0.5*hz*z[i] + f[i]*z[i];
	}
}


/* 
 * Computes r = A*x + B*u - b
 * and      y = max([norm(r,inf), y])
 * and      z -= l'*r
 * where A is stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_MVMSUB3_5_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *z, pendulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;
	pendulum_QP_solver_FLOAT AxBu[5];
	pendulum_QP_solver_FLOAT norm = *y;
	pendulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<5; i++ ){
		AxBu[i] = A[k++]*x[0] + B[m++]*u[0];
	}	
	for( j=1; j<10; j++ ){		
		for( i=0; i<5; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<10; n++ ){
		for( i=0; i<5; i++ ){
			AxBu[i] += B[m++]*u[n];
		}		
	}

	for( i=0; i<5; i++ ){
		r[i] = AxBu[i] - b[i];
		lr += l[i]*r[i];
		if( r[i] > norm ){
			norm = r[i];
		}
		if( -r[i] > norm ){
			norm = -r[i];
		}
	}
	*y = norm;
	*z -= lr;
}


/* 
 * Computes r = A*x + B*u - b
 * and      y = max([norm(r,inf), y])
 * and      z -= l'*r
 * where A is stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *z, pendulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	pendulum_QP_solver_FLOAT AxBu[3];
	pendulum_QP_solver_FLOAT norm = *y;
	pendulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<3; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<3; i++ ){
		r[i] = AxBu[i] - b[i];
		lr += l[i]*r[i];
		if( r[i] > norm ){
			norm = r[i];
		}
		if( -r[i] > norm ){
			norm = -r[i];
		}
	}
	*y = norm;
	*z -= lr;
}


/* 
 * Computes r = A*x + B*u - b
 * and      y = max([norm(r,inf), y])
 * and      z -= l'*r
 * where A is stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *z, pendulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	pendulum_QP_solver_FLOAT AxBu[3];
	pendulum_QP_solver_FLOAT norm = *y;
	pendulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<3; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<3; i++ ){
		r[i] = AxBu[i] - b[i];
		lr += l[i]*r[i];
		if( r[i] > norm ){
			norm = r[i];
		}
		if( -r[i] > norm ){
			norm = -r[i];
		}
	}
	*y = norm;
	*z -= lr;
}


/*
 * Matrix vector multiplication y = M'*x where M is of size [5 x 10]
 * and stored in column major format. Note the transpose of M!
 */
void pendulum_QP_solver_LA_DENSE_MTVM_5_10(pendulum_QP_solver_FLOAT *M, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<10; i++ ){
		y[i] = 0;
		for( j=0; j<5; j++ ){
			y[i] += M[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [3 x 10]
 * and B is of size [5 x 10]
 * and stored in column major format. Note the transposes of A and B!
 */
void pendulum_QP_solver_LA_DENSE_MTVM2_3_10_5(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	int n;
	int m = 0;
	for( i=0; i<10; i++ ){
		z[i] = 0;
		for( j=0; j<3; j++ ){
			z[i] += A[k++]*x[j];
		}
		for( n=0; n<5; n++ ){
			z[i] += B[m++]*y[n];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [3 x 10] and stored in column major format.
 * and B is of size [3 x 10] and stored in diagzero format
 * Note the transposes of A and B!
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	for( i=0; i<3; i++ ){
		z[i] = 0;
		for( j=0; j<3; j++ ){
			z[i] += A[k++]*x[j];
		}
		z[i] += B[i]*y[i];
	}
	for( i=3 ;i<10; i++ ){
		z[i] = 0;
		for( j=0; j<3; j++ ){
			z[i] += A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = M'*x where M is of size [3 x 3]
 * and stored in diagzero format. Note the transpose of M!
 */
void pendulum_QP_solver_LA_DIAGZERO_MTVM_3_3(pendulum_QP_solver_FLOAT *M, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<3; i++ ){
		y[i] = M[i]*x[i];
	}
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 10. Output z is of course scalar.
 */
void pendulum_QP_solver_LA_VSUBADD3_10(pendulum_QP_solver_FLOAT* t, pendulum_QP_solver_FLOAT* u, int* uidx, pendulum_QP_solver_FLOAT* v, pendulum_QP_solver_FLOAT* w, pendulum_QP_solver_FLOAT* y, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* r)
{
	int i;
	pendulum_QP_solver_FLOAT norm = *r;
	pendulum_QP_solver_FLOAT vx = 0;
	pendulum_QP_solver_FLOAT x;
	for( i=0; i<10; i++){
		x = t[i] - u[uidx[i]];
		y[i] = x + w[i];
		vx += v[i]*x;
		if( y[i] > norm ){
			norm = y[i];
		}
		if( -y[i] > norm ){
			norm = -y[i];
		}
	}
	*z -= vx;
	*r = norm;
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 6. Output z is of course scalar.
 */
void pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_FLOAT* t, int* tidx, pendulum_QP_solver_FLOAT* u, pendulum_QP_solver_FLOAT* v, pendulum_QP_solver_FLOAT* w, pendulum_QP_solver_FLOAT* y, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* r)
{
	int i;
	pendulum_QP_solver_FLOAT norm = *r;
	pendulum_QP_solver_FLOAT vx = 0;
	pendulum_QP_solver_FLOAT x;
	for( i=0; i<6; i++){
		x = t[tidx[i]] - u[i];
		y[i] = x + w[i];
		vx += v[i]*x;
		if( y[i] > norm ){
			norm = y[i];
		}
		if( -y[i] > norm ){
			norm = -y[i];
		}
	}
	*z -= vx;
	*r = norm;
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 3. Output z is of course scalar.
 */
void pendulum_QP_solver_LA_VSUBADD3_3(pendulum_QP_solver_FLOAT* t, pendulum_QP_solver_FLOAT* u, int* uidx, pendulum_QP_solver_FLOAT* v, pendulum_QP_solver_FLOAT* w, pendulum_QP_solver_FLOAT* y, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* r)
{
	int i;
	pendulum_QP_solver_FLOAT norm = *r;
	pendulum_QP_solver_FLOAT vx = 0;
	pendulum_QP_solver_FLOAT x;
	for( i=0; i<3; i++){
		x = t[i] - u[uidx[i]];
		y[i] = x + w[i];
		vx += v[i]*x;
		if( y[i] > norm ){
			norm = y[i];
		}
		if( -y[i] > norm ){
			norm = -y[i];
		}
	}
	*z -= vx;
	*r = norm;
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 3. Output z is of course scalar.
 */
void pendulum_QP_solver_LA_VSUBADD2_3(pendulum_QP_solver_FLOAT* t, int* tidx, pendulum_QP_solver_FLOAT* u, pendulum_QP_solver_FLOAT* v, pendulum_QP_solver_FLOAT* w, pendulum_QP_solver_FLOAT* y, pendulum_QP_solver_FLOAT* z, pendulum_QP_solver_FLOAT* r)
{
	int i;
	pendulum_QP_solver_FLOAT norm = *r;
	pendulum_QP_solver_FLOAT vx = 0;
	pendulum_QP_solver_FLOAT x;
	for( i=0; i<3; i++){
		x = t[tidx[i]] - u[i];
		y[i] = x + w[i];
		vx += v[i]*x;
		if( y[i] > norm ){
			norm = y[i];
		}
		if( -y[i] > norm ){
			norm = -y[i];
		}
	}
	*z -= vx;
	*r = norm;
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 10
 * Returns also L/S, a value that is often used elsewhere.
 */
void pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_FLOAT *lu, pendulum_QP_solver_FLOAT *su, pendulum_QP_solver_FLOAT *ru, pendulum_QP_solver_FLOAT *ll, pendulum_QP_solver_FLOAT *sl, pendulum_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, pendulum_QP_solver_FLOAT *grad, pendulum_QP_solver_FLOAT *lubysu, pendulum_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<10; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<10; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<6; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 3
 * Returns also L/S, a value that is often used elsewhere.
 */
void pendulum_QP_solver_LA_INEQ_B_GRAD_3_3_3(pendulum_QP_solver_FLOAT *lu, pendulum_QP_solver_FLOAT *su, pendulum_QP_solver_FLOAT *ru, pendulum_QP_solver_FLOAT *ll, pendulum_QP_solver_FLOAT *sl, pendulum_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, pendulum_QP_solver_FLOAT *grad, pendulum_QP_solver_FLOAT *lubysu, pendulum_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<3; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<3; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<3; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Addition of three vectors  z = u + w + v
 * of length 143.
 */
void pendulum_QP_solver_LA_VVADD3_143(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *w, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<143; i++ ){
		z[i] = u[i] + v[i] + w[i];
	}
}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 10.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_FLOAT *H, pendulum_QP_solver_FLOAT *llbysl, int* lbIdx, pendulum_QP_solver_FLOAT *lubysu, int* ubIdx, pendulum_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* copy  H into PHI */
	for( i=0; i<10; i++ ){
		Phi[i] = H[i];
	}

	/* add llbysl onto Phi where necessary */
	for( i=0; i<10; i++ ){
		Phi[lbIdx[i]] += llbysl[i];
	}

	/* add lubysu onto Phi where necessary */
	for( i=0; i<6; i++){
		Phi[ubIdx[i]] +=  lubysu[i];
	}
	
	/* compute cholesky */
	for(i=0; i<10; i++)
	{
#if pendulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
		if( Phi[i] < 1.0000000000000000E-013 )
		{
            PRINTTEXT("WARNING: small pivot in Cholesky fact. (=%3.1e < eps=%3.1e), regularizing to %3.1e\n",Phi[i],1.0000000000000000E-013,4.0000000000000002E-004);
			Phi[i] = 2.0000000000000000E-002;
		}
		else
		{
			Phi[i] = sqrt(Phi[i]);
		}
#else
		Phi[i] = Phi[i] < 1.0000000000000000E-013 ? 2.0000000000000000E-002 : sqrt(Phi[i]);
#endif
	}

}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [5 x 10],
 * B is given and of size [5 x 10], L is a diagonal
 * matrix of size 5 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_10(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<10; j++){
		for( i=0; i<5; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 10.
 */
void pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<10; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [3 x 10],
 * B is given and of size [3 x 10], L is a diagonal
 * matrix of size 3 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<10; j++){
		for( i=0; i<3; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [5 x 10]
 *  size(B) = [3 x 10]
 * 
 * and all matrices are stored in column major format.
 *
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE.  
 * 
 */
void pendulum_QP_solver_LA_DENSE_MMTM_5_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *C)
{
    int i, j, k;
    pendulum_QP_solver_FLOAT temp;
    
    for( i=0; i<5; i++ ){        
        for( j=0; j<3; j++ ){
            temp = 0; 
            for( k=0; k<10; k++ ){
                temp += A[k*5+i]*B[k*3+j];
            }						
            C[j*5+i] = temp;
        }
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [3 x 10],
 * B is given and of size [3 x 10], L is a diagonal
 *  matrix of size 10 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<10; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [3 x 10]
 *  size(B) = [3 x 10] in diagzero format
 * 
 * A and C matrices are stored in column major format.
 * 
 * 
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *C)
{
    int i, j;
	
	for( i=0; i<3; i++ ){
		for( j=0; j<3; j++){
			C[j*3+i] = B[i*3+j]*A[i];
		}
	}

}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 3.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void pendulum_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_3_3_3(pendulum_QP_solver_FLOAT *H, pendulum_QP_solver_FLOAT *llbysl, int* lbIdx, pendulum_QP_solver_FLOAT *lubysu, int* ubIdx, pendulum_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* compute cholesky */
	for( i=0; i<3; i++ ){
		Phi[i] = H[i] + llbysl[i] + lubysu[i];

#if pendulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
		if( Phi[i] < 1.0000000000000000E-013 )
		{
            PRINTTEXT("WARNING: small pivot in Cholesky fact. (=%3.1e < eps=%3.1e), regularizing to %3.1e\n",Phi[i],1.0000000000000000E-013,4.0000000000000002E-004);
			Phi[i] = 2.0000000000000000E-002;
		}
		else
		{
			Phi[i] = sqrt(Phi[i]);
		}
#else
		Phi[i] = Phi[i] < 1.0000000000000000E-013 ? 2.0000000000000000E-002 : sqrt(Phi[i]);
#endif
	}
	
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [3 x 3],
 * B is given and of size [3 x 3], L is a diagonal
 *  matrix of size 3 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<3; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 3.
 */
void pendulum_QP_solver_LA_DIAG_FORWARDSUB_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<3; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [5 x 10] in column
 * storage format, and B is of size [5 x 10] also in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void pendulum_QP_solver_LA_DENSE_MMT2_5_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    pendulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<10; k++ ){
                ltemp += A[k*5+i]*A[k*5+j];
            }			
			for( k=0; k<10; k++ ){
                ltemp += B[k*5+i]*B[k*5+j];
            }
            L[ii+j] = ltemp;
        }
        ii += ++di;
    }
}


/* 
 * Computes r = b - A*x - B*u
 * where A an B are stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_MVMSUB2_5_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[m++]*u[0];
	}	
	for( j=1; j<10; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
	for( n=1; n<10; n++ ){
		for( i=0; i<5; i++ ){
			r[i] -= B[m++]*u[n];
		}		
	}
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [3 x 10] in column
 * storage format, and B is of size [3 x 10] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    pendulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<3; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<10; k++ ){
                ltemp += A[k*3+i]*A[k*3+j];
            }		
            L[ii+j] = ltemp;
        }
		/* work on the diagonal
		 * there might be i == j, but j has already been incremented so it is i == j-1 */
		L[ii+i] += B[i]*B[i];
        ii += ++di;
    }
}


/* 
 * Computes r = b - A*x - B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [3 x 10] in column
 * storage format, and B is of size [3 x 3] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    pendulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<3; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<10; k++ ){
                ltemp += A[k*3+i]*A[k*3+j];
            }		
            L[ii+j] = ltemp;
        }
		/* work on the diagonal
		 * there might be i == j, but j has already been incremented so it is i == j-1 */
		L[ii+i] += B[i]*B[i];
        ii += ++di;
    }
}


/* 
 * Computes r = b - A*x - B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 5 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void pendulum_QP_solver_LA_DENSE_CHOL_5(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    pendulum_QP_solver_FLOAT l;
    pendulum_QP_solver_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<5; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<5; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if pendulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
        if( Mii < 1.0000000000000000E-013 ){
             PRINTTEXT("WARNING (CHOL): small %d-th pivot in Cholesky fact. (=%3.1e < eps=%3.1e), regularizing to %3.1e\n",i,Mii,1.0000000000000000E-013,4.0000000000000002E-004);
			 L[ii+i] = 2.0000000000000000E-002;
		} else
		{
			L[ii+i] = sqrt(Mii);
		}
#else
		L[ii+i] = Mii < 1.0000000000000000E-013 ? 2.0000000000000000E-002 : sqrt(Mii);
#endif

		jj = ((i+1)*(i+2))/2; dj = i+1;
        for( j=i+1; j<5; j++ ){
            l = 0;            
            for( k=0; k<i; k++ ){
                l += L[jj+k]*L[ii+k];
            }

			/* saturate values for numerical stability */
			l = MIN(l,  BIGMM);
			l = MAX(l, -BIGMM);

            L[jj+i] = (L[jj+i] - l)/L[ii+i];            
			jj += ++dj;
        }
		ii += ++di;
    }	
}


/**
 * Forward substitution to solve L*y = b where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * The dimensions involved are 5.
 */
void pendulum_QP_solver_LA_DENSE_FORWARDSUB_5(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    pendulum_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){
        yel = b[i];        
        for( j=0; j<i; j++ ){
            yel -= y[j]*L[ii+j];
        }

		/* saturate for numerical stability  */
		yel = MIN(yel, BIGM);
		yel = MAX(yel, -BIGM);

        y[i] = yel / L[ii+i];
        ii += ++di;
    }
}


/** 
 * Forward substitution for the matrix equation A*L' = B'
 * where A is to be computed and is of size [3 x 5],
 * B is given and of size [3 x 5], L is a lower tri-
 * angular matrix of size 5 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_5(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    pendulum_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<5; j++ ){        
        for( i=0; i<3; i++ ){
            a = B[i*5+j];
            for( k=0; k<j; k++ ){
                a -= A[k*3+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*3+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 3
 * and A is a dense matrix of size [3 x 5] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void pendulum_QP_solver_LA_DENSE_MMTSUB_3_5(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    pendulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<3; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<5; k++ ){
                ltemp += A[k*3+i]*A[k*3+j];
            }						
            L[ii+j] -= ltemp;
        }
        ii += ++di;
    }
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 3 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    pendulum_QP_solver_FLOAT l;
    pendulum_QP_solver_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<3; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<3; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if pendulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
        if( Mii < 1.0000000000000000E-013 ){
             PRINTTEXT("WARNING (CHOL): small %d-th pivot in Cholesky fact. (=%3.1e < eps=%3.1e), regularizing to %3.1e\n",i,Mii,1.0000000000000000E-013,4.0000000000000002E-004);
			 L[ii+i] = 2.0000000000000000E-002;
		} else
		{
			L[ii+i] = sqrt(Mii);
		}
#else
		L[ii+i] = Mii < 1.0000000000000000E-013 ? 2.0000000000000000E-002 : sqrt(Mii);
#endif

		jj = ((i+1)*(i+2))/2; dj = i+1;
        for( j=i+1; j<3; j++ ){
            l = 0;            
            for( k=0; k<i; k++ ){
                l += L[jj+k]*L[ii+k];
            }

			/* saturate values for numerical stability */
			l = MIN(l,  BIGMM);
			l = MAX(l, -BIGMM);

            L[jj+i] = (L[jj+i] - l)/L[ii+i];            
			jj += ++dj;
        }
		ii += ++di;
    }	
}


/* 
 * Computes r = b - A*x
 * where A is stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_MVMSUB1_3_5(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<5; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Forward substitution to solve L*y = b where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * The dimensions involved are 3.
 */
void pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    pendulum_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<3; i++ ){
        yel = b[i];        
        for( j=0; j<i; j++ ){
            yel -= y[j]*L[ii+j];
        }

		/* saturate for numerical stability  */
		yel = MIN(yel, BIGM);
		yel = MAX(yel, -BIGM);

        y[i] = yel / L[ii+i];
        ii += ++di;
    }
}


/** 
 * Forward substitution for the matrix equation A*L' = B'
 * where A is to be computed and is of size [3 x 3],
 * B is given and of size [3 x 3], L is a lower tri-
 * angular matrix of size 3 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    pendulum_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<3; j++ ){        
        for( i=0; i<3; i++ ){
            a = B[i*3+j];
            for( k=0; k<j; k++ ){
                a -= A[k*3+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*3+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 3
 * and A is a dense matrix of size [3 x 3] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    pendulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<3; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<3; k++ ){
                ltemp += A[k*3+i]*A[k*3+j];
            }						
            L[ii+j] -= ltemp;
        }
        ii += ++di;
    }
}


/* 
 * Computes r = b - A*x
 * where A is stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<3; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 3.
 */
void pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    pendulum_QP_solver_FLOAT xel;    
	int start = 3;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 2;
    for( i=2; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 2;
        for( j=2; j>i; j-- ){
            xel -= x[j]*L[jj+i];
            jj -= dj--;
        }

		/* saturate for numerical stability */
		xel = MIN(xel, BIGM);
		xel = MAX(xel, -BIGM); 

        x[i] = xel / L[ii+i];
        ii -= di--;
    }
}


/*
 * Matrix vector multiplication y = b - M'*x where M is of size [3 x 3]
 * and stored in column major format. Note the transpose of M!
 */
void pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<3; i++ ){
		r[i] = b[i];
		for( j=0; j<3; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = b - M'*x where M is of size [3 x 5]
 * and stored in column major format. Note the transpose of M!
 */
void pendulum_QP_solver_LA_DENSE_MTVMSUB_3_5(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<5; i++ ){
		r[i] = b[i];
		for( j=0; j<3; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 5.
 */
void pendulum_QP_solver_LA_DENSE_BACKWARDSUB_5(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    pendulum_QP_solver_FLOAT xel;    
	int start = 10;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 4;
    for( i=4; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 4;
        for( j=4; j>i; j-- ){
            xel -= x[j]*L[jj+i];
            jj -= dj--;
        }

		/* saturate for numerical stability */
		xel = MIN(xel, BIGM);
		xel = MAX(xel, -BIGM); 

        x[i] = xel / L[ii+i];
        ii -= di--;
    }
}


/*
 * Vector subtraction z = -x - y for vectors of length 143.
 */
void pendulum_QP_solver_LA_VSUB2_143(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<143; i++){
		z[i] = -x[i] - y[i];
	}
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 10 in vector
 * storage format.
 */
void pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<10; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 3 in vector
 * storage format.
 */
void pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_3(pendulum_QP_solver_FLOAT *L, pendulum_QP_solver_FLOAT *b, pendulum_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<3; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 10,
 * and x has length 10 and is indexed through yidx.
 */
void pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_FLOAT *x, int* xidx, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<10; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 10.
 */
void pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *w, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<10; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 10
 * and z, x and yidx are of length 6.
 */
void pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<6; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 6.
 */
void pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *w, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<6; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 3,
 * and x has length 3 and is indexed through yidx.
 */
void pendulum_QP_solver_LA_VSUB_INDEXED_3(pendulum_QP_solver_FLOAT *x, int* xidx, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<3; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 3.
 */
void pendulum_QP_solver_LA_VSUB3_3(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *w, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<3; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 3
 * and z, x and yidx are of length 3.
 */
void pendulum_QP_solver_LA_VSUB2_INDEXED_3(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<3; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/**
 * Backtracking line search.
 * 
 * First determine the maximum line length by a feasibility line
 * search, i.e. a ~= argmax{ a \in [0...1] s.t. l+a*dl >= 0 and s+a*ds >= 0}.
 *
 * The function returns either the number of iterations or exits the error code
 * pendulum_QP_solver_NOPROGRESS (should be negative).
 */
int pendulum_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *dl, pendulum_QP_solver_FLOAT *ds, pendulum_QP_solver_FLOAT *a, pendulum_QP_solver_FLOAT *mu_aff)
{
    int i;
	int lsIt=1;    
    pendulum_QP_solver_FLOAT dltemp;
    pendulum_QP_solver_FLOAT dstemp;
    pendulum_QP_solver_FLOAT mya = 1.0;
    pendulum_QP_solver_FLOAT mymu;
        
    while( 1 ){                        

        /* 
         * Compute both snew and wnew together.
         * We compute also mu_affine along the way here, as the
         * values might be in registers, so it should be cheaper.
         */
        mymu = 0;
        for( i=0; i<230; i++ ){
            dltemp = l[i] + mya*dl[i];
            dstemp = s[i] + mya*ds[i];
            if( dltemp < 0 || dstemp < 0 ){
                lsIt++;
                break;
            } else {                
                mymu += dstemp*dltemp;
            }
        }
        
        /* 
         * If no early termination of the for-loop above occurred, we
         * found the required value of a and we can quit the while loop.
         */
        if( i == 230 ){
            break;
        } else {
            mya *= pendulum_QP_solver_SET_LS_SCALE_AFF;
            if( mya < pendulum_QP_solver_SET_LS_MINSTEP ){
                return pendulum_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* return new values and iteration counter */
    *a = mya;
    *mu_aff = mymu / (pendulum_QP_solver_FLOAT)230;
    return lsIt;
}


/*
 * Vector subtraction x = (u.*v - mu)*sigma where a is a scalar
*  and x,u,v are vectors of length 230.
 */
void pendulum_QP_solver_LA_VSUB5_230(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT mu,  pendulum_QP_solver_FLOAT sigma, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<230; i++){
		x[i] = u[i]*v[i] - mu;
		x[i] *= sigma;
	}
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 10,
 * u, su, uidx are of length 6 and v, sv, vidx are of length 10.
 */
void pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *su, int* uidx, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *sv, int* vidx, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<10; i++ ){
		x[i] = 0;
	}
	for( i=0; i<6; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<10; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A an B are stored in column major format
 */
void pendulum_QP_solver_LA_DENSE_2MVMADD_5_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<5; i++ ){
		r[i] = A[k++]*x[0] + B[m++]*u[0];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<10; n++ ){
		for( i=0; i<5; i++ ){
			r[i] += B[m++]*u[n];
		}		
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 3,
 * u, su, uidx are of length 3 and v, sv, vidx are of length 3.
 */
void pendulum_QP_solver_LA_VSUB6_INDEXED_3_3_3(pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *su, int* uidx, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *sv, int* vidx, pendulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<3; i++ ){
		x[i] = 0;
	}
	for( i=0; i<3; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<3; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_3(pendulum_QP_solver_FLOAT *A, pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *B, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<3; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<10; j++ ){		
		for( i=0; i<3; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Vector subtraction z = x - y for vectors of length 143.
 */
void pendulum_QP_solver_LA_VSUB_143(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<143; i++){
		z[i] = x[i] - y[i];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 10 (length of y >= 10).
 */
void pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<10; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 6 (length of y >= 6).
 */
void pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<6; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 3 (length of y >= 3).
 */
void pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_3(pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<3; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 3 (length of y >= 3).
 */
void pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_3(pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *u, pendulum_QP_solver_FLOAT *y, int* yidx, pendulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<3; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/*
 * Computes ds = -l.\(r + s.*dl) for vectors of length 230.
 */
void pendulum_QP_solver_LA_VSUB7_230(pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *r, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *dl, pendulum_QP_solver_FLOAT *ds)
{
	int i;
	for( i=0; i<230; i++){
		ds[i] = -(r[i] + s[i]*dl[i])/l[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 143.
 */
void pendulum_QP_solver_LA_VADD_143(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<143; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 44.
 */
void pendulum_QP_solver_LA_VADD_44(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<44; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 230.
 */
void pendulum_QP_solver_LA_VADD_230(pendulum_QP_solver_FLOAT *x, pendulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<230; i++){
		x[i] += y[i];
	}
}


/**
 * Backtracking line search for combined predictor/corrector step.
 * Update on variables with safety factor gamma (to keep us away from
 * boundary).
 */
int pendulum_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(pendulum_QP_solver_FLOAT *z, pendulum_QP_solver_FLOAT *v, pendulum_QP_solver_FLOAT *l, pendulum_QP_solver_FLOAT *s, pendulum_QP_solver_FLOAT *dz, pendulum_QP_solver_FLOAT *dv, pendulum_QP_solver_FLOAT *dl, pendulum_QP_solver_FLOAT *ds, pendulum_QP_solver_FLOAT *a, pendulum_QP_solver_FLOAT *mu)
{
    int i, lsIt=1;       
    pendulum_QP_solver_FLOAT dltemp;
    pendulum_QP_solver_FLOAT dstemp;    
    pendulum_QP_solver_FLOAT a_gamma;
            
    *a = 1.0;
    while( 1 ){                        

        /* check whether search criterion is fulfilled */
        for( i=0; i<230; i++ ){
            dltemp = l[i] + (*a)*dl[i];
            dstemp = s[i] + (*a)*ds[i];
            if( dltemp < 0 || dstemp < 0 ){
                lsIt++;
                break;
            }
        }
        
        /* 
         * If no early termination of the for-loop above occurred, we
         * found the required value of a and we can quit the while loop.
         */
        if( i == 230 ){
            break;
        } else {
            *a *= pendulum_QP_solver_SET_LS_SCALE;
            if( *a < pendulum_QP_solver_SET_LS_MINSTEP ){
                return pendulum_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* update variables with safety margin */
    a_gamma = (*a)*pendulum_QP_solver_SET_LS_MAXSTEP;
    
    /* primal variables */
    for( i=0; i<143; i++ ){
        z[i] += a_gamma*dz[i];
    }
    
    /* equality constraint multipliers */
    for( i=0; i<44; i++ ){
        v[i] += a_gamma*dv[i];
    }
    
    /* inequality constraint multipliers & slacks, also update mu */
    *mu = 0;
    for( i=0; i<230; i++ ){
        dltemp = l[i] + a_gamma*dl[i]; l[i] = dltemp;
        dstemp = s[i] + a_gamma*ds[i]; s[i] = dstemp;
        *mu += dltemp*dstemp;
    }
    
    *a = a_gamma;
    *mu /= (pendulum_QP_solver_FLOAT)230;
    return lsIt;
}




/* VARIABLE DEFINITIONS ------------------------------------------------ */
pendulum_QP_solver_FLOAT pendulum_QP_solver_z[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_v[44];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dz_aff[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dv_aff[44];
pendulum_QP_solver_FLOAT pendulum_QP_solver_grad_cost[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_grad_eq[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_rd[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_l[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_s[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_lbys[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dl_aff[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_ds_aff[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dz_cc[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dv_cc[44];
pendulum_QP_solver_FLOAT pendulum_QP_solver_dl_cc[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_ds_cc[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_ccrhs[230];
pendulum_QP_solver_FLOAT pendulum_QP_solver_grad_ineq[143];
pendulum_QP_solver_FLOAT pendulum_QP_solver_H00[10] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z00 = pendulum_QP_solver_z + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff00 = pendulum_QP_solver_dz_aff + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc00 = pendulum_QP_solver_dz_cc + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd00 = pendulum_QP_solver_rd + 0;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd00[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost00 = pendulum_QP_solver_grad_cost + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq00 = pendulum_QP_solver_grad_eq + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq00 = pendulum_QP_solver_grad_ineq + 0;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv00[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v00 = pendulum_QP_solver_v + 0;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re00[5];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta00[5];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc00[5];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff00 = pendulum_QP_solver_dv_aff + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc00 = pendulum_QP_solver_dv_cc + 0;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V00[50];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd00[15];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld00[15];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy00[5];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy00[5];
int pendulum_QP_solver_lbIdx00[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb00 = pendulum_QP_solver_l + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb00 = pendulum_QP_solver_s + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb00 = pendulum_QP_solver_lbys + 0;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb00[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff00 = pendulum_QP_solver_dl_aff + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff00 = pendulum_QP_solver_ds_aff + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc00 = pendulum_QP_solver_dl_cc + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc00 = pendulum_QP_solver_ds_cc + 0;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl00 = pendulum_QP_solver_ccrhs + 0;
int pendulum_QP_solver_ubIdx00[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub00 = pendulum_QP_solver_l + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub00 = pendulum_QP_solver_s + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub00 = pendulum_QP_solver_lbys + 10;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub00[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff00 = pendulum_QP_solver_dl_aff + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff00 = pendulum_QP_solver_ds_aff + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc00 = pendulum_QP_solver_dl_cc + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc00 = pendulum_QP_solver_ds_cc + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub00 = pendulum_QP_solver_ccrhs + 10;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi00[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z01 = pendulum_QP_solver_z + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff01 = pendulum_QP_solver_dz_aff + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc01 = pendulum_QP_solver_dz_cc + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd01 = pendulum_QP_solver_rd + 10;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd01[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost01 = pendulum_QP_solver_grad_cost + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq01 = pendulum_QP_solver_grad_eq + 10;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq01 = pendulum_QP_solver_grad_ineq + 10;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv01[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v01 = pendulum_QP_solver_v + 5;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re01[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta01[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc01[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff01 = pendulum_QP_solver_dv_aff + 5;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc01 = pendulum_QP_solver_dv_cc + 5;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V01[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd01[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld01[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy01[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy01[3];
int pendulum_QP_solver_lbIdx01[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb01 = pendulum_QP_solver_l + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb01 = pendulum_QP_solver_s + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb01 = pendulum_QP_solver_lbys + 16;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb01[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff01 = pendulum_QP_solver_dl_aff + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff01 = pendulum_QP_solver_ds_aff + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc01 = pendulum_QP_solver_dl_cc + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc01 = pendulum_QP_solver_ds_cc + 16;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl01 = pendulum_QP_solver_ccrhs + 16;
int pendulum_QP_solver_ubIdx01[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub01 = pendulum_QP_solver_l + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub01 = pendulum_QP_solver_s + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub01 = pendulum_QP_solver_lbys + 26;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub01[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff01 = pendulum_QP_solver_dl_aff + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff01 = pendulum_QP_solver_ds_aff + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc01 = pendulum_QP_solver_dl_cc + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc01 = pendulum_QP_solver_ds_cc + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub01 = pendulum_QP_solver_ccrhs + 26;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi01[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_D01[50] = {0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
pendulum_QP_solver_FLOAT pendulum_QP_solver_W01[50];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd01[15];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd01[15];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z02 = pendulum_QP_solver_z + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff02 = pendulum_QP_solver_dz_aff + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc02 = pendulum_QP_solver_dz_cc + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd02 = pendulum_QP_solver_rd + 20;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd02[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost02 = pendulum_QP_solver_grad_cost + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq02 = pendulum_QP_solver_grad_eq + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq02 = pendulum_QP_solver_grad_ineq + 20;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv02[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v02 = pendulum_QP_solver_v + 8;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re02[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta02[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc02[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff02 = pendulum_QP_solver_dv_aff + 8;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc02 = pendulum_QP_solver_dv_cc + 8;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V02[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd02[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld02[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy02[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy02[3];
int pendulum_QP_solver_lbIdx02[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb02 = pendulum_QP_solver_l + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb02 = pendulum_QP_solver_s + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb02 = pendulum_QP_solver_lbys + 32;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb02[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff02 = pendulum_QP_solver_dl_aff + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff02 = pendulum_QP_solver_ds_aff + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc02 = pendulum_QP_solver_dl_cc + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc02 = pendulum_QP_solver_ds_cc + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl02 = pendulum_QP_solver_ccrhs + 32;
int pendulum_QP_solver_ubIdx02[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub02 = pendulum_QP_solver_l + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub02 = pendulum_QP_solver_s + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub02 = pendulum_QP_solver_lbys + 42;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub02[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff02 = pendulum_QP_solver_dl_aff + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff02 = pendulum_QP_solver_ds_aff + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc02 = pendulum_QP_solver_dl_cc + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc02 = pendulum_QP_solver_ds_cc + 42;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub02 = pendulum_QP_solver_ccrhs + 42;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi02[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_D02[10] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
pendulum_QP_solver_FLOAT pendulum_QP_solver_W02[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd02[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd02[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z03 = pendulum_QP_solver_z + 30;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff03 = pendulum_QP_solver_dz_aff + 30;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc03 = pendulum_QP_solver_dz_cc + 30;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd03 = pendulum_QP_solver_rd + 30;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd03[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost03 = pendulum_QP_solver_grad_cost + 30;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq03 = pendulum_QP_solver_grad_eq + 30;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq03 = pendulum_QP_solver_grad_ineq + 30;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv03[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v03 = pendulum_QP_solver_v + 11;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re03[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta03[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc03[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff03 = pendulum_QP_solver_dv_aff + 11;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc03 = pendulum_QP_solver_dv_cc + 11;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V03[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd03[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld03[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy03[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy03[3];
int pendulum_QP_solver_lbIdx03[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb03 = pendulum_QP_solver_l + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb03 = pendulum_QP_solver_s + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb03 = pendulum_QP_solver_lbys + 48;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb03[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff03 = pendulum_QP_solver_dl_aff + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff03 = pendulum_QP_solver_ds_aff + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc03 = pendulum_QP_solver_dl_cc + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc03 = pendulum_QP_solver_ds_cc + 48;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl03 = pendulum_QP_solver_ccrhs + 48;
int pendulum_QP_solver_ubIdx03[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub03 = pendulum_QP_solver_l + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub03 = pendulum_QP_solver_s + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub03 = pendulum_QP_solver_lbys + 58;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub03[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff03 = pendulum_QP_solver_dl_aff + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff03 = pendulum_QP_solver_ds_aff + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc03 = pendulum_QP_solver_dl_cc + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc03 = pendulum_QP_solver_ds_cc + 58;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub03 = pendulum_QP_solver_ccrhs + 58;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi03[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W03[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd03[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd03[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z04 = pendulum_QP_solver_z + 40;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff04 = pendulum_QP_solver_dz_aff + 40;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc04 = pendulum_QP_solver_dz_cc + 40;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd04 = pendulum_QP_solver_rd + 40;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd04[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost04 = pendulum_QP_solver_grad_cost + 40;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq04 = pendulum_QP_solver_grad_eq + 40;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq04 = pendulum_QP_solver_grad_ineq + 40;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv04[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v04 = pendulum_QP_solver_v + 14;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re04[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta04[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc04[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff04 = pendulum_QP_solver_dv_aff + 14;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc04 = pendulum_QP_solver_dv_cc + 14;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V04[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd04[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld04[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy04[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy04[3];
int pendulum_QP_solver_lbIdx04[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb04 = pendulum_QP_solver_l + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb04 = pendulum_QP_solver_s + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb04 = pendulum_QP_solver_lbys + 64;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb04[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff04 = pendulum_QP_solver_dl_aff + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff04 = pendulum_QP_solver_ds_aff + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc04 = pendulum_QP_solver_dl_cc + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc04 = pendulum_QP_solver_ds_cc + 64;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl04 = pendulum_QP_solver_ccrhs + 64;
int pendulum_QP_solver_ubIdx04[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub04 = pendulum_QP_solver_l + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub04 = pendulum_QP_solver_s + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub04 = pendulum_QP_solver_lbys + 74;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub04[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff04 = pendulum_QP_solver_dl_aff + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff04 = pendulum_QP_solver_ds_aff + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc04 = pendulum_QP_solver_dl_cc + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc04 = pendulum_QP_solver_ds_cc + 74;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub04 = pendulum_QP_solver_ccrhs + 74;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi04[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W04[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd04[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd04[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z05 = pendulum_QP_solver_z + 50;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff05 = pendulum_QP_solver_dz_aff + 50;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc05 = pendulum_QP_solver_dz_cc + 50;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd05 = pendulum_QP_solver_rd + 50;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd05[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost05 = pendulum_QP_solver_grad_cost + 50;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq05 = pendulum_QP_solver_grad_eq + 50;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq05 = pendulum_QP_solver_grad_ineq + 50;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv05[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v05 = pendulum_QP_solver_v + 17;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re05[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta05[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc05[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff05 = pendulum_QP_solver_dv_aff + 17;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc05 = pendulum_QP_solver_dv_cc + 17;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V05[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd05[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld05[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy05[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy05[3];
int pendulum_QP_solver_lbIdx05[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb05 = pendulum_QP_solver_l + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb05 = pendulum_QP_solver_s + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb05 = pendulum_QP_solver_lbys + 80;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb05[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff05 = pendulum_QP_solver_dl_aff + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff05 = pendulum_QP_solver_ds_aff + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc05 = pendulum_QP_solver_dl_cc + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc05 = pendulum_QP_solver_ds_cc + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl05 = pendulum_QP_solver_ccrhs + 80;
int pendulum_QP_solver_ubIdx05[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub05 = pendulum_QP_solver_l + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub05 = pendulum_QP_solver_s + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub05 = pendulum_QP_solver_lbys + 90;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub05[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff05 = pendulum_QP_solver_dl_aff + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff05 = pendulum_QP_solver_ds_aff + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc05 = pendulum_QP_solver_dl_cc + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc05 = pendulum_QP_solver_ds_cc + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub05 = pendulum_QP_solver_ccrhs + 90;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi05[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W05[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd05[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd05[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z06 = pendulum_QP_solver_z + 60;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff06 = pendulum_QP_solver_dz_aff + 60;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc06 = pendulum_QP_solver_dz_cc + 60;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd06 = pendulum_QP_solver_rd + 60;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd06[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost06 = pendulum_QP_solver_grad_cost + 60;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq06 = pendulum_QP_solver_grad_eq + 60;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq06 = pendulum_QP_solver_grad_ineq + 60;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv06[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v06 = pendulum_QP_solver_v + 20;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re06[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta06[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc06[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff06 = pendulum_QP_solver_dv_aff + 20;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc06 = pendulum_QP_solver_dv_cc + 20;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V06[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd06[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld06[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy06[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy06[3];
int pendulum_QP_solver_lbIdx06[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb06 = pendulum_QP_solver_l + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb06 = pendulum_QP_solver_s + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb06 = pendulum_QP_solver_lbys + 96;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb06[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff06 = pendulum_QP_solver_dl_aff + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff06 = pendulum_QP_solver_ds_aff + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc06 = pendulum_QP_solver_dl_cc + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc06 = pendulum_QP_solver_ds_cc + 96;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl06 = pendulum_QP_solver_ccrhs + 96;
int pendulum_QP_solver_ubIdx06[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub06 = pendulum_QP_solver_l + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub06 = pendulum_QP_solver_s + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub06 = pendulum_QP_solver_lbys + 106;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub06[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff06 = pendulum_QP_solver_dl_aff + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff06 = pendulum_QP_solver_ds_aff + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc06 = pendulum_QP_solver_dl_cc + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc06 = pendulum_QP_solver_ds_cc + 106;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub06 = pendulum_QP_solver_ccrhs + 106;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi06[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W06[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd06[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd06[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z07 = pendulum_QP_solver_z + 70;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff07 = pendulum_QP_solver_dz_aff + 70;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc07 = pendulum_QP_solver_dz_cc + 70;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd07 = pendulum_QP_solver_rd + 70;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd07[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost07 = pendulum_QP_solver_grad_cost + 70;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq07 = pendulum_QP_solver_grad_eq + 70;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq07 = pendulum_QP_solver_grad_ineq + 70;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv07[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v07 = pendulum_QP_solver_v + 23;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re07[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta07[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc07[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff07 = pendulum_QP_solver_dv_aff + 23;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc07 = pendulum_QP_solver_dv_cc + 23;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V07[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd07[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld07[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy07[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy07[3];
int pendulum_QP_solver_lbIdx07[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb07 = pendulum_QP_solver_l + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb07 = pendulum_QP_solver_s + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb07 = pendulum_QP_solver_lbys + 112;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb07[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff07 = pendulum_QP_solver_dl_aff + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff07 = pendulum_QP_solver_ds_aff + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc07 = pendulum_QP_solver_dl_cc + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc07 = pendulum_QP_solver_ds_cc + 112;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl07 = pendulum_QP_solver_ccrhs + 112;
int pendulum_QP_solver_ubIdx07[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub07 = pendulum_QP_solver_l + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub07 = pendulum_QP_solver_s + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub07 = pendulum_QP_solver_lbys + 122;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub07[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff07 = pendulum_QP_solver_dl_aff + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff07 = pendulum_QP_solver_ds_aff + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc07 = pendulum_QP_solver_dl_cc + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc07 = pendulum_QP_solver_ds_cc + 122;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub07 = pendulum_QP_solver_ccrhs + 122;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi07[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W07[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd07[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd07[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z08 = pendulum_QP_solver_z + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff08 = pendulum_QP_solver_dz_aff + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc08 = pendulum_QP_solver_dz_cc + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd08 = pendulum_QP_solver_rd + 80;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd08[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost08 = pendulum_QP_solver_grad_cost + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq08 = pendulum_QP_solver_grad_eq + 80;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq08 = pendulum_QP_solver_grad_ineq + 80;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv08[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v08 = pendulum_QP_solver_v + 26;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re08[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta08[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc08[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff08 = pendulum_QP_solver_dv_aff + 26;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc08 = pendulum_QP_solver_dv_cc + 26;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V08[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd08[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld08[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy08[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy08[3];
int pendulum_QP_solver_lbIdx08[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb08 = pendulum_QP_solver_l + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb08 = pendulum_QP_solver_s + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb08 = pendulum_QP_solver_lbys + 128;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb08[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff08 = pendulum_QP_solver_dl_aff + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff08 = pendulum_QP_solver_ds_aff + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc08 = pendulum_QP_solver_dl_cc + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc08 = pendulum_QP_solver_ds_cc + 128;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl08 = pendulum_QP_solver_ccrhs + 128;
int pendulum_QP_solver_ubIdx08[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub08 = pendulum_QP_solver_l + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub08 = pendulum_QP_solver_s + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub08 = pendulum_QP_solver_lbys + 138;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub08[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff08 = pendulum_QP_solver_dl_aff + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff08 = pendulum_QP_solver_ds_aff + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc08 = pendulum_QP_solver_dl_cc + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc08 = pendulum_QP_solver_ds_cc + 138;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub08 = pendulum_QP_solver_ccrhs + 138;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi08[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W08[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd08[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd08[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z09 = pendulum_QP_solver_z + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff09 = pendulum_QP_solver_dz_aff + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc09 = pendulum_QP_solver_dz_cc + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd09 = pendulum_QP_solver_rd + 90;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd09[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost09 = pendulum_QP_solver_grad_cost + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq09 = pendulum_QP_solver_grad_eq + 90;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq09 = pendulum_QP_solver_grad_ineq + 90;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv09[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v09 = pendulum_QP_solver_v + 29;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re09[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta09[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc09[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff09 = pendulum_QP_solver_dv_aff + 29;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc09 = pendulum_QP_solver_dv_cc + 29;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V09[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd09[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld09[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy09[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy09[3];
int pendulum_QP_solver_lbIdx09[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb09 = pendulum_QP_solver_l + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb09 = pendulum_QP_solver_s + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb09 = pendulum_QP_solver_lbys + 144;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb09[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff09 = pendulum_QP_solver_dl_aff + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff09 = pendulum_QP_solver_ds_aff + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc09 = pendulum_QP_solver_dl_cc + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc09 = pendulum_QP_solver_ds_cc + 144;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl09 = pendulum_QP_solver_ccrhs + 144;
int pendulum_QP_solver_ubIdx09[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub09 = pendulum_QP_solver_l + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub09 = pendulum_QP_solver_s + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub09 = pendulum_QP_solver_lbys + 154;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub09[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff09 = pendulum_QP_solver_dl_aff + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff09 = pendulum_QP_solver_ds_aff + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc09 = pendulum_QP_solver_dl_cc + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc09 = pendulum_QP_solver_ds_cc + 154;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub09 = pendulum_QP_solver_ccrhs + 154;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi09[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W09[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd09[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd09[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z10 = pendulum_QP_solver_z + 100;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff10 = pendulum_QP_solver_dz_aff + 100;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc10 = pendulum_QP_solver_dz_cc + 100;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd10 = pendulum_QP_solver_rd + 100;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd10[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost10 = pendulum_QP_solver_grad_cost + 100;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq10 = pendulum_QP_solver_grad_eq + 100;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq10 = pendulum_QP_solver_grad_ineq + 100;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv10[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v10 = pendulum_QP_solver_v + 32;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re10[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta10[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc10[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff10 = pendulum_QP_solver_dv_aff + 32;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc10 = pendulum_QP_solver_dv_cc + 32;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V10[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd10[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld10[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy10[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy10[3];
int pendulum_QP_solver_lbIdx10[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb10 = pendulum_QP_solver_l + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb10 = pendulum_QP_solver_s + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb10 = pendulum_QP_solver_lbys + 160;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb10[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff10 = pendulum_QP_solver_dl_aff + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff10 = pendulum_QP_solver_ds_aff + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc10 = pendulum_QP_solver_dl_cc + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc10 = pendulum_QP_solver_ds_cc + 160;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl10 = pendulum_QP_solver_ccrhs + 160;
int pendulum_QP_solver_ubIdx10[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub10 = pendulum_QP_solver_l + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub10 = pendulum_QP_solver_s + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub10 = pendulum_QP_solver_lbys + 170;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub10[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff10 = pendulum_QP_solver_dl_aff + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff10 = pendulum_QP_solver_ds_aff + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc10 = pendulum_QP_solver_dl_cc + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc10 = pendulum_QP_solver_ds_cc + 170;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub10 = pendulum_QP_solver_ccrhs + 170;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi10[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W10[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd10[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd10[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z11 = pendulum_QP_solver_z + 110;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff11 = pendulum_QP_solver_dz_aff + 110;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc11 = pendulum_QP_solver_dz_cc + 110;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd11 = pendulum_QP_solver_rd + 110;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd11[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost11 = pendulum_QP_solver_grad_cost + 110;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq11 = pendulum_QP_solver_grad_eq + 110;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq11 = pendulum_QP_solver_grad_ineq + 110;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv11[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v11 = pendulum_QP_solver_v + 35;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re11[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta11[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc11[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff11 = pendulum_QP_solver_dv_aff + 35;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc11 = pendulum_QP_solver_dv_cc + 35;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V11[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd11[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld11[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy11[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy11[3];
int pendulum_QP_solver_lbIdx11[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb11 = pendulum_QP_solver_l + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb11 = pendulum_QP_solver_s + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb11 = pendulum_QP_solver_lbys + 176;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb11[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff11 = pendulum_QP_solver_dl_aff + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff11 = pendulum_QP_solver_ds_aff + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc11 = pendulum_QP_solver_dl_cc + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc11 = pendulum_QP_solver_ds_cc + 176;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl11 = pendulum_QP_solver_ccrhs + 176;
int pendulum_QP_solver_ubIdx11[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub11 = pendulum_QP_solver_l + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub11 = pendulum_QP_solver_s + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub11 = pendulum_QP_solver_lbys + 186;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub11[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff11 = pendulum_QP_solver_dl_aff + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff11 = pendulum_QP_solver_ds_aff + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc11 = pendulum_QP_solver_dl_cc + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc11 = pendulum_QP_solver_ds_cc + 186;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub11 = pendulum_QP_solver_ccrhs + 186;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi11[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W11[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd11[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd11[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z12 = pendulum_QP_solver_z + 120;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff12 = pendulum_QP_solver_dz_aff + 120;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc12 = pendulum_QP_solver_dz_cc + 120;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd12 = pendulum_QP_solver_rd + 120;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd12[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost12 = pendulum_QP_solver_grad_cost + 120;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq12 = pendulum_QP_solver_grad_eq + 120;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq12 = pendulum_QP_solver_grad_ineq + 120;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv12[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v12 = pendulum_QP_solver_v + 38;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re12[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta12[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc12[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff12 = pendulum_QP_solver_dv_aff + 38;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc12 = pendulum_QP_solver_dv_cc + 38;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V12[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd12[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld12[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy12[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy12[3];
int pendulum_QP_solver_lbIdx12[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb12 = pendulum_QP_solver_l + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb12 = pendulum_QP_solver_s + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb12 = pendulum_QP_solver_lbys + 192;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb12[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff12 = pendulum_QP_solver_dl_aff + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff12 = pendulum_QP_solver_ds_aff + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc12 = pendulum_QP_solver_dl_cc + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc12 = pendulum_QP_solver_ds_cc + 192;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl12 = pendulum_QP_solver_ccrhs + 192;
int pendulum_QP_solver_ubIdx12[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub12 = pendulum_QP_solver_l + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub12 = pendulum_QP_solver_s + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub12 = pendulum_QP_solver_lbys + 202;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub12[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff12 = pendulum_QP_solver_dl_aff + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff12 = pendulum_QP_solver_ds_aff + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc12 = pendulum_QP_solver_dl_cc + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc12 = pendulum_QP_solver_ds_cc + 202;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub12 = pendulum_QP_solver_ccrhs + 202;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi12[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W12[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd12[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd12[9];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z13 = pendulum_QP_solver_z + 130;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff13 = pendulum_QP_solver_dz_aff + 130;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc13 = pendulum_QP_solver_dz_cc + 130;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd13 = pendulum_QP_solver_rd + 130;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd13[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost13 = pendulum_QP_solver_grad_cost + 130;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq13 = pendulum_QP_solver_grad_eq + 130;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq13 = pendulum_QP_solver_grad_ineq + 130;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv13[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_v13 = pendulum_QP_solver_v + 41;
pendulum_QP_solver_FLOAT pendulum_QP_solver_re13[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_beta13[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_betacc13[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvaff13 = pendulum_QP_solver_dv_aff + 41;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dvcc13 = pendulum_QP_solver_dv_cc + 41;
pendulum_QP_solver_FLOAT pendulum_QP_solver_V13[30];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Yd13[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ld13[6];
pendulum_QP_solver_FLOAT pendulum_QP_solver_yy13[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_bmy13[3];
int pendulum_QP_solver_lbIdx13[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb13 = pendulum_QP_solver_l + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb13 = pendulum_QP_solver_s + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb13 = pendulum_QP_solver_lbys + 208;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb13[10];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff13 = pendulum_QP_solver_dl_aff + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff13 = pendulum_QP_solver_ds_aff + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc13 = pendulum_QP_solver_dl_cc + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc13 = pendulum_QP_solver_ds_cc + 208;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl13 = pendulum_QP_solver_ccrhs + 208;
int pendulum_QP_solver_ubIdx13[6] = {0, 1, 2, 3, 4, 5};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub13 = pendulum_QP_solver_l + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub13 = pendulum_QP_solver_s + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub13 = pendulum_QP_solver_lbys + 218;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub13[6];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff13 = pendulum_QP_solver_dl_aff + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff13 = pendulum_QP_solver_ds_aff + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc13 = pendulum_QP_solver_dl_cc + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc13 = pendulum_QP_solver_ds_cc + 218;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub13 = pendulum_QP_solver_ccrhs + 218;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi13[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_W13[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Ysd13[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lsd13[9];
pendulum_QP_solver_FLOAT pendulum_QP_solver_H14[3] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
pendulum_QP_solver_FLOAT pendulum_QP_solver_f14[3] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_z14 = pendulum_QP_solver_z + 140;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzaff14 = pendulum_QP_solver_dz_aff + 140;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dzcc14 = pendulum_QP_solver_dz_cc + 140;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_rd14 = pendulum_QP_solver_rd + 140;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Lbyrd14[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_cost14 = pendulum_QP_solver_grad_cost + 140;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_eq14 = pendulum_QP_solver_grad_eq + 140;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_grad_ineq14 = pendulum_QP_solver_grad_ineq + 140;
pendulum_QP_solver_FLOAT pendulum_QP_solver_ctv14[3];
int pendulum_QP_solver_lbIdx14[3] = {0, 1, 2};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llb14 = pendulum_QP_solver_l + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_slb14 = pendulum_QP_solver_s + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_llbbyslb14 = pendulum_QP_solver_lbys + 224;
pendulum_QP_solver_FLOAT pendulum_QP_solver_rilb14[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbaff14 = pendulum_QP_solver_dl_aff + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbaff14 = pendulum_QP_solver_ds_aff + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dllbcc14 = pendulum_QP_solver_dl_cc + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dslbcc14 = pendulum_QP_solver_ds_cc + 224;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsl14 = pendulum_QP_solver_ccrhs + 224;
int pendulum_QP_solver_ubIdx14[3] = {0, 1, 2};
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lub14 = pendulum_QP_solver_l + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_sub14 = pendulum_QP_solver_s + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_lubbysub14 = pendulum_QP_solver_lbys + 227;
pendulum_QP_solver_FLOAT pendulum_QP_solver_riub14[3];
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubaff14 = pendulum_QP_solver_dl_aff + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubaff14 = pendulum_QP_solver_ds_aff + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dlubcc14 = pendulum_QP_solver_dl_cc + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_dsubcc14 = pendulum_QP_solver_ds_cc + 227;
pendulum_QP_solver_FLOAT* pendulum_QP_solver_ccrhsub14 = pendulum_QP_solver_ccrhs + 227;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Phi14[3];
pendulum_QP_solver_FLOAT pendulum_QP_solver_D14[3] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
pendulum_QP_solver_FLOAT pendulum_QP_solver_W14[3];
pendulum_QP_solver_FLOAT musigma;
pendulum_QP_solver_FLOAT sigma_3rdroot;
pendulum_QP_solver_FLOAT pendulum_QP_solver_Diag1_0[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_Diag2_0[10];
pendulum_QP_solver_FLOAT pendulum_QP_solver_L_0[45];




/* SOLVER CODE --------------------------------------------------------- */
int pendulum_QP_solver_solve(pendulum_QP_solver_params* params, pendulum_QP_solver_output* output, pendulum_QP_solver_info* info)
{	
int exitcode;

#if pendulum_QP_solver_SET_TIMING == 1
	pendulum_QP_solver_timer solvertimer;
	pendulum_QP_solver_tic(&solvertimer);
#endif
/* FUNCTION CALLS INTO LA LIBRARY -------------------------------------- */
info->it = 0;
pendulum_QP_solver_LA_INITIALIZEVECTOR_143(pendulum_QP_solver_z, 0);
pendulum_QP_solver_LA_INITIALIZEVECTOR_44(pendulum_QP_solver_v, 1);
pendulum_QP_solver_LA_INITIALIZEVECTOR_230(pendulum_QP_solver_l, 10);
pendulum_QP_solver_LA_INITIALIZEVECTOR_230(pendulum_QP_solver_s, 10);
info->mu = 0;
pendulum_QP_solver_LA_DOTACC_230(pendulum_QP_solver_l, pendulum_QP_solver_s, &info->mu);
info->mu /= 230;
while( 1 ){
info->pobj = 0;
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f1, pendulum_QP_solver_z00, pendulum_QP_solver_grad_cost00, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f2, pendulum_QP_solver_z01, pendulum_QP_solver_grad_cost01, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f3, pendulum_QP_solver_z02, pendulum_QP_solver_grad_cost02, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f4, pendulum_QP_solver_z03, pendulum_QP_solver_grad_cost03, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f5, pendulum_QP_solver_z04, pendulum_QP_solver_grad_cost04, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f6, pendulum_QP_solver_z05, pendulum_QP_solver_grad_cost05, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f7, pendulum_QP_solver_z06, pendulum_QP_solver_grad_cost06, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f8, pendulum_QP_solver_z07, pendulum_QP_solver_grad_cost07, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f9, pendulum_QP_solver_z08, pendulum_QP_solver_grad_cost08, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f10, pendulum_QP_solver_z09, pendulum_QP_solver_grad_cost09, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f11, pendulum_QP_solver_z10, pendulum_QP_solver_grad_cost10, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f12, pendulum_QP_solver_z11, pendulum_QP_solver_grad_cost11, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f13, pendulum_QP_solver_z12, pendulum_QP_solver_grad_cost12, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_10(pendulum_QP_solver_H00, params->f14, pendulum_QP_solver_z13, pendulum_QP_solver_grad_cost13, &info->pobj);
pendulum_QP_solver_LA_DIAG_QUADFCN_3(pendulum_QP_solver_H14, pendulum_QP_solver_f14, pendulum_QP_solver_z14, pendulum_QP_solver_grad_cost14, &info->pobj);
info->res_eq = 0;
info->dgap = 0;
pendulum_QP_solver_LA_DENSE_MVMSUB3_5_10_10(params->C1, pendulum_QP_solver_z00, pendulum_QP_solver_D01, pendulum_QP_solver_z01, params->e1, pendulum_QP_solver_v00, pendulum_QP_solver_re00, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C2, pendulum_QP_solver_z01, pendulum_QP_solver_D02, pendulum_QP_solver_z02, params->e2, pendulum_QP_solver_v01, pendulum_QP_solver_re01, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C3, pendulum_QP_solver_z02, pendulum_QP_solver_D02, pendulum_QP_solver_z03, params->e3, pendulum_QP_solver_v02, pendulum_QP_solver_re02, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C4, pendulum_QP_solver_z03, pendulum_QP_solver_D02, pendulum_QP_solver_z04, params->e4, pendulum_QP_solver_v03, pendulum_QP_solver_re03, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C5, pendulum_QP_solver_z04, pendulum_QP_solver_D02, pendulum_QP_solver_z05, params->e5, pendulum_QP_solver_v04, pendulum_QP_solver_re04, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C6, pendulum_QP_solver_z05, pendulum_QP_solver_D02, pendulum_QP_solver_z06, params->e6, pendulum_QP_solver_v05, pendulum_QP_solver_re05, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C7, pendulum_QP_solver_z06, pendulum_QP_solver_D02, pendulum_QP_solver_z07, params->e7, pendulum_QP_solver_v06, pendulum_QP_solver_re06, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C8, pendulum_QP_solver_z07, pendulum_QP_solver_D02, pendulum_QP_solver_z08, params->e8, pendulum_QP_solver_v07, pendulum_QP_solver_re07, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C9, pendulum_QP_solver_z08, pendulum_QP_solver_D02, pendulum_QP_solver_z09, params->e9, pendulum_QP_solver_v08, pendulum_QP_solver_re08, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C10, pendulum_QP_solver_z09, pendulum_QP_solver_D02, pendulum_QP_solver_z10, params->e10, pendulum_QP_solver_v09, pendulum_QP_solver_re09, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C11, pendulum_QP_solver_z10, pendulum_QP_solver_D02, pendulum_QP_solver_z11, params->e11, pendulum_QP_solver_v10, pendulum_QP_solver_re10, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C12, pendulum_QP_solver_z11, pendulum_QP_solver_D02, pendulum_QP_solver_z12, params->e12, pendulum_QP_solver_v11, pendulum_QP_solver_re11, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_10(params->C13, pendulum_QP_solver_z12, pendulum_QP_solver_D02, pendulum_QP_solver_z13, params->e13, pendulum_QP_solver_v12, pendulum_QP_solver_re12, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_3_10_3(params->C14, pendulum_QP_solver_z13, pendulum_QP_solver_D14, pendulum_QP_solver_z14, params->e14, pendulum_QP_solver_v13, pendulum_QP_solver_re13, &info->dgap, &info->res_eq);
pendulum_QP_solver_LA_DENSE_MTVM_5_10(params->C1, pendulum_QP_solver_v00, pendulum_QP_solver_grad_eq00);
pendulum_QP_solver_LA_DENSE_MTVM2_3_10_5(params->C2, pendulum_QP_solver_v01, pendulum_QP_solver_D01, pendulum_QP_solver_v00, pendulum_QP_solver_grad_eq01);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C3, pendulum_QP_solver_v02, pendulum_QP_solver_D02, pendulum_QP_solver_v01, pendulum_QP_solver_grad_eq02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C4, pendulum_QP_solver_v03, pendulum_QP_solver_D02, pendulum_QP_solver_v02, pendulum_QP_solver_grad_eq03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C5, pendulum_QP_solver_v04, pendulum_QP_solver_D02, pendulum_QP_solver_v03, pendulum_QP_solver_grad_eq04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C6, pendulum_QP_solver_v05, pendulum_QP_solver_D02, pendulum_QP_solver_v04, pendulum_QP_solver_grad_eq05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C7, pendulum_QP_solver_v06, pendulum_QP_solver_D02, pendulum_QP_solver_v05, pendulum_QP_solver_grad_eq06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C8, pendulum_QP_solver_v07, pendulum_QP_solver_D02, pendulum_QP_solver_v06, pendulum_QP_solver_grad_eq07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C9, pendulum_QP_solver_v08, pendulum_QP_solver_D02, pendulum_QP_solver_v07, pendulum_QP_solver_grad_eq08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C10, pendulum_QP_solver_v09, pendulum_QP_solver_D02, pendulum_QP_solver_v08, pendulum_QP_solver_grad_eq09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C11, pendulum_QP_solver_v10, pendulum_QP_solver_D02, pendulum_QP_solver_v09, pendulum_QP_solver_grad_eq10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C12, pendulum_QP_solver_v11, pendulum_QP_solver_D02, pendulum_QP_solver_v10, pendulum_QP_solver_grad_eq11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C13, pendulum_QP_solver_v12, pendulum_QP_solver_D02, pendulum_QP_solver_v11, pendulum_QP_solver_grad_eq12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C14, pendulum_QP_solver_v13, pendulum_QP_solver_D02, pendulum_QP_solver_v12, pendulum_QP_solver_grad_eq13);
pendulum_QP_solver_LA_DIAGZERO_MTVM_3_3(pendulum_QP_solver_D14, pendulum_QP_solver_v13, pendulum_QP_solver_grad_eq14);
info->res_ineq = 0;
pendulum_QP_solver_LA_VSUBADD3_10(params->lb1, pendulum_QP_solver_z00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_llb00, pendulum_QP_solver_slb00, pendulum_QP_solver_rilb00, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z00, pendulum_QP_solver_ubIdx00, params->ub1, pendulum_QP_solver_lub00, pendulum_QP_solver_sub00, pendulum_QP_solver_riub00, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb2, pendulum_QP_solver_z01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_llb01, pendulum_QP_solver_slb01, pendulum_QP_solver_rilb01, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z01, pendulum_QP_solver_ubIdx01, params->ub2, pendulum_QP_solver_lub01, pendulum_QP_solver_sub01, pendulum_QP_solver_riub01, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb3, pendulum_QP_solver_z02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_llb02, pendulum_QP_solver_slb02, pendulum_QP_solver_rilb02, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z02, pendulum_QP_solver_ubIdx02, params->ub3, pendulum_QP_solver_lub02, pendulum_QP_solver_sub02, pendulum_QP_solver_riub02, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb4, pendulum_QP_solver_z03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_llb03, pendulum_QP_solver_slb03, pendulum_QP_solver_rilb03, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z03, pendulum_QP_solver_ubIdx03, params->ub4, pendulum_QP_solver_lub03, pendulum_QP_solver_sub03, pendulum_QP_solver_riub03, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb5, pendulum_QP_solver_z04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_llb04, pendulum_QP_solver_slb04, pendulum_QP_solver_rilb04, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z04, pendulum_QP_solver_ubIdx04, params->ub5, pendulum_QP_solver_lub04, pendulum_QP_solver_sub04, pendulum_QP_solver_riub04, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb6, pendulum_QP_solver_z05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_llb05, pendulum_QP_solver_slb05, pendulum_QP_solver_rilb05, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z05, pendulum_QP_solver_ubIdx05, params->ub6, pendulum_QP_solver_lub05, pendulum_QP_solver_sub05, pendulum_QP_solver_riub05, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb7, pendulum_QP_solver_z06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_llb06, pendulum_QP_solver_slb06, pendulum_QP_solver_rilb06, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z06, pendulum_QP_solver_ubIdx06, params->ub7, pendulum_QP_solver_lub06, pendulum_QP_solver_sub06, pendulum_QP_solver_riub06, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb8, pendulum_QP_solver_z07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_llb07, pendulum_QP_solver_slb07, pendulum_QP_solver_rilb07, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z07, pendulum_QP_solver_ubIdx07, params->ub8, pendulum_QP_solver_lub07, pendulum_QP_solver_sub07, pendulum_QP_solver_riub07, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb9, pendulum_QP_solver_z08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_llb08, pendulum_QP_solver_slb08, pendulum_QP_solver_rilb08, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z08, pendulum_QP_solver_ubIdx08, params->ub9, pendulum_QP_solver_lub08, pendulum_QP_solver_sub08, pendulum_QP_solver_riub08, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb10, pendulum_QP_solver_z09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_llb09, pendulum_QP_solver_slb09, pendulum_QP_solver_rilb09, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z09, pendulum_QP_solver_ubIdx09, params->ub10, pendulum_QP_solver_lub09, pendulum_QP_solver_sub09, pendulum_QP_solver_riub09, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb11, pendulum_QP_solver_z10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_llb10, pendulum_QP_solver_slb10, pendulum_QP_solver_rilb10, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z10, pendulum_QP_solver_ubIdx10, params->ub11, pendulum_QP_solver_lub10, pendulum_QP_solver_sub10, pendulum_QP_solver_riub10, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb12, pendulum_QP_solver_z11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_llb11, pendulum_QP_solver_slb11, pendulum_QP_solver_rilb11, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z11, pendulum_QP_solver_ubIdx11, params->ub12, pendulum_QP_solver_lub11, pendulum_QP_solver_sub11, pendulum_QP_solver_riub11, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb13, pendulum_QP_solver_z12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_llb12, pendulum_QP_solver_slb12, pendulum_QP_solver_rilb12, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z12, pendulum_QP_solver_ubIdx12, params->ub13, pendulum_QP_solver_lub12, pendulum_QP_solver_sub12, pendulum_QP_solver_riub12, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_10(params->lb14, pendulum_QP_solver_z13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_llb13, pendulum_QP_solver_slb13, pendulum_QP_solver_rilb13, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_6(pendulum_QP_solver_z13, pendulum_QP_solver_ubIdx13, params->ub14, pendulum_QP_solver_lub13, pendulum_QP_solver_sub13, pendulum_QP_solver_riub13, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD3_3(params->lb15, pendulum_QP_solver_z14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_llb14, pendulum_QP_solver_slb14, pendulum_QP_solver_rilb14, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_VSUBADD2_3(pendulum_QP_solver_z14, pendulum_QP_solver_ubIdx14, params->ub15, pendulum_QP_solver_lub14, pendulum_QP_solver_sub14, pendulum_QP_solver_riub14, &info->dgap, &info->res_ineq);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub00, pendulum_QP_solver_sub00, pendulum_QP_solver_riub00, pendulum_QP_solver_llb00, pendulum_QP_solver_slb00, pendulum_QP_solver_rilb00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_ubIdx00, pendulum_QP_solver_grad_ineq00, pendulum_QP_solver_lubbysub00, pendulum_QP_solver_llbbyslb00);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub01, pendulum_QP_solver_sub01, pendulum_QP_solver_riub01, pendulum_QP_solver_llb01, pendulum_QP_solver_slb01, pendulum_QP_solver_rilb01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_ubIdx01, pendulum_QP_solver_grad_ineq01, pendulum_QP_solver_lubbysub01, pendulum_QP_solver_llbbyslb01);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub02, pendulum_QP_solver_sub02, pendulum_QP_solver_riub02, pendulum_QP_solver_llb02, pendulum_QP_solver_slb02, pendulum_QP_solver_rilb02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_ubIdx02, pendulum_QP_solver_grad_ineq02, pendulum_QP_solver_lubbysub02, pendulum_QP_solver_llbbyslb02);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub03, pendulum_QP_solver_sub03, pendulum_QP_solver_riub03, pendulum_QP_solver_llb03, pendulum_QP_solver_slb03, pendulum_QP_solver_rilb03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_ubIdx03, pendulum_QP_solver_grad_ineq03, pendulum_QP_solver_lubbysub03, pendulum_QP_solver_llbbyslb03);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub04, pendulum_QP_solver_sub04, pendulum_QP_solver_riub04, pendulum_QP_solver_llb04, pendulum_QP_solver_slb04, pendulum_QP_solver_rilb04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_ubIdx04, pendulum_QP_solver_grad_ineq04, pendulum_QP_solver_lubbysub04, pendulum_QP_solver_llbbyslb04);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub05, pendulum_QP_solver_sub05, pendulum_QP_solver_riub05, pendulum_QP_solver_llb05, pendulum_QP_solver_slb05, pendulum_QP_solver_rilb05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_ubIdx05, pendulum_QP_solver_grad_ineq05, pendulum_QP_solver_lubbysub05, pendulum_QP_solver_llbbyslb05);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub06, pendulum_QP_solver_sub06, pendulum_QP_solver_riub06, pendulum_QP_solver_llb06, pendulum_QP_solver_slb06, pendulum_QP_solver_rilb06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_ubIdx06, pendulum_QP_solver_grad_ineq06, pendulum_QP_solver_lubbysub06, pendulum_QP_solver_llbbyslb06);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub07, pendulum_QP_solver_sub07, pendulum_QP_solver_riub07, pendulum_QP_solver_llb07, pendulum_QP_solver_slb07, pendulum_QP_solver_rilb07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_ubIdx07, pendulum_QP_solver_grad_ineq07, pendulum_QP_solver_lubbysub07, pendulum_QP_solver_llbbyslb07);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub08, pendulum_QP_solver_sub08, pendulum_QP_solver_riub08, pendulum_QP_solver_llb08, pendulum_QP_solver_slb08, pendulum_QP_solver_rilb08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_ubIdx08, pendulum_QP_solver_grad_ineq08, pendulum_QP_solver_lubbysub08, pendulum_QP_solver_llbbyslb08);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub09, pendulum_QP_solver_sub09, pendulum_QP_solver_riub09, pendulum_QP_solver_llb09, pendulum_QP_solver_slb09, pendulum_QP_solver_rilb09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_ubIdx09, pendulum_QP_solver_grad_ineq09, pendulum_QP_solver_lubbysub09, pendulum_QP_solver_llbbyslb09);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub10, pendulum_QP_solver_sub10, pendulum_QP_solver_riub10, pendulum_QP_solver_llb10, pendulum_QP_solver_slb10, pendulum_QP_solver_rilb10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_ubIdx10, pendulum_QP_solver_grad_ineq10, pendulum_QP_solver_lubbysub10, pendulum_QP_solver_llbbyslb10);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub11, pendulum_QP_solver_sub11, pendulum_QP_solver_riub11, pendulum_QP_solver_llb11, pendulum_QP_solver_slb11, pendulum_QP_solver_rilb11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_ubIdx11, pendulum_QP_solver_grad_ineq11, pendulum_QP_solver_lubbysub11, pendulum_QP_solver_llbbyslb11);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub12, pendulum_QP_solver_sub12, pendulum_QP_solver_riub12, pendulum_QP_solver_llb12, pendulum_QP_solver_slb12, pendulum_QP_solver_rilb12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_ubIdx12, pendulum_QP_solver_grad_ineq12, pendulum_QP_solver_lubbysub12, pendulum_QP_solver_llbbyslb12);
pendulum_QP_solver_LA_INEQ_B_GRAD_10_10_6(pendulum_QP_solver_lub13, pendulum_QP_solver_sub13, pendulum_QP_solver_riub13, pendulum_QP_solver_llb13, pendulum_QP_solver_slb13, pendulum_QP_solver_rilb13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_ubIdx13, pendulum_QP_solver_grad_ineq13, pendulum_QP_solver_lubbysub13, pendulum_QP_solver_llbbyslb13);
pendulum_QP_solver_LA_INEQ_B_GRAD_3_3_3(pendulum_QP_solver_lub14, pendulum_QP_solver_sub14, pendulum_QP_solver_riub14, pendulum_QP_solver_llb14, pendulum_QP_solver_slb14, pendulum_QP_solver_rilb14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_ubIdx14, pendulum_QP_solver_grad_ineq14, pendulum_QP_solver_lubbysub14, pendulum_QP_solver_llbbyslb14);
info->dobj = info->pobj - info->dgap;
info->rdgap = info->pobj ? info->dgap / info->pobj : 1e6;
if( info->rdgap < 0 ) info->rdgap = -info->rdgap;
if( info->mu < pendulum_QP_solver_SET_ACC_KKTCOMPL
    && (info->rdgap < pendulum_QP_solver_SET_ACC_RDGAP || info->dgap < pendulum_QP_solver_SET_ACC_KKTCOMPL)
    && info->res_eq < pendulum_QP_solver_SET_ACC_RESEQ
    && info->res_ineq < pendulum_QP_solver_SET_ACC_RESINEQ ){
exitcode = pendulum_QP_solver_OPTIMAL; break; }
if( info->it == pendulum_QP_solver_SET_MAXIT ){
exitcode = pendulum_QP_solver_MAXITREACHED; break; }
pendulum_QP_solver_LA_VVADD3_143(pendulum_QP_solver_grad_cost, pendulum_QP_solver_grad_eq, pendulum_QP_solver_grad_ineq, pendulum_QP_solver_rd);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_lubbysub00, pendulum_QP_solver_ubIdx00, pendulum_QP_solver_Phi00);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_10(pendulum_QP_solver_Phi00, params->C1, pendulum_QP_solver_V00);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi00, pendulum_QP_solver_rd00, pendulum_QP_solver_Lbyrd00);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_lubbysub01, pendulum_QP_solver_ubIdx01, pendulum_QP_solver_Phi01);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi01, params->C2, pendulum_QP_solver_V01);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_10(pendulum_QP_solver_Phi01, pendulum_QP_solver_D01, pendulum_QP_solver_W01);
pendulum_QP_solver_LA_DENSE_MMTM_5_10_3(pendulum_QP_solver_W01, pendulum_QP_solver_V01, pendulum_QP_solver_Ysd01);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi01, pendulum_QP_solver_rd01, pendulum_QP_solver_Lbyrd01);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_lubbysub02, pendulum_QP_solver_ubIdx02, pendulum_QP_solver_Phi02);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi02, params->C3, pendulum_QP_solver_V02);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi02, pendulum_QP_solver_D02, pendulum_QP_solver_W02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W02, pendulum_QP_solver_V02, pendulum_QP_solver_Ysd02);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi02, pendulum_QP_solver_rd02, pendulum_QP_solver_Lbyrd02);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_lubbysub03, pendulum_QP_solver_ubIdx03, pendulum_QP_solver_Phi03);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi03, params->C4, pendulum_QP_solver_V03);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi03, pendulum_QP_solver_D02, pendulum_QP_solver_W03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W03, pendulum_QP_solver_V03, pendulum_QP_solver_Ysd03);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi03, pendulum_QP_solver_rd03, pendulum_QP_solver_Lbyrd03);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_lubbysub04, pendulum_QP_solver_ubIdx04, pendulum_QP_solver_Phi04);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi04, params->C5, pendulum_QP_solver_V04);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi04, pendulum_QP_solver_D02, pendulum_QP_solver_W04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W04, pendulum_QP_solver_V04, pendulum_QP_solver_Ysd04);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi04, pendulum_QP_solver_rd04, pendulum_QP_solver_Lbyrd04);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_lubbysub05, pendulum_QP_solver_ubIdx05, pendulum_QP_solver_Phi05);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi05, params->C6, pendulum_QP_solver_V05);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi05, pendulum_QP_solver_D02, pendulum_QP_solver_W05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W05, pendulum_QP_solver_V05, pendulum_QP_solver_Ysd05);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi05, pendulum_QP_solver_rd05, pendulum_QP_solver_Lbyrd05);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_lubbysub06, pendulum_QP_solver_ubIdx06, pendulum_QP_solver_Phi06);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi06, params->C7, pendulum_QP_solver_V06);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi06, pendulum_QP_solver_D02, pendulum_QP_solver_W06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W06, pendulum_QP_solver_V06, pendulum_QP_solver_Ysd06);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi06, pendulum_QP_solver_rd06, pendulum_QP_solver_Lbyrd06);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_lubbysub07, pendulum_QP_solver_ubIdx07, pendulum_QP_solver_Phi07);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi07, params->C8, pendulum_QP_solver_V07);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi07, pendulum_QP_solver_D02, pendulum_QP_solver_W07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W07, pendulum_QP_solver_V07, pendulum_QP_solver_Ysd07);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi07, pendulum_QP_solver_rd07, pendulum_QP_solver_Lbyrd07);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_lubbysub08, pendulum_QP_solver_ubIdx08, pendulum_QP_solver_Phi08);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi08, params->C9, pendulum_QP_solver_V08);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi08, pendulum_QP_solver_D02, pendulum_QP_solver_W08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W08, pendulum_QP_solver_V08, pendulum_QP_solver_Ysd08);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi08, pendulum_QP_solver_rd08, pendulum_QP_solver_Lbyrd08);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_lubbysub09, pendulum_QP_solver_ubIdx09, pendulum_QP_solver_Phi09);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi09, params->C10, pendulum_QP_solver_V09);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi09, pendulum_QP_solver_D02, pendulum_QP_solver_W09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W09, pendulum_QP_solver_V09, pendulum_QP_solver_Ysd09);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi09, pendulum_QP_solver_rd09, pendulum_QP_solver_Lbyrd09);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_lubbysub10, pendulum_QP_solver_ubIdx10, pendulum_QP_solver_Phi10);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi10, params->C11, pendulum_QP_solver_V10);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi10, pendulum_QP_solver_D02, pendulum_QP_solver_W10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W10, pendulum_QP_solver_V10, pendulum_QP_solver_Ysd10);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi10, pendulum_QP_solver_rd10, pendulum_QP_solver_Lbyrd10);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_lubbysub11, pendulum_QP_solver_ubIdx11, pendulum_QP_solver_Phi11);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi11, params->C12, pendulum_QP_solver_V11);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi11, pendulum_QP_solver_D02, pendulum_QP_solver_W11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W11, pendulum_QP_solver_V11, pendulum_QP_solver_Ysd11);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi11, pendulum_QP_solver_rd11, pendulum_QP_solver_Lbyrd11);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_lubbysub12, pendulum_QP_solver_ubIdx12, pendulum_QP_solver_Phi12);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi12, params->C13, pendulum_QP_solver_V12);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi12, pendulum_QP_solver_D02, pendulum_QP_solver_W12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W12, pendulum_QP_solver_V12, pendulum_QP_solver_Ysd12);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi12, pendulum_QP_solver_rd12, pendulum_QP_solver_Lbyrd12);
pendulum_QP_solver_LA_DIAG_CHOL_LBUB_10_10_6(pendulum_QP_solver_H00, pendulum_QP_solver_llbbyslb13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_lubbysub13, pendulum_QP_solver_ubIdx13, pendulum_QP_solver_Phi13);
pendulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_3_10(pendulum_QP_solver_Phi13, params->C14, pendulum_QP_solver_V13);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_10(pendulum_QP_solver_Phi13, pendulum_QP_solver_D02, pendulum_QP_solver_W13);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_3_10_3(pendulum_QP_solver_W13, pendulum_QP_solver_V13, pendulum_QP_solver_Ysd13);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi13, pendulum_QP_solver_rd13, pendulum_QP_solver_Lbyrd13);
pendulum_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_3_3_3(pendulum_QP_solver_H14, pendulum_QP_solver_llbbyslb14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_lubbysub14, pendulum_QP_solver_ubIdx14, pendulum_QP_solver_Phi14);
pendulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Phi14, pendulum_QP_solver_D14, pendulum_QP_solver_W14);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_3(pendulum_QP_solver_Phi14, pendulum_QP_solver_rd14, pendulum_QP_solver_Lbyrd14);
pendulum_QP_solver_LA_DENSE_MMT2_5_10_10(pendulum_QP_solver_V00, pendulum_QP_solver_W01, pendulum_QP_solver_Yd00);
pendulum_QP_solver_LA_DENSE_MVMSUB2_5_10_10(pendulum_QP_solver_V00, pendulum_QP_solver_Lbyrd00, pendulum_QP_solver_W01, pendulum_QP_solver_Lbyrd01, pendulum_QP_solver_re00, pendulum_QP_solver_beta00);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V01, pendulum_QP_solver_W02, pendulum_QP_solver_Yd01);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V01, pendulum_QP_solver_Lbyrd01, pendulum_QP_solver_W02, pendulum_QP_solver_Lbyrd02, pendulum_QP_solver_re01, pendulum_QP_solver_beta01);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V02, pendulum_QP_solver_W03, pendulum_QP_solver_Yd02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V02, pendulum_QP_solver_Lbyrd02, pendulum_QP_solver_W03, pendulum_QP_solver_Lbyrd03, pendulum_QP_solver_re02, pendulum_QP_solver_beta02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V03, pendulum_QP_solver_W04, pendulum_QP_solver_Yd03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V03, pendulum_QP_solver_Lbyrd03, pendulum_QP_solver_W04, pendulum_QP_solver_Lbyrd04, pendulum_QP_solver_re03, pendulum_QP_solver_beta03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V04, pendulum_QP_solver_W05, pendulum_QP_solver_Yd04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V04, pendulum_QP_solver_Lbyrd04, pendulum_QP_solver_W05, pendulum_QP_solver_Lbyrd05, pendulum_QP_solver_re04, pendulum_QP_solver_beta04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V05, pendulum_QP_solver_W06, pendulum_QP_solver_Yd05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V05, pendulum_QP_solver_Lbyrd05, pendulum_QP_solver_W06, pendulum_QP_solver_Lbyrd06, pendulum_QP_solver_re05, pendulum_QP_solver_beta05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V06, pendulum_QP_solver_W07, pendulum_QP_solver_Yd06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V06, pendulum_QP_solver_Lbyrd06, pendulum_QP_solver_W07, pendulum_QP_solver_Lbyrd07, pendulum_QP_solver_re06, pendulum_QP_solver_beta06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V07, pendulum_QP_solver_W08, pendulum_QP_solver_Yd07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V07, pendulum_QP_solver_Lbyrd07, pendulum_QP_solver_W08, pendulum_QP_solver_Lbyrd08, pendulum_QP_solver_re07, pendulum_QP_solver_beta07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V08, pendulum_QP_solver_W09, pendulum_QP_solver_Yd08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V08, pendulum_QP_solver_Lbyrd08, pendulum_QP_solver_W09, pendulum_QP_solver_Lbyrd09, pendulum_QP_solver_re08, pendulum_QP_solver_beta08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V09, pendulum_QP_solver_W10, pendulum_QP_solver_Yd09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V09, pendulum_QP_solver_Lbyrd09, pendulum_QP_solver_W10, pendulum_QP_solver_Lbyrd10, pendulum_QP_solver_re09, pendulum_QP_solver_beta09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V10, pendulum_QP_solver_W11, pendulum_QP_solver_Yd10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V10, pendulum_QP_solver_Lbyrd10, pendulum_QP_solver_W11, pendulum_QP_solver_Lbyrd11, pendulum_QP_solver_re10, pendulum_QP_solver_beta10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V11, pendulum_QP_solver_W12, pendulum_QP_solver_Yd11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V11, pendulum_QP_solver_Lbyrd11, pendulum_QP_solver_W12, pendulum_QP_solver_Lbyrd12, pendulum_QP_solver_re11, pendulum_QP_solver_beta11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_10(pendulum_QP_solver_V12, pendulum_QP_solver_W13, pendulum_QP_solver_Yd12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_10(pendulum_QP_solver_V12, pendulum_QP_solver_Lbyrd12, pendulum_QP_solver_W13, pendulum_QP_solver_Lbyrd13, pendulum_QP_solver_re12, pendulum_QP_solver_beta12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_3_10_3(pendulum_QP_solver_V13, pendulum_QP_solver_W14, pendulum_QP_solver_Yd13);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_3_10_3(pendulum_QP_solver_V13, pendulum_QP_solver_Lbyrd13, pendulum_QP_solver_W14, pendulum_QP_solver_Lbyrd14, pendulum_QP_solver_re13, pendulum_QP_solver_beta13);
pendulum_QP_solver_LA_DENSE_CHOL_5(pendulum_QP_solver_Yd00, pendulum_QP_solver_Ld00);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_5(pendulum_QP_solver_Ld00, pendulum_QP_solver_beta00, pendulum_QP_solver_yy00);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_5(pendulum_QP_solver_Ld00, pendulum_QP_solver_Ysd01, pendulum_QP_solver_Lsd01);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_5(pendulum_QP_solver_Lsd01, pendulum_QP_solver_Yd01);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd01, pendulum_QP_solver_Ld01);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_5(pendulum_QP_solver_Lsd01, pendulum_QP_solver_yy00, pendulum_QP_solver_beta01, pendulum_QP_solver_bmy01);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld01, pendulum_QP_solver_bmy01, pendulum_QP_solver_yy01);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld01, pendulum_QP_solver_Ysd02, pendulum_QP_solver_Lsd02);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd02, pendulum_QP_solver_Yd02);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd02, pendulum_QP_solver_Ld02);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd02, pendulum_QP_solver_yy01, pendulum_QP_solver_beta02, pendulum_QP_solver_bmy02);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld02, pendulum_QP_solver_bmy02, pendulum_QP_solver_yy02);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld02, pendulum_QP_solver_Ysd03, pendulum_QP_solver_Lsd03);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd03, pendulum_QP_solver_Yd03);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd03, pendulum_QP_solver_Ld03);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd03, pendulum_QP_solver_yy02, pendulum_QP_solver_beta03, pendulum_QP_solver_bmy03);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld03, pendulum_QP_solver_bmy03, pendulum_QP_solver_yy03);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld03, pendulum_QP_solver_Ysd04, pendulum_QP_solver_Lsd04);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd04, pendulum_QP_solver_Yd04);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd04, pendulum_QP_solver_Ld04);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd04, pendulum_QP_solver_yy03, pendulum_QP_solver_beta04, pendulum_QP_solver_bmy04);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld04, pendulum_QP_solver_bmy04, pendulum_QP_solver_yy04);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld04, pendulum_QP_solver_Ysd05, pendulum_QP_solver_Lsd05);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd05, pendulum_QP_solver_Yd05);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd05, pendulum_QP_solver_Ld05);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd05, pendulum_QP_solver_yy04, pendulum_QP_solver_beta05, pendulum_QP_solver_bmy05);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld05, pendulum_QP_solver_bmy05, pendulum_QP_solver_yy05);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld05, pendulum_QP_solver_Ysd06, pendulum_QP_solver_Lsd06);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd06, pendulum_QP_solver_Yd06);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd06, pendulum_QP_solver_Ld06);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd06, pendulum_QP_solver_yy05, pendulum_QP_solver_beta06, pendulum_QP_solver_bmy06);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld06, pendulum_QP_solver_bmy06, pendulum_QP_solver_yy06);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld06, pendulum_QP_solver_Ysd07, pendulum_QP_solver_Lsd07);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd07, pendulum_QP_solver_Yd07);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd07, pendulum_QP_solver_Ld07);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd07, pendulum_QP_solver_yy06, pendulum_QP_solver_beta07, pendulum_QP_solver_bmy07);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld07, pendulum_QP_solver_bmy07, pendulum_QP_solver_yy07);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld07, pendulum_QP_solver_Ysd08, pendulum_QP_solver_Lsd08);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd08, pendulum_QP_solver_Yd08);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd08, pendulum_QP_solver_Ld08);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd08, pendulum_QP_solver_yy07, pendulum_QP_solver_beta08, pendulum_QP_solver_bmy08);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld08, pendulum_QP_solver_bmy08, pendulum_QP_solver_yy08);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld08, pendulum_QP_solver_Ysd09, pendulum_QP_solver_Lsd09);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd09, pendulum_QP_solver_Yd09);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd09, pendulum_QP_solver_Ld09);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd09, pendulum_QP_solver_yy08, pendulum_QP_solver_beta09, pendulum_QP_solver_bmy09);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld09, pendulum_QP_solver_bmy09, pendulum_QP_solver_yy09);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld09, pendulum_QP_solver_Ysd10, pendulum_QP_solver_Lsd10);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd10, pendulum_QP_solver_Yd10);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd10, pendulum_QP_solver_Ld10);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd10, pendulum_QP_solver_yy09, pendulum_QP_solver_beta10, pendulum_QP_solver_bmy10);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld10, pendulum_QP_solver_bmy10, pendulum_QP_solver_yy10);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld10, pendulum_QP_solver_Ysd11, pendulum_QP_solver_Lsd11);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd11, pendulum_QP_solver_Yd11);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd11, pendulum_QP_solver_Ld11);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd11, pendulum_QP_solver_yy10, pendulum_QP_solver_beta11, pendulum_QP_solver_bmy11);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld11, pendulum_QP_solver_bmy11, pendulum_QP_solver_yy11);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld11, pendulum_QP_solver_Ysd12, pendulum_QP_solver_Lsd12);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd12, pendulum_QP_solver_Yd12);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd12, pendulum_QP_solver_Ld12);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd12, pendulum_QP_solver_yy11, pendulum_QP_solver_beta12, pendulum_QP_solver_bmy12);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld12, pendulum_QP_solver_bmy12, pendulum_QP_solver_yy12);
pendulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_3_3(pendulum_QP_solver_Ld12, pendulum_QP_solver_Ysd13, pendulum_QP_solver_Lsd13);
pendulum_QP_solver_LA_DENSE_MMTSUB_3_3(pendulum_QP_solver_Lsd13, pendulum_QP_solver_Yd13);
pendulum_QP_solver_LA_DENSE_CHOL_3(pendulum_QP_solver_Yd13, pendulum_QP_solver_Ld13);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd13, pendulum_QP_solver_yy12, pendulum_QP_solver_beta13, pendulum_QP_solver_bmy13);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld13, pendulum_QP_solver_bmy13, pendulum_QP_solver_yy13);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld13, pendulum_QP_solver_yy13, pendulum_QP_solver_dvaff13);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd13, pendulum_QP_solver_dvaff13, pendulum_QP_solver_yy12, pendulum_QP_solver_bmy12);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld12, pendulum_QP_solver_bmy12, pendulum_QP_solver_dvaff12);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd12, pendulum_QP_solver_dvaff12, pendulum_QP_solver_yy11, pendulum_QP_solver_bmy11);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld11, pendulum_QP_solver_bmy11, pendulum_QP_solver_dvaff11);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd11, pendulum_QP_solver_dvaff11, pendulum_QP_solver_yy10, pendulum_QP_solver_bmy10);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld10, pendulum_QP_solver_bmy10, pendulum_QP_solver_dvaff10);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd10, pendulum_QP_solver_dvaff10, pendulum_QP_solver_yy09, pendulum_QP_solver_bmy09);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld09, pendulum_QP_solver_bmy09, pendulum_QP_solver_dvaff09);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd09, pendulum_QP_solver_dvaff09, pendulum_QP_solver_yy08, pendulum_QP_solver_bmy08);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld08, pendulum_QP_solver_bmy08, pendulum_QP_solver_dvaff08);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd08, pendulum_QP_solver_dvaff08, pendulum_QP_solver_yy07, pendulum_QP_solver_bmy07);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld07, pendulum_QP_solver_bmy07, pendulum_QP_solver_dvaff07);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd07, pendulum_QP_solver_dvaff07, pendulum_QP_solver_yy06, pendulum_QP_solver_bmy06);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld06, pendulum_QP_solver_bmy06, pendulum_QP_solver_dvaff06);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd06, pendulum_QP_solver_dvaff06, pendulum_QP_solver_yy05, pendulum_QP_solver_bmy05);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld05, pendulum_QP_solver_bmy05, pendulum_QP_solver_dvaff05);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd05, pendulum_QP_solver_dvaff05, pendulum_QP_solver_yy04, pendulum_QP_solver_bmy04);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld04, pendulum_QP_solver_bmy04, pendulum_QP_solver_dvaff04);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd04, pendulum_QP_solver_dvaff04, pendulum_QP_solver_yy03, pendulum_QP_solver_bmy03);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld03, pendulum_QP_solver_bmy03, pendulum_QP_solver_dvaff03);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd03, pendulum_QP_solver_dvaff03, pendulum_QP_solver_yy02, pendulum_QP_solver_bmy02);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld02, pendulum_QP_solver_bmy02, pendulum_QP_solver_dvaff02);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd02, pendulum_QP_solver_dvaff02, pendulum_QP_solver_yy01, pendulum_QP_solver_bmy01);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld01, pendulum_QP_solver_bmy01, pendulum_QP_solver_dvaff01);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_5(pendulum_QP_solver_Lsd01, pendulum_QP_solver_dvaff01, pendulum_QP_solver_yy00, pendulum_QP_solver_bmy00);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_5(pendulum_QP_solver_Ld00, pendulum_QP_solver_bmy00, pendulum_QP_solver_dvaff00);
pendulum_QP_solver_LA_DENSE_MTVM_5_10(params->C1, pendulum_QP_solver_dvaff00, pendulum_QP_solver_grad_eq00);
pendulum_QP_solver_LA_DENSE_MTVM2_3_10_5(params->C2, pendulum_QP_solver_dvaff01, pendulum_QP_solver_D01, pendulum_QP_solver_dvaff00, pendulum_QP_solver_grad_eq01);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C3, pendulum_QP_solver_dvaff02, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff01, pendulum_QP_solver_grad_eq02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C4, pendulum_QP_solver_dvaff03, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff02, pendulum_QP_solver_grad_eq03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C5, pendulum_QP_solver_dvaff04, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff03, pendulum_QP_solver_grad_eq04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C6, pendulum_QP_solver_dvaff05, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff04, pendulum_QP_solver_grad_eq05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C7, pendulum_QP_solver_dvaff06, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff05, pendulum_QP_solver_grad_eq06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C8, pendulum_QP_solver_dvaff07, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff06, pendulum_QP_solver_grad_eq07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C9, pendulum_QP_solver_dvaff08, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff07, pendulum_QP_solver_grad_eq08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C10, pendulum_QP_solver_dvaff09, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff08, pendulum_QP_solver_grad_eq09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C11, pendulum_QP_solver_dvaff10, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff09, pendulum_QP_solver_grad_eq10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C12, pendulum_QP_solver_dvaff11, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff10, pendulum_QP_solver_grad_eq11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C13, pendulum_QP_solver_dvaff12, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff11, pendulum_QP_solver_grad_eq12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C14, pendulum_QP_solver_dvaff13, pendulum_QP_solver_D02, pendulum_QP_solver_dvaff12, pendulum_QP_solver_grad_eq13);
pendulum_QP_solver_LA_DIAGZERO_MTVM_3_3(pendulum_QP_solver_D14, pendulum_QP_solver_dvaff13, pendulum_QP_solver_grad_eq14);
pendulum_QP_solver_LA_VSUB2_143(pendulum_QP_solver_rd, pendulum_QP_solver_grad_eq, pendulum_QP_solver_rd);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi00, pendulum_QP_solver_rd00, pendulum_QP_solver_dzaff00);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi01, pendulum_QP_solver_rd01, pendulum_QP_solver_dzaff01);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi02, pendulum_QP_solver_rd02, pendulum_QP_solver_dzaff02);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi03, pendulum_QP_solver_rd03, pendulum_QP_solver_dzaff03);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi04, pendulum_QP_solver_rd04, pendulum_QP_solver_dzaff04);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi05, pendulum_QP_solver_rd05, pendulum_QP_solver_dzaff05);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi06, pendulum_QP_solver_rd06, pendulum_QP_solver_dzaff06);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi07, pendulum_QP_solver_rd07, pendulum_QP_solver_dzaff07);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi08, pendulum_QP_solver_rd08, pendulum_QP_solver_dzaff08);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi09, pendulum_QP_solver_rd09, pendulum_QP_solver_dzaff09);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi10, pendulum_QP_solver_rd10, pendulum_QP_solver_dzaff10);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi11, pendulum_QP_solver_rd11, pendulum_QP_solver_dzaff11);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi12, pendulum_QP_solver_rd12, pendulum_QP_solver_dzaff12);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi13, pendulum_QP_solver_rd13, pendulum_QP_solver_dzaff13);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_3(pendulum_QP_solver_Phi14, pendulum_QP_solver_rd14, pendulum_QP_solver_dzaff14);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_rilb00, pendulum_QP_solver_dslbaff00);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb00, pendulum_QP_solver_dslbaff00, pendulum_QP_solver_llb00, pendulum_QP_solver_dllbaff00);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub00, pendulum_QP_solver_dzaff00, pendulum_QP_solver_ubIdx00, pendulum_QP_solver_dsubaff00);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub00, pendulum_QP_solver_dsubaff00, pendulum_QP_solver_lub00, pendulum_QP_solver_dlubaff00);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_rilb01, pendulum_QP_solver_dslbaff01);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb01, pendulum_QP_solver_dslbaff01, pendulum_QP_solver_llb01, pendulum_QP_solver_dllbaff01);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub01, pendulum_QP_solver_dzaff01, pendulum_QP_solver_ubIdx01, pendulum_QP_solver_dsubaff01);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub01, pendulum_QP_solver_dsubaff01, pendulum_QP_solver_lub01, pendulum_QP_solver_dlubaff01);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_rilb02, pendulum_QP_solver_dslbaff02);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb02, pendulum_QP_solver_dslbaff02, pendulum_QP_solver_llb02, pendulum_QP_solver_dllbaff02);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub02, pendulum_QP_solver_dzaff02, pendulum_QP_solver_ubIdx02, pendulum_QP_solver_dsubaff02);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub02, pendulum_QP_solver_dsubaff02, pendulum_QP_solver_lub02, pendulum_QP_solver_dlubaff02);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_rilb03, pendulum_QP_solver_dslbaff03);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb03, pendulum_QP_solver_dslbaff03, pendulum_QP_solver_llb03, pendulum_QP_solver_dllbaff03);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub03, pendulum_QP_solver_dzaff03, pendulum_QP_solver_ubIdx03, pendulum_QP_solver_dsubaff03);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub03, pendulum_QP_solver_dsubaff03, pendulum_QP_solver_lub03, pendulum_QP_solver_dlubaff03);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_rilb04, pendulum_QP_solver_dslbaff04);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb04, pendulum_QP_solver_dslbaff04, pendulum_QP_solver_llb04, pendulum_QP_solver_dllbaff04);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub04, pendulum_QP_solver_dzaff04, pendulum_QP_solver_ubIdx04, pendulum_QP_solver_dsubaff04);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub04, pendulum_QP_solver_dsubaff04, pendulum_QP_solver_lub04, pendulum_QP_solver_dlubaff04);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_rilb05, pendulum_QP_solver_dslbaff05);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb05, pendulum_QP_solver_dslbaff05, pendulum_QP_solver_llb05, pendulum_QP_solver_dllbaff05);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub05, pendulum_QP_solver_dzaff05, pendulum_QP_solver_ubIdx05, pendulum_QP_solver_dsubaff05);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub05, pendulum_QP_solver_dsubaff05, pendulum_QP_solver_lub05, pendulum_QP_solver_dlubaff05);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_rilb06, pendulum_QP_solver_dslbaff06);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb06, pendulum_QP_solver_dslbaff06, pendulum_QP_solver_llb06, pendulum_QP_solver_dllbaff06);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub06, pendulum_QP_solver_dzaff06, pendulum_QP_solver_ubIdx06, pendulum_QP_solver_dsubaff06);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub06, pendulum_QP_solver_dsubaff06, pendulum_QP_solver_lub06, pendulum_QP_solver_dlubaff06);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_rilb07, pendulum_QP_solver_dslbaff07);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb07, pendulum_QP_solver_dslbaff07, pendulum_QP_solver_llb07, pendulum_QP_solver_dllbaff07);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub07, pendulum_QP_solver_dzaff07, pendulum_QP_solver_ubIdx07, pendulum_QP_solver_dsubaff07);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub07, pendulum_QP_solver_dsubaff07, pendulum_QP_solver_lub07, pendulum_QP_solver_dlubaff07);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_rilb08, pendulum_QP_solver_dslbaff08);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb08, pendulum_QP_solver_dslbaff08, pendulum_QP_solver_llb08, pendulum_QP_solver_dllbaff08);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub08, pendulum_QP_solver_dzaff08, pendulum_QP_solver_ubIdx08, pendulum_QP_solver_dsubaff08);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub08, pendulum_QP_solver_dsubaff08, pendulum_QP_solver_lub08, pendulum_QP_solver_dlubaff08);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_rilb09, pendulum_QP_solver_dslbaff09);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb09, pendulum_QP_solver_dslbaff09, pendulum_QP_solver_llb09, pendulum_QP_solver_dllbaff09);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub09, pendulum_QP_solver_dzaff09, pendulum_QP_solver_ubIdx09, pendulum_QP_solver_dsubaff09);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub09, pendulum_QP_solver_dsubaff09, pendulum_QP_solver_lub09, pendulum_QP_solver_dlubaff09);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_rilb10, pendulum_QP_solver_dslbaff10);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb10, pendulum_QP_solver_dslbaff10, pendulum_QP_solver_llb10, pendulum_QP_solver_dllbaff10);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub10, pendulum_QP_solver_dzaff10, pendulum_QP_solver_ubIdx10, pendulum_QP_solver_dsubaff10);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub10, pendulum_QP_solver_dsubaff10, pendulum_QP_solver_lub10, pendulum_QP_solver_dlubaff10);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_rilb11, pendulum_QP_solver_dslbaff11);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb11, pendulum_QP_solver_dslbaff11, pendulum_QP_solver_llb11, pendulum_QP_solver_dllbaff11);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub11, pendulum_QP_solver_dzaff11, pendulum_QP_solver_ubIdx11, pendulum_QP_solver_dsubaff11);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub11, pendulum_QP_solver_dsubaff11, pendulum_QP_solver_lub11, pendulum_QP_solver_dlubaff11);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_rilb12, pendulum_QP_solver_dslbaff12);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb12, pendulum_QP_solver_dslbaff12, pendulum_QP_solver_llb12, pendulum_QP_solver_dllbaff12);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub12, pendulum_QP_solver_dzaff12, pendulum_QP_solver_ubIdx12, pendulum_QP_solver_dsubaff12);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub12, pendulum_QP_solver_dsubaff12, pendulum_QP_solver_lub12, pendulum_QP_solver_dlubaff12);
pendulum_QP_solver_LA_VSUB_INDEXED_10(pendulum_QP_solver_dzaff13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_rilb13, pendulum_QP_solver_dslbaff13);
pendulum_QP_solver_LA_VSUB3_10(pendulum_QP_solver_llbbyslb13, pendulum_QP_solver_dslbaff13, pendulum_QP_solver_llb13, pendulum_QP_solver_dllbaff13);
pendulum_QP_solver_LA_VSUB2_INDEXED_6(pendulum_QP_solver_riub13, pendulum_QP_solver_dzaff13, pendulum_QP_solver_ubIdx13, pendulum_QP_solver_dsubaff13);
pendulum_QP_solver_LA_VSUB3_6(pendulum_QP_solver_lubbysub13, pendulum_QP_solver_dsubaff13, pendulum_QP_solver_lub13, pendulum_QP_solver_dlubaff13);
pendulum_QP_solver_LA_VSUB_INDEXED_3(pendulum_QP_solver_dzaff14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_rilb14, pendulum_QP_solver_dslbaff14);
pendulum_QP_solver_LA_VSUB3_3(pendulum_QP_solver_llbbyslb14, pendulum_QP_solver_dslbaff14, pendulum_QP_solver_llb14, pendulum_QP_solver_dllbaff14);
pendulum_QP_solver_LA_VSUB2_INDEXED_3(pendulum_QP_solver_riub14, pendulum_QP_solver_dzaff14, pendulum_QP_solver_ubIdx14, pendulum_QP_solver_dsubaff14);
pendulum_QP_solver_LA_VSUB3_3(pendulum_QP_solver_lubbysub14, pendulum_QP_solver_dsubaff14, pendulum_QP_solver_lub14, pendulum_QP_solver_dlubaff14);
info->lsit_aff = pendulum_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(pendulum_QP_solver_l, pendulum_QP_solver_s, pendulum_QP_solver_dl_aff, pendulum_QP_solver_ds_aff, &info->step_aff, &info->mu_aff);
if( info->lsit_aff == pendulum_QP_solver_NOPROGRESS ){
exitcode = pendulum_QP_solver_NOPROGRESS; break;
}
sigma_3rdroot = info->mu_aff / info->mu;
info->sigma = sigma_3rdroot*sigma_3rdroot*sigma_3rdroot;
musigma = info->mu * info->sigma;
pendulum_QP_solver_LA_VSUB5_230(pendulum_QP_solver_ds_aff, pendulum_QP_solver_dl_aff, info->mu, info->sigma, pendulum_QP_solver_ccrhs);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub00, pendulum_QP_solver_sub00, pendulum_QP_solver_ubIdx00, pendulum_QP_solver_ccrhsl00, pendulum_QP_solver_slb00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_rd00);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub01, pendulum_QP_solver_sub01, pendulum_QP_solver_ubIdx01, pendulum_QP_solver_ccrhsl01, pendulum_QP_solver_slb01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_rd01);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi00, pendulum_QP_solver_rd00, pendulum_QP_solver_Lbyrd00);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi01, pendulum_QP_solver_rd01, pendulum_QP_solver_Lbyrd01);
pendulum_QP_solver_LA_DENSE_2MVMADD_5_10_10(pendulum_QP_solver_V00, pendulum_QP_solver_Lbyrd00, pendulum_QP_solver_W01, pendulum_QP_solver_Lbyrd01, pendulum_QP_solver_beta00);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_5(pendulum_QP_solver_Ld00, pendulum_QP_solver_beta00, pendulum_QP_solver_yy00);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub02, pendulum_QP_solver_sub02, pendulum_QP_solver_ubIdx02, pendulum_QP_solver_ccrhsl02, pendulum_QP_solver_slb02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_rd02);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi02, pendulum_QP_solver_rd02, pendulum_QP_solver_Lbyrd02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V01, pendulum_QP_solver_Lbyrd01, pendulum_QP_solver_W02, pendulum_QP_solver_Lbyrd02, pendulum_QP_solver_beta01);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_5(pendulum_QP_solver_Lsd01, pendulum_QP_solver_yy00, pendulum_QP_solver_beta01, pendulum_QP_solver_bmy01);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld01, pendulum_QP_solver_bmy01, pendulum_QP_solver_yy01);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub03, pendulum_QP_solver_sub03, pendulum_QP_solver_ubIdx03, pendulum_QP_solver_ccrhsl03, pendulum_QP_solver_slb03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_rd03);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi03, pendulum_QP_solver_rd03, pendulum_QP_solver_Lbyrd03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V02, pendulum_QP_solver_Lbyrd02, pendulum_QP_solver_W03, pendulum_QP_solver_Lbyrd03, pendulum_QP_solver_beta02);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd02, pendulum_QP_solver_yy01, pendulum_QP_solver_beta02, pendulum_QP_solver_bmy02);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld02, pendulum_QP_solver_bmy02, pendulum_QP_solver_yy02);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub04, pendulum_QP_solver_sub04, pendulum_QP_solver_ubIdx04, pendulum_QP_solver_ccrhsl04, pendulum_QP_solver_slb04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_rd04);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi04, pendulum_QP_solver_rd04, pendulum_QP_solver_Lbyrd04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V03, pendulum_QP_solver_Lbyrd03, pendulum_QP_solver_W04, pendulum_QP_solver_Lbyrd04, pendulum_QP_solver_beta03);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd03, pendulum_QP_solver_yy02, pendulum_QP_solver_beta03, pendulum_QP_solver_bmy03);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld03, pendulum_QP_solver_bmy03, pendulum_QP_solver_yy03);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub05, pendulum_QP_solver_sub05, pendulum_QP_solver_ubIdx05, pendulum_QP_solver_ccrhsl05, pendulum_QP_solver_slb05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_rd05);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi05, pendulum_QP_solver_rd05, pendulum_QP_solver_Lbyrd05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V04, pendulum_QP_solver_Lbyrd04, pendulum_QP_solver_W05, pendulum_QP_solver_Lbyrd05, pendulum_QP_solver_beta04);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd04, pendulum_QP_solver_yy03, pendulum_QP_solver_beta04, pendulum_QP_solver_bmy04);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld04, pendulum_QP_solver_bmy04, pendulum_QP_solver_yy04);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub06, pendulum_QP_solver_sub06, pendulum_QP_solver_ubIdx06, pendulum_QP_solver_ccrhsl06, pendulum_QP_solver_slb06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_rd06);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi06, pendulum_QP_solver_rd06, pendulum_QP_solver_Lbyrd06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V05, pendulum_QP_solver_Lbyrd05, pendulum_QP_solver_W06, pendulum_QP_solver_Lbyrd06, pendulum_QP_solver_beta05);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd05, pendulum_QP_solver_yy04, pendulum_QP_solver_beta05, pendulum_QP_solver_bmy05);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld05, pendulum_QP_solver_bmy05, pendulum_QP_solver_yy05);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub07, pendulum_QP_solver_sub07, pendulum_QP_solver_ubIdx07, pendulum_QP_solver_ccrhsl07, pendulum_QP_solver_slb07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_rd07);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi07, pendulum_QP_solver_rd07, pendulum_QP_solver_Lbyrd07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V06, pendulum_QP_solver_Lbyrd06, pendulum_QP_solver_W07, pendulum_QP_solver_Lbyrd07, pendulum_QP_solver_beta06);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd06, pendulum_QP_solver_yy05, pendulum_QP_solver_beta06, pendulum_QP_solver_bmy06);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld06, pendulum_QP_solver_bmy06, pendulum_QP_solver_yy06);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub08, pendulum_QP_solver_sub08, pendulum_QP_solver_ubIdx08, pendulum_QP_solver_ccrhsl08, pendulum_QP_solver_slb08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_rd08);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi08, pendulum_QP_solver_rd08, pendulum_QP_solver_Lbyrd08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V07, pendulum_QP_solver_Lbyrd07, pendulum_QP_solver_W08, pendulum_QP_solver_Lbyrd08, pendulum_QP_solver_beta07);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd07, pendulum_QP_solver_yy06, pendulum_QP_solver_beta07, pendulum_QP_solver_bmy07);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld07, pendulum_QP_solver_bmy07, pendulum_QP_solver_yy07);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub09, pendulum_QP_solver_sub09, pendulum_QP_solver_ubIdx09, pendulum_QP_solver_ccrhsl09, pendulum_QP_solver_slb09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_rd09);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi09, pendulum_QP_solver_rd09, pendulum_QP_solver_Lbyrd09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V08, pendulum_QP_solver_Lbyrd08, pendulum_QP_solver_W09, pendulum_QP_solver_Lbyrd09, pendulum_QP_solver_beta08);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd08, pendulum_QP_solver_yy07, pendulum_QP_solver_beta08, pendulum_QP_solver_bmy08);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld08, pendulum_QP_solver_bmy08, pendulum_QP_solver_yy08);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub10, pendulum_QP_solver_sub10, pendulum_QP_solver_ubIdx10, pendulum_QP_solver_ccrhsl10, pendulum_QP_solver_slb10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_rd10);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi10, pendulum_QP_solver_rd10, pendulum_QP_solver_Lbyrd10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V09, pendulum_QP_solver_Lbyrd09, pendulum_QP_solver_W10, pendulum_QP_solver_Lbyrd10, pendulum_QP_solver_beta09);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd09, pendulum_QP_solver_yy08, pendulum_QP_solver_beta09, pendulum_QP_solver_bmy09);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld09, pendulum_QP_solver_bmy09, pendulum_QP_solver_yy09);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub11, pendulum_QP_solver_sub11, pendulum_QP_solver_ubIdx11, pendulum_QP_solver_ccrhsl11, pendulum_QP_solver_slb11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_rd11);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi11, pendulum_QP_solver_rd11, pendulum_QP_solver_Lbyrd11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V10, pendulum_QP_solver_Lbyrd10, pendulum_QP_solver_W11, pendulum_QP_solver_Lbyrd11, pendulum_QP_solver_beta10);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd10, pendulum_QP_solver_yy09, pendulum_QP_solver_beta10, pendulum_QP_solver_bmy10);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld10, pendulum_QP_solver_bmy10, pendulum_QP_solver_yy10);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub12, pendulum_QP_solver_sub12, pendulum_QP_solver_ubIdx12, pendulum_QP_solver_ccrhsl12, pendulum_QP_solver_slb12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_rd12);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi12, pendulum_QP_solver_rd12, pendulum_QP_solver_Lbyrd12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V11, pendulum_QP_solver_Lbyrd11, pendulum_QP_solver_W12, pendulum_QP_solver_Lbyrd12, pendulum_QP_solver_beta11);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd11, pendulum_QP_solver_yy10, pendulum_QP_solver_beta11, pendulum_QP_solver_bmy11);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld11, pendulum_QP_solver_bmy11, pendulum_QP_solver_yy11);
pendulum_QP_solver_LA_VSUB6_INDEXED_10_6_10(pendulum_QP_solver_ccrhsub13, pendulum_QP_solver_sub13, pendulum_QP_solver_ubIdx13, pendulum_QP_solver_ccrhsl13, pendulum_QP_solver_slb13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_rd13);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_10(pendulum_QP_solver_Phi13, pendulum_QP_solver_rd13, pendulum_QP_solver_Lbyrd13);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_10(pendulum_QP_solver_V12, pendulum_QP_solver_Lbyrd12, pendulum_QP_solver_W13, pendulum_QP_solver_Lbyrd13, pendulum_QP_solver_beta12);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd12, pendulum_QP_solver_yy11, pendulum_QP_solver_beta12, pendulum_QP_solver_bmy12);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld12, pendulum_QP_solver_bmy12, pendulum_QP_solver_yy12);
pendulum_QP_solver_LA_VSUB6_INDEXED_3_3_3(pendulum_QP_solver_ccrhsub14, pendulum_QP_solver_sub14, pendulum_QP_solver_ubIdx14, pendulum_QP_solver_ccrhsl14, pendulum_QP_solver_slb14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_rd14);
pendulum_QP_solver_LA_DIAG_FORWARDSUB_3(pendulum_QP_solver_Phi14, pendulum_QP_solver_rd14, pendulum_QP_solver_Lbyrd14);
pendulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_3_10_3(pendulum_QP_solver_V13, pendulum_QP_solver_Lbyrd13, pendulum_QP_solver_W14, pendulum_QP_solver_Lbyrd14, pendulum_QP_solver_beta13);
pendulum_QP_solver_LA_DENSE_MVMSUB1_3_3(pendulum_QP_solver_Lsd13, pendulum_QP_solver_yy12, pendulum_QP_solver_beta13, pendulum_QP_solver_bmy13);
pendulum_QP_solver_LA_DENSE_FORWARDSUB_3(pendulum_QP_solver_Ld13, pendulum_QP_solver_bmy13, pendulum_QP_solver_yy13);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld13, pendulum_QP_solver_yy13, pendulum_QP_solver_dvcc13);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd13, pendulum_QP_solver_dvcc13, pendulum_QP_solver_yy12, pendulum_QP_solver_bmy12);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld12, pendulum_QP_solver_bmy12, pendulum_QP_solver_dvcc12);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd12, pendulum_QP_solver_dvcc12, pendulum_QP_solver_yy11, pendulum_QP_solver_bmy11);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld11, pendulum_QP_solver_bmy11, pendulum_QP_solver_dvcc11);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd11, pendulum_QP_solver_dvcc11, pendulum_QP_solver_yy10, pendulum_QP_solver_bmy10);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld10, pendulum_QP_solver_bmy10, pendulum_QP_solver_dvcc10);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd10, pendulum_QP_solver_dvcc10, pendulum_QP_solver_yy09, pendulum_QP_solver_bmy09);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld09, pendulum_QP_solver_bmy09, pendulum_QP_solver_dvcc09);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd09, pendulum_QP_solver_dvcc09, pendulum_QP_solver_yy08, pendulum_QP_solver_bmy08);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld08, pendulum_QP_solver_bmy08, pendulum_QP_solver_dvcc08);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd08, pendulum_QP_solver_dvcc08, pendulum_QP_solver_yy07, pendulum_QP_solver_bmy07);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld07, pendulum_QP_solver_bmy07, pendulum_QP_solver_dvcc07);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd07, pendulum_QP_solver_dvcc07, pendulum_QP_solver_yy06, pendulum_QP_solver_bmy06);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld06, pendulum_QP_solver_bmy06, pendulum_QP_solver_dvcc06);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd06, pendulum_QP_solver_dvcc06, pendulum_QP_solver_yy05, pendulum_QP_solver_bmy05);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld05, pendulum_QP_solver_bmy05, pendulum_QP_solver_dvcc05);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd05, pendulum_QP_solver_dvcc05, pendulum_QP_solver_yy04, pendulum_QP_solver_bmy04);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld04, pendulum_QP_solver_bmy04, pendulum_QP_solver_dvcc04);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd04, pendulum_QP_solver_dvcc04, pendulum_QP_solver_yy03, pendulum_QP_solver_bmy03);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld03, pendulum_QP_solver_bmy03, pendulum_QP_solver_dvcc03);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd03, pendulum_QP_solver_dvcc03, pendulum_QP_solver_yy02, pendulum_QP_solver_bmy02);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld02, pendulum_QP_solver_bmy02, pendulum_QP_solver_dvcc02);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_3(pendulum_QP_solver_Lsd02, pendulum_QP_solver_dvcc02, pendulum_QP_solver_yy01, pendulum_QP_solver_bmy01);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_3(pendulum_QP_solver_Ld01, pendulum_QP_solver_bmy01, pendulum_QP_solver_dvcc01);
pendulum_QP_solver_LA_DENSE_MTVMSUB_3_5(pendulum_QP_solver_Lsd01, pendulum_QP_solver_dvcc01, pendulum_QP_solver_yy00, pendulum_QP_solver_bmy00);
pendulum_QP_solver_LA_DENSE_BACKWARDSUB_5(pendulum_QP_solver_Ld00, pendulum_QP_solver_bmy00, pendulum_QP_solver_dvcc00);
pendulum_QP_solver_LA_DENSE_MTVM_5_10(params->C1, pendulum_QP_solver_dvcc00, pendulum_QP_solver_grad_eq00);
pendulum_QP_solver_LA_DENSE_MTVM2_3_10_5(params->C2, pendulum_QP_solver_dvcc01, pendulum_QP_solver_D01, pendulum_QP_solver_dvcc00, pendulum_QP_solver_grad_eq01);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C3, pendulum_QP_solver_dvcc02, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc01, pendulum_QP_solver_grad_eq02);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C4, pendulum_QP_solver_dvcc03, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc02, pendulum_QP_solver_grad_eq03);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C5, pendulum_QP_solver_dvcc04, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc03, pendulum_QP_solver_grad_eq04);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C6, pendulum_QP_solver_dvcc05, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc04, pendulum_QP_solver_grad_eq05);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C7, pendulum_QP_solver_dvcc06, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc05, pendulum_QP_solver_grad_eq06);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C8, pendulum_QP_solver_dvcc07, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc06, pendulum_QP_solver_grad_eq07);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C9, pendulum_QP_solver_dvcc08, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc07, pendulum_QP_solver_grad_eq08);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C10, pendulum_QP_solver_dvcc09, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc08, pendulum_QP_solver_grad_eq09);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C11, pendulum_QP_solver_dvcc10, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc09, pendulum_QP_solver_grad_eq10);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C12, pendulum_QP_solver_dvcc11, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc10, pendulum_QP_solver_grad_eq11);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C13, pendulum_QP_solver_dvcc12, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc11, pendulum_QP_solver_grad_eq12);
pendulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_3_10_3(params->C14, pendulum_QP_solver_dvcc13, pendulum_QP_solver_D02, pendulum_QP_solver_dvcc12, pendulum_QP_solver_grad_eq13);
pendulum_QP_solver_LA_DIAGZERO_MTVM_3_3(pendulum_QP_solver_D14, pendulum_QP_solver_dvcc13, pendulum_QP_solver_grad_eq14);
pendulum_QP_solver_LA_VSUB_143(pendulum_QP_solver_rd, pendulum_QP_solver_grad_eq, pendulum_QP_solver_rd);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi00, pendulum_QP_solver_rd00, pendulum_QP_solver_dzcc00);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi01, pendulum_QP_solver_rd01, pendulum_QP_solver_dzcc01);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi02, pendulum_QP_solver_rd02, pendulum_QP_solver_dzcc02);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi03, pendulum_QP_solver_rd03, pendulum_QP_solver_dzcc03);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi04, pendulum_QP_solver_rd04, pendulum_QP_solver_dzcc04);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi05, pendulum_QP_solver_rd05, pendulum_QP_solver_dzcc05);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi06, pendulum_QP_solver_rd06, pendulum_QP_solver_dzcc06);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi07, pendulum_QP_solver_rd07, pendulum_QP_solver_dzcc07);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi08, pendulum_QP_solver_rd08, pendulum_QP_solver_dzcc08);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi09, pendulum_QP_solver_rd09, pendulum_QP_solver_dzcc09);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi10, pendulum_QP_solver_rd10, pendulum_QP_solver_dzcc10);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi11, pendulum_QP_solver_rd11, pendulum_QP_solver_dzcc11);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi12, pendulum_QP_solver_rd12, pendulum_QP_solver_dzcc12);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_10(pendulum_QP_solver_Phi13, pendulum_QP_solver_rd13, pendulum_QP_solver_dzcc13);
pendulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_3(pendulum_QP_solver_Phi14, pendulum_QP_solver_rd14, pendulum_QP_solver_dzcc14);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl00, pendulum_QP_solver_slb00, pendulum_QP_solver_llbbyslb00, pendulum_QP_solver_dzcc00, pendulum_QP_solver_lbIdx00, pendulum_QP_solver_dllbcc00);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub00, pendulum_QP_solver_sub00, pendulum_QP_solver_lubbysub00, pendulum_QP_solver_dzcc00, pendulum_QP_solver_ubIdx00, pendulum_QP_solver_dlubcc00);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl01, pendulum_QP_solver_slb01, pendulum_QP_solver_llbbyslb01, pendulum_QP_solver_dzcc01, pendulum_QP_solver_lbIdx01, pendulum_QP_solver_dllbcc01);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub01, pendulum_QP_solver_sub01, pendulum_QP_solver_lubbysub01, pendulum_QP_solver_dzcc01, pendulum_QP_solver_ubIdx01, pendulum_QP_solver_dlubcc01);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl02, pendulum_QP_solver_slb02, pendulum_QP_solver_llbbyslb02, pendulum_QP_solver_dzcc02, pendulum_QP_solver_lbIdx02, pendulum_QP_solver_dllbcc02);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub02, pendulum_QP_solver_sub02, pendulum_QP_solver_lubbysub02, pendulum_QP_solver_dzcc02, pendulum_QP_solver_ubIdx02, pendulum_QP_solver_dlubcc02);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl03, pendulum_QP_solver_slb03, pendulum_QP_solver_llbbyslb03, pendulum_QP_solver_dzcc03, pendulum_QP_solver_lbIdx03, pendulum_QP_solver_dllbcc03);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub03, pendulum_QP_solver_sub03, pendulum_QP_solver_lubbysub03, pendulum_QP_solver_dzcc03, pendulum_QP_solver_ubIdx03, pendulum_QP_solver_dlubcc03);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl04, pendulum_QP_solver_slb04, pendulum_QP_solver_llbbyslb04, pendulum_QP_solver_dzcc04, pendulum_QP_solver_lbIdx04, pendulum_QP_solver_dllbcc04);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub04, pendulum_QP_solver_sub04, pendulum_QP_solver_lubbysub04, pendulum_QP_solver_dzcc04, pendulum_QP_solver_ubIdx04, pendulum_QP_solver_dlubcc04);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl05, pendulum_QP_solver_slb05, pendulum_QP_solver_llbbyslb05, pendulum_QP_solver_dzcc05, pendulum_QP_solver_lbIdx05, pendulum_QP_solver_dllbcc05);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub05, pendulum_QP_solver_sub05, pendulum_QP_solver_lubbysub05, pendulum_QP_solver_dzcc05, pendulum_QP_solver_ubIdx05, pendulum_QP_solver_dlubcc05);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl06, pendulum_QP_solver_slb06, pendulum_QP_solver_llbbyslb06, pendulum_QP_solver_dzcc06, pendulum_QP_solver_lbIdx06, pendulum_QP_solver_dllbcc06);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub06, pendulum_QP_solver_sub06, pendulum_QP_solver_lubbysub06, pendulum_QP_solver_dzcc06, pendulum_QP_solver_ubIdx06, pendulum_QP_solver_dlubcc06);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl07, pendulum_QP_solver_slb07, pendulum_QP_solver_llbbyslb07, pendulum_QP_solver_dzcc07, pendulum_QP_solver_lbIdx07, pendulum_QP_solver_dllbcc07);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub07, pendulum_QP_solver_sub07, pendulum_QP_solver_lubbysub07, pendulum_QP_solver_dzcc07, pendulum_QP_solver_ubIdx07, pendulum_QP_solver_dlubcc07);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl08, pendulum_QP_solver_slb08, pendulum_QP_solver_llbbyslb08, pendulum_QP_solver_dzcc08, pendulum_QP_solver_lbIdx08, pendulum_QP_solver_dllbcc08);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub08, pendulum_QP_solver_sub08, pendulum_QP_solver_lubbysub08, pendulum_QP_solver_dzcc08, pendulum_QP_solver_ubIdx08, pendulum_QP_solver_dlubcc08);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl09, pendulum_QP_solver_slb09, pendulum_QP_solver_llbbyslb09, pendulum_QP_solver_dzcc09, pendulum_QP_solver_lbIdx09, pendulum_QP_solver_dllbcc09);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub09, pendulum_QP_solver_sub09, pendulum_QP_solver_lubbysub09, pendulum_QP_solver_dzcc09, pendulum_QP_solver_ubIdx09, pendulum_QP_solver_dlubcc09);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl10, pendulum_QP_solver_slb10, pendulum_QP_solver_llbbyslb10, pendulum_QP_solver_dzcc10, pendulum_QP_solver_lbIdx10, pendulum_QP_solver_dllbcc10);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub10, pendulum_QP_solver_sub10, pendulum_QP_solver_lubbysub10, pendulum_QP_solver_dzcc10, pendulum_QP_solver_ubIdx10, pendulum_QP_solver_dlubcc10);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl11, pendulum_QP_solver_slb11, pendulum_QP_solver_llbbyslb11, pendulum_QP_solver_dzcc11, pendulum_QP_solver_lbIdx11, pendulum_QP_solver_dllbcc11);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub11, pendulum_QP_solver_sub11, pendulum_QP_solver_lubbysub11, pendulum_QP_solver_dzcc11, pendulum_QP_solver_ubIdx11, pendulum_QP_solver_dlubcc11);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl12, pendulum_QP_solver_slb12, pendulum_QP_solver_llbbyslb12, pendulum_QP_solver_dzcc12, pendulum_QP_solver_lbIdx12, pendulum_QP_solver_dllbcc12);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub12, pendulum_QP_solver_sub12, pendulum_QP_solver_lubbysub12, pendulum_QP_solver_dzcc12, pendulum_QP_solver_ubIdx12, pendulum_QP_solver_dlubcc12);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_10(pendulum_QP_solver_ccrhsl13, pendulum_QP_solver_slb13, pendulum_QP_solver_llbbyslb13, pendulum_QP_solver_dzcc13, pendulum_QP_solver_lbIdx13, pendulum_QP_solver_dllbcc13);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_6(pendulum_QP_solver_ccrhsub13, pendulum_QP_solver_sub13, pendulum_QP_solver_lubbysub13, pendulum_QP_solver_dzcc13, pendulum_QP_solver_ubIdx13, pendulum_QP_solver_dlubcc13);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_3(pendulum_QP_solver_ccrhsl14, pendulum_QP_solver_slb14, pendulum_QP_solver_llbbyslb14, pendulum_QP_solver_dzcc14, pendulum_QP_solver_lbIdx14, pendulum_QP_solver_dllbcc14);
pendulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_3(pendulum_QP_solver_ccrhsub14, pendulum_QP_solver_sub14, pendulum_QP_solver_lubbysub14, pendulum_QP_solver_dzcc14, pendulum_QP_solver_ubIdx14, pendulum_QP_solver_dlubcc14);
pendulum_QP_solver_LA_VSUB7_230(pendulum_QP_solver_l, pendulum_QP_solver_ccrhs, pendulum_QP_solver_s, pendulum_QP_solver_dl_cc, pendulum_QP_solver_ds_cc);
pendulum_QP_solver_LA_VADD_143(pendulum_QP_solver_dz_cc, pendulum_QP_solver_dz_aff);
pendulum_QP_solver_LA_VADD_44(pendulum_QP_solver_dv_cc, pendulum_QP_solver_dv_aff);
pendulum_QP_solver_LA_VADD_230(pendulum_QP_solver_dl_cc, pendulum_QP_solver_dl_aff);
pendulum_QP_solver_LA_VADD_230(pendulum_QP_solver_ds_cc, pendulum_QP_solver_ds_aff);
info->lsit_cc = pendulum_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(pendulum_QP_solver_z, pendulum_QP_solver_v, pendulum_QP_solver_l, pendulum_QP_solver_s, pendulum_QP_solver_dz_cc, pendulum_QP_solver_dv_cc, pendulum_QP_solver_dl_cc, pendulum_QP_solver_ds_cc, &info->step_cc, &info->mu);
if( info->lsit_cc == pendulum_QP_solver_NOPROGRESS ){
exitcode = pendulum_QP_solver_NOPROGRESS; break;
}
info->it++;
}
output->z1[0] = pendulum_QP_solver_z00[0];
output->z1[1] = pendulum_QP_solver_z00[1];
output->z1[2] = pendulum_QP_solver_z00[2];
output->z1[3] = pendulum_QP_solver_z00[3];
output->z1[4] = pendulum_QP_solver_z00[4];
output->z1[5] = pendulum_QP_solver_z00[5];
output->z2[0] = pendulum_QP_solver_z01[0];
output->z2[1] = pendulum_QP_solver_z01[1];
output->z2[2] = pendulum_QP_solver_z01[2];
output->z2[3] = pendulum_QP_solver_z01[3];
output->z2[4] = pendulum_QP_solver_z01[4];
output->z2[5] = pendulum_QP_solver_z01[5];
output->z3[0] = pendulum_QP_solver_z02[0];
output->z3[1] = pendulum_QP_solver_z02[1];
output->z3[2] = pendulum_QP_solver_z02[2];
output->z3[3] = pendulum_QP_solver_z02[3];
output->z3[4] = pendulum_QP_solver_z02[4];
output->z3[5] = pendulum_QP_solver_z02[5];
output->z4[0] = pendulum_QP_solver_z03[0];
output->z4[1] = pendulum_QP_solver_z03[1];
output->z4[2] = pendulum_QP_solver_z03[2];
output->z4[3] = pendulum_QP_solver_z03[3];
output->z4[4] = pendulum_QP_solver_z03[4];
output->z4[5] = pendulum_QP_solver_z03[5];
output->z5[0] = pendulum_QP_solver_z04[0];
output->z5[1] = pendulum_QP_solver_z04[1];
output->z5[2] = pendulum_QP_solver_z04[2];
output->z5[3] = pendulum_QP_solver_z04[3];
output->z5[4] = pendulum_QP_solver_z04[4];
output->z5[5] = pendulum_QP_solver_z04[5];
output->z6[0] = pendulum_QP_solver_z05[0];
output->z6[1] = pendulum_QP_solver_z05[1];
output->z6[2] = pendulum_QP_solver_z05[2];
output->z6[3] = pendulum_QP_solver_z05[3];
output->z6[4] = pendulum_QP_solver_z05[4];
output->z6[5] = pendulum_QP_solver_z05[5];
output->z7[0] = pendulum_QP_solver_z06[0];
output->z7[1] = pendulum_QP_solver_z06[1];
output->z7[2] = pendulum_QP_solver_z06[2];
output->z7[3] = pendulum_QP_solver_z06[3];
output->z7[4] = pendulum_QP_solver_z06[4];
output->z7[5] = pendulum_QP_solver_z06[5];
output->z8[0] = pendulum_QP_solver_z07[0];
output->z8[1] = pendulum_QP_solver_z07[1];
output->z8[2] = pendulum_QP_solver_z07[2];
output->z8[3] = pendulum_QP_solver_z07[3];
output->z8[4] = pendulum_QP_solver_z07[4];
output->z8[5] = pendulum_QP_solver_z07[5];
output->z9[0] = pendulum_QP_solver_z08[0];
output->z9[1] = pendulum_QP_solver_z08[1];
output->z9[2] = pendulum_QP_solver_z08[2];
output->z9[3] = pendulum_QP_solver_z08[3];
output->z9[4] = pendulum_QP_solver_z08[4];
output->z9[5] = pendulum_QP_solver_z08[5];
output->z10[0] = pendulum_QP_solver_z09[0];
output->z10[1] = pendulum_QP_solver_z09[1];
output->z10[2] = pendulum_QP_solver_z09[2];
output->z10[3] = pendulum_QP_solver_z09[3];
output->z10[4] = pendulum_QP_solver_z09[4];
output->z10[5] = pendulum_QP_solver_z09[5];
output->z11[0] = pendulum_QP_solver_z10[0];
output->z11[1] = pendulum_QP_solver_z10[1];
output->z11[2] = pendulum_QP_solver_z10[2];
output->z11[3] = pendulum_QP_solver_z10[3];
output->z11[4] = pendulum_QP_solver_z10[4];
output->z11[5] = pendulum_QP_solver_z10[5];
output->z12[0] = pendulum_QP_solver_z11[0];
output->z12[1] = pendulum_QP_solver_z11[1];
output->z12[2] = pendulum_QP_solver_z11[2];
output->z12[3] = pendulum_QP_solver_z11[3];
output->z12[4] = pendulum_QP_solver_z11[4];
output->z12[5] = pendulum_QP_solver_z11[5];
output->z13[0] = pendulum_QP_solver_z12[0];
output->z13[1] = pendulum_QP_solver_z12[1];
output->z13[2] = pendulum_QP_solver_z12[2];
output->z13[3] = pendulum_QP_solver_z12[3];
output->z13[4] = pendulum_QP_solver_z12[4];
output->z13[5] = pendulum_QP_solver_z12[5];
output->z14[0] = pendulum_QP_solver_z13[0];
output->z14[1] = pendulum_QP_solver_z13[1];
output->z14[2] = pendulum_QP_solver_z13[2];
output->z14[3] = pendulum_QP_solver_z13[3];
output->z14[4] = pendulum_QP_solver_z13[4];
output->z14[5] = pendulum_QP_solver_z13[5];
output->z15[0] = pendulum_QP_solver_z14[0];
output->z15[1] = pendulum_QP_solver_z14[1];
output->z15[2] = pendulum_QP_solver_z14[2];

#if pendulum_QP_solver_SET_TIMING == 1
info->solvetime = pendulum_QP_solver_toc(&solvertimer);
#if pendulum_QP_solver_SET_PRINTLEVEL > 0 && pendulum_QP_solver_SET_TIMING == 1
if( info->it > 1 ){
	PRINTTEXT("Solve time: %5.3f ms (%d iterations)\n\n", info->solvetime*1000, info->it);
} else {
	PRINTTEXT("Solve time: %5.3f ms (%d iteration)\n\n", info->solvetime*1000, info->it);
}
#endif
#else
info->solvetime = -1;
#endif
return exitcode;
}
