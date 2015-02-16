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

#include "doublependulum_QP_solver.h"

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
 * Initializes a vector of length 271 with a value.
 */
void doublependulum_QP_solver_LA_INITIALIZEVECTOR_271(doublependulum_QP_solver_FLOAT* vec, doublependulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<271; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 74 with a value.
 */
void doublependulum_QP_solver_LA_INITIALIZEVECTOR_74(doublependulum_QP_solver_FLOAT* vec, doublependulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<74; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 430 with a value.
 */
void doublependulum_QP_solver_LA_INITIALIZEVECTOR_430(doublependulum_QP_solver_FLOAT* vec, doublependulum_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<430; i++ )
	{
		vec[i] = value;
	}
}


/* 
 * Calculates a dot product and adds it to a variable: z += x'*y; 
 * This function is for vectors of length 430.
 */
void doublependulum_QP_solver_LA_DOTACC_430(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<430; i++ ){
		*z += x[i]*y[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [19 x 19]
 *             f  - column vector of size 19
 *             z  - column vector of size 19
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 19
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_FLOAT* H, doublependulum_QP_solver_FLOAT* f, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* grad, doublependulum_QP_solver_FLOAT* value)
{
	int i;
	doublependulum_QP_solver_FLOAT hz;	
	for( i=0; i<19; i++){
		hz = H[i]*z[i];
		grad[i] = hz + f[i];
		*value += 0.5*hz*z[i] + f[i]*z[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [5 x 5]
 *             f  - column vector of size 5
 *             z  - column vector of size 5
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 5
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void doublependulum_QP_solver_LA_DIAG_QUADFCN_5(doublependulum_QP_solver_FLOAT* H, doublependulum_QP_solver_FLOAT* f, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* grad, doublependulum_QP_solver_FLOAT* value)
{
	int i;
	doublependulum_QP_solver_FLOAT hz;	
	for( i=0; i<5; i++){
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
void doublependulum_QP_solver_LA_DENSE_MVMSUB3_9_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *z, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;
	doublependulum_QP_solver_FLOAT AxBu[9];
	doublependulum_QP_solver_FLOAT norm = *y;
	doublependulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<9; i++ ){
		AxBu[i] = A[k++]*x[0] + B[m++]*u[0];
	}	
	for( j=1; j<19; j++ ){		
		for( i=0; i<9; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<19; n++ ){
		for( i=0; i<9; i++ ){
			AxBu[i] += B[m++]*u[n];
		}		
	}

	for( i=0; i<9; i++ ){
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
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *z, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	doublependulum_QP_solver_FLOAT AxBu[5];
	doublependulum_QP_solver_FLOAT norm = *y;
	doublependulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<5; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			AxBu[i] += A[k++]*x[j];
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
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *z, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	doublependulum_QP_solver_FLOAT AxBu[5];
	doublependulum_QP_solver_FLOAT norm = *y;
	doublependulum_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<5; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			AxBu[i] += A[k++]*x[j];
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
 * Matrix vector multiplication y = M'*x where M is of size [9 x 19]
 * and stored in column major format. Note the transpose of M!
 */
void doublependulum_QP_solver_LA_DENSE_MTVM_9_19(doublependulum_QP_solver_FLOAT *M, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<19; i++ ){
		y[i] = 0;
		for( j=0; j<9; j++ ){
			y[i] += M[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [5 x 19]
 * and B is of size [9 x 19]
 * and stored in column major format. Note the transposes of A and B!
 */
void doublependulum_QP_solver_LA_DENSE_MTVM2_5_19_9(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	int n;
	int m = 0;
	for( i=0; i<19; i++ ){
		z[i] = 0;
		for( j=0; j<5; j++ ){
			z[i] += A[k++]*x[j];
		}
		for( n=0; n<9; n++ ){
			z[i] += B[m++]*y[n];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [5 x 19] and stored in column major format.
 * and B is of size [5 x 19] and stored in diagzero format
 * Note the transposes of A and B!
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	for( i=0; i<5; i++ ){
		z[i] = 0;
		for( j=0; j<5; j++ ){
			z[i] += A[k++]*x[j];
		}
		z[i] += B[i]*y[i];
	}
	for( i=5 ;i<19; i++ ){
		z[i] = 0;
		for( j=0; j<5; j++ ){
			z[i] += A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = M'*x where M is of size [5 x 5]
 * and stored in diagzero format. Note the transpose of M!
 */
void doublependulum_QP_solver_LA_DIAGZERO_MTVM_5_5(doublependulum_QP_solver_FLOAT *M, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<5; i++ ){
		y[i] = M[i]*x[i];
	}
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 19. Output z is of course scalar.
 */
void doublependulum_QP_solver_LA_VSUBADD3_19(doublependulum_QP_solver_FLOAT* t, doublependulum_QP_solver_FLOAT* u, int* uidx, doublependulum_QP_solver_FLOAT* v, doublependulum_QP_solver_FLOAT* w, doublependulum_QP_solver_FLOAT* y, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_FLOAT norm = *r;
	doublependulum_QP_solver_FLOAT vx = 0;
	doublependulum_QP_solver_FLOAT x;
	for( i=0; i<19; i++){
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
 * for vectors of length 11. Output z is of course scalar.
 */
void doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_FLOAT* t, int* tidx, doublependulum_QP_solver_FLOAT* u, doublependulum_QP_solver_FLOAT* v, doublependulum_QP_solver_FLOAT* w, doublependulum_QP_solver_FLOAT* y, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_FLOAT norm = *r;
	doublependulum_QP_solver_FLOAT vx = 0;
	doublependulum_QP_solver_FLOAT x;
	for( i=0; i<11; i++){
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
 * for vectors of length 5. Output z is of course scalar.
 */
void doublependulum_QP_solver_LA_VSUBADD3_5(doublependulum_QP_solver_FLOAT* t, doublependulum_QP_solver_FLOAT* u, int* uidx, doublependulum_QP_solver_FLOAT* v, doublependulum_QP_solver_FLOAT* w, doublependulum_QP_solver_FLOAT* y, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_FLOAT norm = *r;
	doublependulum_QP_solver_FLOAT vx = 0;
	doublependulum_QP_solver_FLOAT x;
	for( i=0; i<5; i++){
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
 * for vectors of length 5. Output z is of course scalar.
 */
void doublependulum_QP_solver_LA_VSUBADD2_5(doublependulum_QP_solver_FLOAT* t, int* tidx, doublependulum_QP_solver_FLOAT* u, doublependulum_QP_solver_FLOAT* v, doublependulum_QP_solver_FLOAT* w, doublependulum_QP_solver_FLOAT* y, doublependulum_QP_solver_FLOAT* z, doublependulum_QP_solver_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_FLOAT norm = *r;
	doublependulum_QP_solver_FLOAT vx = 0;
	doublependulum_QP_solver_FLOAT x;
	for( i=0; i<5; i++){
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
 * Special function for box constraints of length 19
 * Returns also L/S, a value that is often used elsewhere.
 */
void doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_FLOAT *lu, doublependulum_QP_solver_FLOAT *su, doublependulum_QP_solver_FLOAT *ru, doublependulum_QP_solver_FLOAT *ll, doublependulum_QP_solver_FLOAT *sl, doublependulum_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, doublependulum_QP_solver_FLOAT *grad, doublependulum_QP_solver_FLOAT *lubysu, doublependulum_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<19; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<19; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<11; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 5
 * Returns also L/S, a value that is often used elsewhere.
 */
void doublependulum_QP_solver_LA_INEQ_B_GRAD_5_5_5(doublependulum_QP_solver_FLOAT *lu, doublependulum_QP_solver_FLOAT *su, doublependulum_QP_solver_FLOAT *ru, doublependulum_QP_solver_FLOAT *ll, doublependulum_QP_solver_FLOAT *sl, doublependulum_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, doublependulum_QP_solver_FLOAT *grad, doublependulum_QP_solver_FLOAT *lubysu, doublependulum_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<5; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<5; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<5; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Addition of three vectors  z = u + w + v
 * of length 271.
 */
void doublependulum_QP_solver_LA_VVADD3_271(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *w, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<271; i++ ){
		z[i] = u[i] + v[i] + w[i];
	}
}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 19.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_FLOAT *H, doublependulum_QP_solver_FLOAT *llbysl, int* lbIdx, doublependulum_QP_solver_FLOAT *lubysu, int* ubIdx, doublependulum_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* copy  H into PHI */
	for( i=0; i<19; i++ ){
		Phi[i] = H[i];
	}

	/* add llbysl onto Phi where necessary */
	for( i=0; i<19; i++ ){
		Phi[lbIdx[i]] += llbysl[i];
	}

	/* add lubysu onto Phi where necessary */
	for( i=0; i<11; i++){
		Phi[ubIdx[i]] +=  lubysu[i];
	}
	
	/* compute cholesky */
	for(i=0; i<19; i++)
	{
#if doublependulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
 * where A is to be computed and is of size [9 x 19],
 * B is given and of size [9 x 19], L is a diagonal
 * matrix of size 9 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_19(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<19; j++){
		for( i=0; i<9; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 19.
 */
void doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<19; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [5 x 19],
 * B is given and of size [5 x 19], L is a diagonal
 * matrix of size 5 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<19; j++){
		for( i=0; i<5; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [9 x 19]
 *  size(B) = [5 x 19]
 * 
 * and all matrices are stored in column major format.
 *
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE.  
 * 
 */
void doublependulum_QP_solver_LA_DENSE_MMTM_9_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *C)
{
    int i, j, k;
    doublependulum_QP_solver_FLOAT temp;
    
    for( i=0; i<9; i++ ){        
        for( j=0; j<5; j++ ){
            temp = 0; 
            for( k=0; k<19; k++ ){
                temp += A[k*9+i]*B[k*5+j];
            }						
            C[j*9+i] = temp;
        }
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [5 x 19],
 * B is given and of size [5 x 19], L is a diagonal
 *  matrix of size 19 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<19; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [5 x 19]
 *  size(B) = [5 x 19] in diagzero format
 * 
 * A and C matrices are stored in column major format.
 * 
 * 
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *C)
{
    int i, j;
	
	for( i=0; i<5; i++ ){
		for( j=0; j<5; j++){
			C[j*5+i] = B[i*5+j]*A[i];
		}
	}

}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 5.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void doublependulum_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_5_5_5(doublependulum_QP_solver_FLOAT *H, doublependulum_QP_solver_FLOAT *llbysl, int* lbIdx, doublependulum_QP_solver_FLOAT *lubysu, int* ubIdx, doublependulum_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* compute cholesky */
	for( i=0; i<5; i++ ){
		Phi[i] = H[i] + llbysl[i] + lubysu[i];

#if doublependulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
 * where A is to be computed and is of size [5 x 5],
 * B is given and of size [5 x 5], L is a diagonal
 *  matrix of size 5 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<5; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 5.
 */
void doublependulum_QP_solver_LA_DIAG_FORWARDSUB_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<5; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [9 x 19] in column
 * storage format, and B is of size [9 x 19] also in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_LA_DENSE_MMT2_9_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<9; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<19; k++ ){
                ltemp += A[k*9+i]*A[k*9+j];
            }			
			for( k=0; k<19; k++ ){
                ltemp += B[k*9+i]*B[k*9+j];
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
void doublependulum_QP_solver_LA_DENSE_MVMSUB2_9_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<9; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[m++]*u[0];
	}	
	for( j=1; j<19; j++ ){		
		for( i=0; i<9; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
	for( n=1; n<19; n++ ){
		for( i=0; i<9; i++ ){
			r[i] -= B[m++]*u[n];
		}		
	}
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [5 x 19] in column
 * storage format, and B is of size [5 x 19] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<19; k++ ){
                ltemp += A[k*5+i]*A[k*5+j];
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
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [5 x 19] in column
 * storage format, and B is of size [5 x 5] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<19; k++ ){
                ltemp += A[k*5+i]*A[k*5+j];
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
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 9 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void doublependulum_QP_solver_LA_DENSE_CHOL_9(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    doublependulum_QP_solver_FLOAT l;
    doublependulum_QP_solver_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<9; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<9; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if doublependulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
        for( j=i+1; j<9; j++ ){
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
 * The dimensions involved are 9.
 */
void doublependulum_QP_solver_LA_DENSE_FORWARDSUB_9(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    doublependulum_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<9; i++ ){
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
 * where A is to be computed and is of size [5 x 9],
 * B is given and of size [5 x 9], L is a lower tri-
 * angular matrix of size 9 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_9(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    doublependulum_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<9; j++ ){        
        for( i=0; i<5; i++ ){
            a = B[i*9+j];
            for( k=0; k<j; k++ ){
                a -= A[k*5+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*5+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 5
 * and A is a dense matrix of size [5 x 9] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_LA_DENSE_MMTSUB_5_9(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<9; k++ ){
                ltemp += A[k*5+i]*A[k*5+j];
            }						
            L[ii+j] -= ltemp;
        }
        ii += ++di;
    }
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 5 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    doublependulum_QP_solver_FLOAT l;
    doublependulum_QP_solver_FLOAT Mii;

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
        
#if doublependulum_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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


/* 
 * Computes r = b - A*x
 * where A is stored in column major format
 */
void doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_9(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<9; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Forward substitution to solve L*y = b where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * The dimensions involved are 5.
 */
void doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    doublependulum_QP_solver_FLOAT yel;
            
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
 * where A is to be computed and is of size [5 x 5],
 * B is given and of size [5 x 5], L is a lower tri-
 * angular matrix of size 5 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    doublependulum_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<5; j++ ){        
        for( i=0; i<5; i++ ){
            a = B[i*5+j];
            for( k=0; k<j; k++ ){
                a -= A[k*5+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*5+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 5
 * and A is a dense matrix of size [5 x 5] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<5; k++ ){
                ltemp += A[k*5+i]*A[k*5+j];
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
void doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<5; j++ ){		
		for( i=0; i<5; i++ ){
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
void doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    doublependulum_QP_solver_FLOAT xel;    
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
 * Matrix vector multiplication y = b - M'*x where M is of size [5 x 5]
 * and stored in column major format. Note the transpose of M!
 */
void doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<5; i++ ){
		r[i] = b[i];
		for( j=0; j<5; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = b - M'*x where M is of size [5 x 9]
 * and stored in column major format. Note the transpose of M!
 */
void doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_9(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<9; i++ ){
		r[i] = b[i];
		for( j=0; j<5; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 9.
 */
void doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_9(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    doublependulum_QP_solver_FLOAT xel;    
	int start = 36;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 8;
    for( i=8; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 8;
        for( j=8; j>i; j-- ){
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
 * Vector subtraction z = -x - y for vectors of length 271.
 */
void doublependulum_QP_solver_LA_VSUB2_271(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<271; i++){
		z[i] = -x[i] - y[i];
	}
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 19 in vector
 * storage format.
 */
void doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<19; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 5 in vector
 * storage format.
 */
void doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(doublependulum_QP_solver_FLOAT *L, doublependulum_QP_solver_FLOAT *b, doublependulum_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<5; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 19,
 * and x has length 19 and is indexed through yidx.
 */
void doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_FLOAT *x, int* xidx, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<19; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 19.
 */
void doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *w, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<19; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 19
 * and z, x and yidx are of length 11.
 */
void doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<11; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 11.
 */
void doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *w, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<11; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 5,
 * and x has length 5 and is indexed through yidx.
 */
void doublependulum_QP_solver_LA_VSUB_INDEXED_5(doublependulum_QP_solver_FLOAT *x, int* xidx, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 5.
 */
void doublependulum_QP_solver_LA_VSUB3_5(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *w, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<5; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 5
 * and z, x and yidx are of length 5.
 */
void doublependulum_QP_solver_LA_VSUB2_INDEXED_5(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++){
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
 * doublependulum_QP_solver_NOPROGRESS (should be negative).
 */
int doublependulum_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *dl, doublependulum_QP_solver_FLOAT *ds, doublependulum_QP_solver_FLOAT *a, doublependulum_QP_solver_FLOAT *mu_aff)
{
    int i;
	int lsIt=1;    
    doublependulum_QP_solver_FLOAT dltemp;
    doublependulum_QP_solver_FLOAT dstemp;
    doublependulum_QP_solver_FLOAT mya = 1.0;
    doublependulum_QP_solver_FLOAT mymu;
        
    while( 1 ){                        

        /* 
         * Compute both snew and wnew together.
         * We compute also mu_affine along the way here, as the
         * values might be in registers, so it should be cheaper.
         */
        mymu = 0;
        for( i=0; i<430; i++ ){
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
        if( i == 430 ){
            break;
        } else {
            mya *= doublependulum_QP_solver_SET_LS_SCALE_AFF;
            if( mya < doublependulum_QP_solver_SET_LS_MINSTEP ){
                return doublependulum_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* return new values and iteration counter */
    *a = mya;
    *mu_aff = mymu / (doublependulum_QP_solver_FLOAT)430;
    return lsIt;
}


/*
 * Vector subtraction x = (u.*v - mu)*sigma where a is a scalar
*  and x,u,v are vectors of length 430.
 */
void doublependulum_QP_solver_LA_VSUB5_430(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT mu,  doublependulum_QP_solver_FLOAT sigma, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<430; i++){
		x[i] = u[i]*v[i] - mu;
		x[i] *= sigma;
	}
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 19,
 * u, su, uidx are of length 11 and v, sv, vidx are of length 19.
 */
void doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *su, int* uidx, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *sv, int* vidx, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<19; i++ ){
		x[i] = 0;
	}
	for( i=0; i<11; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<19; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A an B are stored in column major format
 */
void doublependulum_QP_solver_LA_DENSE_2MVMADD_9_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<9; i++ ){
		r[i] = A[k++]*x[0] + B[m++]*u[0];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<9; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<19; n++ ){
		for( i=0; i<9; i++ ){
			r[i] += B[m++]*u[n];
		}		
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 5,
 * u, su, uidx are of length 5 and v, sv, vidx are of length 5.
 */
void doublependulum_QP_solver_LA_VSUB6_INDEXED_5_5_5(doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *su, int* uidx, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *sv, int* vidx, doublependulum_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<5; i++ ){
		x[i] = 0;
	}
	for( i=0; i<5; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<5; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_5(doublependulum_QP_solver_FLOAT *A, doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *B, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<19; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Vector subtraction z = x - y for vectors of length 271.
 */
void doublependulum_QP_solver_LA_VSUB_271(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<271; i++){
		z[i] = x[i] - y[i];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 19 (length of y >= 19).
 */
void doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<19; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 11 (length of y >= 11).
 */
void doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<11; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 5 (length of y >= 5).
 */
void doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_5(doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 5 (length of y >= 5).
 */
void doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_5(doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *u, doublependulum_QP_solver_FLOAT *y, int* yidx, doublependulum_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/*
 * Computes ds = -l.\(r + s.*dl) for vectors of length 430.
 */
void doublependulum_QP_solver_LA_VSUB7_430(doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *r, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *dl, doublependulum_QP_solver_FLOAT *ds)
{
	int i;
	for( i=0; i<430; i++){
		ds[i] = -(r[i] + s[i]*dl[i])/l[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 271.
 */
void doublependulum_QP_solver_LA_VADD_271(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<271; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 74.
 */
void doublependulum_QP_solver_LA_VADD_74(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<74; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 430.
 */
void doublependulum_QP_solver_LA_VADD_430(doublependulum_QP_solver_FLOAT *x, doublependulum_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<430; i++){
		x[i] += y[i];
	}
}


/**
 * Backtracking line search for combined predictor/corrector step.
 * Update on variables with safety factor gamma (to keep us away from
 * boundary).
 */
int doublependulum_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(doublependulum_QP_solver_FLOAT *z, doublependulum_QP_solver_FLOAT *v, doublependulum_QP_solver_FLOAT *l, doublependulum_QP_solver_FLOAT *s, doublependulum_QP_solver_FLOAT *dz, doublependulum_QP_solver_FLOAT *dv, doublependulum_QP_solver_FLOAT *dl, doublependulum_QP_solver_FLOAT *ds, doublependulum_QP_solver_FLOAT *a, doublependulum_QP_solver_FLOAT *mu)
{
    int i, lsIt=1;       
    doublependulum_QP_solver_FLOAT dltemp;
    doublependulum_QP_solver_FLOAT dstemp;    
    doublependulum_QP_solver_FLOAT a_gamma;
            
    *a = 1.0;
    while( 1 ){                        

        /* check whether search criterion is fulfilled */
        for( i=0; i<430; i++ ){
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
        if( i == 430 ){
            break;
        } else {
            *a *= doublependulum_QP_solver_SET_LS_SCALE;
            if( *a < doublependulum_QP_solver_SET_LS_MINSTEP ){
                return doublependulum_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* update variables with safety margin */
    a_gamma = (*a)*doublependulum_QP_solver_SET_LS_MAXSTEP;
    
    /* primal variables */
    for( i=0; i<271; i++ ){
        z[i] += a_gamma*dz[i];
    }
    
    /* equality constraint multipliers */
    for( i=0; i<74; i++ ){
        v[i] += a_gamma*dv[i];
    }
    
    /* inequality constraint multipliers & slacks, also update mu */
    *mu = 0;
    for( i=0; i<430; i++ ){
        dltemp = l[i] + a_gamma*dl[i]; l[i] = dltemp;
        dstemp = s[i] + a_gamma*ds[i]; s[i] = dstemp;
        *mu += dltemp*dstemp;
    }
    
    *a = a_gamma;
    *mu /= (doublependulum_QP_solver_FLOAT)430;
    return lsIt;
}




/* VARIABLE DEFINITIONS ------------------------------------------------ */
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_z[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_v[74];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dz_aff[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dv_aff[74];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_grad_cost[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_grad_eq[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rd[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_l[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_s[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_lbys[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dl_aff[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ds_aff[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dz_cc[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dv_cc[74];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_dl_cc[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ds_cc[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ccrhs[430];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_grad_ineq[271];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_H00[19] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z00 = doublependulum_QP_solver_z + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff00 = doublependulum_QP_solver_dz_aff + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc00 = doublependulum_QP_solver_dz_cc + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd00 = doublependulum_QP_solver_rd + 0;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd00[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost00 = doublependulum_QP_solver_grad_cost + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq00 = doublependulum_QP_solver_grad_eq + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq00 = doublependulum_QP_solver_grad_ineq + 0;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv00[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v00 = doublependulum_QP_solver_v + 0;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re00[9];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta00[9];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc00[9];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff00 = doublependulum_QP_solver_dv_aff + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc00 = doublependulum_QP_solver_dv_cc + 0;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V00[171];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd00[45];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld00[45];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy00[9];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy00[9];
int doublependulum_QP_solver_lbIdx00[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb00 = doublependulum_QP_solver_l + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb00 = doublependulum_QP_solver_s + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb00 = doublependulum_QP_solver_lbys + 0;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb00[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff00 = doublependulum_QP_solver_dl_aff + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff00 = doublependulum_QP_solver_ds_aff + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc00 = doublependulum_QP_solver_dl_cc + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc00 = doublependulum_QP_solver_ds_cc + 0;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl00 = doublependulum_QP_solver_ccrhs + 0;
int doublependulum_QP_solver_ubIdx00[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub00 = doublependulum_QP_solver_l + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub00 = doublependulum_QP_solver_s + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub00 = doublependulum_QP_solver_lbys + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub00[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff00 = doublependulum_QP_solver_dl_aff + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff00 = doublependulum_QP_solver_ds_aff + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc00 = doublependulum_QP_solver_dl_cc + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc00 = doublependulum_QP_solver_ds_cc + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub00 = doublependulum_QP_solver_ccrhs + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi00[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z01 = doublependulum_QP_solver_z + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff01 = doublependulum_QP_solver_dz_aff + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc01 = doublependulum_QP_solver_dz_cc + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd01 = doublependulum_QP_solver_rd + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd01[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost01 = doublependulum_QP_solver_grad_cost + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq01 = doublependulum_QP_solver_grad_eq + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq01 = doublependulum_QP_solver_grad_ineq + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv01[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v01 = doublependulum_QP_solver_v + 9;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re01[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta01[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc01[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff01 = doublependulum_QP_solver_dv_aff + 9;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc01 = doublependulum_QP_solver_dv_cc + 9;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V01[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd01[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld01[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy01[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy01[5];
int doublependulum_QP_solver_lbIdx01[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb01 = doublependulum_QP_solver_l + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb01 = doublependulum_QP_solver_s + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb01 = doublependulum_QP_solver_lbys + 30;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb01[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff01 = doublependulum_QP_solver_dl_aff + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff01 = doublependulum_QP_solver_ds_aff + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc01 = doublependulum_QP_solver_dl_cc + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc01 = doublependulum_QP_solver_ds_cc + 30;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl01 = doublependulum_QP_solver_ccrhs + 30;
int doublependulum_QP_solver_ubIdx01[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub01 = doublependulum_QP_solver_l + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub01 = doublependulum_QP_solver_s + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub01 = doublependulum_QP_solver_lbys + 49;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub01[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff01 = doublependulum_QP_solver_dl_aff + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff01 = doublependulum_QP_solver_ds_aff + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc01 = doublependulum_QP_solver_dl_cc + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc01 = doublependulum_QP_solver_ds_cc + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub01 = doublependulum_QP_solver_ccrhs + 49;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi01[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_D01[171] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W01[171];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd01[45];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd01[45];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z02 = doublependulum_QP_solver_z + 38;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff02 = doublependulum_QP_solver_dz_aff + 38;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc02 = doublependulum_QP_solver_dz_cc + 38;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd02 = doublependulum_QP_solver_rd + 38;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd02[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost02 = doublependulum_QP_solver_grad_cost + 38;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq02 = doublependulum_QP_solver_grad_eq + 38;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq02 = doublependulum_QP_solver_grad_ineq + 38;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv02[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v02 = doublependulum_QP_solver_v + 14;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re02[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta02[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc02[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff02 = doublependulum_QP_solver_dv_aff + 14;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc02 = doublependulum_QP_solver_dv_cc + 14;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V02[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd02[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld02[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy02[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy02[5];
int doublependulum_QP_solver_lbIdx02[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb02 = doublependulum_QP_solver_l + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb02 = doublependulum_QP_solver_s + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb02 = doublependulum_QP_solver_lbys + 60;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb02[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff02 = doublependulum_QP_solver_dl_aff + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff02 = doublependulum_QP_solver_ds_aff + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc02 = doublependulum_QP_solver_dl_cc + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc02 = doublependulum_QP_solver_ds_cc + 60;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl02 = doublependulum_QP_solver_ccrhs + 60;
int doublependulum_QP_solver_ubIdx02[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub02 = doublependulum_QP_solver_l + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub02 = doublependulum_QP_solver_s + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub02 = doublependulum_QP_solver_lbys + 79;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub02[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff02 = doublependulum_QP_solver_dl_aff + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff02 = doublependulum_QP_solver_ds_aff + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc02 = doublependulum_QP_solver_dl_cc + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc02 = doublependulum_QP_solver_ds_cc + 79;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub02 = doublependulum_QP_solver_ccrhs + 79;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi02[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_D02[19] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W02[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd02[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd02[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z03 = doublependulum_QP_solver_z + 57;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff03 = doublependulum_QP_solver_dz_aff + 57;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc03 = doublependulum_QP_solver_dz_cc + 57;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd03 = doublependulum_QP_solver_rd + 57;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd03[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost03 = doublependulum_QP_solver_grad_cost + 57;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq03 = doublependulum_QP_solver_grad_eq + 57;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq03 = doublependulum_QP_solver_grad_ineq + 57;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv03[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v03 = doublependulum_QP_solver_v + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re03[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta03[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc03[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff03 = doublependulum_QP_solver_dv_aff + 19;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc03 = doublependulum_QP_solver_dv_cc + 19;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V03[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd03[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld03[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy03[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy03[5];
int doublependulum_QP_solver_lbIdx03[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb03 = doublependulum_QP_solver_l + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb03 = doublependulum_QP_solver_s + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb03 = doublependulum_QP_solver_lbys + 90;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb03[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff03 = doublependulum_QP_solver_dl_aff + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff03 = doublependulum_QP_solver_ds_aff + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc03 = doublependulum_QP_solver_dl_cc + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc03 = doublependulum_QP_solver_ds_cc + 90;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl03 = doublependulum_QP_solver_ccrhs + 90;
int doublependulum_QP_solver_ubIdx03[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub03 = doublependulum_QP_solver_l + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub03 = doublependulum_QP_solver_s + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub03 = doublependulum_QP_solver_lbys + 109;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub03[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff03 = doublependulum_QP_solver_dl_aff + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff03 = doublependulum_QP_solver_ds_aff + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc03 = doublependulum_QP_solver_dl_cc + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc03 = doublependulum_QP_solver_ds_cc + 109;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub03 = doublependulum_QP_solver_ccrhs + 109;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi03[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W03[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd03[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd03[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z04 = doublependulum_QP_solver_z + 76;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff04 = doublependulum_QP_solver_dz_aff + 76;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc04 = doublependulum_QP_solver_dz_cc + 76;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd04 = doublependulum_QP_solver_rd + 76;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd04[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost04 = doublependulum_QP_solver_grad_cost + 76;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq04 = doublependulum_QP_solver_grad_eq + 76;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq04 = doublependulum_QP_solver_grad_ineq + 76;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv04[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v04 = doublependulum_QP_solver_v + 24;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re04[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta04[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc04[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff04 = doublependulum_QP_solver_dv_aff + 24;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc04 = doublependulum_QP_solver_dv_cc + 24;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V04[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd04[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld04[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy04[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy04[5];
int doublependulum_QP_solver_lbIdx04[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb04 = doublependulum_QP_solver_l + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb04 = doublependulum_QP_solver_s + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb04 = doublependulum_QP_solver_lbys + 120;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb04[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff04 = doublependulum_QP_solver_dl_aff + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff04 = doublependulum_QP_solver_ds_aff + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc04 = doublependulum_QP_solver_dl_cc + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc04 = doublependulum_QP_solver_ds_cc + 120;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl04 = doublependulum_QP_solver_ccrhs + 120;
int doublependulum_QP_solver_ubIdx04[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub04 = doublependulum_QP_solver_l + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub04 = doublependulum_QP_solver_s + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub04 = doublependulum_QP_solver_lbys + 139;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub04[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff04 = doublependulum_QP_solver_dl_aff + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff04 = doublependulum_QP_solver_ds_aff + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc04 = doublependulum_QP_solver_dl_cc + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc04 = doublependulum_QP_solver_ds_cc + 139;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub04 = doublependulum_QP_solver_ccrhs + 139;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi04[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W04[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd04[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd04[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z05 = doublependulum_QP_solver_z + 95;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff05 = doublependulum_QP_solver_dz_aff + 95;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc05 = doublependulum_QP_solver_dz_cc + 95;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd05 = doublependulum_QP_solver_rd + 95;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd05[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost05 = doublependulum_QP_solver_grad_cost + 95;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq05 = doublependulum_QP_solver_grad_eq + 95;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq05 = doublependulum_QP_solver_grad_ineq + 95;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv05[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v05 = doublependulum_QP_solver_v + 29;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re05[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta05[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc05[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff05 = doublependulum_QP_solver_dv_aff + 29;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc05 = doublependulum_QP_solver_dv_cc + 29;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V05[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd05[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld05[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy05[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy05[5];
int doublependulum_QP_solver_lbIdx05[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb05 = doublependulum_QP_solver_l + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb05 = doublependulum_QP_solver_s + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb05 = doublependulum_QP_solver_lbys + 150;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb05[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff05 = doublependulum_QP_solver_dl_aff + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff05 = doublependulum_QP_solver_ds_aff + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc05 = doublependulum_QP_solver_dl_cc + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc05 = doublependulum_QP_solver_ds_cc + 150;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl05 = doublependulum_QP_solver_ccrhs + 150;
int doublependulum_QP_solver_ubIdx05[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub05 = doublependulum_QP_solver_l + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub05 = doublependulum_QP_solver_s + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub05 = doublependulum_QP_solver_lbys + 169;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub05[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff05 = doublependulum_QP_solver_dl_aff + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff05 = doublependulum_QP_solver_ds_aff + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc05 = doublependulum_QP_solver_dl_cc + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc05 = doublependulum_QP_solver_ds_cc + 169;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub05 = doublependulum_QP_solver_ccrhs + 169;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi05[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W05[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd05[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd05[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z06 = doublependulum_QP_solver_z + 114;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff06 = doublependulum_QP_solver_dz_aff + 114;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc06 = doublependulum_QP_solver_dz_cc + 114;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd06 = doublependulum_QP_solver_rd + 114;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd06[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost06 = doublependulum_QP_solver_grad_cost + 114;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq06 = doublependulum_QP_solver_grad_eq + 114;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq06 = doublependulum_QP_solver_grad_ineq + 114;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv06[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v06 = doublependulum_QP_solver_v + 34;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re06[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta06[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc06[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff06 = doublependulum_QP_solver_dv_aff + 34;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc06 = doublependulum_QP_solver_dv_cc + 34;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V06[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd06[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld06[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy06[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy06[5];
int doublependulum_QP_solver_lbIdx06[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb06 = doublependulum_QP_solver_l + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb06 = doublependulum_QP_solver_s + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb06 = doublependulum_QP_solver_lbys + 180;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb06[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff06 = doublependulum_QP_solver_dl_aff + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff06 = doublependulum_QP_solver_ds_aff + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc06 = doublependulum_QP_solver_dl_cc + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc06 = doublependulum_QP_solver_ds_cc + 180;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl06 = doublependulum_QP_solver_ccrhs + 180;
int doublependulum_QP_solver_ubIdx06[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub06 = doublependulum_QP_solver_l + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub06 = doublependulum_QP_solver_s + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub06 = doublependulum_QP_solver_lbys + 199;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub06[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff06 = doublependulum_QP_solver_dl_aff + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff06 = doublependulum_QP_solver_ds_aff + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc06 = doublependulum_QP_solver_dl_cc + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc06 = doublependulum_QP_solver_ds_cc + 199;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub06 = doublependulum_QP_solver_ccrhs + 199;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi06[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W06[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd06[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd06[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z07 = doublependulum_QP_solver_z + 133;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff07 = doublependulum_QP_solver_dz_aff + 133;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc07 = doublependulum_QP_solver_dz_cc + 133;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd07 = doublependulum_QP_solver_rd + 133;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd07[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost07 = doublependulum_QP_solver_grad_cost + 133;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq07 = doublependulum_QP_solver_grad_eq + 133;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq07 = doublependulum_QP_solver_grad_ineq + 133;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv07[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v07 = doublependulum_QP_solver_v + 39;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re07[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta07[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc07[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff07 = doublependulum_QP_solver_dv_aff + 39;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc07 = doublependulum_QP_solver_dv_cc + 39;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V07[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd07[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld07[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy07[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy07[5];
int doublependulum_QP_solver_lbIdx07[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb07 = doublependulum_QP_solver_l + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb07 = doublependulum_QP_solver_s + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb07 = doublependulum_QP_solver_lbys + 210;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb07[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff07 = doublependulum_QP_solver_dl_aff + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff07 = doublependulum_QP_solver_ds_aff + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc07 = doublependulum_QP_solver_dl_cc + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc07 = doublependulum_QP_solver_ds_cc + 210;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl07 = doublependulum_QP_solver_ccrhs + 210;
int doublependulum_QP_solver_ubIdx07[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub07 = doublependulum_QP_solver_l + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub07 = doublependulum_QP_solver_s + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub07 = doublependulum_QP_solver_lbys + 229;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub07[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff07 = doublependulum_QP_solver_dl_aff + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff07 = doublependulum_QP_solver_ds_aff + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc07 = doublependulum_QP_solver_dl_cc + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc07 = doublependulum_QP_solver_ds_cc + 229;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub07 = doublependulum_QP_solver_ccrhs + 229;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi07[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W07[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd07[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd07[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z08 = doublependulum_QP_solver_z + 152;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff08 = doublependulum_QP_solver_dz_aff + 152;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc08 = doublependulum_QP_solver_dz_cc + 152;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd08 = doublependulum_QP_solver_rd + 152;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd08[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost08 = doublependulum_QP_solver_grad_cost + 152;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq08 = doublependulum_QP_solver_grad_eq + 152;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq08 = doublependulum_QP_solver_grad_ineq + 152;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv08[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v08 = doublependulum_QP_solver_v + 44;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re08[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta08[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc08[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff08 = doublependulum_QP_solver_dv_aff + 44;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc08 = doublependulum_QP_solver_dv_cc + 44;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V08[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd08[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld08[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy08[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy08[5];
int doublependulum_QP_solver_lbIdx08[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb08 = doublependulum_QP_solver_l + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb08 = doublependulum_QP_solver_s + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb08 = doublependulum_QP_solver_lbys + 240;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb08[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff08 = doublependulum_QP_solver_dl_aff + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff08 = doublependulum_QP_solver_ds_aff + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc08 = doublependulum_QP_solver_dl_cc + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc08 = doublependulum_QP_solver_ds_cc + 240;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl08 = doublependulum_QP_solver_ccrhs + 240;
int doublependulum_QP_solver_ubIdx08[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub08 = doublependulum_QP_solver_l + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub08 = doublependulum_QP_solver_s + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub08 = doublependulum_QP_solver_lbys + 259;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub08[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff08 = doublependulum_QP_solver_dl_aff + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff08 = doublependulum_QP_solver_ds_aff + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc08 = doublependulum_QP_solver_dl_cc + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc08 = doublependulum_QP_solver_ds_cc + 259;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub08 = doublependulum_QP_solver_ccrhs + 259;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi08[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W08[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd08[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd08[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z09 = doublependulum_QP_solver_z + 171;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff09 = doublependulum_QP_solver_dz_aff + 171;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc09 = doublependulum_QP_solver_dz_cc + 171;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd09 = doublependulum_QP_solver_rd + 171;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd09[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost09 = doublependulum_QP_solver_grad_cost + 171;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq09 = doublependulum_QP_solver_grad_eq + 171;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq09 = doublependulum_QP_solver_grad_ineq + 171;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv09[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v09 = doublependulum_QP_solver_v + 49;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re09[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta09[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc09[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff09 = doublependulum_QP_solver_dv_aff + 49;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc09 = doublependulum_QP_solver_dv_cc + 49;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V09[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd09[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld09[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy09[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy09[5];
int doublependulum_QP_solver_lbIdx09[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb09 = doublependulum_QP_solver_l + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb09 = doublependulum_QP_solver_s + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb09 = doublependulum_QP_solver_lbys + 270;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb09[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff09 = doublependulum_QP_solver_dl_aff + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff09 = doublependulum_QP_solver_ds_aff + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc09 = doublependulum_QP_solver_dl_cc + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc09 = doublependulum_QP_solver_ds_cc + 270;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl09 = doublependulum_QP_solver_ccrhs + 270;
int doublependulum_QP_solver_ubIdx09[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub09 = doublependulum_QP_solver_l + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub09 = doublependulum_QP_solver_s + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub09 = doublependulum_QP_solver_lbys + 289;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub09[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff09 = doublependulum_QP_solver_dl_aff + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff09 = doublependulum_QP_solver_ds_aff + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc09 = doublependulum_QP_solver_dl_cc + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc09 = doublependulum_QP_solver_ds_cc + 289;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub09 = doublependulum_QP_solver_ccrhs + 289;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi09[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W09[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd09[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd09[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z10 = doublependulum_QP_solver_z + 190;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff10 = doublependulum_QP_solver_dz_aff + 190;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc10 = doublependulum_QP_solver_dz_cc + 190;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd10 = doublependulum_QP_solver_rd + 190;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd10[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost10 = doublependulum_QP_solver_grad_cost + 190;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq10 = doublependulum_QP_solver_grad_eq + 190;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq10 = doublependulum_QP_solver_grad_ineq + 190;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv10[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v10 = doublependulum_QP_solver_v + 54;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re10[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta10[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc10[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff10 = doublependulum_QP_solver_dv_aff + 54;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc10 = doublependulum_QP_solver_dv_cc + 54;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V10[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd10[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld10[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy10[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy10[5];
int doublependulum_QP_solver_lbIdx10[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb10 = doublependulum_QP_solver_l + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb10 = doublependulum_QP_solver_s + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb10 = doublependulum_QP_solver_lbys + 300;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb10[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff10 = doublependulum_QP_solver_dl_aff + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff10 = doublependulum_QP_solver_ds_aff + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc10 = doublependulum_QP_solver_dl_cc + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc10 = doublependulum_QP_solver_ds_cc + 300;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl10 = doublependulum_QP_solver_ccrhs + 300;
int doublependulum_QP_solver_ubIdx10[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub10 = doublependulum_QP_solver_l + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub10 = doublependulum_QP_solver_s + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub10 = doublependulum_QP_solver_lbys + 319;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub10[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff10 = doublependulum_QP_solver_dl_aff + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff10 = doublependulum_QP_solver_ds_aff + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc10 = doublependulum_QP_solver_dl_cc + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc10 = doublependulum_QP_solver_ds_cc + 319;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub10 = doublependulum_QP_solver_ccrhs + 319;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi10[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W10[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd10[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd10[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z11 = doublependulum_QP_solver_z + 209;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff11 = doublependulum_QP_solver_dz_aff + 209;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc11 = doublependulum_QP_solver_dz_cc + 209;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd11 = doublependulum_QP_solver_rd + 209;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd11[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost11 = doublependulum_QP_solver_grad_cost + 209;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq11 = doublependulum_QP_solver_grad_eq + 209;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq11 = doublependulum_QP_solver_grad_ineq + 209;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv11[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v11 = doublependulum_QP_solver_v + 59;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re11[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta11[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc11[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff11 = doublependulum_QP_solver_dv_aff + 59;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc11 = doublependulum_QP_solver_dv_cc + 59;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V11[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd11[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld11[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy11[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy11[5];
int doublependulum_QP_solver_lbIdx11[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb11 = doublependulum_QP_solver_l + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb11 = doublependulum_QP_solver_s + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb11 = doublependulum_QP_solver_lbys + 330;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb11[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff11 = doublependulum_QP_solver_dl_aff + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff11 = doublependulum_QP_solver_ds_aff + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc11 = doublependulum_QP_solver_dl_cc + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc11 = doublependulum_QP_solver_ds_cc + 330;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl11 = doublependulum_QP_solver_ccrhs + 330;
int doublependulum_QP_solver_ubIdx11[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub11 = doublependulum_QP_solver_l + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub11 = doublependulum_QP_solver_s + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub11 = doublependulum_QP_solver_lbys + 349;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub11[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff11 = doublependulum_QP_solver_dl_aff + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff11 = doublependulum_QP_solver_ds_aff + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc11 = doublependulum_QP_solver_dl_cc + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc11 = doublependulum_QP_solver_ds_cc + 349;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub11 = doublependulum_QP_solver_ccrhs + 349;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi11[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W11[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd11[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd11[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z12 = doublependulum_QP_solver_z + 228;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff12 = doublependulum_QP_solver_dz_aff + 228;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc12 = doublependulum_QP_solver_dz_cc + 228;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd12 = doublependulum_QP_solver_rd + 228;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd12[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost12 = doublependulum_QP_solver_grad_cost + 228;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq12 = doublependulum_QP_solver_grad_eq + 228;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq12 = doublependulum_QP_solver_grad_ineq + 228;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv12[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v12 = doublependulum_QP_solver_v + 64;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re12[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta12[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc12[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff12 = doublependulum_QP_solver_dv_aff + 64;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc12 = doublependulum_QP_solver_dv_cc + 64;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V12[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd12[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld12[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy12[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy12[5];
int doublependulum_QP_solver_lbIdx12[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb12 = doublependulum_QP_solver_l + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb12 = doublependulum_QP_solver_s + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb12 = doublependulum_QP_solver_lbys + 360;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb12[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff12 = doublependulum_QP_solver_dl_aff + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff12 = doublependulum_QP_solver_ds_aff + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc12 = doublependulum_QP_solver_dl_cc + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc12 = doublependulum_QP_solver_ds_cc + 360;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl12 = doublependulum_QP_solver_ccrhs + 360;
int doublependulum_QP_solver_ubIdx12[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub12 = doublependulum_QP_solver_l + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub12 = doublependulum_QP_solver_s + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub12 = doublependulum_QP_solver_lbys + 379;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub12[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff12 = doublependulum_QP_solver_dl_aff + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff12 = doublependulum_QP_solver_ds_aff + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc12 = doublependulum_QP_solver_dl_cc + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc12 = doublependulum_QP_solver_ds_cc + 379;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub12 = doublependulum_QP_solver_ccrhs + 379;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi12[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W12[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd12[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd12[25];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z13 = doublependulum_QP_solver_z + 247;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff13 = doublependulum_QP_solver_dz_aff + 247;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc13 = doublependulum_QP_solver_dz_cc + 247;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd13 = doublependulum_QP_solver_rd + 247;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd13[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost13 = doublependulum_QP_solver_grad_cost + 247;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq13 = doublependulum_QP_solver_grad_eq + 247;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq13 = doublependulum_QP_solver_grad_ineq + 247;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv13[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_v13 = doublependulum_QP_solver_v + 69;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_re13[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_beta13[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_betacc13[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvaff13 = doublependulum_QP_solver_dv_aff + 69;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dvcc13 = doublependulum_QP_solver_dv_cc + 69;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_V13[95];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Yd13[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ld13[15];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_yy13[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_bmy13[5];
int doublependulum_QP_solver_lbIdx13[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb13 = doublependulum_QP_solver_l + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb13 = doublependulum_QP_solver_s + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb13 = doublependulum_QP_solver_lbys + 390;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb13[19];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff13 = doublependulum_QP_solver_dl_aff + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff13 = doublependulum_QP_solver_ds_aff + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc13 = doublependulum_QP_solver_dl_cc + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc13 = doublependulum_QP_solver_ds_cc + 390;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl13 = doublependulum_QP_solver_ccrhs + 390;
int doublependulum_QP_solver_ubIdx13[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub13 = doublependulum_QP_solver_l + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub13 = doublependulum_QP_solver_s + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub13 = doublependulum_QP_solver_lbys + 409;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub13[11];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff13 = doublependulum_QP_solver_dl_aff + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff13 = doublependulum_QP_solver_ds_aff + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc13 = doublependulum_QP_solver_dl_cc + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc13 = doublependulum_QP_solver_ds_cc + 409;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub13 = doublependulum_QP_solver_ccrhs + 409;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi13[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W13[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Ysd13[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lsd13[25];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_H14[5] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_f14[5] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_z14 = doublependulum_QP_solver_z + 266;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzaff14 = doublependulum_QP_solver_dz_aff + 266;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dzcc14 = doublependulum_QP_solver_dz_cc + 266;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_rd14 = doublependulum_QP_solver_rd + 266;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Lbyrd14[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_cost14 = doublependulum_QP_solver_grad_cost + 266;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_eq14 = doublependulum_QP_solver_grad_eq + 266;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_grad_ineq14 = doublependulum_QP_solver_grad_ineq + 266;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_ctv14[5];
int doublependulum_QP_solver_lbIdx14[5] = {0, 1, 2, 3, 4};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llb14 = doublependulum_QP_solver_l + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_slb14 = doublependulum_QP_solver_s + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_llbbyslb14 = doublependulum_QP_solver_lbys + 420;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_rilb14[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbaff14 = doublependulum_QP_solver_dl_aff + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbaff14 = doublependulum_QP_solver_ds_aff + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dllbcc14 = doublependulum_QP_solver_dl_cc + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dslbcc14 = doublependulum_QP_solver_ds_cc + 420;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsl14 = doublependulum_QP_solver_ccrhs + 420;
int doublependulum_QP_solver_ubIdx14[5] = {0, 1, 2, 3, 4};
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lub14 = doublependulum_QP_solver_l + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_sub14 = doublependulum_QP_solver_s + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_lubbysub14 = doublependulum_QP_solver_lbys + 425;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_riub14[5];
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubaff14 = doublependulum_QP_solver_dl_aff + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubaff14 = doublependulum_QP_solver_ds_aff + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dlubcc14 = doublependulum_QP_solver_dl_cc + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_dsubcc14 = doublependulum_QP_solver_ds_cc + 425;
doublependulum_QP_solver_FLOAT* doublependulum_QP_solver_ccrhsub14 = doublependulum_QP_solver_ccrhs + 425;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Phi14[5];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_D14[5] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_W14[5];
doublependulum_QP_solver_FLOAT musigma;
doublependulum_QP_solver_FLOAT sigma_3rdroot;
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Diag1_0[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_Diag2_0[19];
doublependulum_QP_solver_FLOAT doublependulum_QP_solver_L_0[171];




/* SOLVER CODE --------------------------------------------------------- */
int doublependulum_QP_solver_solve(doublependulum_QP_solver_params* params, doublependulum_QP_solver_output* output, doublependulum_QP_solver_info* info)
{	
int exitcode;

#if doublependulum_QP_solver_SET_TIMING == 1
	doublependulum_QP_solver_timer solvertimer;
	doublependulum_QP_solver_tic(&solvertimer);
#endif
/* FUNCTION CALLS INTO LA LIBRARY -------------------------------------- */
info->it = 0;
doublependulum_QP_solver_LA_INITIALIZEVECTOR_271(doublependulum_QP_solver_z, 0);
doublependulum_QP_solver_LA_INITIALIZEVECTOR_74(doublependulum_QP_solver_v, 1);
doublependulum_QP_solver_LA_INITIALIZEVECTOR_430(doublependulum_QP_solver_l, 10);
doublependulum_QP_solver_LA_INITIALIZEVECTOR_430(doublependulum_QP_solver_s, 10);
info->mu = 0;
doublependulum_QP_solver_LA_DOTACC_430(doublependulum_QP_solver_l, doublependulum_QP_solver_s, &info->mu);
info->mu /= 430;
while( 1 ){
info->pobj = 0;
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f1, doublependulum_QP_solver_z00, doublependulum_QP_solver_grad_cost00, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f2, doublependulum_QP_solver_z01, doublependulum_QP_solver_grad_cost01, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f3, doublependulum_QP_solver_z02, doublependulum_QP_solver_grad_cost02, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f4, doublependulum_QP_solver_z03, doublependulum_QP_solver_grad_cost03, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f5, doublependulum_QP_solver_z04, doublependulum_QP_solver_grad_cost04, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f6, doublependulum_QP_solver_z05, doublependulum_QP_solver_grad_cost05, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f7, doublependulum_QP_solver_z06, doublependulum_QP_solver_grad_cost06, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f8, doublependulum_QP_solver_z07, doublependulum_QP_solver_grad_cost07, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f9, doublependulum_QP_solver_z08, doublependulum_QP_solver_grad_cost08, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f10, doublependulum_QP_solver_z09, doublependulum_QP_solver_grad_cost09, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f11, doublependulum_QP_solver_z10, doublependulum_QP_solver_grad_cost10, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f12, doublependulum_QP_solver_z11, doublependulum_QP_solver_grad_cost11, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f13, doublependulum_QP_solver_z12, doublependulum_QP_solver_grad_cost12, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_19(doublependulum_QP_solver_H00, params->f14, doublependulum_QP_solver_z13, doublependulum_QP_solver_grad_cost13, &info->pobj);
doublependulum_QP_solver_LA_DIAG_QUADFCN_5(doublependulum_QP_solver_H14, doublependulum_QP_solver_f14, doublependulum_QP_solver_z14, doublependulum_QP_solver_grad_cost14, &info->pobj);
info->res_eq = 0;
info->dgap = 0;
doublependulum_QP_solver_LA_DENSE_MVMSUB3_9_19_19(params->C1, doublependulum_QP_solver_z00, doublependulum_QP_solver_D01, doublependulum_QP_solver_z01, params->e1, doublependulum_QP_solver_v00, doublependulum_QP_solver_re00, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C2, doublependulum_QP_solver_z01, doublependulum_QP_solver_D02, doublependulum_QP_solver_z02, params->e2, doublependulum_QP_solver_v01, doublependulum_QP_solver_re01, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C3, doublependulum_QP_solver_z02, doublependulum_QP_solver_D02, doublependulum_QP_solver_z03, params->e3, doublependulum_QP_solver_v02, doublependulum_QP_solver_re02, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C4, doublependulum_QP_solver_z03, doublependulum_QP_solver_D02, doublependulum_QP_solver_z04, params->e4, doublependulum_QP_solver_v03, doublependulum_QP_solver_re03, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C5, doublependulum_QP_solver_z04, doublependulum_QP_solver_D02, doublependulum_QP_solver_z05, params->e5, doublependulum_QP_solver_v04, doublependulum_QP_solver_re04, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C6, doublependulum_QP_solver_z05, doublependulum_QP_solver_D02, doublependulum_QP_solver_z06, params->e6, doublependulum_QP_solver_v05, doublependulum_QP_solver_re05, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C7, doublependulum_QP_solver_z06, doublependulum_QP_solver_D02, doublependulum_QP_solver_z07, params->e7, doublependulum_QP_solver_v06, doublependulum_QP_solver_re06, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C8, doublependulum_QP_solver_z07, doublependulum_QP_solver_D02, doublependulum_QP_solver_z08, params->e8, doublependulum_QP_solver_v07, doublependulum_QP_solver_re07, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C9, doublependulum_QP_solver_z08, doublependulum_QP_solver_D02, doublependulum_QP_solver_z09, params->e9, doublependulum_QP_solver_v08, doublependulum_QP_solver_re08, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C10, doublependulum_QP_solver_z09, doublependulum_QP_solver_D02, doublependulum_QP_solver_z10, params->e10, doublependulum_QP_solver_v09, doublependulum_QP_solver_re09, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C11, doublependulum_QP_solver_z10, doublependulum_QP_solver_D02, doublependulum_QP_solver_z11, params->e11, doublependulum_QP_solver_v10, doublependulum_QP_solver_re10, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C12, doublependulum_QP_solver_z11, doublependulum_QP_solver_D02, doublependulum_QP_solver_z12, params->e12, doublependulum_QP_solver_v11, doublependulum_QP_solver_re11, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_19(params->C13, doublependulum_QP_solver_z12, doublependulum_QP_solver_D02, doublependulum_QP_solver_z13, params->e13, doublependulum_QP_solver_v12, doublependulum_QP_solver_re12, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_19_5(params->C14, doublependulum_QP_solver_z13, doublependulum_QP_solver_D14, doublependulum_QP_solver_z14, params->e14, doublependulum_QP_solver_v13, doublependulum_QP_solver_re13, &info->dgap, &info->res_eq);
doublependulum_QP_solver_LA_DENSE_MTVM_9_19(params->C1, doublependulum_QP_solver_v00, doublependulum_QP_solver_grad_eq00);
doublependulum_QP_solver_LA_DENSE_MTVM2_5_19_9(params->C2, doublependulum_QP_solver_v01, doublependulum_QP_solver_D01, doublependulum_QP_solver_v00, doublependulum_QP_solver_grad_eq01);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C3, doublependulum_QP_solver_v02, doublependulum_QP_solver_D02, doublependulum_QP_solver_v01, doublependulum_QP_solver_grad_eq02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C4, doublependulum_QP_solver_v03, doublependulum_QP_solver_D02, doublependulum_QP_solver_v02, doublependulum_QP_solver_grad_eq03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C5, doublependulum_QP_solver_v04, doublependulum_QP_solver_D02, doublependulum_QP_solver_v03, doublependulum_QP_solver_grad_eq04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C6, doublependulum_QP_solver_v05, doublependulum_QP_solver_D02, doublependulum_QP_solver_v04, doublependulum_QP_solver_grad_eq05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C7, doublependulum_QP_solver_v06, doublependulum_QP_solver_D02, doublependulum_QP_solver_v05, doublependulum_QP_solver_grad_eq06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C8, doublependulum_QP_solver_v07, doublependulum_QP_solver_D02, doublependulum_QP_solver_v06, doublependulum_QP_solver_grad_eq07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C9, doublependulum_QP_solver_v08, doublependulum_QP_solver_D02, doublependulum_QP_solver_v07, doublependulum_QP_solver_grad_eq08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C10, doublependulum_QP_solver_v09, doublependulum_QP_solver_D02, doublependulum_QP_solver_v08, doublependulum_QP_solver_grad_eq09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C11, doublependulum_QP_solver_v10, doublependulum_QP_solver_D02, doublependulum_QP_solver_v09, doublependulum_QP_solver_grad_eq10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C12, doublependulum_QP_solver_v11, doublependulum_QP_solver_D02, doublependulum_QP_solver_v10, doublependulum_QP_solver_grad_eq11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C13, doublependulum_QP_solver_v12, doublependulum_QP_solver_D02, doublependulum_QP_solver_v11, doublependulum_QP_solver_grad_eq12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C14, doublependulum_QP_solver_v13, doublependulum_QP_solver_D02, doublependulum_QP_solver_v12, doublependulum_QP_solver_grad_eq13);
doublependulum_QP_solver_LA_DIAGZERO_MTVM_5_5(doublependulum_QP_solver_D14, doublependulum_QP_solver_v13, doublependulum_QP_solver_grad_eq14);
info->res_ineq = 0;
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb1, doublependulum_QP_solver_z00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_llb00, doublependulum_QP_solver_slb00, doublependulum_QP_solver_rilb00, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z00, doublependulum_QP_solver_ubIdx00, params->ub1, doublependulum_QP_solver_lub00, doublependulum_QP_solver_sub00, doublependulum_QP_solver_riub00, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb2, doublependulum_QP_solver_z01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_llb01, doublependulum_QP_solver_slb01, doublependulum_QP_solver_rilb01, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z01, doublependulum_QP_solver_ubIdx01, params->ub2, doublependulum_QP_solver_lub01, doublependulum_QP_solver_sub01, doublependulum_QP_solver_riub01, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb3, doublependulum_QP_solver_z02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_llb02, doublependulum_QP_solver_slb02, doublependulum_QP_solver_rilb02, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z02, doublependulum_QP_solver_ubIdx02, params->ub3, doublependulum_QP_solver_lub02, doublependulum_QP_solver_sub02, doublependulum_QP_solver_riub02, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb4, doublependulum_QP_solver_z03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_llb03, doublependulum_QP_solver_slb03, doublependulum_QP_solver_rilb03, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z03, doublependulum_QP_solver_ubIdx03, params->ub4, doublependulum_QP_solver_lub03, doublependulum_QP_solver_sub03, doublependulum_QP_solver_riub03, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb5, doublependulum_QP_solver_z04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_llb04, doublependulum_QP_solver_slb04, doublependulum_QP_solver_rilb04, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z04, doublependulum_QP_solver_ubIdx04, params->ub5, doublependulum_QP_solver_lub04, doublependulum_QP_solver_sub04, doublependulum_QP_solver_riub04, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb6, doublependulum_QP_solver_z05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_llb05, doublependulum_QP_solver_slb05, doublependulum_QP_solver_rilb05, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z05, doublependulum_QP_solver_ubIdx05, params->ub6, doublependulum_QP_solver_lub05, doublependulum_QP_solver_sub05, doublependulum_QP_solver_riub05, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb7, doublependulum_QP_solver_z06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_llb06, doublependulum_QP_solver_slb06, doublependulum_QP_solver_rilb06, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z06, doublependulum_QP_solver_ubIdx06, params->ub7, doublependulum_QP_solver_lub06, doublependulum_QP_solver_sub06, doublependulum_QP_solver_riub06, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb8, doublependulum_QP_solver_z07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_llb07, doublependulum_QP_solver_slb07, doublependulum_QP_solver_rilb07, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z07, doublependulum_QP_solver_ubIdx07, params->ub8, doublependulum_QP_solver_lub07, doublependulum_QP_solver_sub07, doublependulum_QP_solver_riub07, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb9, doublependulum_QP_solver_z08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_llb08, doublependulum_QP_solver_slb08, doublependulum_QP_solver_rilb08, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z08, doublependulum_QP_solver_ubIdx08, params->ub9, doublependulum_QP_solver_lub08, doublependulum_QP_solver_sub08, doublependulum_QP_solver_riub08, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb10, doublependulum_QP_solver_z09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_llb09, doublependulum_QP_solver_slb09, doublependulum_QP_solver_rilb09, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z09, doublependulum_QP_solver_ubIdx09, params->ub10, doublependulum_QP_solver_lub09, doublependulum_QP_solver_sub09, doublependulum_QP_solver_riub09, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb11, doublependulum_QP_solver_z10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_llb10, doublependulum_QP_solver_slb10, doublependulum_QP_solver_rilb10, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z10, doublependulum_QP_solver_ubIdx10, params->ub11, doublependulum_QP_solver_lub10, doublependulum_QP_solver_sub10, doublependulum_QP_solver_riub10, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb12, doublependulum_QP_solver_z11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_llb11, doublependulum_QP_solver_slb11, doublependulum_QP_solver_rilb11, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z11, doublependulum_QP_solver_ubIdx11, params->ub12, doublependulum_QP_solver_lub11, doublependulum_QP_solver_sub11, doublependulum_QP_solver_riub11, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb13, doublependulum_QP_solver_z12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_llb12, doublependulum_QP_solver_slb12, doublependulum_QP_solver_rilb12, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z12, doublependulum_QP_solver_ubIdx12, params->ub13, doublependulum_QP_solver_lub12, doublependulum_QP_solver_sub12, doublependulum_QP_solver_riub12, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_19(params->lb14, doublependulum_QP_solver_z13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_llb13, doublependulum_QP_solver_slb13, doublependulum_QP_solver_rilb13, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_11(doublependulum_QP_solver_z13, doublependulum_QP_solver_ubIdx13, params->ub14, doublependulum_QP_solver_lub13, doublependulum_QP_solver_sub13, doublependulum_QP_solver_riub13, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD3_5(params->lb15, doublependulum_QP_solver_z14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_llb14, doublependulum_QP_solver_slb14, doublependulum_QP_solver_rilb14, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_VSUBADD2_5(doublependulum_QP_solver_z14, doublependulum_QP_solver_ubIdx14, params->ub15, doublependulum_QP_solver_lub14, doublependulum_QP_solver_sub14, doublependulum_QP_solver_riub14, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub00, doublependulum_QP_solver_sub00, doublependulum_QP_solver_riub00, doublependulum_QP_solver_llb00, doublependulum_QP_solver_slb00, doublependulum_QP_solver_rilb00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_ubIdx00, doublependulum_QP_solver_grad_ineq00, doublependulum_QP_solver_lubbysub00, doublependulum_QP_solver_llbbyslb00);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub01, doublependulum_QP_solver_sub01, doublependulum_QP_solver_riub01, doublependulum_QP_solver_llb01, doublependulum_QP_solver_slb01, doublependulum_QP_solver_rilb01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_ubIdx01, doublependulum_QP_solver_grad_ineq01, doublependulum_QP_solver_lubbysub01, doublependulum_QP_solver_llbbyslb01);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub02, doublependulum_QP_solver_sub02, doublependulum_QP_solver_riub02, doublependulum_QP_solver_llb02, doublependulum_QP_solver_slb02, doublependulum_QP_solver_rilb02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_ubIdx02, doublependulum_QP_solver_grad_ineq02, doublependulum_QP_solver_lubbysub02, doublependulum_QP_solver_llbbyslb02);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub03, doublependulum_QP_solver_sub03, doublependulum_QP_solver_riub03, doublependulum_QP_solver_llb03, doublependulum_QP_solver_slb03, doublependulum_QP_solver_rilb03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_ubIdx03, doublependulum_QP_solver_grad_ineq03, doublependulum_QP_solver_lubbysub03, doublependulum_QP_solver_llbbyslb03);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub04, doublependulum_QP_solver_sub04, doublependulum_QP_solver_riub04, doublependulum_QP_solver_llb04, doublependulum_QP_solver_slb04, doublependulum_QP_solver_rilb04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_ubIdx04, doublependulum_QP_solver_grad_ineq04, doublependulum_QP_solver_lubbysub04, doublependulum_QP_solver_llbbyslb04);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub05, doublependulum_QP_solver_sub05, doublependulum_QP_solver_riub05, doublependulum_QP_solver_llb05, doublependulum_QP_solver_slb05, doublependulum_QP_solver_rilb05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_ubIdx05, doublependulum_QP_solver_grad_ineq05, doublependulum_QP_solver_lubbysub05, doublependulum_QP_solver_llbbyslb05);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub06, doublependulum_QP_solver_sub06, doublependulum_QP_solver_riub06, doublependulum_QP_solver_llb06, doublependulum_QP_solver_slb06, doublependulum_QP_solver_rilb06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_ubIdx06, doublependulum_QP_solver_grad_ineq06, doublependulum_QP_solver_lubbysub06, doublependulum_QP_solver_llbbyslb06);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub07, doublependulum_QP_solver_sub07, doublependulum_QP_solver_riub07, doublependulum_QP_solver_llb07, doublependulum_QP_solver_slb07, doublependulum_QP_solver_rilb07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_ubIdx07, doublependulum_QP_solver_grad_ineq07, doublependulum_QP_solver_lubbysub07, doublependulum_QP_solver_llbbyslb07);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub08, doublependulum_QP_solver_sub08, doublependulum_QP_solver_riub08, doublependulum_QP_solver_llb08, doublependulum_QP_solver_slb08, doublependulum_QP_solver_rilb08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_ubIdx08, doublependulum_QP_solver_grad_ineq08, doublependulum_QP_solver_lubbysub08, doublependulum_QP_solver_llbbyslb08);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub09, doublependulum_QP_solver_sub09, doublependulum_QP_solver_riub09, doublependulum_QP_solver_llb09, doublependulum_QP_solver_slb09, doublependulum_QP_solver_rilb09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_ubIdx09, doublependulum_QP_solver_grad_ineq09, doublependulum_QP_solver_lubbysub09, doublependulum_QP_solver_llbbyslb09);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub10, doublependulum_QP_solver_sub10, doublependulum_QP_solver_riub10, doublependulum_QP_solver_llb10, doublependulum_QP_solver_slb10, doublependulum_QP_solver_rilb10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_ubIdx10, doublependulum_QP_solver_grad_ineq10, doublependulum_QP_solver_lubbysub10, doublependulum_QP_solver_llbbyslb10);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub11, doublependulum_QP_solver_sub11, doublependulum_QP_solver_riub11, doublependulum_QP_solver_llb11, doublependulum_QP_solver_slb11, doublependulum_QP_solver_rilb11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_ubIdx11, doublependulum_QP_solver_grad_ineq11, doublependulum_QP_solver_lubbysub11, doublependulum_QP_solver_llbbyslb11);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub12, doublependulum_QP_solver_sub12, doublependulum_QP_solver_riub12, doublependulum_QP_solver_llb12, doublependulum_QP_solver_slb12, doublependulum_QP_solver_rilb12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_ubIdx12, doublependulum_QP_solver_grad_ineq12, doublependulum_QP_solver_lubbysub12, doublependulum_QP_solver_llbbyslb12);
doublependulum_QP_solver_LA_INEQ_B_GRAD_19_19_11(doublependulum_QP_solver_lub13, doublependulum_QP_solver_sub13, doublependulum_QP_solver_riub13, doublependulum_QP_solver_llb13, doublependulum_QP_solver_slb13, doublependulum_QP_solver_rilb13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_ubIdx13, doublependulum_QP_solver_grad_ineq13, doublependulum_QP_solver_lubbysub13, doublependulum_QP_solver_llbbyslb13);
doublependulum_QP_solver_LA_INEQ_B_GRAD_5_5_5(doublependulum_QP_solver_lub14, doublependulum_QP_solver_sub14, doublependulum_QP_solver_riub14, doublependulum_QP_solver_llb14, doublependulum_QP_solver_slb14, doublependulum_QP_solver_rilb14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_ubIdx14, doublependulum_QP_solver_grad_ineq14, doublependulum_QP_solver_lubbysub14, doublependulum_QP_solver_llbbyslb14);
info->dobj = info->pobj - info->dgap;
info->rdgap = info->pobj ? info->dgap / info->pobj : 1e6;
if( info->rdgap < 0 ) info->rdgap = -info->rdgap;
if( info->mu < doublependulum_QP_solver_SET_ACC_KKTCOMPL
    && (info->rdgap < doublependulum_QP_solver_SET_ACC_RDGAP || info->dgap < doublependulum_QP_solver_SET_ACC_KKTCOMPL)
    && info->res_eq < doublependulum_QP_solver_SET_ACC_RESEQ
    && info->res_ineq < doublependulum_QP_solver_SET_ACC_RESINEQ ){
exitcode = doublependulum_QP_solver_OPTIMAL; break; }
if( info->it == doublependulum_QP_solver_SET_MAXIT ){
exitcode = doublependulum_QP_solver_MAXITREACHED; break; }
doublependulum_QP_solver_LA_VVADD3_271(doublependulum_QP_solver_grad_cost, doublependulum_QP_solver_grad_eq, doublependulum_QP_solver_grad_ineq, doublependulum_QP_solver_rd);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_lubbysub00, doublependulum_QP_solver_ubIdx00, doublependulum_QP_solver_Phi00);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_19(doublependulum_QP_solver_Phi00, params->C1, doublependulum_QP_solver_V00);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi00, doublependulum_QP_solver_rd00, doublependulum_QP_solver_Lbyrd00);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_lubbysub01, doublependulum_QP_solver_ubIdx01, doublependulum_QP_solver_Phi01);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi01, params->C2, doublependulum_QP_solver_V01);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_19(doublependulum_QP_solver_Phi01, doublependulum_QP_solver_D01, doublependulum_QP_solver_W01);
doublependulum_QP_solver_LA_DENSE_MMTM_9_19_5(doublependulum_QP_solver_W01, doublependulum_QP_solver_V01, doublependulum_QP_solver_Ysd01);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi01, doublependulum_QP_solver_rd01, doublependulum_QP_solver_Lbyrd01);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_lubbysub02, doublependulum_QP_solver_ubIdx02, doublependulum_QP_solver_Phi02);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi02, params->C3, doublependulum_QP_solver_V02);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi02, doublependulum_QP_solver_D02, doublependulum_QP_solver_W02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W02, doublependulum_QP_solver_V02, doublependulum_QP_solver_Ysd02);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi02, doublependulum_QP_solver_rd02, doublependulum_QP_solver_Lbyrd02);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_lubbysub03, doublependulum_QP_solver_ubIdx03, doublependulum_QP_solver_Phi03);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi03, params->C4, doublependulum_QP_solver_V03);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi03, doublependulum_QP_solver_D02, doublependulum_QP_solver_W03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W03, doublependulum_QP_solver_V03, doublependulum_QP_solver_Ysd03);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi03, doublependulum_QP_solver_rd03, doublependulum_QP_solver_Lbyrd03);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_lubbysub04, doublependulum_QP_solver_ubIdx04, doublependulum_QP_solver_Phi04);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi04, params->C5, doublependulum_QP_solver_V04);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi04, doublependulum_QP_solver_D02, doublependulum_QP_solver_W04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W04, doublependulum_QP_solver_V04, doublependulum_QP_solver_Ysd04);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi04, doublependulum_QP_solver_rd04, doublependulum_QP_solver_Lbyrd04);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_lubbysub05, doublependulum_QP_solver_ubIdx05, doublependulum_QP_solver_Phi05);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi05, params->C6, doublependulum_QP_solver_V05);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi05, doublependulum_QP_solver_D02, doublependulum_QP_solver_W05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W05, doublependulum_QP_solver_V05, doublependulum_QP_solver_Ysd05);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi05, doublependulum_QP_solver_rd05, doublependulum_QP_solver_Lbyrd05);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_lubbysub06, doublependulum_QP_solver_ubIdx06, doublependulum_QP_solver_Phi06);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi06, params->C7, doublependulum_QP_solver_V06);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi06, doublependulum_QP_solver_D02, doublependulum_QP_solver_W06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W06, doublependulum_QP_solver_V06, doublependulum_QP_solver_Ysd06);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi06, doublependulum_QP_solver_rd06, doublependulum_QP_solver_Lbyrd06);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_lubbysub07, doublependulum_QP_solver_ubIdx07, doublependulum_QP_solver_Phi07);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi07, params->C8, doublependulum_QP_solver_V07);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi07, doublependulum_QP_solver_D02, doublependulum_QP_solver_W07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W07, doublependulum_QP_solver_V07, doublependulum_QP_solver_Ysd07);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi07, doublependulum_QP_solver_rd07, doublependulum_QP_solver_Lbyrd07);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_lubbysub08, doublependulum_QP_solver_ubIdx08, doublependulum_QP_solver_Phi08);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi08, params->C9, doublependulum_QP_solver_V08);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi08, doublependulum_QP_solver_D02, doublependulum_QP_solver_W08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W08, doublependulum_QP_solver_V08, doublependulum_QP_solver_Ysd08);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi08, doublependulum_QP_solver_rd08, doublependulum_QP_solver_Lbyrd08);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_lubbysub09, doublependulum_QP_solver_ubIdx09, doublependulum_QP_solver_Phi09);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi09, params->C10, doublependulum_QP_solver_V09);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi09, doublependulum_QP_solver_D02, doublependulum_QP_solver_W09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W09, doublependulum_QP_solver_V09, doublependulum_QP_solver_Ysd09);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi09, doublependulum_QP_solver_rd09, doublependulum_QP_solver_Lbyrd09);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_lubbysub10, doublependulum_QP_solver_ubIdx10, doublependulum_QP_solver_Phi10);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi10, params->C11, doublependulum_QP_solver_V10);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi10, doublependulum_QP_solver_D02, doublependulum_QP_solver_W10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W10, doublependulum_QP_solver_V10, doublependulum_QP_solver_Ysd10);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi10, doublependulum_QP_solver_rd10, doublependulum_QP_solver_Lbyrd10);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_lubbysub11, doublependulum_QP_solver_ubIdx11, doublependulum_QP_solver_Phi11);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi11, params->C12, doublependulum_QP_solver_V11);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi11, doublependulum_QP_solver_D02, doublependulum_QP_solver_W11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W11, doublependulum_QP_solver_V11, doublependulum_QP_solver_Ysd11);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi11, doublependulum_QP_solver_rd11, doublependulum_QP_solver_Lbyrd11);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_lubbysub12, doublependulum_QP_solver_ubIdx12, doublependulum_QP_solver_Phi12);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi12, params->C13, doublependulum_QP_solver_V12);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi12, doublependulum_QP_solver_D02, doublependulum_QP_solver_W12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W12, doublependulum_QP_solver_V12, doublependulum_QP_solver_Ysd12);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi12, doublependulum_QP_solver_rd12, doublependulum_QP_solver_Lbyrd12);
doublependulum_QP_solver_LA_DIAG_CHOL_LBUB_19_19_11(doublependulum_QP_solver_H00, doublependulum_QP_solver_llbbyslb13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_lubbysub13, doublependulum_QP_solver_ubIdx13, doublependulum_QP_solver_Phi13);
doublependulum_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_19(doublependulum_QP_solver_Phi13, params->C14, doublependulum_QP_solver_V13);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_19(doublependulum_QP_solver_Phi13, doublependulum_QP_solver_D02, doublependulum_QP_solver_W13);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_19_5(doublependulum_QP_solver_W13, doublependulum_QP_solver_V13, doublependulum_QP_solver_Ysd13);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi13, doublependulum_QP_solver_rd13, doublependulum_QP_solver_Lbyrd13);
doublependulum_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_5_5_5(doublependulum_QP_solver_H14, doublependulum_QP_solver_llbbyslb14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_lubbysub14, doublependulum_QP_solver_ubIdx14, doublependulum_QP_solver_Phi14);
doublependulum_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Phi14, doublependulum_QP_solver_D14, doublependulum_QP_solver_W14);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_5(doublependulum_QP_solver_Phi14, doublependulum_QP_solver_rd14, doublependulum_QP_solver_Lbyrd14);
doublependulum_QP_solver_LA_DENSE_MMT2_9_19_19(doublependulum_QP_solver_V00, doublependulum_QP_solver_W01, doublependulum_QP_solver_Yd00);
doublependulum_QP_solver_LA_DENSE_MVMSUB2_9_19_19(doublependulum_QP_solver_V00, doublependulum_QP_solver_Lbyrd00, doublependulum_QP_solver_W01, doublependulum_QP_solver_Lbyrd01, doublependulum_QP_solver_re00, doublependulum_QP_solver_beta00);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V01, doublependulum_QP_solver_W02, doublependulum_QP_solver_Yd01);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V01, doublependulum_QP_solver_Lbyrd01, doublependulum_QP_solver_W02, doublependulum_QP_solver_Lbyrd02, doublependulum_QP_solver_re01, doublependulum_QP_solver_beta01);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V02, doublependulum_QP_solver_W03, doublependulum_QP_solver_Yd02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V02, doublependulum_QP_solver_Lbyrd02, doublependulum_QP_solver_W03, doublependulum_QP_solver_Lbyrd03, doublependulum_QP_solver_re02, doublependulum_QP_solver_beta02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V03, doublependulum_QP_solver_W04, doublependulum_QP_solver_Yd03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V03, doublependulum_QP_solver_Lbyrd03, doublependulum_QP_solver_W04, doublependulum_QP_solver_Lbyrd04, doublependulum_QP_solver_re03, doublependulum_QP_solver_beta03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V04, doublependulum_QP_solver_W05, doublependulum_QP_solver_Yd04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V04, doublependulum_QP_solver_Lbyrd04, doublependulum_QP_solver_W05, doublependulum_QP_solver_Lbyrd05, doublependulum_QP_solver_re04, doublependulum_QP_solver_beta04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V05, doublependulum_QP_solver_W06, doublependulum_QP_solver_Yd05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V05, doublependulum_QP_solver_Lbyrd05, doublependulum_QP_solver_W06, doublependulum_QP_solver_Lbyrd06, doublependulum_QP_solver_re05, doublependulum_QP_solver_beta05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V06, doublependulum_QP_solver_W07, doublependulum_QP_solver_Yd06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V06, doublependulum_QP_solver_Lbyrd06, doublependulum_QP_solver_W07, doublependulum_QP_solver_Lbyrd07, doublependulum_QP_solver_re06, doublependulum_QP_solver_beta06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V07, doublependulum_QP_solver_W08, doublependulum_QP_solver_Yd07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V07, doublependulum_QP_solver_Lbyrd07, doublependulum_QP_solver_W08, doublependulum_QP_solver_Lbyrd08, doublependulum_QP_solver_re07, doublependulum_QP_solver_beta07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V08, doublependulum_QP_solver_W09, doublependulum_QP_solver_Yd08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V08, doublependulum_QP_solver_Lbyrd08, doublependulum_QP_solver_W09, doublependulum_QP_solver_Lbyrd09, doublependulum_QP_solver_re08, doublependulum_QP_solver_beta08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V09, doublependulum_QP_solver_W10, doublependulum_QP_solver_Yd09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V09, doublependulum_QP_solver_Lbyrd09, doublependulum_QP_solver_W10, doublependulum_QP_solver_Lbyrd10, doublependulum_QP_solver_re09, doublependulum_QP_solver_beta09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V10, doublependulum_QP_solver_W11, doublependulum_QP_solver_Yd10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V10, doublependulum_QP_solver_Lbyrd10, doublependulum_QP_solver_W11, doublependulum_QP_solver_Lbyrd11, doublependulum_QP_solver_re10, doublependulum_QP_solver_beta10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V11, doublependulum_QP_solver_W12, doublependulum_QP_solver_Yd11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V11, doublependulum_QP_solver_Lbyrd11, doublependulum_QP_solver_W12, doublependulum_QP_solver_Lbyrd12, doublependulum_QP_solver_re11, doublependulum_QP_solver_beta11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_19(doublependulum_QP_solver_V12, doublependulum_QP_solver_W13, doublependulum_QP_solver_Yd12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_19(doublependulum_QP_solver_V12, doublependulum_QP_solver_Lbyrd12, doublependulum_QP_solver_W13, doublependulum_QP_solver_Lbyrd13, doublependulum_QP_solver_re12, doublependulum_QP_solver_beta12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_19_5(doublependulum_QP_solver_V13, doublependulum_QP_solver_W14, doublependulum_QP_solver_Yd13);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_19_5(doublependulum_QP_solver_V13, doublependulum_QP_solver_Lbyrd13, doublependulum_QP_solver_W14, doublependulum_QP_solver_Lbyrd14, doublependulum_QP_solver_re13, doublependulum_QP_solver_beta13);
doublependulum_QP_solver_LA_DENSE_CHOL_9(doublependulum_QP_solver_Yd00, doublependulum_QP_solver_Ld00);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_9(doublependulum_QP_solver_Ld00, doublependulum_QP_solver_beta00, doublependulum_QP_solver_yy00);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_9(doublependulum_QP_solver_Ld00, doublependulum_QP_solver_Ysd01, doublependulum_QP_solver_Lsd01);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_9(doublependulum_QP_solver_Lsd01, doublependulum_QP_solver_Yd01);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd01, doublependulum_QP_solver_Ld01);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_9(doublependulum_QP_solver_Lsd01, doublependulum_QP_solver_yy00, doublependulum_QP_solver_beta01, doublependulum_QP_solver_bmy01);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld01, doublependulum_QP_solver_bmy01, doublependulum_QP_solver_yy01);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld01, doublependulum_QP_solver_Ysd02, doublependulum_QP_solver_Lsd02);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd02, doublependulum_QP_solver_Yd02);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd02, doublependulum_QP_solver_Ld02);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd02, doublependulum_QP_solver_yy01, doublependulum_QP_solver_beta02, doublependulum_QP_solver_bmy02);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld02, doublependulum_QP_solver_bmy02, doublependulum_QP_solver_yy02);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld02, doublependulum_QP_solver_Ysd03, doublependulum_QP_solver_Lsd03);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd03, doublependulum_QP_solver_Yd03);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd03, doublependulum_QP_solver_Ld03);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd03, doublependulum_QP_solver_yy02, doublependulum_QP_solver_beta03, doublependulum_QP_solver_bmy03);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld03, doublependulum_QP_solver_bmy03, doublependulum_QP_solver_yy03);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld03, doublependulum_QP_solver_Ysd04, doublependulum_QP_solver_Lsd04);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd04, doublependulum_QP_solver_Yd04);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd04, doublependulum_QP_solver_Ld04);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd04, doublependulum_QP_solver_yy03, doublependulum_QP_solver_beta04, doublependulum_QP_solver_bmy04);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld04, doublependulum_QP_solver_bmy04, doublependulum_QP_solver_yy04);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld04, doublependulum_QP_solver_Ysd05, doublependulum_QP_solver_Lsd05);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd05, doublependulum_QP_solver_Yd05);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd05, doublependulum_QP_solver_Ld05);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd05, doublependulum_QP_solver_yy04, doublependulum_QP_solver_beta05, doublependulum_QP_solver_bmy05);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld05, doublependulum_QP_solver_bmy05, doublependulum_QP_solver_yy05);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld05, doublependulum_QP_solver_Ysd06, doublependulum_QP_solver_Lsd06);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd06, doublependulum_QP_solver_Yd06);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd06, doublependulum_QP_solver_Ld06);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd06, doublependulum_QP_solver_yy05, doublependulum_QP_solver_beta06, doublependulum_QP_solver_bmy06);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld06, doublependulum_QP_solver_bmy06, doublependulum_QP_solver_yy06);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld06, doublependulum_QP_solver_Ysd07, doublependulum_QP_solver_Lsd07);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd07, doublependulum_QP_solver_Yd07);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd07, doublependulum_QP_solver_Ld07);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd07, doublependulum_QP_solver_yy06, doublependulum_QP_solver_beta07, doublependulum_QP_solver_bmy07);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld07, doublependulum_QP_solver_bmy07, doublependulum_QP_solver_yy07);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld07, doublependulum_QP_solver_Ysd08, doublependulum_QP_solver_Lsd08);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd08, doublependulum_QP_solver_Yd08);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd08, doublependulum_QP_solver_Ld08);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd08, doublependulum_QP_solver_yy07, doublependulum_QP_solver_beta08, doublependulum_QP_solver_bmy08);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld08, doublependulum_QP_solver_bmy08, doublependulum_QP_solver_yy08);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld08, doublependulum_QP_solver_Ysd09, doublependulum_QP_solver_Lsd09);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd09, doublependulum_QP_solver_Yd09);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd09, doublependulum_QP_solver_Ld09);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd09, doublependulum_QP_solver_yy08, doublependulum_QP_solver_beta09, doublependulum_QP_solver_bmy09);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld09, doublependulum_QP_solver_bmy09, doublependulum_QP_solver_yy09);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld09, doublependulum_QP_solver_Ysd10, doublependulum_QP_solver_Lsd10);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd10, doublependulum_QP_solver_Yd10);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd10, doublependulum_QP_solver_Ld10);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd10, doublependulum_QP_solver_yy09, doublependulum_QP_solver_beta10, doublependulum_QP_solver_bmy10);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld10, doublependulum_QP_solver_bmy10, doublependulum_QP_solver_yy10);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld10, doublependulum_QP_solver_Ysd11, doublependulum_QP_solver_Lsd11);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd11, doublependulum_QP_solver_Yd11);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd11, doublependulum_QP_solver_Ld11);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd11, doublependulum_QP_solver_yy10, doublependulum_QP_solver_beta11, doublependulum_QP_solver_bmy11);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld11, doublependulum_QP_solver_bmy11, doublependulum_QP_solver_yy11);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld11, doublependulum_QP_solver_Ysd12, doublependulum_QP_solver_Lsd12);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd12, doublependulum_QP_solver_Yd12);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd12, doublependulum_QP_solver_Ld12);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd12, doublependulum_QP_solver_yy11, doublependulum_QP_solver_beta12, doublependulum_QP_solver_bmy12);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld12, doublependulum_QP_solver_bmy12, doublependulum_QP_solver_yy12);
doublependulum_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(doublependulum_QP_solver_Ld12, doublependulum_QP_solver_Ysd13, doublependulum_QP_solver_Lsd13);
doublependulum_QP_solver_LA_DENSE_MMTSUB_5_5(doublependulum_QP_solver_Lsd13, doublependulum_QP_solver_Yd13);
doublependulum_QP_solver_LA_DENSE_CHOL_5(doublependulum_QP_solver_Yd13, doublependulum_QP_solver_Ld13);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd13, doublependulum_QP_solver_yy12, doublependulum_QP_solver_beta13, doublependulum_QP_solver_bmy13);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld13, doublependulum_QP_solver_bmy13, doublependulum_QP_solver_yy13);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld13, doublependulum_QP_solver_yy13, doublependulum_QP_solver_dvaff13);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd13, doublependulum_QP_solver_dvaff13, doublependulum_QP_solver_yy12, doublependulum_QP_solver_bmy12);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld12, doublependulum_QP_solver_bmy12, doublependulum_QP_solver_dvaff12);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd12, doublependulum_QP_solver_dvaff12, doublependulum_QP_solver_yy11, doublependulum_QP_solver_bmy11);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld11, doublependulum_QP_solver_bmy11, doublependulum_QP_solver_dvaff11);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd11, doublependulum_QP_solver_dvaff11, doublependulum_QP_solver_yy10, doublependulum_QP_solver_bmy10);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld10, doublependulum_QP_solver_bmy10, doublependulum_QP_solver_dvaff10);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd10, doublependulum_QP_solver_dvaff10, doublependulum_QP_solver_yy09, doublependulum_QP_solver_bmy09);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld09, doublependulum_QP_solver_bmy09, doublependulum_QP_solver_dvaff09);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd09, doublependulum_QP_solver_dvaff09, doublependulum_QP_solver_yy08, doublependulum_QP_solver_bmy08);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld08, doublependulum_QP_solver_bmy08, doublependulum_QP_solver_dvaff08);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd08, doublependulum_QP_solver_dvaff08, doublependulum_QP_solver_yy07, doublependulum_QP_solver_bmy07);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld07, doublependulum_QP_solver_bmy07, doublependulum_QP_solver_dvaff07);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd07, doublependulum_QP_solver_dvaff07, doublependulum_QP_solver_yy06, doublependulum_QP_solver_bmy06);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld06, doublependulum_QP_solver_bmy06, doublependulum_QP_solver_dvaff06);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd06, doublependulum_QP_solver_dvaff06, doublependulum_QP_solver_yy05, doublependulum_QP_solver_bmy05);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld05, doublependulum_QP_solver_bmy05, doublependulum_QP_solver_dvaff05);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd05, doublependulum_QP_solver_dvaff05, doublependulum_QP_solver_yy04, doublependulum_QP_solver_bmy04);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld04, doublependulum_QP_solver_bmy04, doublependulum_QP_solver_dvaff04);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd04, doublependulum_QP_solver_dvaff04, doublependulum_QP_solver_yy03, doublependulum_QP_solver_bmy03);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld03, doublependulum_QP_solver_bmy03, doublependulum_QP_solver_dvaff03);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd03, doublependulum_QP_solver_dvaff03, doublependulum_QP_solver_yy02, doublependulum_QP_solver_bmy02);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld02, doublependulum_QP_solver_bmy02, doublependulum_QP_solver_dvaff02);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd02, doublependulum_QP_solver_dvaff02, doublependulum_QP_solver_yy01, doublependulum_QP_solver_bmy01);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld01, doublependulum_QP_solver_bmy01, doublependulum_QP_solver_dvaff01);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_9(doublependulum_QP_solver_Lsd01, doublependulum_QP_solver_dvaff01, doublependulum_QP_solver_yy00, doublependulum_QP_solver_bmy00);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_9(doublependulum_QP_solver_Ld00, doublependulum_QP_solver_bmy00, doublependulum_QP_solver_dvaff00);
doublependulum_QP_solver_LA_DENSE_MTVM_9_19(params->C1, doublependulum_QP_solver_dvaff00, doublependulum_QP_solver_grad_eq00);
doublependulum_QP_solver_LA_DENSE_MTVM2_5_19_9(params->C2, doublependulum_QP_solver_dvaff01, doublependulum_QP_solver_D01, doublependulum_QP_solver_dvaff00, doublependulum_QP_solver_grad_eq01);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C3, doublependulum_QP_solver_dvaff02, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff01, doublependulum_QP_solver_grad_eq02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C4, doublependulum_QP_solver_dvaff03, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff02, doublependulum_QP_solver_grad_eq03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C5, doublependulum_QP_solver_dvaff04, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff03, doublependulum_QP_solver_grad_eq04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C6, doublependulum_QP_solver_dvaff05, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff04, doublependulum_QP_solver_grad_eq05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C7, doublependulum_QP_solver_dvaff06, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff05, doublependulum_QP_solver_grad_eq06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C8, doublependulum_QP_solver_dvaff07, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff06, doublependulum_QP_solver_grad_eq07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C9, doublependulum_QP_solver_dvaff08, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff07, doublependulum_QP_solver_grad_eq08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C10, doublependulum_QP_solver_dvaff09, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff08, doublependulum_QP_solver_grad_eq09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C11, doublependulum_QP_solver_dvaff10, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff09, doublependulum_QP_solver_grad_eq10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C12, doublependulum_QP_solver_dvaff11, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff10, doublependulum_QP_solver_grad_eq11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C13, doublependulum_QP_solver_dvaff12, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff11, doublependulum_QP_solver_grad_eq12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C14, doublependulum_QP_solver_dvaff13, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvaff12, doublependulum_QP_solver_grad_eq13);
doublependulum_QP_solver_LA_DIAGZERO_MTVM_5_5(doublependulum_QP_solver_D14, doublependulum_QP_solver_dvaff13, doublependulum_QP_solver_grad_eq14);
doublependulum_QP_solver_LA_VSUB2_271(doublependulum_QP_solver_rd, doublependulum_QP_solver_grad_eq, doublependulum_QP_solver_rd);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi00, doublependulum_QP_solver_rd00, doublependulum_QP_solver_dzaff00);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi01, doublependulum_QP_solver_rd01, doublependulum_QP_solver_dzaff01);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi02, doublependulum_QP_solver_rd02, doublependulum_QP_solver_dzaff02);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi03, doublependulum_QP_solver_rd03, doublependulum_QP_solver_dzaff03);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi04, doublependulum_QP_solver_rd04, doublependulum_QP_solver_dzaff04);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi05, doublependulum_QP_solver_rd05, doublependulum_QP_solver_dzaff05);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi06, doublependulum_QP_solver_rd06, doublependulum_QP_solver_dzaff06);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi07, doublependulum_QP_solver_rd07, doublependulum_QP_solver_dzaff07);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi08, doublependulum_QP_solver_rd08, doublependulum_QP_solver_dzaff08);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi09, doublependulum_QP_solver_rd09, doublependulum_QP_solver_dzaff09);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi10, doublependulum_QP_solver_rd10, doublependulum_QP_solver_dzaff10);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi11, doublependulum_QP_solver_rd11, doublependulum_QP_solver_dzaff11);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi12, doublependulum_QP_solver_rd12, doublependulum_QP_solver_dzaff12);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi13, doublependulum_QP_solver_rd13, doublependulum_QP_solver_dzaff13);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(doublependulum_QP_solver_Phi14, doublependulum_QP_solver_rd14, doublependulum_QP_solver_dzaff14);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_rilb00, doublependulum_QP_solver_dslbaff00);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb00, doublependulum_QP_solver_dslbaff00, doublependulum_QP_solver_llb00, doublependulum_QP_solver_dllbaff00);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub00, doublependulum_QP_solver_dzaff00, doublependulum_QP_solver_ubIdx00, doublependulum_QP_solver_dsubaff00);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub00, doublependulum_QP_solver_dsubaff00, doublependulum_QP_solver_lub00, doublependulum_QP_solver_dlubaff00);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_rilb01, doublependulum_QP_solver_dslbaff01);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb01, doublependulum_QP_solver_dslbaff01, doublependulum_QP_solver_llb01, doublependulum_QP_solver_dllbaff01);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub01, doublependulum_QP_solver_dzaff01, doublependulum_QP_solver_ubIdx01, doublependulum_QP_solver_dsubaff01);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub01, doublependulum_QP_solver_dsubaff01, doublependulum_QP_solver_lub01, doublependulum_QP_solver_dlubaff01);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_rilb02, doublependulum_QP_solver_dslbaff02);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb02, doublependulum_QP_solver_dslbaff02, doublependulum_QP_solver_llb02, doublependulum_QP_solver_dllbaff02);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub02, doublependulum_QP_solver_dzaff02, doublependulum_QP_solver_ubIdx02, doublependulum_QP_solver_dsubaff02);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub02, doublependulum_QP_solver_dsubaff02, doublependulum_QP_solver_lub02, doublependulum_QP_solver_dlubaff02);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_rilb03, doublependulum_QP_solver_dslbaff03);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb03, doublependulum_QP_solver_dslbaff03, doublependulum_QP_solver_llb03, doublependulum_QP_solver_dllbaff03);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub03, doublependulum_QP_solver_dzaff03, doublependulum_QP_solver_ubIdx03, doublependulum_QP_solver_dsubaff03);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub03, doublependulum_QP_solver_dsubaff03, doublependulum_QP_solver_lub03, doublependulum_QP_solver_dlubaff03);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_rilb04, doublependulum_QP_solver_dslbaff04);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb04, doublependulum_QP_solver_dslbaff04, doublependulum_QP_solver_llb04, doublependulum_QP_solver_dllbaff04);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub04, doublependulum_QP_solver_dzaff04, doublependulum_QP_solver_ubIdx04, doublependulum_QP_solver_dsubaff04);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub04, doublependulum_QP_solver_dsubaff04, doublependulum_QP_solver_lub04, doublependulum_QP_solver_dlubaff04);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_rilb05, doublependulum_QP_solver_dslbaff05);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb05, doublependulum_QP_solver_dslbaff05, doublependulum_QP_solver_llb05, doublependulum_QP_solver_dllbaff05);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub05, doublependulum_QP_solver_dzaff05, doublependulum_QP_solver_ubIdx05, doublependulum_QP_solver_dsubaff05);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub05, doublependulum_QP_solver_dsubaff05, doublependulum_QP_solver_lub05, doublependulum_QP_solver_dlubaff05);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_rilb06, doublependulum_QP_solver_dslbaff06);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb06, doublependulum_QP_solver_dslbaff06, doublependulum_QP_solver_llb06, doublependulum_QP_solver_dllbaff06);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub06, doublependulum_QP_solver_dzaff06, doublependulum_QP_solver_ubIdx06, doublependulum_QP_solver_dsubaff06);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub06, doublependulum_QP_solver_dsubaff06, doublependulum_QP_solver_lub06, doublependulum_QP_solver_dlubaff06);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_rilb07, doublependulum_QP_solver_dslbaff07);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb07, doublependulum_QP_solver_dslbaff07, doublependulum_QP_solver_llb07, doublependulum_QP_solver_dllbaff07);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub07, doublependulum_QP_solver_dzaff07, doublependulum_QP_solver_ubIdx07, doublependulum_QP_solver_dsubaff07);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub07, doublependulum_QP_solver_dsubaff07, doublependulum_QP_solver_lub07, doublependulum_QP_solver_dlubaff07);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_rilb08, doublependulum_QP_solver_dslbaff08);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb08, doublependulum_QP_solver_dslbaff08, doublependulum_QP_solver_llb08, doublependulum_QP_solver_dllbaff08);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub08, doublependulum_QP_solver_dzaff08, doublependulum_QP_solver_ubIdx08, doublependulum_QP_solver_dsubaff08);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub08, doublependulum_QP_solver_dsubaff08, doublependulum_QP_solver_lub08, doublependulum_QP_solver_dlubaff08);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_rilb09, doublependulum_QP_solver_dslbaff09);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb09, doublependulum_QP_solver_dslbaff09, doublependulum_QP_solver_llb09, doublependulum_QP_solver_dllbaff09);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub09, doublependulum_QP_solver_dzaff09, doublependulum_QP_solver_ubIdx09, doublependulum_QP_solver_dsubaff09);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub09, doublependulum_QP_solver_dsubaff09, doublependulum_QP_solver_lub09, doublependulum_QP_solver_dlubaff09);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_rilb10, doublependulum_QP_solver_dslbaff10);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb10, doublependulum_QP_solver_dslbaff10, doublependulum_QP_solver_llb10, doublependulum_QP_solver_dllbaff10);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub10, doublependulum_QP_solver_dzaff10, doublependulum_QP_solver_ubIdx10, doublependulum_QP_solver_dsubaff10);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub10, doublependulum_QP_solver_dsubaff10, doublependulum_QP_solver_lub10, doublependulum_QP_solver_dlubaff10);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_rilb11, doublependulum_QP_solver_dslbaff11);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb11, doublependulum_QP_solver_dslbaff11, doublependulum_QP_solver_llb11, doublependulum_QP_solver_dllbaff11);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub11, doublependulum_QP_solver_dzaff11, doublependulum_QP_solver_ubIdx11, doublependulum_QP_solver_dsubaff11);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub11, doublependulum_QP_solver_dsubaff11, doublependulum_QP_solver_lub11, doublependulum_QP_solver_dlubaff11);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_rilb12, doublependulum_QP_solver_dslbaff12);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb12, doublependulum_QP_solver_dslbaff12, doublependulum_QP_solver_llb12, doublependulum_QP_solver_dllbaff12);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub12, doublependulum_QP_solver_dzaff12, doublependulum_QP_solver_ubIdx12, doublependulum_QP_solver_dsubaff12);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub12, doublependulum_QP_solver_dsubaff12, doublependulum_QP_solver_lub12, doublependulum_QP_solver_dlubaff12);
doublependulum_QP_solver_LA_VSUB_INDEXED_19(doublependulum_QP_solver_dzaff13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_rilb13, doublependulum_QP_solver_dslbaff13);
doublependulum_QP_solver_LA_VSUB3_19(doublependulum_QP_solver_llbbyslb13, doublependulum_QP_solver_dslbaff13, doublependulum_QP_solver_llb13, doublependulum_QP_solver_dllbaff13);
doublependulum_QP_solver_LA_VSUB2_INDEXED_11(doublependulum_QP_solver_riub13, doublependulum_QP_solver_dzaff13, doublependulum_QP_solver_ubIdx13, doublependulum_QP_solver_dsubaff13);
doublependulum_QP_solver_LA_VSUB3_11(doublependulum_QP_solver_lubbysub13, doublependulum_QP_solver_dsubaff13, doublependulum_QP_solver_lub13, doublependulum_QP_solver_dlubaff13);
doublependulum_QP_solver_LA_VSUB_INDEXED_5(doublependulum_QP_solver_dzaff14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_rilb14, doublependulum_QP_solver_dslbaff14);
doublependulum_QP_solver_LA_VSUB3_5(doublependulum_QP_solver_llbbyslb14, doublependulum_QP_solver_dslbaff14, doublependulum_QP_solver_llb14, doublependulum_QP_solver_dllbaff14);
doublependulum_QP_solver_LA_VSUB2_INDEXED_5(doublependulum_QP_solver_riub14, doublependulum_QP_solver_dzaff14, doublependulum_QP_solver_ubIdx14, doublependulum_QP_solver_dsubaff14);
doublependulum_QP_solver_LA_VSUB3_5(doublependulum_QP_solver_lubbysub14, doublependulum_QP_solver_dsubaff14, doublependulum_QP_solver_lub14, doublependulum_QP_solver_dlubaff14);
info->lsit_aff = doublependulum_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(doublependulum_QP_solver_l, doublependulum_QP_solver_s, doublependulum_QP_solver_dl_aff, doublependulum_QP_solver_ds_aff, &info->step_aff, &info->mu_aff);
if( info->lsit_aff == doublependulum_QP_solver_NOPROGRESS ){
exitcode = doublependulum_QP_solver_NOPROGRESS; break;
}
sigma_3rdroot = info->mu_aff / info->mu;
info->sigma = sigma_3rdroot*sigma_3rdroot*sigma_3rdroot;
musigma = info->mu * info->sigma;
doublependulum_QP_solver_LA_VSUB5_430(doublependulum_QP_solver_ds_aff, doublependulum_QP_solver_dl_aff, info->mu, info->sigma, doublependulum_QP_solver_ccrhs);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub00, doublependulum_QP_solver_sub00, doublependulum_QP_solver_ubIdx00, doublependulum_QP_solver_ccrhsl00, doublependulum_QP_solver_slb00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_rd00);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub01, doublependulum_QP_solver_sub01, doublependulum_QP_solver_ubIdx01, doublependulum_QP_solver_ccrhsl01, doublependulum_QP_solver_slb01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_rd01);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi00, doublependulum_QP_solver_rd00, doublependulum_QP_solver_Lbyrd00);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi01, doublependulum_QP_solver_rd01, doublependulum_QP_solver_Lbyrd01);
doublependulum_QP_solver_LA_DENSE_2MVMADD_9_19_19(doublependulum_QP_solver_V00, doublependulum_QP_solver_Lbyrd00, doublependulum_QP_solver_W01, doublependulum_QP_solver_Lbyrd01, doublependulum_QP_solver_beta00);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_9(doublependulum_QP_solver_Ld00, doublependulum_QP_solver_beta00, doublependulum_QP_solver_yy00);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub02, doublependulum_QP_solver_sub02, doublependulum_QP_solver_ubIdx02, doublependulum_QP_solver_ccrhsl02, doublependulum_QP_solver_slb02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_rd02);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi02, doublependulum_QP_solver_rd02, doublependulum_QP_solver_Lbyrd02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V01, doublependulum_QP_solver_Lbyrd01, doublependulum_QP_solver_W02, doublependulum_QP_solver_Lbyrd02, doublependulum_QP_solver_beta01);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_9(doublependulum_QP_solver_Lsd01, doublependulum_QP_solver_yy00, doublependulum_QP_solver_beta01, doublependulum_QP_solver_bmy01);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld01, doublependulum_QP_solver_bmy01, doublependulum_QP_solver_yy01);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub03, doublependulum_QP_solver_sub03, doublependulum_QP_solver_ubIdx03, doublependulum_QP_solver_ccrhsl03, doublependulum_QP_solver_slb03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_rd03);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi03, doublependulum_QP_solver_rd03, doublependulum_QP_solver_Lbyrd03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V02, doublependulum_QP_solver_Lbyrd02, doublependulum_QP_solver_W03, doublependulum_QP_solver_Lbyrd03, doublependulum_QP_solver_beta02);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd02, doublependulum_QP_solver_yy01, doublependulum_QP_solver_beta02, doublependulum_QP_solver_bmy02);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld02, doublependulum_QP_solver_bmy02, doublependulum_QP_solver_yy02);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub04, doublependulum_QP_solver_sub04, doublependulum_QP_solver_ubIdx04, doublependulum_QP_solver_ccrhsl04, doublependulum_QP_solver_slb04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_rd04);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi04, doublependulum_QP_solver_rd04, doublependulum_QP_solver_Lbyrd04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V03, doublependulum_QP_solver_Lbyrd03, doublependulum_QP_solver_W04, doublependulum_QP_solver_Lbyrd04, doublependulum_QP_solver_beta03);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd03, doublependulum_QP_solver_yy02, doublependulum_QP_solver_beta03, doublependulum_QP_solver_bmy03);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld03, doublependulum_QP_solver_bmy03, doublependulum_QP_solver_yy03);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub05, doublependulum_QP_solver_sub05, doublependulum_QP_solver_ubIdx05, doublependulum_QP_solver_ccrhsl05, doublependulum_QP_solver_slb05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_rd05);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi05, doublependulum_QP_solver_rd05, doublependulum_QP_solver_Lbyrd05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V04, doublependulum_QP_solver_Lbyrd04, doublependulum_QP_solver_W05, doublependulum_QP_solver_Lbyrd05, doublependulum_QP_solver_beta04);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd04, doublependulum_QP_solver_yy03, doublependulum_QP_solver_beta04, doublependulum_QP_solver_bmy04);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld04, doublependulum_QP_solver_bmy04, doublependulum_QP_solver_yy04);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub06, doublependulum_QP_solver_sub06, doublependulum_QP_solver_ubIdx06, doublependulum_QP_solver_ccrhsl06, doublependulum_QP_solver_slb06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_rd06);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi06, doublependulum_QP_solver_rd06, doublependulum_QP_solver_Lbyrd06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V05, doublependulum_QP_solver_Lbyrd05, doublependulum_QP_solver_W06, doublependulum_QP_solver_Lbyrd06, doublependulum_QP_solver_beta05);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd05, doublependulum_QP_solver_yy04, doublependulum_QP_solver_beta05, doublependulum_QP_solver_bmy05);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld05, doublependulum_QP_solver_bmy05, doublependulum_QP_solver_yy05);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub07, doublependulum_QP_solver_sub07, doublependulum_QP_solver_ubIdx07, doublependulum_QP_solver_ccrhsl07, doublependulum_QP_solver_slb07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_rd07);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi07, doublependulum_QP_solver_rd07, doublependulum_QP_solver_Lbyrd07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V06, doublependulum_QP_solver_Lbyrd06, doublependulum_QP_solver_W07, doublependulum_QP_solver_Lbyrd07, doublependulum_QP_solver_beta06);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd06, doublependulum_QP_solver_yy05, doublependulum_QP_solver_beta06, doublependulum_QP_solver_bmy06);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld06, doublependulum_QP_solver_bmy06, doublependulum_QP_solver_yy06);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub08, doublependulum_QP_solver_sub08, doublependulum_QP_solver_ubIdx08, doublependulum_QP_solver_ccrhsl08, doublependulum_QP_solver_slb08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_rd08);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi08, doublependulum_QP_solver_rd08, doublependulum_QP_solver_Lbyrd08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V07, doublependulum_QP_solver_Lbyrd07, doublependulum_QP_solver_W08, doublependulum_QP_solver_Lbyrd08, doublependulum_QP_solver_beta07);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd07, doublependulum_QP_solver_yy06, doublependulum_QP_solver_beta07, doublependulum_QP_solver_bmy07);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld07, doublependulum_QP_solver_bmy07, doublependulum_QP_solver_yy07);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub09, doublependulum_QP_solver_sub09, doublependulum_QP_solver_ubIdx09, doublependulum_QP_solver_ccrhsl09, doublependulum_QP_solver_slb09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_rd09);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi09, doublependulum_QP_solver_rd09, doublependulum_QP_solver_Lbyrd09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V08, doublependulum_QP_solver_Lbyrd08, doublependulum_QP_solver_W09, doublependulum_QP_solver_Lbyrd09, doublependulum_QP_solver_beta08);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd08, doublependulum_QP_solver_yy07, doublependulum_QP_solver_beta08, doublependulum_QP_solver_bmy08);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld08, doublependulum_QP_solver_bmy08, doublependulum_QP_solver_yy08);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub10, doublependulum_QP_solver_sub10, doublependulum_QP_solver_ubIdx10, doublependulum_QP_solver_ccrhsl10, doublependulum_QP_solver_slb10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_rd10);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi10, doublependulum_QP_solver_rd10, doublependulum_QP_solver_Lbyrd10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V09, doublependulum_QP_solver_Lbyrd09, doublependulum_QP_solver_W10, doublependulum_QP_solver_Lbyrd10, doublependulum_QP_solver_beta09);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd09, doublependulum_QP_solver_yy08, doublependulum_QP_solver_beta09, doublependulum_QP_solver_bmy09);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld09, doublependulum_QP_solver_bmy09, doublependulum_QP_solver_yy09);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub11, doublependulum_QP_solver_sub11, doublependulum_QP_solver_ubIdx11, doublependulum_QP_solver_ccrhsl11, doublependulum_QP_solver_slb11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_rd11);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi11, doublependulum_QP_solver_rd11, doublependulum_QP_solver_Lbyrd11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V10, doublependulum_QP_solver_Lbyrd10, doublependulum_QP_solver_W11, doublependulum_QP_solver_Lbyrd11, doublependulum_QP_solver_beta10);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd10, doublependulum_QP_solver_yy09, doublependulum_QP_solver_beta10, doublependulum_QP_solver_bmy10);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld10, doublependulum_QP_solver_bmy10, doublependulum_QP_solver_yy10);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub12, doublependulum_QP_solver_sub12, doublependulum_QP_solver_ubIdx12, doublependulum_QP_solver_ccrhsl12, doublependulum_QP_solver_slb12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_rd12);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi12, doublependulum_QP_solver_rd12, doublependulum_QP_solver_Lbyrd12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V11, doublependulum_QP_solver_Lbyrd11, doublependulum_QP_solver_W12, doublependulum_QP_solver_Lbyrd12, doublependulum_QP_solver_beta11);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd11, doublependulum_QP_solver_yy10, doublependulum_QP_solver_beta11, doublependulum_QP_solver_bmy11);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld11, doublependulum_QP_solver_bmy11, doublependulum_QP_solver_yy11);
doublependulum_QP_solver_LA_VSUB6_INDEXED_19_11_19(doublependulum_QP_solver_ccrhsub13, doublependulum_QP_solver_sub13, doublependulum_QP_solver_ubIdx13, doublependulum_QP_solver_ccrhsl13, doublependulum_QP_solver_slb13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_rd13);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_19(doublependulum_QP_solver_Phi13, doublependulum_QP_solver_rd13, doublependulum_QP_solver_Lbyrd13);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_19(doublependulum_QP_solver_V12, doublependulum_QP_solver_Lbyrd12, doublependulum_QP_solver_W13, doublependulum_QP_solver_Lbyrd13, doublependulum_QP_solver_beta12);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd12, doublependulum_QP_solver_yy11, doublependulum_QP_solver_beta12, doublependulum_QP_solver_bmy12);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld12, doublependulum_QP_solver_bmy12, doublependulum_QP_solver_yy12);
doublependulum_QP_solver_LA_VSUB6_INDEXED_5_5_5(doublependulum_QP_solver_ccrhsub14, doublependulum_QP_solver_sub14, doublependulum_QP_solver_ubIdx14, doublependulum_QP_solver_ccrhsl14, doublependulum_QP_solver_slb14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_rd14);
doublependulum_QP_solver_LA_DIAG_FORWARDSUB_5(doublependulum_QP_solver_Phi14, doublependulum_QP_solver_rd14, doublependulum_QP_solver_Lbyrd14);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_19_5(doublependulum_QP_solver_V13, doublependulum_QP_solver_Lbyrd13, doublependulum_QP_solver_W14, doublependulum_QP_solver_Lbyrd14, doublependulum_QP_solver_beta13);
doublependulum_QP_solver_LA_DENSE_MVMSUB1_5_5(doublependulum_QP_solver_Lsd13, doublependulum_QP_solver_yy12, doublependulum_QP_solver_beta13, doublependulum_QP_solver_bmy13);
doublependulum_QP_solver_LA_DENSE_FORWARDSUB_5(doublependulum_QP_solver_Ld13, doublependulum_QP_solver_bmy13, doublependulum_QP_solver_yy13);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld13, doublependulum_QP_solver_yy13, doublependulum_QP_solver_dvcc13);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd13, doublependulum_QP_solver_dvcc13, doublependulum_QP_solver_yy12, doublependulum_QP_solver_bmy12);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld12, doublependulum_QP_solver_bmy12, doublependulum_QP_solver_dvcc12);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd12, doublependulum_QP_solver_dvcc12, doublependulum_QP_solver_yy11, doublependulum_QP_solver_bmy11);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld11, doublependulum_QP_solver_bmy11, doublependulum_QP_solver_dvcc11);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd11, doublependulum_QP_solver_dvcc11, doublependulum_QP_solver_yy10, doublependulum_QP_solver_bmy10);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld10, doublependulum_QP_solver_bmy10, doublependulum_QP_solver_dvcc10);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd10, doublependulum_QP_solver_dvcc10, doublependulum_QP_solver_yy09, doublependulum_QP_solver_bmy09);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld09, doublependulum_QP_solver_bmy09, doublependulum_QP_solver_dvcc09);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd09, doublependulum_QP_solver_dvcc09, doublependulum_QP_solver_yy08, doublependulum_QP_solver_bmy08);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld08, doublependulum_QP_solver_bmy08, doublependulum_QP_solver_dvcc08);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd08, doublependulum_QP_solver_dvcc08, doublependulum_QP_solver_yy07, doublependulum_QP_solver_bmy07);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld07, doublependulum_QP_solver_bmy07, doublependulum_QP_solver_dvcc07);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd07, doublependulum_QP_solver_dvcc07, doublependulum_QP_solver_yy06, doublependulum_QP_solver_bmy06);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld06, doublependulum_QP_solver_bmy06, doublependulum_QP_solver_dvcc06);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd06, doublependulum_QP_solver_dvcc06, doublependulum_QP_solver_yy05, doublependulum_QP_solver_bmy05);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld05, doublependulum_QP_solver_bmy05, doublependulum_QP_solver_dvcc05);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd05, doublependulum_QP_solver_dvcc05, doublependulum_QP_solver_yy04, doublependulum_QP_solver_bmy04);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld04, doublependulum_QP_solver_bmy04, doublependulum_QP_solver_dvcc04);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd04, doublependulum_QP_solver_dvcc04, doublependulum_QP_solver_yy03, doublependulum_QP_solver_bmy03);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld03, doublependulum_QP_solver_bmy03, doublependulum_QP_solver_dvcc03);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd03, doublependulum_QP_solver_dvcc03, doublependulum_QP_solver_yy02, doublependulum_QP_solver_bmy02);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld02, doublependulum_QP_solver_bmy02, doublependulum_QP_solver_dvcc02);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_5(doublependulum_QP_solver_Lsd02, doublependulum_QP_solver_dvcc02, doublependulum_QP_solver_yy01, doublependulum_QP_solver_bmy01);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_5(doublependulum_QP_solver_Ld01, doublependulum_QP_solver_bmy01, doublependulum_QP_solver_dvcc01);
doublependulum_QP_solver_LA_DENSE_MTVMSUB_5_9(doublependulum_QP_solver_Lsd01, doublependulum_QP_solver_dvcc01, doublependulum_QP_solver_yy00, doublependulum_QP_solver_bmy00);
doublependulum_QP_solver_LA_DENSE_BACKWARDSUB_9(doublependulum_QP_solver_Ld00, doublependulum_QP_solver_bmy00, doublependulum_QP_solver_dvcc00);
doublependulum_QP_solver_LA_DENSE_MTVM_9_19(params->C1, doublependulum_QP_solver_dvcc00, doublependulum_QP_solver_grad_eq00);
doublependulum_QP_solver_LA_DENSE_MTVM2_5_19_9(params->C2, doublependulum_QP_solver_dvcc01, doublependulum_QP_solver_D01, doublependulum_QP_solver_dvcc00, doublependulum_QP_solver_grad_eq01);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C3, doublependulum_QP_solver_dvcc02, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc01, doublependulum_QP_solver_grad_eq02);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C4, doublependulum_QP_solver_dvcc03, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc02, doublependulum_QP_solver_grad_eq03);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C5, doublependulum_QP_solver_dvcc04, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc03, doublependulum_QP_solver_grad_eq04);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C6, doublependulum_QP_solver_dvcc05, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc04, doublependulum_QP_solver_grad_eq05);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C7, doublependulum_QP_solver_dvcc06, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc05, doublependulum_QP_solver_grad_eq06);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C8, doublependulum_QP_solver_dvcc07, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc06, doublependulum_QP_solver_grad_eq07);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C9, doublependulum_QP_solver_dvcc08, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc07, doublependulum_QP_solver_grad_eq08);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C10, doublependulum_QP_solver_dvcc09, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc08, doublependulum_QP_solver_grad_eq09);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C11, doublependulum_QP_solver_dvcc10, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc09, doublependulum_QP_solver_grad_eq10);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C12, doublependulum_QP_solver_dvcc11, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc10, doublependulum_QP_solver_grad_eq11);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C13, doublependulum_QP_solver_dvcc12, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc11, doublependulum_QP_solver_grad_eq12);
doublependulum_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_19_5(params->C14, doublependulum_QP_solver_dvcc13, doublependulum_QP_solver_D02, doublependulum_QP_solver_dvcc12, doublependulum_QP_solver_grad_eq13);
doublependulum_QP_solver_LA_DIAGZERO_MTVM_5_5(doublependulum_QP_solver_D14, doublependulum_QP_solver_dvcc13, doublependulum_QP_solver_grad_eq14);
doublependulum_QP_solver_LA_VSUB_271(doublependulum_QP_solver_rd, doublependulum_QP_solver_grad_eq, doublependulum_QP_solver_rd);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi00, doublependulum_QP_solver_rd00, doublependulum_QP_solver_dzcc00);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi01, doublependulum_QP_solver_rd01, doublependulum_QP_solver_dzcc01);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi02, doublependulum_QP_solver_rd02, doublependulum_QP_solver_dzcc02);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi03, doublependulum_QP_solver_rd03, doublependulum_QP_solver_dzcc03);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi04, doublependulum_QP_solver_rd04, doublependulum_QP_solver_dzcc04);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi05, doublependulum_QP_solver_rd05, doublependulum_QP_solver_dzcc05);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi06, doublependulum_QP_solver_rd06, doublependulum_QP_solver_dzcc06);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi07, doublependulum_QP_solver_rd07, doublependulum_QP_solver_dzcc07);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi08, doublependulum_QP_solver_rd08, doublependulum_QP_solver_dzcc08);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi09, doublependulum_QP_solver_rd09, doublependulum_QP_solver_dzcc09);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi10, doublependulum_QP_solver_rd10, doublependulum_QP_solver_dzcc10);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi11, doublependulum_QP_solver_rd11, doublependulum_QP_solver_dzcc11);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi12, doublependulum_QP_solver_rd12, doublependulum_QP_solver_dzcc12);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_19(doublependulum_QP_solver_Phi13, doublependulum_QP_solver_rd13, doublependulum_QP_solver_dzcc13);
doublependulum_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(doublependulum_QP_solver_Phi14, doublependulum_QP_solver_rd14, doublependulum_QP_solver_dzcc14);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl00, doublependulum_QP_solver_slb00, doublependulum_QP_solver_llbbyslb00, doublependulum_QP_solver_dzcc00, doublependulum_QP_solver_lbIdx00, doublependulum_QP_solver_dllbcc00);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub00, doublependulum_QP_solver_sub00, doublependulum_QP_solver_lubbysub00, doublependulum_QP_solver_dzcc00, doublependulum_QP_solver_ubIdx00, doublependulum_QP_solver_dlubcc00);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl01, doublependulum_QP_solver_slb01, doublependulum_QP_solver_llbbyslb01, doublependulum_QP_solver_dzcc01, doublependulum_QP_solver_lbIdx01, doublependulum_QP_solver_dllbcc01);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub01, doublependulum_QP_solver_sub01, doublependulum_QP_solver_lubbysub01, doublependulum_QP_solver_dzcc01, doublependulum_QP_solver_ubIdx01, doublependulum_QP_solver_dlubcc01);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl02, doublependulum_QP_solver_slb02, doublependulum_QP_solver_llbbyslb02, doublependulum_QP_solver_dzcc02, doublependulum_QP_solver_lbIdx02, doublependulum_QP_solver_dllbcc02);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub02, doublependulum_QP_solver_sub02, doublependulum_QP_solver_lubbysub02, doublependulum_QP_solver_dzcc02, doublependulum_QP_solver_ubIdx02, doublependulum_QP_solver_dlubcc02);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl03, doublependulum_QP_solver_slb03, doublependulum_QP_solver_llbbyslb03, doublependulum_QP_solver_dzcc03, doublependulum_QP_solver_lbIdx03, doublependulum_QP_solver_dllbcc03);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub03, doublependulum_QP_solver_sub03, doublependulum_QP_solver_lubbysub03, doublependulum_QP_solver_dzcc03, doublependulum_QP_solver_ubIdx03, doublependulum_QP_solver_dlubcc03);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl04, doublependulum_QP_solver_slb04, doublependulum_QP_solver_llbbyslb04, doublependulum_QP_solver_dzcc04, doublependulum_QP_solver_lbIdx04, doublependulum_QP_solver_dllbcc04);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub04, doublependulum_QP_solver_sub04, doublependulum_QP_solver_lubbysub04, doublependulum_QP_solver_dzcc04, doublependulum_QP_solver_ubIdx04, doublependulum_QP_solver_dlubcc04);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl05, doublependulum_QP_solver_slb05, doublependulum_QP_solver_llbbyslb05, doublependulum_QP_solver_dzcc05, doublependulum_QP_solver_lbIdx05, doublependulum_QP_solver_dllbcc05);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub05, doublependulum_QP_solver_sub05, doublependulum_QP_solver_lubbysub05, doublependulum_QP_solver_dzcc05, doublependulum_QP_solver_ubIdx05, doublependulum_QP_solver_dlubcc05);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl06, doublependulum_QP_solver_slb06, doublependulum_QP_solver_llbbyslb06, doublependulum_QP_solver_dzcc06, doublependulum_QP_solver_lbIdx06, doublependulum_QP_solver_dllbcc06);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub06, doublependulum_QP_solver_sub06, doublependulum_QP_solver_lubbysub06, doublependulum_QP_solver_dzcc06, doublependulum_QP_solver_ubIdx06, doublependulum_QP_solver_dlubcc06);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl07, doublependulum_QP_solver_slb07, doublependulum_QP_solver_llbbyslb07, doublependulum_QP_solver_dzcc07, doublependulum_QP_solver_lbIdx07, doublependulum_QP_solver_dllbcc07);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub07, doublependulum_QP_solver_sub07, doublependulum_QP_solver_lubbysub07, doublependulum_QP_solver_dzcc07, doublependulum_QP_solver_ubIdx07, doublependulum_QP_solver_dlubcc07);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl08, doublependulum_QP_solver_slb08, doublependulum_QP_solver_llbbyslb08, doublependulum_QP_solver_dzcc08, doublependulum_QP_solver_lbIdx08, doublependulum_QP_solver_dllbcc08);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub08, doublependulum_QP_solver_sub08, doublependulum_QP_solver_lubbysub08, doublependulum_QP_solver_dzcc08, doublependulum_QP_solver_ubIdx08, doublependulum_QP_solver_dlubcc08);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl09, doublependulum_QP_solver_slb09, doublependulum_QP_solver_llbbyslb09, doublependulum_QP_solver_dzcc09, doublependulum_QP_solver_lbIdx09, doublependulum_QP_solver_dllbcc09);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub09, doublependulum_QP_solver_sub09, doublependulum_QP_solver_lubbysub09, doublependulum_QP_solver_dzcc09, doublependulum_QP_solver_ubIdx09, doublependulum_QP_solver_dlubcc09);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl10, doublependulum_QP_solver_slb10, doublependulum_QP_solver_llbbyslb10, doublependulum_QP_solver_dzcc10, doublependulum_QP_solver_lbIdx10, doublependulum_QP_solver_dllbcc10);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub10, doublependulum_QP_solver_sub10, doublependulum_QP_solver_lubbysub10, doublependulum_QP_solver_dzcc10, doublependulum_QP_solver_ubIdx10, doublependulum_QP_solver_dlubcc10);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl11, doublependulum_QP_solver_slb11, doublependulum_QP_solver_llbbyslb11, doublependulum_QP_solver_dzcc11, doublependulum_QP_solver_lbIdx11, doublependulum_QP_solver_dllbcc11);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub11, doublependulum_QP_solver_sub11, doublependulum_QP_solver_lubbysub11, doublependulum_QP_solver_dzcc11, doublependulum_QP_solver_ubIdx11, doublependulum_QP_solver_dlubcc11);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl12, doublependulum_QP_solver_slb12, doublependulum_QP_solver_llbbyslb12, doublependulum_QP_solver_dzcc12, doublependulum_QP_solver_lbIdx12, doublependulum_QP_solver_dllbcc12);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub12, doublependulum_QP_solver_sub12, doublependulum_QP_solver_lubbysub12, doublependulum_QP_solver_dzcc12, doublependulum_QP_solver_ubIdx12, doublependulum_QP_solver_dlubcc12);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_19(doublependulum_QP_solver_ccrhsl13, doublependulum_QP_solver_slb13, doublependulum_QP_solver_llbbyslb13, doublependulum_QP_solver_dzcc13, doublependulum_QP_solver_lbIdx13, doublependulum_QP_solver_dllbcc13);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_11(doublependulum_QP_solver_ccrhsub13, doublependulum_QP_solver_sub13, doublependulum_QP_solver_lubbysub13, doublependulum_QP_solver_dzcc13, doublependulum_QP_solver_ubIdx13, doublependulum_QP_solver_dlubcc13);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_5(doublependulum_QP_solver_ccrhsl14, doublependulum_QP_solver_slb14, doublependulum_QP_solver_llbbyslb14, doublependulum_QP_solver_dzcc14, doublependulum_QP_solver_lbIdx14, doublependulum_QP_solver_dllbcc14);
doublependulum_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_5(doublependulum_QP_solver_ccrhsub14, doublependulum_QP_solver_sub14, doublependulum_QP_solver_lubbysub14, doublependulum_QP_solver_dzcc14, doublependulum_QP_solver_ubIdx14, doublependulum_QP_solver_dlubcc14);
doublependulum_QP_solver_LA_VSUB7_430(doublependulum_QP_solver_l, doublependulum_QP_solver_ccrhs, doublependulum_QP_solver_s, doublependulum_QP_solver_dl_cc, doublependulum_QP_solver_ds_cc);
doublependulum_QP_solver_LA_VADD_271(doublependulum_QP_solver_dz_cc, doublependulum_QP_solver_dz_aff);
doublependulum_QP_solver_LA_VADD_74(doublependulum_QP_solver_dv_cc, doublependulum_QP_solver_dv_aff);
doublependulum_QP_solver_LA_VADD_430(doublependulum_QP_solver_dl_cc, doublependulum_QP_solver_dl_aff);
doublependulum_QP_solver_LA_VADD_430(doublependulum_QP_solver_ds_cc, doublependulum_QP_solver_ds_aff);
info->lsit_cc = doublependulum_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(doublependulum_QP_solver_z, doublependulum_QP_solver_v, doublependulum_QP_solver_l, doublependulum_QP_solver_s, doublependulum_QP_solver_dz_cc, doublependulum_QP_solver_dv_cc, doublependulum_QP_solver_dl_cc, doublependulum_QP_solver_ds_cc, &info->step_cc, &info->mu);
if( info->lsit_cc == doublependulum_QP_solver_NOPROGRESS ){
exitcode = doublependulum_QP_solver_NOPROGRESS; break;
}
info->it++;
}
output->z1[0] = doublependulum_QP_solver_z00[0];
output->z1[1] = doublependulum_QP_solver_z00[1];
output->z1[2] = doublependulum_QP_solver_z00[2];
output->z1[3] = doublependulum_QP_solver_z00[3];
output->z1[4] = doublependulum_QP_solver_z00[4];
output->z1[5] = doublependulum_QP_solver_z00[5];
output->z1[6] = doublependulum_QP_solver_z00[6];
output->z1[7] = doublependulum_QP_solver_z00[7];
output->z1[8] = doublependulum_QP_solver_z00[8];
output->z1[9] = doublependulum_QP_solver_z00[9];
output->z1[10] = doublependulum_QP_solver_z00[10];
output->z2[0] = doublependulum_QP_solver_z01[0];
output->z2[1] = doublependulum_QP_solver_z01[1];
output->z2[2] = doublependulum_QP_solver_z01[2];
output->z2[3] = doublependulum_QP_solver_z01[3];
output->z2[4] = doublependulum_QP_solver_z01[4];
output->z2[5] = doublependulum_QP_solver_z01[5];
output->z2[6] = doublependulum_QP_solver_z01[6];
output->z2[7] = doublependulum_QP_solver_z01[7];
output->z2[8] = doublependulum_QP_solver_z01[8];
output->z2[9] = doublependulum_QP_solver_z01[9];
output->z2[10] = doublependulum_QP_solver_z01[10];
output->z3[0] = doublependulum_QP_solver_z02[0];
output->z3[1] = doublependulum_QP_solver_z02[1];
output->z3[2] = doublependulum_QP_solver_z02[2];
output->z3[3] = doublependulum_QP_solver_z02[3];
output->z3[4] = doublependulum_QP_solver_z02[4];
output->z3[5] = doublependulum_QP_solver_z02[5];
output->z3[6] = doublependulum_QP_solver_z02[6];
output->z3[7] = doublependulum_QP_solver_z02[7];
output->z3[8] = doublependulum_QP_solver_z02[8];
output->z3[9] = doublependulum_QP_solver_z02[9];
output->z3[10] = doublependulum_QP_solver_z02[10];
output->z4[0] = doublependulum_QP_solver_z03[0];
output->z4[1] = doublependulum_QP_solver_z03[1];
output->z4[2] = doublependulum_QP_solver_z03[2];
output->z4[3] = doublependulum_QP_solver_z03[3];
output->z4[4] = doublependulum_QP_solver_z03[4];
output->z4[5] = doublependulum_QP_solver_z03[5];
output->z4[6] = doublependulum_QP_solver_z03[6];
output->z4[7] = doublependulum_QP_solver_z03[7];
output->z4[8] = doublependulum_QP_solver_z03[8];
output->z4[9] = doublependulum_QP_solver_z03[9];
output->z4[10] = doublependulum_QP_solver_z03[10];
output->z5[0] = doublependulum_QP_solver_z04[0];
output->z5[1] = doublependulum_QP_solver_z04[1];
output->z5[2] = doublependulum_QP_solver_z04[2];
output->z5[3] = doublependulum_QP_solver_z04[3];
output->z5[4] = doublependulum_QP_solver_z04[4];
output->z5[5] = doublependulum_QP_solver_z04[5];
output->z5[6] = doublependulum_QP_solver_z04[6];
output->z5[7] = doublependulum_QP_solver_z04[7];
output->z5[8] = doublependulum_QP_solver_z04[8];
output->z5[9] = doublependulum_QP_solver_z04[9];
output->z5[10] = doublependulum_QP_solver_z04[10];
output->z6[0] = doublependulum_QP_solver_z05[0];
output->z6[1] = doublependulum_QP_solver_z05[1];
output->z6[2] = doublependulum_QP_solver_z05[2];
output->z6[3] = doublependulum_QP_solver_z05[3];
output->z6[4] = doublependulum_QP_solver_z05[4];
output->z6[5] = doublependulum_QP_solver_z05[5];
output->z6[6] = doublependulum_QP_solver_z05[6];
output->z6[7] = doublependulum_QP_solver_z05[7];
output->z6[8] = doublependulum_QP_solver_z05[8];
output->z6[9] = doublependulum_QP_solver_z05[9];
output->z6[10] = doublependulum_QP_solver_z05[10];
output->z7[0] = doublependulum_QP_solver_z06[0];
output->z7[1] = doublependulum_QP_solver_z06[1];
output->z7[2] = doublependulum_QP_solver_z06[2];
output->z7[3] = doublependulum_QP_solver_z06[3];
output->z7[4] = doublependulum_QP_solver_z06[4];
output->z7[5] = doublependulum_QP_solver_z06[5];
output->z7[6] = doublependulum_QP_solver_z06[6];
output->z7[7] = doublependulum_QP_solver_z06[7];
output->z7[8] = doublependulum_QP_solver_z06[8];
output->z7[9] = doublependulum_QP_solver_z06[9];
output->z7[10] = doublependulum_QP_solver_z06[10];
output->z8[0] = doublependulum_QP_solver_z07[0];
output->z8[1] = doublependulum_QP_solver_z07[1];
output->z8[2] = doublependulum_QP_solver_z07[2];
output->z8[3] = doublependulum_QP_solver_z07[3];
output->z8[4] = doublependulum_QP_solver_z07[4];
output->z8[5] = doublependulum_QP_solver_z07[5];
output->z8[6] = doublependulum_QP_solver_z07[6];
output->z8[7] = doublependulum_QP_solver_z07[7];
output->z8[8] = doublependulum_QP_solver_z07[8];
output->z8[9] = doublependulum_QP_solver_z07[9];
output->z8[10] = doublependulum_QP_solver_z07[10];
output->z9[0] = doublependulum_QP_solver_z08[0];
output->z9[1] = doublependulum_QP_solver_z08[1];
output->z9[2] = doublependulum_QP_solver_z08[2];
output->z9[3] = doublependulum_QP_solver_z08[3];
output->z9[4] = doublependulum_QP_solver_z08[4];
output->z9[5] = doublependulum_QP_solver_z08[5];
output->z9[6] = doublependulum_QP_solver_z08[6];
output->z9[7] = doublependulum_QP_solver_z08[7];
output->z9[8] = doublependulum_QP_solver_z08[8];
output->z9[9] = doublependulum_QP_solver_z08[9];
output->z9[10] = doublependulum_QP_solver_z08[10];
output->z10[0] = doublependulum_QP_solver_z09[0];
output->z10[1] = doublependulum_QP_solver_z09[1];
output->z10[2] = doublependulum_QP_solver_z09[2];
output->z10[3] = doublependulum_QP_solver_z09[3];
output->z10[4] = doublependulum_QP_solver_z09[4];
output->z10[5] = doublependulum_QP_solver_z09[5];
output->z10[6] = doublependulum_QP_solver_z09[6];
output->z10[7] = doublependulum_QP_solver_z09[7];
output->z10[8] = doublependulum_QP_solver_z09[8];
output->z10[9] = doublependulum_QP_solver_z09[9];
output->z10[10] = doublependulum_QP_solver_z09[10];
output->z11[0] = doublependulum_QP_solver_z10[0];
output->z11[1] = doublependulum_QP_solver_z10[1];
output->z11[2] = doublependulum_QP_solver_z10[2];
output->z11[3] = doublependulum_QP_solver_z10[3];
output->z11[4] = doublependulum_QP_solver_z10[4];
output->z11[5] = doublependulum_QP_solver_z10[5];
output->z11[6] = doublependulum_QP_solver_z10[6];
output->z11[7] = doublependulum_QP_solver_z10[7];
output->z11[8] = doublependulum_QP_solver_z10[8];
output->z11[9] = doublependulum_QP_solver_z10[9];
output->z11[10] = doublependulum_QP_solver_z10[10];
output->z12[0] = doublependulum_QP_solver_z11[0];
output->z12[1] = doublependulum_QP_solver_z11[1];
output->z12[2] = doublependulum_QP_solver_z11[2];
output->z12[3] = doublependulum_QP_solver_z11[3];
output->z12[4] = doublependulum_QP_solver_z11[4];
output->z12[5] = doublependulum_QP_solver_z11[5];
output->z12[6] = doublependulum_QP_solver_z11[6];
output->z12[7] = doublependulum_QP_solver_z11[7];
output->z12[8] = doublependulum_QP_solver_z11[8];
output->z12[9] = doublependulum_QP_solver_z11[9];
output->z12[10] = doublependulum_QP_solver_z11[10];
output->z13[0] = doublependulum_QP_solver_z12[0];
output->z13[1] = doublependulum_QP_solver_z12[1];
output->z13[2] = doublependulum_QP_solver_z12[2];
output->z13[3] = doublependulum_QP_solver_z12[3];
output->z13[4] = doublependulum_QP_solver_z12[4];
output->z13[5] = doublependulum_QP_solver_z12[5];
output->z13[6] = doublependulum_QP_solver_z12[6];
output->z13[7] = doublependulum_QP_solver_z12[7];
output->z13[8] = doublependulum_QP_solver_z12[8];
output->z13[9] = doublependulum_QP_solver_z12[9];
output->z13[10] = doublependulum_QP_solver_z12[10];
output->z14[0] = doublependulum_QP_solver_z13[0];
output->z14[1] = doublependulum_QP_solver_z13[1];
output->z14[2] = doublependulum_QP_solver_z13[2];
output->z14[3] = doublependulum_QP_solver_z13[3];
output->z14[4] = doublependulum_QP_solver_z13[4];
output->z14[5] = doublependulum_QP_solver_z13[5];
output->z14[6] = doublependulum_QP_solver_z13[6];
output->z14[7] = doublependulum_QP_solver_z13[7];
output->z14[8] = doublependulum_QP_solver_z13[8];
output->z14[9] = doublependulum_QP_solver_z13[9];
output->z14[10] = doublependulum_QP_solver_z13[10];
output->z15[0] = doublependulum_QP_solver_z14[0];
output->z15[1] = doublependulum_QP_solver_z14[1];
output->z15[2] = doublependulum_QP_solver_z14[2];
output->z15[3] = doublependulum_QP_solver_z14[3];
output->z15[4] = doublependulum_QP_solver_z14[4];

#if doublependulum_QP_solver_SET_TIMING == 1
info->solvetime = doublependulum_QP_solver_toc(&solvertimer);
#if doublependulum_QP_solver_SET_PRINTLEVEL > 0 && doublependulum_QP_solver_SET_TIMING == 1
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
