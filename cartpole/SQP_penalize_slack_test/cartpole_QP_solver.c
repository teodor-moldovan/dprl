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

#include "cartpole_QP_solver.h"

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
 * Initializes a vector of length 257 with a value.
 */
void cartpole_QP_solver_LA_INITIALIZEVECTOR_257(cartpole_QP_solver_FLOAT* vec, cartpole_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<257; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 74 with a value.
 */
void cartpole_QP_solver_LA_INITIALIZEVECTOR_74(cartpole_QP_solver_FLOAT* vec, cartpole_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<74; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 402 with a value.
 */
void cartpole_QP_solver_LA_INITIALIZEVECTOR_402(cartpole_QP_solver_FLOAT* vec, cartpole_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<402; i++ )
	{
		vec[i] = value;
	}
}


/* 
 * Calculates a dot product and adds it to a variable: z += x'*y; 
 * This function is for vectors of length 402.
 */
void cartpole_QP_solver_LA_DOTACC_402(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<402; i++ ){
		*z += x[i]*y[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [18 x 18]
 *             f  - column vector of size 18
 *             z  - column vector of size 18
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 18
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void cartpole_QP_solver_LA_DIAG_QUADFCN_18(cartpole_QP_solver_FLOAT* H, cartpole_QP_solver_FLOAT* f, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* grad, cartpole_QP_solver_FLOAT* value)
{
	int i;
	cartpole_QP_solver_FLOAT hz;	
	for( i=0; i<18; i++){
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
void cartpole_QP_solver_LA_DIAG_QUADFCN_5(cartpole_QP_solver_FLOAT* H, cartpole_QP_solver_FLOAT* f, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* grad, cartpole_QP_solver_FLOAT* value)
{
	int i;
	cartpole_QP_solver_FLOAT hz;	
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
void cartpole_QP_solver_LA_DENSE_MVMSUB3_9_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *z, cartpole_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;
	cartpole_QP_solver_FLOAT AxBu[9];
	cartpole_QP_solver_FLOAT norm = *y;
	cartpole_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<9; i++ ){
		AxBu[i] = A[k++]*x[0] + B[m++]*u[0];
	}	
	for( j=1; j<18; j++ ){		
		for( i=0; i<9; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<18; n++ ){
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *z, cartpole_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	cartpole_QP_solver_FLOAT AxBu[5];
	cartpole_QP_solver_FLOAT norm = *y;
	cartpole_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<5; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *z, cartpole_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	cartpole_QP_solver_FLOAT AxBu[5];
	cartpole_QP_solver_FLOAT norm = *y;
	cartpole_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<5; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
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
 * Matrix vector multiplication y = M'*x where M is of size [9 x 18]
 * and stored in column major format. Note the transpose of M!
 */
void cartpole_QP_solver_LA_DENSE_MTVM_9_18(cartpole_QP_solver_FLOAT *M, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<18; i++ ){
		y[i] = 0;
		for( j=0; j<9; j++ ){
			y[i] += M[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [5 x 18]
 * and B is of size [9 x 18]
 * and stored in column major format. Note the transposes of A and B!
 */
void cartpole_QP_solver_LA_DENSE_MTVM2_5_18_9(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	int n;
	int m = 0;
	for( i=0; i<18; i++ ){
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
 * where A is of size [5 x 18] and stored in column major format.
 * and B is of size [5 x 18] and stored in diagzero format
 * Note the transposes of A and B!
 */
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
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
	for( i=5 ;i<18; i++ ){
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
void cartpole_QP_solver_LA_DIAGZERO_MTVM_5_5(cartpole_QP_solver_FLOAT *M, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y)
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
 * for vectors of length 18. Output z is of course scalar.
 */
void cartpole_QP_solver_LA_VSUBADD3_18(cartpole_QP_solver_FLOAT* t, cartpole_QP_solver_FLOAT* u, int* uidx, cartpole_QP_solver_FLOAT* v, cartpole_QP_solver_FLOAT* w, cartpole_QP_solver_FLOAT* y, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* r)
{
	int i;
	cartpole_QP_solver_FLOAT norm = *r;
	cartpole_QP_solver_FLOAT vx = 0;
	cartpole_QP_solver_FLOAT x;
	for( i=0; i<18; i++){
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
 * for vectors of length 10. Output z is of course scalar.
 */
void cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_FLOAT* t, int* tidx, cartpole_QP_solver_FLOAT* u, cartpole_QP_solver_FLOAT* v, cartpole_QP_solver_FLOAT* w, cartpole_QP_solver_FLOAT* y, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* r)
{
	int i;
	cartpole_QP_solver_FLOAT norm = *r;
	cartpole_QP_solver_FLOAT vx = 0;
	cartpole_QP_solver_FLOAT x;
	for( i=0; i<10; i++){
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
void cartpole_QP_solver_LA_VSUBADD3_5(cartpole_QP_solver_FLOAT* t, cartpole_QP_solver_FLOAT* u, int* uidx, cartpole_QP_solver_FLOAT* v, cartpole_QP_solver_FLOAT* w, cartpole_QP_solver_FLOAT* y, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* r)
{
	int i;
	cartpole_QP_solver_FLOAT norm = *r;
	cartpole_QP_solver_FLOAT vx = 0;
	cartpole_QP_solver_FLOAT x;
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
void cartpole_QP_solver_LA_VSUBADD2_5(cartpole_QP_solver_FLOAT* t, int* tidx, cartpole_QP_solver_FLOAT* u, cartpole_QP_solver_FLOAT* v, cartpole_QP_solver_FLOAT* w, cartpole_QP_solver_FLOAT* y, cartpole_QP_solver_FLOAT* z, cartpole_QP_solver_FLOAT* r)
{
	int i;
	cartpole_QP_solver_FLOAT norm = *r;
	cartpole_QP_solver_FLOAT vx = 0;
	cartpole_QP_solver_FLOAT x;
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
 * Special function for box constraints of length 18
 * Returns also L/S, a value that is often used elsewhere.
 */
void cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_FLOAT *lu, cartpole_QP_solver_FLOAT *su, cartpole_QP_solver_FLOAT *ru, cartpole_QP_solver_FLOAT *ll, cartpole_QP_solver_FLOAT *sl, cartpole_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, cartpole_QP_solver_FLOAT *grad, cartpole_QP_solver_FLOAT *lubysu, cartpole_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<18; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<18; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<10; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 5
 * Returns also L/S, a value that is often used elsewhere.
 */
void cartpole_QP_solver_LA_INEQ_B_GRAD_5_5_5(cartpole_QP_solver_FLOAT *lu, cartpole_QP_solver_FLOAT *su, cartpole_QP_solver_FLOAT *ru, cartpole_QP_solver_FLOAT *ll, cartpole_QP_solver_FLOAT *sl, cartpole_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, cartpole_QP_solver_FLOAT *grad, cartpole_QP_solver_FLOAT *lubysu, cartpole_QP_solver_FLOAT *llbysl)
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
 * of length 257.
 */
void cartpole_QP_solver_LA_VVADD3_257(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *w, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<257; i++ ){
		z[i] = u[i] + v[i] + w[i];
	}
}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 18.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(cartpole_QP_solver_FLOAT *H, cartpole_QP_solver_FLOAT *llbysl, int* lbIdx, cartpole_QP_solver_FLOAT *lubysu, int* ubIdx, cartpole_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* copy  H into PHI */
	for( i=0; i<18; i++ ){
		Phi[i] = H[i];
	}

	/* add llbysl onto Phi where necessary */
	for( i=0; i<18; i++ ){
		Phi[lbIdx[i]] += llbysl[i];
	}

	/* add lubysu onto Phi where necessary */
	for( i=0; i<10; i++){
		Phi[ubIdx[i]] +=  lubysu[i];
	}
	
	/* compute cholesky */
	for(i=0; i<18; i++)
	{
#if cartpole_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
 * where A is to be computed and is of size [9 x 18],
 * B is given and of size [9 x 18], L is a diagonal
 * matrix of size 9 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_18(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<18; j++){
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
 * The dimensions involved are 18.
 */
void cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<18; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [5 x 18],
 * B is given and of size [5 x 18], L is a diagonal
 * matrix of size 5 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<18; j++){
		for( i=0; i<5; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [9 x 18]
 *  size(B) = [5 x 18]
 * 
 * and all matrices are stored in column major format.
 *
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE.  
 * 
 */
void cartpole_QP_solver_LA_DENSE_MMTM_9_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *C)
{
    int i, j, k;
    cartpole_QP_solver_FLOAT temp;
    
    for( i=0; i<9; i++ ){        
        for( j=0; j<5; j++ ){
            temp = 0; 
            for( k=0; k<18; k++ ){
                temp += A[k*9+i]*B[k*5+j];
            }						
            C[j*9+i] = temp;
        }
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [5 x 18],
 * B is given and of size [5 x 18], L is a diagonal
 *  matrix of size 18 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<18; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [5 x 18]
 *  size(B) = [5 x 18] in diagzero format
 * 
 * A and C matrices are stored in column major format.
 * 
 * 
 */
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *C)
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
void cartpole_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_5_5_5(cartpole_QP_solver_FLOAT *H, cartpole_QP_solver_FLOAT *llbysl, int* lbIdx, cartpole_QP_solver_FLOAT *lubysu, int* ubIdx, cartpole_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* compute cholesky */
	for( i=0; i<5; i++ ){
		Phi[i] = H[i] + llbysl[i] + lubysu[i];

#if cartpole_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
void cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
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
void cartpole_QP_solver_LA_DIAG_FORWARDSUB_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<5; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [9 x 18] in column
 * storage format, and B is of size [9 x 18] also in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void cartpole_QP_solver_LA_DENSE_MMT2_9_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    cartpole_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<9; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<18; k++ ){
                ltemp += A[k*9+i]*A[k*9+j];
            }			
			for( k=0; k<18; k++ ){
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
void cartpole_QP_solver_LA_DENSE_MVMSUB2_9_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<9; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[m++]*u[0];
	}	
	for( j=1; j<18; j++ ){		
		for( i=0; i<9; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
	for( n=1; n<18; n++ ){
		for( i=0; i<9; i++ ){
			r[i] -= B[m++]*u[n];
		}		
	}
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [5 x 18] in column
 * storage format, and B is of size [5 x 18] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    cartpole_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<18; k++ ){
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [5 x 18] in column
 * storage format, and B is of size [5 x 5] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    cartpole_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<5; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<18; k++ ){
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
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
void cartpole_QP_solver_LA_DENSE_CHOL_9(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    cartpole_QP_solver_FLOAT l;
    cartpole_QP_solver_FLOAT Mii;

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
        
#if cartpole_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
void cartpole_QP_solver_LA_DENSE_FORWARDSUB_9(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    cartpole_QP_solver_FLOAT yel;
            
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
void cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_9(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    cartpole_QP_solver_FLOAT a;
    
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
void cartpole_QP_solver_LA_DENSE_MMTSUB_5_9(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    cartpole_QP_solver_FLOAT ltemp;
    
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
void cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    cartpole_QP_solver_FLOAT l;
    cartpole_QP_solver_FLOAT Mii;

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
        
#if cartpole_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
void cartpole_QP_solver_LA_DENSE_MVMSUB1_5_9(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
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
void cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    cartpole_QP_solver_FLOAT yel;
            
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
void cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    cartpole_QP_solver_FLOAT a;
    
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
void cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    cartpole_QP_solver_FLOAT ltemp;
    
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
void cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
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
void cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    cartpole_QP_solver_FLOAT xel;    
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
void cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
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
void cartpole_QP_solver_LA_DENSE_MTVMSUB_5_9(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *r)
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
void cartpole_QP_solver_LA_DENSE_BACKWARDSUB_9(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    cartpole_QP_solver_FLOAT xel;    
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
 * Vector subtraction z = -x - y for vectors of length 257.
 */
void cartpole_QP_solver_LA_VSUB2_257(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<257; i++){
		z[i] = -x[i] - y[i];
	}
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 18 in vector
 * storage format.
 */
void cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<18; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 5 in vector
 * storage format.
 */
void cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(cartpole_QP_solver_FLOAT *L, cartpole_QP_solver_FLOAT *b, cartpole_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<5; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 18,
 * and x has length 18 and is indexed through yidx.
 */
void cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_FLOAT *x, int* xidx, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<18; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 18.
 */
void cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *w, cartpole_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<18; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 18
 * and z, x and yidx are of length 10.
 */
void cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<10; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 10.
 */
void cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *w, cartpole_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<10; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 5,
 * and x has length 5 and is indexed through yidx.
 */
void cartpole_QP_solver_LA_VSUB_INDEXED_5(cartpole_QP_solver_FLOAT *x, int* xidx, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 5.
 */
void cartpole_QP_solver_LA_VSUB3_5(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *w, cartpole_QP_solver_FLOAT *x)
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
void cartpole_QP_solver_LA_VSUB2_INDEXED_5(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
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
 * cartpole_QP_solver_NOPROGRESS (should be negative).
 */
int cartpole_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *dl, cartpole_QP_solver_FLOAT *ds, cartpole_QP_solver_FLOAT *a, cartpole_QP_solver_FLOAT *mu_aff)
{
    int i;
	int lsIt=1;    
    cartpole_QP_solver_FLOAT dltemp;
    cartpole_QP_solver_FLOAT dstemp;
    cartpole_QP_solver_FLOAT mya = 1.0;
    cartpole_QP_solver_FLOAT mymu;
        
    while( 1 ){                        

        /* 
         * Compute both snew and wnew together.
         * We compute also mu_affine along the way here, as the
         * values might be in registers, so it should be cheaper.
         */
        mymu = 0;
        for( i=0; i<402; i++ ){
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
        if( i == 402 ){
            break;
        } else {
            mya *= cartpole_QP_solver_SET_LS_SCALE_AFF;
            if( mya < cartpole_QP_solver_SET_LS_MINSTEP ){
                return cartpole_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* return new values and iteration counter */
    *a = mya;
    *mu_aff = mymu / (cartpole_QP_solver_FLOAT)402;
    return lsIt;
}


/*
 * Vector subtraction x = (u.*v - mu)*sigma where a is a scalar
*  and x,u,v are vectors of length 402.
 */
void cartpole_QP_solver_LA_VSUB5_402(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT mu,  cartpole_QP_solver_FLOAT sigma, cartpole_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<402; i++){
		x[i] = u[i]*v[i] - mu;
		x[i] *= sigma;
	}
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 18,
 * u, su, uidx are of length 10 and v, sv, vidx are of length 18.
 */
void cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *su, int* uidx, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *sv, int* vidx, cartpole_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<18; i++ ){
		x[i] = 0;
	}
	for( i=0; i<10; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<18; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A an B are stored in column major format
 */
void cartpole_QP_solver_LA_DENSE_2MVMADD_9_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<9; i++ ){
		r[i] = A[k++]*x[0] + B[m++]*u[0];
	}	

	for( j=1; j<18; j++ ){		
		for( i=0; i<9; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<18; n++ ){
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 5,
 * u, su, uidx are of length 5 and v, sv, vidx are of length 5.
 */
void cartpole_QP_solver_LA_VSUB6_INDEXED_5_5_5(cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *su, int* uidx, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *sv, int* vidx, cartpole_QP_solver_FLOAT *x)
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
void cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_5(cartpole_QP_solver_FLOAT *A, cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *B, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<5; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<18; j++ ){		
		for( i=0; i<5; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Vector subtraction z = x - y for vectors of length 257.
 */
void cartpole_QP_solver_LA_VSUB_257(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<257; i++){
		z[i] = x[i] - y[i];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 18 (length of y >= 18).
 */
void cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<18; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 10 (length of y >= 10).
 */
void cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<10; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 5 (length of y >= 5).
 */
void cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_5(cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
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
void cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_5(cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *u, cartpole_QP_solver_FLOAT *y, int* yidx, cartpole_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<5; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/*
 * Computes ds = -l.\(r + s.*dl) for vectors of length 402.
 */
void cartpole_QP_solver_LA_VSUB7_402(cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *r, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *dl, cartpole_QP_solver_FLOAT *ds)
{
	int i;
	for( i=0; i<402; i++){
		ds[i] = -(r[i] + s[i]*dl[i])/l[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 257.
 */
void cartpole_QP_solver_LA_VADD_257(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<257; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 74.
 */
void cartpole_QP_solver_LA_VADD_74(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<74; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 402.
 */
void cartpole_QP_solver_LA_VADD_402(cartpole_QP_solver_FLOAT *x, cartpole_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<402; i++){
		x[i] += y[i];
	}
}


/**
 * Backtracking line search for combined predictor/corrector step.
 * Update on variables with safety factor gamma (to keep us away from
 * boundary).
 */
int cartpole_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(cartpole_QP_solver_FLOAT *z, cartpole_QP_solver_FLOAT *v, cartpole_QP_solver_FLOAT *l, cartpole_QP_solver_FLOAT *s, cartpole_QP_solver_FLOAT *dz, cartpole_QP_solver_FLOAT *dv, cartpole_QP_solver_FLOAT *dl, cartpole_QP_solver_FLOAT *ds, cartpole_QP_solver_FLOAT *a, cartpole_QP_solver_FLOAT *mu)
{
    int i, lsIt=1;       
    cartpole_QP_solver_FLOAT dltemp;
    cartpole_QP_solver_FLOAT dstemp;    
    cartpole_QP_solver_FLOAT a_gamma;
            
    *a = 1.0;
    while( 1 ){                        

        /* check whether search criterion is fulfilled */
        for( i=0; i<402; i++ ){
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
        if( i == 402 ){
            break;
        } else {
            *a *= cartpole_QP_solver_SET_LS_SCALE;
            if( *a < cartpole_QP_solver_SET_LS_MINSTEP ){
                return cartpole_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* update variables with safety margin */
    a_gamma = (*a)*cartpole_QP_solver_SET_LS_MAXSTEP;
    
    /* primal variables */
    for( i=0; i<257; i++ ){
        z[i] += a_gamma*dz[i];
    }
    
    /* equality constraint multipliers */
    for( i=0; i<74; i++ ){
        v[i] += a_gamma*dv[i];
    }
    
    /* inequality constraint multipliers & slacks, also update mu */
    *mu = 0;
    for( i=0; i<402; i++ ){
        dltemp = l[i] + a_gamma*dl[i]; l[i] = dltemp;
        dstemp = s[i] + a_gamma*ds[i]; s[i] = dstemp;
        *mu += dltemp*dstemp;
    }
    
    *a = a_gamma;
    *mu /= (cartpole_QP_solver_FLOAT)402;
    return lsIt;
}




/* VARIABLE DEFINITIONS ------------------------------------------------ */
cartpole_QP_solver_FLOAT cartpole_QP_solver_z[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_v[74];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dz_aff[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dv_aff[74];
cartpole_QP_solver_FLOAT cartpole_QP_solver_grad_cost[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_grad_eq[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_rd[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_l[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_s[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_lbys[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dl_aff[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_ds_aff[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dz_cc[257];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dv_cc[74];
cartpole_QP_solver_FLOAT cartpole_QP_solver_dl_cc[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_ds_cc[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_ccrhs[402];
cartpole_QP_solver_FLOAT cartpole_QP_solver_grad_ineq[257];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z00 = cartpole_QP_solver_z + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff00 = cartpole_QP_solver_dz_aff + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc00 = cartpole_QP_solver_dz_cc + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd00 = cartpole_QP_solver_rd + 0;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd00[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost00 = cartpole_QP_solver_grad_cost + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq00 = cartpole_QP_solver_grad_eq + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq00 = cartpole_QP_solver_grad_ineq + 0;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv00[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v00 = cartpole_QP_solver_v + 0;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re00[9];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta00[9];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc00[9];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff00 = cartpole_QP_solver_dv_aff + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc00 = cartpole_QP_solver_dv_cc + 0;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V00[162];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd00[45];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld00[45];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy00[9];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy00[9];
int cartpole_QP_solver_lbIdx00[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb00 = cartpole_QP_solver_l + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb00 = cartpole_QP_solver_s + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb00 = cartpole_QP_solver_lbys + 0;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb00[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff00 = cartpole_QP_solver_dl_aff + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff00 = cartpole_QP_solver_ds_aff + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc00 = cartpole_QP_solver_dl_cc + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc00 = cartpole_QP_solver_ds_cc + 0;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl00 = cartpole_QP_solver_ccrhs + 0;
int cartpole_QP_solver_ubIdx00[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub00 = cartpole_QP_solver_l + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub00 = cartpole_QP_solver_s + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub00 = cartpole_QP_solver_lbys + 18;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub00[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff00 = cartpole_QP_solver_dl_aff + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff00 = cartpole_QP_solver_ds_aff + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc00 = cartpole_QP_solver_dl_cc + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc00 = cartpole_QP_solver_ds_cc + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub00 = cartpole_QP_solver_ccrhs + 18;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi00[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z01 = cartpole_QP_solver_z + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff01 = cartpole_QP_solver_dz_aff + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc01 = cartpole_QP_solver_dz_cc + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd01 = cartpole_QP_solver_rd + 18;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd01[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost01 = cartpole_QP_solver_grad_cost + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq01 = cartpole_QP_solver_grad_eq + 18;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq01 = cartpole_QP_solver_grad_ineq + 18;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv01[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v01 = cartpole_QP_solver_v + 9;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re01[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta01[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc01[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff01 = cartpole_QP_solver_dv_aff + 9;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc01 = cartpole_QP_solver_dv_cc + 9;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V01[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd01[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld01[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy01[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy01[5];
int cartpole_QP_solver_lbIdx01[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb01 = cartpole_QP_solver_l + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb01 = cartpole_QP_solver_s + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb01 = cartpole_QP_solver_lbys + 28;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb01[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff01 = cartpole_QP_solver_dl_aff + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff01 = cartpole_QP_solver_ds_aff + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc01 = cartpole_QP_solver_dl_cc + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc01 = cartpole_QP_solver_ds_cc + 28;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl01 = cartpole_QP_solver_ccrhs + 28;
int cartpole_QP_solver_ubIdx01[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub01 = cartpole_QP_solver_l + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub01 = cartpole_QP_solver_s + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub01 = cartpole_QP_solver_lbys + 46;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub01[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff01 = cartpole_QP_solver_dl_aff + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff01 = cartpole_QP_solver_ds_aff + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc01 = cartpole_QP_solver_dl_cc + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc01 = cartpole_QP_solver_ds_cc + 46;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub01 = cartpole_QP_solver_ccrhs + 46;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi01[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_D01[162] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
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
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
cartpole_QP_solver_FLOAT cartpole_QP_solver_W01[162];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd01[45];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd01[45];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z02 = cartpole_QP_solver_z + 36;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff02 = cartpole_QP_solver_dz_aff + 36;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc02 = cartpole_QP_solver_dz_cc + 36;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd02 = cartpole_QP_solver_rd + 36;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd02[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost02 = cartpole_QP_solver_grad_cost + 36;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq02 = cartpole_QP_solver_grad_eq + 36;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq02 = cartpole_QP_solver_grad_ineq + 36;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv02[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v02 = cartpole_QP_solver_v + 14;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re02[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta02[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc02[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff02 = cartpole_QP_solver_dv_aff + 14;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc02 = cartpole_QP_solver_dv_cc + 14;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V02[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd02[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld02[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy02[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy02[5];
int cartpole_QP_solver_lbIdx02[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb02 = cartpole_QP_solver_l + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb02 = cartpole_QP_solver_s + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb02 = cartpole_QP_solver_lbys + 56;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb02[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff02 = cartpole_QP_solver_dl_aff + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff02 = cartpole_QP_solver_ds_aff + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc02 = cartpole_QP_solver_dl_cc + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc02 = cartpole_QP_solver_ds_cc + 56;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl02 = cartpole_QP_solver_ccrhs + 56;
int cartpole_QP_solver_ubIdx02[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub02 = cartpole_QP_solver_l + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub02 = cartpole_QP_solver_s + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub02 = cartpole_QP_solver_lbys + 74;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub02[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff02 = cartpole_QP_solver_dl_aff + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff02 = cartpole_QP_solver_ds_aff + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc02 = cartpole_QP_solver_dl_cc + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc02 = cartpole_QP_solver_ds_cc + 74;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub02 = cartpole_QP_solver_ccrhs + 74;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi02[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_D02[18] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
cartpole_QP_solver_FLOAT cartpole_QP_solver_W02[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd02[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd02[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z03 = cartpole_QP_solver_z + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff03 = cartpole_QP_solver_dz_aff + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc03 = cartpole_QP_solver_dz_cc + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd03 = cartpole_QP_solver_rd + 54;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd03[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost03 = cartpole_QP_solver_grad_cost + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq03 = cartpole_QP_solver_grad_eq + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq03 = cartpole_QP_solver_grad_ineq + 54;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv03[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v03 = cartpole_QP_solver_v + 19;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re03[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta03[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc03[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff03 = cartpole_QP_solver_dv_aff + 19;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc03 = cartpole_QP_solver_dv_cc + 19;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V03[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd03[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld03[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy03[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy03[5];
int cartpole_QP_solver_lbIdx03[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb03 = cartpole_QP_solver_l + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb03 = cartpole_QP_solver_s + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb03 = cartpole_QP_solver_lbys + 84;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb03[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff03 = cartpole_QP_solver_dl_aff + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff03 = cartpole_QP_solver_ds_aff + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc03 = cartpole_QP_solver_dl_cc + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc03 = cartpole_QP_solver_ds_cc + 84;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl03 = cartpole_QP_solver_ccrhs + 84;
int cartpole_QP_solver_ubIdx03[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub03 = cartpole_QP_solver_l + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub03 = cartpole_QP_solver_s + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub03 = cartpole_QP_solver_lbys + 102;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub03[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff03 = cartpole_QP_solver_dl_aff + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff03 = cartpole_QP_solver_ds_aff + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc03 = cartpole_QP_solver_dl_cc + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc03 = cartpole_QP_solver_ds_cc + 102;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub03 = cartpole_QP_solver_ccrhs + 102;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi03[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W03[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd03[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd03[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z04 = cartpole_QP_solver_z + 72;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff04 = cartpole_QP_solver_dz_aff + 72;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc04 = cartpole_QP_solver_dz_cc + 72;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd04 = cartpole_QP_solver_rd + 72;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd04[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost04 = cartpole_QP_solver_grad_cost + 72;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq04 = cartpole_QP_solver_grad_eq + 72;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq04 = cartpole_QP_solver_grad_ineq + 72;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv04[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v04 = cartpole_QP_solver_v + 24;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re04[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta04[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc04[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff04 = cartpole_QP_solver_dv_aff + 24;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc04 = cartpole_QP_solver_dv_cc + 24;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V04[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd04[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld04[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy04[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy04[5];
int cartpole_QP_solver_lbIdx04[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb04 = cartpole_QP_solver_l + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb04 = cartpole_QP_solver_s + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb04 = cartpole_QP_solver_lbys + 112;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb04[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff04 = cartpole_QP_solver_dl_aff + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff04 = cartpole_QP_solver_ds_aff + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc04 = cartpole_QP_solver_dl_cc + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc04 = cartpole_QP_solver_ds_cc + 112;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl04 = cartpole_QP_solver_ccrhs + 112;
int cartpole_QP_solver_ubIdx04[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub04 = cartpole_QP_solver_l + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub04 = cartpole_QP_solver_s + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub04 = cartpole_QP_solver_lbys + 130;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub04[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff04 = cartpole_QP_solver_dl_aff + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff04 = cartpole_QP_solver_ds_aff + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc04 = cartpole_QP_solver_dl_cc + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc04 = cartpole_QP_solver_ds_cc + 130;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub04 = cartpole_QP_solver_ccrhs + 130;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi04[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W04[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd04[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd04[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z05 = cartpole_QP_solver_z + 90;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff05 = cartpole_QP_solver_dz_aff + 90;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc05 = cartpole_QP_solver_dz_cc + 90;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd05 = cartpole_QP_solver_rd + 90;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd05[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost05 = cartpole_QP_solver_grad_cost + 90;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq05 = cartpole_QP_solver_grad_eq + 90;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq05 = cartpole_QP_solver_grad_ineq + 90;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv05[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v05 = cartpole_QP_solver_v + 29;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re05[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta05[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc05[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff05 = cartpole_QP_solver_dv_aff + 29;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc05 = cartpole_QP_solver_dv_cc + 29;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V05[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd05[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld05[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy05[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy05[5];
int cartpole_QP_solver_lbIdx05[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb05 = cartpole_QP_solver_l + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb05 = cartpole_QP_solver_s + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb05 = cartpole_QP_solver_lbys + 140;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb05[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff05 = cartpole_QP_solver_dl_aff + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff05 = cartpole_QP_solver_ds_aff + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc05 = cartpole_QP_solver_dl_cc + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc05 = cartpole_QP_solver_ds_cc + 140;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl05 = cartpole_QP_solver_ccrhs + 140;
int cartpole_QP_solver_ubIdx05[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub05 = cartpole_QP_solver_l + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub05 = cartpole_QP_solver_s + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub05 = cartpole_QP_solver_lbys + 158;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub05[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff05 = cartpole_QP_solver_dl_aff + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff05 = cartpole_QP_solver_ds_aff + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc05 = cartpole_QP_solver_dl_cc + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc05 = cartpole_QP_solver_ds_cc + 158;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub05 = cartpole_QP_solver_ccrhs + 158;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi05[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W05[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd05[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd05[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z06 = cartpole_QP_solver_z + 108;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff06 = cartpole_QP_solver_dz_aff + 108;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc06 = cartpole_QP_solver_dz_cc + 108;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd06 = cartpole_QP_solver_rd + 108;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd06[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost06 = cartpole_QP_solver_grad_cost + 108;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq06 = cartpole_QP_solver_grad_eq + 108;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq06 = cartpole_QP_solver_grad_ineq + 108;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv06[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v06 = cartpole_QP_solver_v + 34;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re06[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta06[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc06[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff06 = cartpole_QP_solver_dv_aff + 34;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc06 = cartpole_QP_solver_dv_cc + 34;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V06[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd06[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld06[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy06[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy06[5];
int cartpole_QP_solver_lbIdx06[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb06 = cartpole_QP_solver_l + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb06 = cartpole_QP_solver_s + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb06 = cartpole_QP_solver_lbys + 168;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb06[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff06 = cartpole_QP_solver_dl_aff + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff06 = cartpole_QP_solver_ds_aff + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc06 = cartpole_QP_solver_dl_cc + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc06 = cartpole_QP_solver_ds_cc + 168;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl06 = cartpole_QP_solver_ccrhs + 168;
int cartpole_QP_solver_ubIdx06[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub06 = cartpole_QP_solver_l + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub06 = cartpole_QP_solver_s + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub06 = cartpole_QP_solver_lbys + 186;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub06[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff06 = cartpole_QP_solver_dl_aff + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff06 = cartpole_QP_solver_ds_aff + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc06 = cartpole_QP_solver_dl_cc + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc06 = cartpole_QP_solver_ds_cc + 186;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub06 = cartpole_QP_solver_ccrhs + 186;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi06[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W06[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd06[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd06[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z07 = cartpole_QP_solver_z + 126;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff07 = cartpole_QP_solver_dz_aff + 126;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc07 = cartpole_QP_solver_dz_cc + 126;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd07 = cartpole_QP_solver_rd + 126;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd07[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost07 = cartpole_QP_solver_grad_cost + 126;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq07 = cartpole_QP_solver_grad_eq + 126;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq07 = cartpole_QP_solver_grad_ineq + 126;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv07[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v07 = cartpole_QP_solver_v + 39;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re07[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta07[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc07[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff07 = cartpole_QP_solver_dv_aff + 39;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc07 = cartpole_QP_solver_dv_cc + 39;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V07[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd07[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld07[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy07[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy07[5];
int cartpole_QP_solver_lbIdx07[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb07 = cartpole_QP_solver_l + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb07 = cartpole_QP_solver_s + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb07 = cartpole_QP_solver_lbys + 196;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb07[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff07 = cartpole_QP_solver_dl_aff + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff07 = cartpole_QP_solver_ds_aff + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc07 = cartpole_QP_solver_dl_cc + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc07 = cartpole_QP_solver_ds_cc + 196;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl07 = cartpole_QP_solver_ccrhs + 196;
int cartpole_QP_solver_ubIdx07[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub07 = cartpole_QP_solver_l + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub07 = cartpole_QP_solver_s + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub07 = cartpole_QP_solver_lbys + 214;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub07[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff07 = cartpole_QP_solver_dl_aff + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff07 = cartpole_QP_solver_ds_aff + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc07 = cartpole_QP_solver_dl_cc + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc07 = cartpole_QP_solver_ds_cc + 214;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub07 = cartpole_QP_solver_ccrhs + 214;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi07[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W07[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd07[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd07[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z08 = cartpole_QP_solver_z + 144;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff08 = cartpole_QP_solver_dz_aff + 144;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc08 = cartpole_QP_solver_dz_cc + 144;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd08 = cartpole_QP_solver_rd + 144;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd08[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost08 = cartpole_QP_solver_grad_cost + 144;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq08 = cartpole_QP_solver_grad_eq + 144;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq08 = cartpole_QP_solver_grad_ineq + 144;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv08[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v08 = cartpole_QP_solver_v + 44;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re08[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta08[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc08[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff08 = cartpole_QP_solver_dv_aff + 44;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc08 = cartpole_QP_solver_dv_cc + 44;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V08[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd08[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld08[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy08[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy08[5];
int cartpole_QP_solver_lbIdx08[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb08 = cartpole_QP_solver_l + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb08 = cartpole_QP_solver_s + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb08 = cartpole_QP_solver_lbys + 224;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb08[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff08 = cartpole_QP_solver_dl_aff + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff08 = cartpole_QP_solver_ds_aff + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc08 = cartpole_QP_solver_dl_cc + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc08 = cartpole_QP_solver_ds_cc + 224;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl08 = cartpole_QP_solver_ccrhs + 224;
int cartpole_QP_solver_ubIdx08[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub08 = cartpole_QP_solver_l + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub08 = cartpole_QP_solver_s + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub08 = cartpole_QP_solver_lbys + 242;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub08[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff08 = cartpole_QP_solver_dl_aff + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff08 = cartpole_QP_solver_ds_aff + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc08 = cartpole_QP_solver_dl_cc + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc08 = cartpole_QP_solver_ds_cc + 242;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub08 = cartpole_QP_solver_ccrhs + 242;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi08[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W08[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd08[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd08[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z09 = cartpole_QP_solver_z + 162;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff09 = cartpole_QP_solver_dz_aff + 162;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc09 = cartpole_QP_solver_dz_cc + 162;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd09 = cartpole_QP_solver_rd + 162;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd09[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost09 = cartpole_QP_solver_grad_cost + 162;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq09 = cartpole_QP_solver_grad_eq + 162;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq09 = cartpole_QP_solver_grad_ineq + 162;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv09[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v09 = cartpole_QP_solver_v + 49;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re09[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta09[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc09[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff09 = cartpole_QP_solver_dv_aff + 49;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc09 = cartpole_QP_solver_dv_cc + 49;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V09[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd09[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld09[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy09[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy09[5];
int cartpole_QP_solver_lbIdx09[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb09 = cartpole_QP_solver_l + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb09 = cartpole_QP_solver_s + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb09 = cartpole_QP_solver_lbys + 252;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb09[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff09 = cartpole_QP_solver_dl_aff + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff09 = cartpole_QP_solver_ds_aff + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc09 = cartpole_QP_solver_dl_cc + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc09 = cartpole_QP_solver_ds_cc + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl09 = cartpole_QP_solver_ccrhs + 252;
int cartpole_QP_solver_ubIdx09[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub09 = cartpole_QP_solver_l + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub09 = cartpole_QP_solver_s + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub09 = cartpole_QP_solver_lbys + 270;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub09[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff09 = cartpole_QP_solver_dl_aff + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff09 = cartpole_QP_solver_ds_aff + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc09 = cartpole_QP_solver_dl_cc + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc09 = cartpole_QP_solver_ds_cc + 270;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub09 = cartpole_QP_solver_ccrhs + 270;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi09[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W09[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd09[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd09[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z10 = cartpole_QP_solver_z + 180;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff10 = cartpole_QP_solver_dz_aff + 180;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc10 = cartpole_QP_solver_dz_cc + 180;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd10 = cartpole_QP_solver_rd + 180;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd10[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost10 = cartpole_QP_solver_grad_cost + 180;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq10 = cartpole_QP_solver_grad_eq + 180;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq10 = cartpole_QP_solver_grad_ineq + 180;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv10[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v10 = cartpole_QP_solver_v + 54;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re10[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta10[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc10[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff10 = cartpole_QP_solver_dv_aff + 54;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc10 = cartpole_QP_solver_dv_cc + 54;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V10[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd10[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld10[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy10[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy10[5];
int cartpole_QP_solver_lbIdx10[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb10 = cartpole_QP_solver_l + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb10 = cartpole_QP_solver_s + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb10 = cartpole_QP_solver_lbys + 280;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb10[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff10 = cartpole_QP_solver_dl_aff + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff10 = cartpole_QP_solver_ds_aff + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc10 = cartpole_QP_solver_dl_cc + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc10 = cartpole_QP_solver_ds_cc + 280;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl10 = cartpole_QP_solver_ccrhs + 280;
int cartpole_QP_solver_ubIdx10[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub10 = cartpole_QP_solver_l + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub10 = cartpole_QP_solver_s + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub10 = cartpole_QP_solver_lbys + 298;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub10[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff10 = cartpole_QP_solver_dl_aff + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff10 = cartpole_QP_solver_ds_aff + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc10 = cartpole_QP_solver_dl_cc + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc10 = cartpole_QP_solver_ds_cc + 298;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub10 = cartpole_QP_solver_ccrhs + 298;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi10[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W10[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd10[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd10[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z11 = cartpole_QP_solver_z + 198;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff11 = cartpole_QP_solver_dz_aff + 198;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc11 = cartpole_QP_solver_dz_cc + 198;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd11 = cartpole_QP_solver_rd + 198;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd11[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost11 = cartpole_QP_solver_grad_cost + 198;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq11 = cartpole_QP_solver_grad_eq + 198;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq11 = cartpole_QP_solver_grad_ineq + 198;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv11[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v11 = cartpole_QP_solver_v + 59;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re11[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta11[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc11[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff11 = cartpole_QP_solver_dv_aff + 59;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc11 = cartpole_QP_solver_dv_cc + 59;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V11[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd11[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld11[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy11[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy11[5];
int cartpole_QP_solver_lbIdx11[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb11 = cartpole_QP_solver_l + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb11 = cartpole_QP_solver_s + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb11 = cartpole_QP_solver_lbys + 308;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb11[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff11 = cartpole_QP_solver_dl_aff + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff11 = cartpole_QP_solver_ds_aff + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc11 = cartpole_QP_solver_dl_cc + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc11 = cartpole_QP_solver_ds_cc + 308;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl11 = cartpole_QP_solver_ccrhs + 308;
int cartpole_QP_solver_ubIdx11[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub11 = cartpole_QP_solver_l + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub11 = cartpole_QP_solver_s + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub11 = cartpole_QP_solver_lbys + 326;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub11[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff11 = cartpole_QP_solver_dl_aff + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff11 = cartpole_QP_solver_ds_aff + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc11 = cartpole_QP_solver_dl_cc + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc11 = cartpole_QP_solver_ds_cc + 326;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub11 = cartpole_QP_solver_ccrhs + 326;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi11[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W11[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd11[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd11[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z12 = cartpole_QP_solver_z + 216;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff12 = cartpole_QP_solver_dz_aff + 216;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc12 = cartpole_QP_solver_dz_cc + 216;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd12 = cartpole_QP_solver_rd + 216;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd12[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost12 = cartpole_QP_solver_grad_cost + 216;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq12 = cartpole_QP_solver_grad_eq + 216;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq12 = cartpole_QP_solver_grad_ineq + 216;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv12[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v12 = cartpole_QP_solver_v + 64;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re12[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta12[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc12[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff12 = cartpole_QP_solver_dv_aff + 64;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc12 = cartpole_QP_solver_dv_cc + 64;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V12[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd12[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld12[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy12[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy12[5];
int cartpole_QP_solver_lbIdx12[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb12 = cartpole_QP_solver_l + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb12 = cartpole_QP_solver_s + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb12 = cartpole_QP_solver_lbys + 336;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb12[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff12 = cartpole_QP_solver_dl_aff + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff12 = cartpole_QP_solver_ds_aff + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc12 = cartpole_QP_solver_dl_cc + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc12 = cartpole_QP_solver_ds_cc + 336;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl12 = cartpole_QP_solver_ccrhs + 336;
int cartpole_QP_solver_ubIdx12[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub12 = cartpole_QP_solver_l + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub12 = cartpole_QP_solver_s + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub12 = cartpole_QP_solver_lbys + 354;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub12[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff12 = cartpole_QP_solver_dl_aff + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff12 = cartpole_QP_solver_ds_aff + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc12 = cartpole_QP_solver_dl_cc + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc12 = cartpole_QP_solver_ds_cc + 354;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub12 = cartpole_QP_solver_ccrhs + 354;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi12[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W12[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd12[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd12[25];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z13 = cartpole_QP_solver_z + 234;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff13 = cartpole_QP_solver_dz_aff + 234;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc13 = cartpole_QP_solver_dz_cc + 234;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd13 = cartpole_QP_solver_rd + 234;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd13[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost13 = cartpole_QP_solver_grad_cost + 234;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq13 = cartpole_QP_solver_grad_eq + 234;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq13 = cartpole_QP_solver_grad_ineq + 234;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv13[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_v13 = cartpole_QP_solver_v + 69;
cartpole_QP_solver_FLOAT cartpole_QP_solver_re13[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_beta13[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_betacc13[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvaff13 = cartpole_QP_solver_dv_aff + 69;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dvcc13 = cartpole_QP_solver_dv_cc + 69;
cartpole_QP_solver_FLOAT cartpole_QP_solver_V13[90];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Yd13[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ld13[15];
cartpole_QP_solver_FLOAT cartpole_QP_solver_yy13[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_bmy13[5];
int cartpole_QP_solver_lbIdx13[18] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb13 = cartpole_QP_solver_l + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb13 = cartpole_QP_solver_s + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb13 = cartpole_QP_solver_lbys + 364;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb13[18];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff13 = cartpole_QP_solver_dl_aff + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff13 = cartpole_QP_solver_ds_aff + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc13 = cartpole_QP_solver_dl_cc + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc13 = cartpole_QP_solver_ds_cc + 364;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl13 = cartpole_QP_solver_ccrhs + 364;
int cartpole_QP_solver_ubIdx13[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub13 = cartpole_QP_solver_l + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub13 = cartpole_QP_solver_s + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub13 = cartpole_QP_solver_lbys + 382;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub13[10];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff13 = cartpole_QP_solver_dl_aff + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff13 = cartpole_QP_solver_ds_aff + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc13 = cartpole_QP_solver_dl_cc + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc13 = cartpole_QP_solver_ds_cc + 382;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub13 = cartpole_QP_solver_ccrhs + 382;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi13[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_W13[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Ysd13[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lsd13[25];
cartpole_QP_solver_FLOAT cartpole_QP_solver_H14[5] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
cartpole_QP_solver_FLOAT cartpole_QP_solver_f14[5] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_z14 = cartpole_QP_solver_z + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzaff14 = cartpole_QP_solver_dz_aff + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dzcc14 = cartpole_QP_solver_dz_cc + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_rd14 = cartpole_QP_solver_rd + 252;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Lbyrd14[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_cost14 = cartpole_QP_solver_grad_cost + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_eq14 = cartpole_QP_solver_grad_eq + 252;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_grad_ineq14 = cartpole_QP_solver_grad_ineq + 252;
cartpole_QP_solver_FLOAT cartpole_QP_solver_ctv14[5];
int cartpole_QP_solver_lbIdx14[5] = {0, 1, 2, 3, 4};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llb14 = cartpole_QP_solver_l + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_slb14 = cartpole_QP_solver_s + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_llbbyslb14 = cartpole_QP_solver_lbys + 392;
cartpole_QP_solver_FLOAT cartpole_QP_solver_rilb14[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbaff14 = cartpole_QP_solver_dl_aff + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbaff14 = cartpole_QP_solver_ds_aff + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dllbcc14 = cartpole_QP_solver_dl_cc + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dslbcc14 = cartpole_QP_solver_ds_cc + 392;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsl14 = cartpole_QP_solver_ccrhs + 392;
int cartpole_QP_solver_ubIdx14[5] = {0, 1, 2, 3, 4};
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lub14 = cartpole_QP_solver_l + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_sub14 = cartpole_QP_solver_s + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_lubbysub14 = cartpole_QP_solver_lbys + 397;
cartpole_QP_solver_FLOAT cartpole_QP_solver_riub14[5];
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubaff14 = cartpole_QP_solver_dl_aff + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubaff14 = cartpole_QP_solver_ds_aff + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dlubcc14 = cartpole_QP_solver_dl_cc + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_dsubcc14 = cartpole_QP_solver_ds_cc + 397;
cartpole_QP_solver_FLOAT* cartpole_QP_solver_ccrhsub14 = cartpole_QP_solver_ccrhs + 397;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Phi14[5];
cartpole_QP_solver_FLOAT cartpole_QP_solver_D14[5] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
cartpole_QP_solver_FLOAT cartpole_QP_solver_W14[5];
cartpole_QP_solver_FLOAT musigma;
cartpole_QP_solver_FLOAT sigma_3rdroot;
cartpole_QP_solver_FLOAT cartpole_QP_solver_Diag1_0[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_Diag2_0[18];
cartpole_QP_solver_FLOAT cartpole_QP_solver_L_0[153];




/* SOLVER CODE --------------------------------------------------------- */
int cartpole_QP_solver_solve(cartpole_QP_solver_params* params, cartpole_QP_solver_output* output, cartpole_QP_solver_info* info)
{	
int exitcode;

#if cartpole_QP_solver_SET_TIMING == 1
	cartpole_QP_solver_timer solvertimer;
	cartpole_QP_solver_tic(&solvertimer);
#endif
/* FUNCTION CALLS INTO LA LIBRARY -------------------------------------- */
info->it = 0;
cartpole_QP_solver_LA_INITIALIZEVECTOR_257(cartpole_QP_solver_z, 0);
cartpole_QP_solver_LA_INITIALIZEVECTOR_74(cartpole_QP_solver_v, 1);
cartpole_QP_solver_LA_INITIALIZEVECTOR_402(cartpole_QP_solver_l, 10);
cartpole_QP_solver_LA_INITIALIZEVECTOR_402(cartpole_QP_solver_s, 10);
info->mu = 0;
cartpole_QP_solver_LA_DOTACC_402(cartpole_QP_solver_l, cartpole_QP_solver_s, &info->mu);
info->mu /= 402;
while( 1 ){
info->pobj = 0;
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H1, params->f1, cartpole_QP_solver_z00, cartpole_QP_solver_grad_cost00, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H2, params->f2, cartpole_QP_solver_z01, cartpole_QP_solver_grad_cost01, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H3, params->f3, cartpole_QP_solver_z02, cartpole_QP_solver_grad_cost02, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H4, params->f4, cartpole_QP_solver_z03, cartpole_QP_solver_grad_cost03, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H5, params->f5, cartpole_QP_solver_z04, cartpole_QP_solver_grad_cost04, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H6, params->f6, cartpole_QP_solver_z05, cartpole_QP_solver_grad_cost05, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H7, params->f7, cartpole_QP_solver_z06, cartpole_QP_solver_grad_cost06, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H8, params->f8, cartpole_QP_solver_z07, cartpole_QP_solver_grad_cost07, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H9, params->f9, cartpole_QP_solver_z08, cartpole_QP_solver_grad_cost08, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H10, params->f10, cartpole_QP_solver_z09, cartpole_QP_solver_grad_cost09, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H11, params->f11, cartpole_QP_solver_z10, cartpole_QP_solver_grad_cost10, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H12, params->f12, cartpole_QP_solver_z11, cartpole_QP_solver_grad_cost11, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H13, params->f13, cartpole_QP_solver_z12, cartpole_QP_solver_grad_cost12, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_18(params->H14, params->f14, cartpole_QP_solver_z13, cartpole_QP_solver_grad_cost13, &info->pobj);
cartpole_QP_solver_LA_DIAG_QUADFCN_5(cartpole_QP_solver_H14, cartpole_QP_solver_f14, cartpole_QP_solver_z14, cartpole_QP_solver_grad_cost14, &info->pobj);
info->res_eq = 0;
info->dgap = 0;
cartpole_QP_solver_LA_DENSE_MVMSUB3_9_18_18(params->C1, cartpole_QP_solver_z00, cartpole_QP_solver_D01, cartpole_QP_solver_z01, params->e1, cartpole_QP_solver_v00, cartpole_QP_solver_re00, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C2, cartpole_QP_solver_z01, cartpole_QP_solver_D02, cartpole_QP_solver_z02, params->e2, cartpole_QP_solver_v01, cartpole_QP_solver_re01, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C3, cartpole_QP_solver_z02, cartpole_QP_solver_D02, cartpole_QP_solver_z03, params->e3, cartpole_QP_solver_v02, cartpole_QP_solver_re02, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C4, cartpole_QP_solver_z03, cartpole_QP_solver_D02, cartpole_QP_solver_z04, params->e4, cartpole_QP_solver_v03, cartpole_QP_solver_re03, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C5, cartpole_QP_solver_z04, cartpole_QP_solver_D02, cartpole_QP_solver_z05, params->e5, cartpole_QP_solver_v04, cartpole_QP_solver_re04, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C6, cartpole_QP_solver_z05, cartpole_QP_solver_D02, cartpole_QP_solver_z06, params->e6, cartpole_QP_solver_v05, cartpole_QP_solver_re05, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C7, cartpole_QP_solver_z06, cartpole_QP_solver_D02, cartpole_QP_solver_z07, params->e7, cartpole_QP_solver_v06, cartpole_QP_solver_re06, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C8, cartpole_QP_solver_z07, cartpole_QP_solver_D02, cartpole_QP_solver_z08, params->e8, cartpole_QP_solver_v07, cartpole_QP_solver_re07, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C9, cartpole_QP_solver_z08, cartpole_QP_solver_D02, cartpole_QP_solver_z09, params->e9, cartpole_QP_solver_v08, cartpole_QP_solver_re08, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C10, cartpole_QP_solver_z09, cartpole_QP_solver_D02, cartpole_QP_solver_z10, params->e10, cartpole_QP_solver_v09, cartpole_QP_solver_re09, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C11, cartpole_QP_solver_z10, cartpole_QP_solver_D02, cartpole_QP_solver_z11, params->e11, cartpole_QP_solver_v10, cartpole_QP_solver_re10, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C12, cartpole_QP_solver_z11, cartpole_QP_solver_D02, cartpole_QP_solver_z12, params->e12, cartpole_QP_solver_v11, cartpole_QP_solver_re11, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_18(params->C13, cartpole_QP_solver_z12, cartpole_QP_solver_D02, cartpole_QP_solver_z13, params->e13, cartpole_QP_solver_v12, cartpole_QP_solver_re12, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_5_18_5(params->C14, cartpole_QP_solver_z13, cartpole_QP_solver_D14, cartpole_QP_solver_z14, params->e14, cartpole_QP_solver_v13, cartpole_QP_solver_re13, &info->dgap, &info->res_eq);
cartpole_QP_solver_LA_DENSE_MTVM_9_18(params->C1, cartpole_QP_solver_v00, cartpole_QP_solver_grad_eq00);
cartpole_QP_solver_LA_DENSE_MTVM2_5_18_9(params->C2, cartpole_QP_solver_v01, cartpole_QP_solver_D01, cartpole_QP_solver_v00, cartpole_QP_solver_grad_eq01);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C3, cartpole_QP_solver_v02, cartpole_QP_solver_D02, cartpole_QP_solver_v01, cartpole_QP_solver_grad_eq02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C4, cartpole_QP_solver_v03, cartpole_QP_solver_D02, cartpole_QP_solver_v02, cartpole_QP_solver_grad_eq03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C5, cartpole_QP_solver_v04, cartpole_QP_solver_D02, cartpole_QP_solver_v03, cartpole_QP_solver_grad_eq04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C6, cartpole_QP_solver_v05, cartpole_QP_solver_D02, cartpole_QP_solver_v04, cartpole_QP_solver_grad_eq05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C7, cartpole_QP_solver_v06, cartpole_QP_solver_D02, cartpole_QP_solver_v05, cartpole_QP_solver_grad_eq06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C8, cartpole_QP_solver_v07, cartpole_QP_solver_D02, cartpole_QP_solver_v06, cartpole_QP_solver_grad_eq07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C9, cartpole_QP_solver_v08, cartpole_QP_solver_D02, cartpole_QP_solver_v07, cartpole_QP_solver_grad_eq08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C10, cartpole_QP_solver_v09, cartpole_QP_solver_D02, cartpole_QP_solver_v08, cartpole_QP_solver_grad_eq09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C11, cartpole_QP_solver_v10, cartpole_QP_solver_D02, cartpole_QP_solver_v09, cartpole_QP_solver_grad_eq10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C12, cartpole_QP_solver_v11, cartpole_QP_solver_D02, cartpole_QP_solver_v10, cartpole_QP_solver_grad_eq11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C13, cartpole_QP_solver_v12, cartpole_QP_solver_D02, cartpole_QP_solver_v11, cartpole_QP_solver_grad_eq12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C14, cartpole_QP_solver_v13, cartpole_QP_solver_D02, cartpole_QP_solver_v12, cartpole_QP_solver_grad_eq13);
cartpole_QP_solver_LA_DIAGZERO_MTVM_5_5(cartpole_QP_solver_D14, cartpole_QP_solver_v13, cartpole_QP_solver_grad_eq14);
info->res_ineq = 0;
cartpole_QP_solver_LA_VSUBADD3_18(params->lb1, cartpole_QP_solver_z00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_llb00, cartpole_QP_solver_slb00, cartpole_QP_solver_rilb00, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z00, cartpole_QP_solver_ubIdx00, params->ub1, cartpole_QP_solver_lub00, cartpole_QP_solver_sub00, cartpole_QP_solver_riub00, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb2, cartpole_QP_solver_z01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_llb01, cartpole_QP_solver_slb01, cartpole_QP_solver_rilb01, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z01, cartpole_QP_solver_ubIdx01, params->ub2, cartpole_QP_solver_lub01, cartpole_QP_solver_sub01, cartpole_QP_solver_riub01, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb3, cartpole_QP_solver_z02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_llb02, cartpole_QP_solver_slb02, cartpole_QP_solver_rilb02, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z02, cartpole_QP_solver_ubIdx02, params->ub3, cartpole_QP_solver_lub02, cartpole_QP_solver_sub02, cartpole_QP_solver_riub02, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb4, cartpole_QP_solver_z03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_llb03, cartpole_QP_solver_slb03, cartpole_QP_solver_rilb03, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z03, cartpole_QP_solver_ubIdx03, params->ub4, cartpole_QP_solver_lub03, cartpole_QP_solver_sub03, cartpole_QP_solver_riub03, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb5, cartpole_QP_solver_z04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_llb04, cartpole_QP_solver_slb04, cartpole_QP_solver_rilb04, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z04, cartpole_QP_solver_ubIdx04, params->ub5, cartpole_QP_solver_lub04, cartpole_QP_solver_sub04, cartpole_QP_solver_riub04, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb6, cartpole_QP_solver_z05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_llb05, cartpole_QP_solver_slb05, cartpole_QP_solver_rilb05, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z05, cartpole_QP_solver_ubIdx05, params->ub6, cartpole_QP_solver_lub05, cartpole_QP_solver_sub05, cartpole_QP_solver_riub05, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb7, cartpole_QP_solver_z06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_llb06, cartpole_QP_solver_slb06, cartpole_QP_solver_rilb06, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z06, cartpole_QP_solver_ubIdx06, params->ub7, cartpole_QP_solver_lub06, cartpole_QP_solver_sub06, cartpole_QP_solver_riub06, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb8, cartpole_QP_solver_z07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_llb07, cartpole_QP_solver_slb07, cartpole_QP_solver_rilb07, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z07, cartpole_QP_solver_ubIdx07, params->ub8, cartpole_QP_solver_lub07, cartpole_QP_solver_sub07, cartpole_QP_solver_riub07, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb9, cartpole_QP_solver_z08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_llb08, cartpole_QP_solver_slb08, cartpole_QP_solver_rilb08, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z08, cartpole_QP_solver_ubIdx08, params->ub9, cartpole_QP_solver_lub08, cartpole_QP_solver_sub08, cartpole_QP_solver_riub08, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb10, cartpole_QP_solver_z09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_llb09, cartpole_QP_solver_slb09, cartpole_QP_solver_rilb09, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z09, cartpole_QP_solver_ubIdx09, params->ub10, cartpole_QP_solver_lub09, cartpole_QP_solver_sub09, cartpole_QP_solver_riub09, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb11, cartpole_QP_solver_z10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_llb10, cartpole_QP_solver_slb10, cartpole_QP_solver_rilb10, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z10, cartpole_QP_solver_ubIdx10, params->ub11, cartpole_QP_solver_lub10, cartpole_QP_solver_sub10, cartpole_QP_solver_riub10, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb12, cartpole_QP_solver_z11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_llb11, cartpole_QP_solver_slb11, cartpole_QP_solver_rilb11, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z11, cartpole_QP_solver_ubIdx11, params->ub12, cartpole_QP_solver_lub11, cartpole_QP_solver_sub11, cartpole_QP_solver_riub11, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb13, cartpole_QP_solver_z12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_llb12, cartpole_QP_solver_slb12, cartpole_QP_solver_rilb12, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z12, cartpole_QP_solver_ubIdx12, params->ub13, cartpole_QP_solver_lub12, cartpole_QP_solver_sub12, cartpole_QP_solver_riub12, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_18(params->lb14, cartpole_QP_solver_z13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_llb13, cartpole_QP_solver_slb13, cartpole_QP_solver_rilb13, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_10(cartpole_QP_solver_z13, cartpole_QP_solver_ubIdx13, params->ub14, cartpole_QP_solver_lub13, cartpole_QP_solver_sub13, cartpole_QP_solver_riub13, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD3_5(params->lb15, cartpole_QP_solver_z14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_llb14, cartpole_QP_solver_slb14, cartpole_QP_solver_rilb14, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_VSUBADD2_5(cartpole_QP_solver_z14, cartpole_QP_solver_ubIdx14, params->ub15, cartpole_QP_solver_lub14, cartpole_QP_solver_sub14, cartpole_QP_solver_riub14, &info->dgap, &info->res_ineq);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub00, cartpole_QP_solver_sub00, cartpole_QP_solver_riub00, cartpole_QP_solver_llb00, cartpole_QP_solver_slb00, cartpole_QP_solver_rilb00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_ubIdx00, cartpole_QP_solver_grad_ineq00, cartpole_QP_solver_lubbysub00, cartpole_QP_solver_llbbyslb00);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub01, cartpole_QP_solver_sub01, cartpole_QP_solver_riub01, cartpole_QP_solver_llb01, cartpole_QP_solver_slb01, cartpole_QP_solver_rilb01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_ubIdx01, cartpole_QP_solver_grad_ineq01, cartpole_QP_solver_lubbysub01, cartpole_QP_solver_llbbyslb01);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub02, cartpole_QP_solver_sub02, cartpole_QP_solver_riub02, cartpole_QP_solver_llb02, cartpole_QP_solver_slb02, cartpole_QP_solver_rilb02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_ubIdx02, cartpole_QP_solver_grad_ineq02, cartpole_QP_solver_lubbysub02, cartpole_QP_solver_llbbyslb02);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub03, cartpole_QP_solver_sub03, cartpole_QP_solver_riub03, cartpole_QP_solver_llb03, cartpole_QP_solver_slb03, cartpole_QP_solver_rilb03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_ubIdx03, cartpole_QP_solver_grad_ineq03, cartpole_QP_solver_lubbysub03, cartpole_QP_solver_llbbyslb03);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub04, cartpole_QP_solver_sub04, cartpole_QP_solver_riub04, cartpole_QP_solver_llb04, cartpole_QP_solver_slb04, cartpole_QP_solver_rilb04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_ubIdx04, cartpole_QP_solver_grad_ineq04, cartpole_QP_solver_lubbysub04, cartpole_QP_solver_llbbyslb04);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub05, cartpole_QP_solver_sub05, cartpole_QP_solver_riub05, cartpole_QP_solver_llb05, cartpole_QP_solver_slb05, cartpole_QP_solver_rilb05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_ubIdx05, cartpole_QP_solver_grad_ineq05, cartpole_QP_solver_lubbysub05, cartpole_QP_solver_llbbyslb05);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub06, cartpole_QP_solver_sub06, cartpole_QP_solver_riub06, cartpole_QP_solver_llb06, cartpole_QP_solver_slb06, cartpole_QP_solver_rilb06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_ubIdx06, cartpole_QP_solver_grad_ineq06, cartpole_QP_solver_lubbysub06, cartpole_QP_solver_llbbyslb06);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub07, cartpole_QP_solver_sub07, cartpole_QP_solver_riub07, cartpole_QP_solver_llb07, cartpole_QP_solver_slb07, cartpole_QP_solver_rilb07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_ubIdx07, cartpole_QP_solver_grad_ineq07, cartpole_QP_solver_lubbysub07, cartpole_QP_solver_llbbyslb07);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub08, cartpole_QP_solver_sub08, cartpole_QP_solver_riub08, cartpole_QP_solver_llb08, cartpole_QP_solver_slb08, cartpole_QP_solver_rilb08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_ubIdx08, cartpole_QP_solver_grad_ineq08, cartpole_QP_solver_lubbysub08, cartpole_QP_solver_llbbyslb08);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub09, cartpole_QP_solver_sub09, cartpole_QP_solver_riub09, cartpole_QP_solver_llb09, cartpole_QP_solver_slb09, cartpole_QP_solver_rilb09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_ubIdx09, cartpole_QP_solver_grad_ineq09, cartpole_QP_solver_lubbysub09, cartpole_QP_solver_llbbyslb09);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub10, cartpole_QP_solver_sub10, cartpole_QP_solver_riub10, cartpole_QP_solver_llb10, cartpole_QP_solver_slb10, cartpole_QP_solver_rilb10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_ubIdx10, cartpole_QP_solver_grad_ineq10, cartpole_QP_solver_lubbysub10, cartpole_QP_solver_llbbyslb10);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub11, cartpole_QP_solver_sub11, cartpole_QP_solver_riub11, cartpole_QP_solver_llb11, cartpole_QP_solver_slb11, cartpole_QP_solver_rilb11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_ubIdx11, cartpole_QP_solver_grad_ineq11, cartpole_QP_solver_lubbysub11, cartpole_QP_solver_llbbyslb11);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub12, cartpole_QP_solver_sub12, cartpole_QP_solver_riub12, cartpole_QP_solver_llb12, cartpole_QP_solver_slb12, cartpole_QP_solver_rilb12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_ubIdx12, cartpole_QP_solver_grad_ineq12, cartpole_QP_solver_lubbysub12, cartpole_QP_solver_llbbyslb12);
cartpole_QP_solver_LA_INEQ_B_GRAD_18_18_10(cartpole_QP_solver_lub13, cartpole_QP_solver_sub13, cartpole_QP_solver_riub13, cartpole_QP_solver_llb13, cartpole_QP_solver_slb13, cartpole_QP_solver_rilb13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_ubIdx13, cartpole_QP_solver_grad_ineq13, cartpole_QP_solver_lubbysub13, cartpole_QP_solver_llbbyslb13);
cartpole_QP_solver_LA_INEQ_B_GRAD_5_5_5(cartpole_QP_solver_lub14, cartpole_QP_solver_sub14, cartpole_QP_solver_riub14, cartpole_QP_solver_llb14, cartpole_QP_solver_slb14, cartpole_QP_solver_rilb14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_ubIdx14, cartpole_QP_solver_grad_ineq14, cartpole_QP_solver_lubbysub14, cartpole_QP_solver_llbbyslb14);
info->dobj = info->pobj - info->dgap;
info->rdgap = info->pobj ? info->dgap / info->pobj : 1e6;
if( info->rdgap < 0 ) info->rdgap = -info->rdgap;
if( info->mu < cartpole_QP_solver_SET_ACC_KKTCOMPL
    && (info->rdgap < cartpole_QP_solver_SET_ACC_RDGAP || info->dgap < cartpole_QP_solver_SET_ACC_KKTCOMPL)
    && info->res_eq < cartpole_QP_solver_SET_ACC_RESEQ
    && info->res_ineq < cartpole_QP_solver_SET_ACC_RESINEQ ){
exitcode = cartpole_QP_solver_OPTIMAL; break; }
if( info->it == cartpole_QP_solver_SET_MAXIT ){
exitcode = cartpole_QP_solver_MAXITREACHED; break; }
cartpole_QP_solver_LA_VVADD3_257(cartpole_QP_solver_grad_cost, cartpole_QP_solver_grad_eq, cartpole_QP_solver_grad_ineq, cartpole_QP_solver_rd);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H1, cartpole_QP_solver_llbbyslb00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_lubbysub00, cartpole_QP_solver_ubIdx00, cartpole_QP_solver_Phi00);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_18(cartpole_QP_solver_Phi00, params->C1, cartpole_QP_solver_V00);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi00, cartpole_QP_solver_rd00, cartpole_QP_solver_Lbyrd00);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H2, cartpole_QP_solver_llbbyslb01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_lubbysub01, cartpole_QP_solver_ubIdx01, cartpole_QP_solver_Phi01);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi01, params->C2, cartpole_QP_solver_V01);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_9_18(cartpole_QP_solver_Phi01, cartpole_QP_solver_D01, cartpole_QP_solver_W01);
cartpole_QP_solver_LA_DENSE_MMTM_9_18_5(cartpole_QP_solver_W01, cartpole_QP_solver_V01, cartpole_QP_solver_Ysd01);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi01, cartpole_QP_solver_rd01, cartpole_QP_solver_Lbyrd01);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H3, cartpole_QP_solver_llbbyslb02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_lubbysub02, cartpole_QP_solver_ubIdx02, cartpole_QP_solver_Phi02);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi02, params->C3, cartpole_QP_solver_V02);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi02, cartpole_QP_solver_D02, cartpole_QP_solver_W02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W02, cartpole_QP_solver_V02, cartpole_QP_solver_Ysd02);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi02, cartpole_QP_solver_rd02, cartpole_QP_solver_Lbyrd02);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H4, cartpole_QP_solver_llbbyslb03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_lubbysub03, cartpole_QP_solver_ubIdx03, cartpole_QP_solver_Phi03);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi03, params->C4, cartpole_QP_solver_V03);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi03, cartpole_QP_solver_D02, cartpole_QP_solver_W03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W03, cartpole_QP_solver_V03, cartpole_QP_solver_Ysd03);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi03, cartpole_QP_solver_rd03, cartpole_QP_solver_Lbyrd03);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H5, cartpole_QP_solver_llbbyslb04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_lubbysub04, cartpole_QP_solver_ubIdx04, cartpole_QP_solver_Phi04);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi04, params->C5, cartpole_QP_solver_V04);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi04, cartpole_QP_solver_D02, cartpole_QP_solver_W04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W04, cartpole_QP_solver_V04, cartpole_QP_solver_Ysd04);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi04, cartpole_QP_solver_rd04, cartpole_QP_solver_Lbyrd04);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H6, cartpole_QP_solver_llbbyslb05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_lubbysub05, cartpole_QP_solver_ubIdx05, cartpole_QP_solver_Phi05);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi05, params->C6, cartpole_QP_solver_V05);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi05, cartpole_QP_solver_D02, cartpole_QP_solver_W05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W05, cartpole_QP_solver_V05, cartpole_QP_solver_Ysd05);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi05, cartpole_QP_solver_rd05, cartpole_QP_solver_Lbyrd05);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H7, cartpole_QP_solver_llbbyslb06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_lubbysub06, cartpole_QP_solver_ubIdx06, cartpole_QP_solver_Phi06);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi06, params->C7, cartpole_QP_solver_V06);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi06, cartpole_QP_solver_D02, cartpole_QP_solver_W06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W06, cartpole_QP_solver_V06, cartpole_QP_solver_Ysd06);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi06, cartpole_QP_solver_rd06, cartpole_QP_solver_Lbyrd06);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H8, cartpole_QP_solver_llbbyslb07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_lubbysub07, cartpole_QP_solver_ubIdx07, cartpole_QP_solver_Phi07);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi07, params->C8, cartpole_QP_solver_V07);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi07, cartpole_QP_solver_D02, cartpole_QP_solver_W07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W07, cartpole_QP_solver_V07, cartpole_QP_solver_Ysd07);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi07, cartpole_QP_solver_rd07, cartpole_QP_solver_Lbyrd07);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H9, cartpole_QP_solver_llbbyslb08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_lubbysub08, cartpole_QP_solver_ubIdx08, cartpole_QP_solver_Phi08);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi08, params->C9, cartpole_QP_solver_V08);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi08, cartpole_QP_solver_D02, cartpole_QP_solver_W08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W08, cartpole_QP_solver_V08, cartpole_QP_solver_Ysd08);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi08, cartpole_QP_solver_rd08, cartpole_QP_solver_Lbyrd08);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H10, cartpole_QP_solver_llbbyslb09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_lubbysub09, cartpole_QP_solver_ubIdx09, cartpole_QP_solver_Phi09);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi09, params->C10, cartpole_QP_solver_V09);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi09, cartpole_QP_solver_D02, cartpole_QP_solver_W09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W09, cartpole_QP_solver_V09, cartpole_QP_solver_Ysd09);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi09, cartpole_QP_solver_rd09, cartpole_QP_solver_Lbyrd09);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H11, cartpole_QP_solver_llbbyslb10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_lubbysub10, cartpole_QP_solver_ubIdx10, cartpole_QP_solver_Phi10);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi10, params->C11, cartpole_QP_solver_V10);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi10, cartpole_QP_solver_D02, cartpole_QP_solver_W10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W10, cartpole_QP_solver_V10, cartpole_QP_solver_Ysd10);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi10, cartpole_QP_solver_rd10, cartpole_QP_solver_Lbyrd10);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H12, cartpole_QP_solver_llbbyslb11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_lubbysub11, cartpole_QP_solver_ubIdx11, cartpole_QP_solver_Phi11);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi11, params->C12, cartpole_QP_solver_V11);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi11, cartpole_QP_solver_D02, cartpole_QP_solver_W11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W11, cartpole_QP_solver_V11, cartpole_QP_solver_Ysd11);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi11, cartpole_QP_solver_rd11, cartpole_QP_solver_Lbyrd11);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H13, cartpole_QP_solver_llbbyslb12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_lubbysub12, cartpole_QP_solver_ubIdx12, cartpole_QP_solver_Phi12);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi12, params->C13, cartpole_QP_solver_V12);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi12, cartpole_QP_solver_D02, cartpole_QP_solver_W12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W12, cartpole_QP_solver_V12, cartpole_QP_solver_Ysd12);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi12, cartpole_QP_solver_rd12, cartpole_QP_solver_Lbyrd12);
cartpole_QP_solver_LA_DIAG_CHOL_LBUB_18_18_10(params->H14, cartpole_QP_solver_llbbyslb13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_lubbysub13, cartpole_QP_solver_ubIdx13, cartpole_QP_solver_Phi13);
cartpole_QP_solver_LA_DIAG_MATRIXFORWARDSUB_5_18(cartpole_QP_solver_Phi13, params->C14, cartpole_QP_solver_V13);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_18(cartpole_QP_solver_Phi13, cartpole_QP_solver_D02, cartpole_QP_solver_W13);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMTM_5_18_5(cartpole_QP_solver_W13, cartpole_QP_solver_V13, cartpole_QP_solver_Ysd13);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi13, cartpole_QP_solver_rd13, cartpole_QP_solver_Lbyrd13);
cartpole_QP_solver_LA_DIAG_CHOL_ONELOOP_LBUB_5_5_5(cartpole_QP_solver_H14, cartpole_QP_solver_llbbyslb14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_lubbysub14, cartpole_QP_solver_ubIdx14, cartpole_QP_solver_Phi14);
cartpole_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Phi14, cartpole_QP_solver_D14, cartpole_QP_solver_W14);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_5(cartpole_QP_solver_Phi14, cartpole_QP_solver_rd14, cartpole_QP_solver_Lbyrd14);
cartpole_QP_solver_LA_DENSE_MMT2_9_18_18(cartpole_QP_solver_V00, cartpole_QP_solver_W01, cartpole_QP_solver_Yd00);
cartpole_QP_solver_LA_DENSE_MVMSUB2_9_18_18(cartpole_QP_solver_V00, cartpole_QP_solver_Lbyrd00, cartpole_QP_solver_W01, cartpole_QP_solver_Lbyrd01, cartpole_QP_solver_re00, cartpole_QP_solver_beta00);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V01, cartpole_QP_solver_W02, cartpole_QP_solver_Yd01);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V01, cartpole_QP_solver_Lbyrd01, cartpole_QP_solver_W02, cartpole_QP_solver_Lbyrd02, cartpole_QP_solver_re01, cartpole_QP_solver_beta01);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V02, cartpole_QP_solver_W03, cartpole_QP_solver_Yd02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V02, cartpole_QP_solver_Lbyrd02, cartpole_QP_solver_W03, cartpole_QP_solver_Lbyrd03, cartpole_QP_solver_re02, cartpole_QP_solver_beta02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V03, cartpole_QP_solver_W04, cartpole_QP_solver_Yd03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V03, cartpole_QP_solver_Lbyrd03, cartpole_QP_solver_W04, cartpole_QP_solver_Lbyrd04, cartpole_QP_solver_re03, cartpole_QP_solver_beta03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V04, cartpole_QP_solver_W05, cartpole_QP_solver_Yd04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V04, cartpole_QP_solver_Lbyrd04, cartpole_QP_solver_W05, cartpole_QP_solver_Lbyrd05, cartpole_QP_solver_re04, cartpole_QP_solver_beta04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V05, cartpole_QP_solver_W06, cartpole_QP_solver_Yd05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V05, cartpole_QP_solver_Lbyrd05, cartpole_QP_solver_W06, cartpole_QP_solver_Lbyrd06, cartpole_QP_solver_re05, cartpole_QP_solver_beta05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V06, cartpole_QP_solver_W07, cartpole_QP_solver_Yd06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V06, cartpole_QP_solver_Lbyrd06, cartpole_QP_solver_W07, cartpole_QP_solver_Lbyrd07, cartpole_QP_solver_re06, cartpole_QP_solver_beta06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V07, cartpole_QP_solver_W08, cartpole_QP_solver_Yd07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V07, cartpole_QP_solver_Lbyrd07, cartpole_QP_solver_W08, cartpole_QP_solver_Lbyrd08, cartpole_QP_solver_re07, cartpole_QP_solver_beta07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V08, cartpole_QP_solver_W09, cartpole_QP_solver_Yd08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V08, cartpole_QP_solver_Lbyrd08, cartpole_QP_solver_W09, cartpole_QP_solver_Lbyrd09, cartpole_QP_solver_re08, cartpole_QP_solver_beta08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V09, cartpole_QP_solver_W10, cartpole_QP_solver_Yd09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V09, cartpole_QP_solver_Lbyrd09, cartpole_QP_solver_W10, cartpole_QP_solver_Lbyrd10, cartpole_QP_solver_re09, cartpole_QP_solver_beta09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V10, cartpole_QP_solver_W11, cartpole_QP_solver_Yd10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V10, cartpole_QP_solver_Lbyrd10, cartpole_QP_solver_W11, cartpole_QP_solver_Lbyrd11, cartpole_QP_solver_re10, cartpole_QP_solver_beta10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V11, cartpole_QP_solver_W12, cartpole_QP_solver_Yd11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V11, cartpole_QP_solver_Lbyrd11, cartpole_QP_solver_W12, cartpole_QP_solver_Lbyrd12, cartpole_QP_solver_re11, cartpole_QP_solver_beta11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_18(cartpole_QP_solver_V12, cartpole_QP_solver_W13, cartpole_QP_solver_Yd12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_18(cartpole_QP_solver_V12, cartpole_QP_solver_Lbyrd12, cartpole_QP_solver_W13, cartpole_QP_solver_Lbyrd13, cartpole_QP_solver_re12, cartpole_QP_solver_beta12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MMT2_5_18_5(cartpole_QP_solver_V13, cartpole_QP_solver_W14, cartpole_QP_solver_Yd13);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_5_18_5(cartpole_QP_solver_V13, cartpole_QP_solver_Lbyrd13, cartpole_QP_solver_W14, cartpole_QP_solver_Lbyrd14, cartpole_QP_solver_re13, cartpole_QP_solver_beta13);
cartpole_QP_solver_LA_DENSE_CHOL_9(cartpole_QP_solver_Yd00, cartpole_QP_solver_Ld00);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_9(cartpole_QP_solver_Ld00, cartpole_QP_solver_beta00, cartpole_QP_solver_yy00);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_9(cartpole_QP_solver_Ld00, cartpole_QP_solver_Ysd01, cartpole_QP_solver_Lsd01);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_9(cartpole_QP_solver_Lsd01, cartpole_QP_solver_Yd01);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd01, cartpole_QP_solver_Ld01);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_9(cartpole_QP_solver_Lsd01, cartpole_QP_solver_yy00, cartpole_QP_solver_beta01, cartpole_QP_solver_bmy01);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld01, cartpole_QP_solver_bmy01, cartpole_QP_solver_yy01);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld01, cartpole_QP_solver_Ysd02, cartpole_QP_solver_Lsd02);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd02, cartpole_QP_solver_Yd02);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd02, cartpole_QP_solver_Ld02);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd02, cartpole_QP_solver_yy01, cartpole_QP_solver_beta02, cartpole_QP_solver_bmy02);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld02, cartpole_QP_solver_bmy02, cartpole_QP_solver_yy02);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld02, cartpole_QP_solver_Ysd03, cartpole_QP_solver_Lsd03);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd03, cartpole_QP_solver_Yd03);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd03, cartpole_QP_solver_Ld03);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd03, cartpole_QP_solver_yy02, cartpole_QP_solver_beta03, cartpole_QP_solver_bmy03);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld03, cartpole_QP_solver_bmy03, cartpole_QP_solver_yy03);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld03, cartpole_QP_solver_Ysd04, cartpole_QP_solver_Lsd04);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd04, cartpole_QP_solver_Yd04);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd04, cartpole_QP_solver_Ld04);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd04, cartpole_QP_solver_yy03, cartpole_QP_solver_beta04, cartpole_QP_solver_bmy04);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld04, cartpole_QP_solver_bmy04, cartpole_QP_solver_yy04);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld04, cartpole_QP_solver_Ysd05, cartpole_QP_solver_Lsd05);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd05, cartpole_QP_solver_Yd05);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd05, cartpole_QP_solver_Ld05);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd05, cartpole_QP_solver_yy04, cartpole_QP_solver_beta05, cartpole_QP_solver_bmy05);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld05, cartpole_QP_solver_bmy05, cartpole_QP_solver_yy05);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld05, cartpole_QP_solver_Ysd06, cartpole_QP_solver_Lsd06);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd06, cartpole_QP_solver_Yd06);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd06, cartpole_QP_solver_Ld06);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd06, cartpole_QP_solver_yy05, cartpole_QP_solver_beta06, cartpole_QP_solver_bmy06);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld06, cartpole_QP_solver_bmy06, cartpole_QP_solver_yy06);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld06, cartpole_QP_solver_Ysd07, cartpole_QP_solver_Lsd07);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd07, cartpole_QP_solver_Yd07);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd07, cartpole_QP_solver_Ld07);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd07, cartpole_QP_solver_yy06, cartpole_QP_solver_beta07, cartpole_QP_solver_bmy07);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld07, cartpole_QP_solver_bmy07, cartpole_QP_solver_yy07);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld07, cartpole_QP_solver_Ysd08, cartpole_QP_solver_Lsd08);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd08, cartpole_QP_solver_Yd08);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd08, cartpole_QP_solver_Ld08);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd08, cartpole_QP_solver_yy07, cartpole_QP_solver_beta08, cartpole_QP_solver_bmy08);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld08, cartpole_QP_solver_bmy08, cartpole_QP_solver_yy08);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld08, cartpole_QP_solver_Ysd09, cartpole_QP_solver_Lsd09);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd09, cartpole_QP_solver_Yd09);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd09, cartpole_QP_solver_Ld09);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd09, cartpole_QP_solver_yy08, cartpole_QP_solver_beta09, cartpole_QP_solver_bmy09);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld09, cartpole_QP_solver_bmy09, cartpole_QP_solver_yy09);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld09, cartpole_QP_solver_Ysd10, cartpole_QP_solver_Lsd10);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd10, cartpole_QP_solver_Yd10);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd10, cartpole_QP_solver_Ld10);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd10, cartpole_QP_solver_yy09, cartpole_QP_solver_beta10, cartpole_QP_solver_bmy10);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld10, cartpole_QP_solver_bmy10, cartpole_QP_solver_yy10);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld10, cartpole_QP_solver_Ysd11, cartpole_QP_solver_Lsd11);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd11, cartpole_QP_solver_Yd11);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd11, cartpole_QP_solver_Ld11);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd11, cartpole_QP_solver_yy10, cartpole_QP_solver_beta11, cartpole_QP_solver_bmy11);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld11, cartpole_QP_solver_bmy11, cartpole_QP_solver_yy11);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld11, cartpole_QP_solver_Ysd12, cartpole_QP_solver_Lsd12);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd12, cartpole_QP_solver_Yd12);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd12, cartpole_QP_solver_Ld12);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd12, cartpole_QP_solver_yy11, cartpole_QP_solver_beta12, cartpole_QP_solver_bmy12);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld12, cartpole_QP_solver_bmy12, cartpole_QP_solver_yy12);
cartpole_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_5_5(cartpole_QP_solver_Ld12, cartpole_QP_solver_Ysd13, cartpole_QP_solver_Lsd13);
cartpole_QP_solver_LA_DENSE_MMTSUB_5_5(cartpole_QP_solver_Lsd13, cartpole_QP_solver_Yd13);
cartpole_QP_solver_LA_DENSE_CHOL_5(cartpole_QP_solver_Yd13, cartpole_QP_solver_Ld13);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd13, cartpole_QP_solver_yy12, cartpole_QP_solver_beta13, cartpole_QP_solver_bmy13);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld13, cartpole_QP_solver_bmy13, cartpole_QP_solver_yy13);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld13, cartpole_QP_solver_yy13, cartpole_QP_solver_dvaff13);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd13, cartpole_QP_solver_dvaff13, cartpole_QP_solver_yy12, cartpole_QP_solver_bmy12);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld12, cartpole_QP_solver_bmy12, cartpole_QP_solver_dvaff12);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd12, cartpole_QP_solver_dvaff12, cartpole_QP_solver_yy11, cartpole_QP_solver_bmy11);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld11, cartpole_QP_solver_bmy11, cartpole_QP_solver_dvaff11);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd11, cartpole_QP_solver_dvaff11, cartpole_QP_solver_yy10, cartpole_QP_solver_bmy10);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld10, cartpole_QP_solver_bmy10, cartpole_QP_solver_dvaff10);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd10, cartpole_QP_solver_dvaff10, cartpole_QP_solver_yy09, cartpole_QP_solver_bmy09);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld09, cartpole_QP_solver_bmy09, cartpole_QP_solver_dvaff09);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd09, cartpole_QP_solver_dvaff09, cartpole_QP_solver_yy08, cartpole_QP_solver_bmy08);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld08, cartpole_QP_solver_bmy08, cartpole_QP_solver_dvaff08);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd08, cartpole_QP_solver_dvaff08, cartpole_QP_solver_yy07, cartpole_QP_solver_bmy07);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld07, cartpole_QP_solver_bmy07, cartpole_QP_solver_dvaff07);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd07, cartpole_QP_solver_dvaff07, cartpole_QP_solver_yy06, cartpole_QP_solver_bmy06);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld06, cartpole_QP_solver_bmy06, cartpole_QP_solver_dvaff06);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd06, cartpole_QP_solver_dvaff06, cartpole_QP_solver_yy05, cartpole_QP_solver_bmy05);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld05, cartpole_QP_solver_bmy05, cartpole_QP_solver_dvaff05);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd05, cartpole_QP_solver_dvaff05, cartpole_QP_solver_yy04, cartpole_QP_solver_bmy04);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld04, cartpole_QP_solver_bmy04, cartpole_QP_solver_dvaff04);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd04, cartpole_QP_solver_dvaff04, cartpole_QP_solver_yy03, cartpole_QP_solver_bmy03);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld03, cartpole_QP_solver_bmy03, cartpole_QP_solver_dvaff03);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd03, cartpole_QP_solver_dvaff03, cartpole_QP_solver_yy02, cartpole_QP_solver_bmy02);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld02, cartpole_QP_solver_bmy02, cartpole_QP_solver_dvaff02);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd02, cartpole_QP_solver_dvaff02, cartpole_QP_solver_yy01, cartpole_QP_solver_bmy01);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld01, cartpole_QP_solver_bmy01, cartpole_QP_solver_dvaff01);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_9(cartpole_QP_solver_Lsd01, cartpole_QP_solver_dvaff01, cartpole_QP_solver_yy00, cartpole_QP_solver_bmy00);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_9(cartpole_QP_solver_Ld00, cartpole_QP_solver_bmy00, cartpole_QP_solver_dvaff00);
cartpole_QP_solver_LA_DENSE_MTVM_9_18(params->C1, cartpole_QP_solver_dvaff00, cartpole_QP_solver_grad_eq00);
cartpole_QP_solver_LA_DENSE_MTVM2_5_18_9(params->C2, cartpole_QP_solver_dvaff01, cartpole_QP_solver_D01, cartpole_QP_solver_dvaff00, cartpole_QP_solver_grad_eq01);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C3, cartpole_QP_solver_dvaff02, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff01, cartpole_QP_solver_grad_eq02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C4, cartpole_QP_solver_dvaff03, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff02, cartpole_QP_solver_grad_eq03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C5, cartpole_QP_solver_dvaff04, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff03, cartpole_QP_solver_grad_eq04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C6, cartpole_QP_solver_dvaff05, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff04, cartpole_QP_solver_grad_eq05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C7, cartpole_QP_solver_dvaff06, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff05, cartpole_QP_solver_grad_eq06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C8, cartpole_QP_solver_dvaff07, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff06, cartpole_QP_solver_grad_eq07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C9, cartpole_QP_solver_dvaff08, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff07, cartpole_QP_solver_grad_eq08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C10, cartpole_QP_solver_dvaff09, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff08, cartpole_QP_solver_grad_eq09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C11, cartpole_QP_solver_dvaff10, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff09, cartpole_QP_solver_grad_eq10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C12, cartpole_QP_solver_dvaff11, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff10, cartpole_QP_solver_grad_eq11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C13, cartpole_QP_solver_dvaff12, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff11, cartpole_QP_solver_grad_eq12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C14, cartpole_QP_solver_dvaff13, cartpole_QP_solver_D02, cartpole_QP_solver_dvaff12, cartpole_QP_solver_grad_eq13);
cartpole_QP_solver_LA_DIAGZERO_MTVM_5_5(cartpole_QP_solver_D14, cartpole_QP_solver_dvaff13, cartpole_QP_solver_grad_eq14);
cartpole_QP_solver_LA_VSUB2_257(cartpole_QP_solver_rd, cartpole_QP_solver_grad_eq, cartpole_QP_solver_rd);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi00, cartpole_QP_solver_rd00, cartpole_QP_solver_dzaff00);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi01, cartpole_QP_solver_rd01, cartpole_QP_solver_dzaff01);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi02, cartpole_QP_solver_rd02, cartpole_QP_solver_dzaff02);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi03, cartpole_QP_solver_rd03, cartpole_QP_solver_dzaff03);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi04, cartpole_QP_solver_rd04, cartpole_QP_solver_dzaff04);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi05, cartpole_QP_solver_rd05, cartpole_QP_solver_dzaff05);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi06, cartpole_QP_solver_rd06, cartpole_QP_solver_dzaff06);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi07, cartpole_QP_solver_rd07, cartpole_QP_solver_dzaff07);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi08, cartpole_QP_solver_rd08, cartpole_QP_solver_dzaff08);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi09, cartpole_QP_solver_rd09, cartpole_QP_solver_dzaff09);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi10, cartpole_QP_solver_rd10, cartpole_QP_solver_dzaff10);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi11, cartpole_QP_solver_rd11, cartpole_QP_solver_dzaff11);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi12, cartpole_QP_solver_rd12, cartpole_QP_solver_dzaff12);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi13, cartpole_QP_solver_rd13, cartpole_QP_solver_dzaff13);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(cartpole_QP_solver_Phi14, cartpole_QP_solver_rd14, cartpole_QP_solver_dzaff14);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_rilb00, cartpole_QP_solver_dslbaff00);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb00, cartpole_QP_solver_dslbaff00, cartpole_QP_solver_llb00, cartpole_QP_solver_dllbaff00);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub00, cartpole_QP_solver_dzaff00, cartpole_QP_solver_ubIdx00, cartpole_QP_solver_dsubaff00);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub00, cartpole_QP_solver_dsubaff00, cartpole_QP_solver_lub00, cartpole_QP_solver_dlubaff00);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_rilb01, cartpole_QP_solver_dslbaff01);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb01, cartpole_QP_solver_dslbaff01, cartpole_QP_solver_llb01, cartpole_QP_solver_dllbaff01);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub01, cartpole_QP_solver_dzaff01, cartpole_QP_solver_ubIdx01, cartpole_QP_solver_dsubaff01);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub01, cartpole_QP_solver_dsubaff01, cartpole_QP_solver_lub01, cartpole_QP_solver_dlubaff01);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_rilb02, cartpole_QP_solver_dslbaff02);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb02, cartpole_QP_solver_dslbaff02, cartpole_QP_solver_llb02, cartpole_QP_solver_dllbaff02);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub02, cartpole_QP_solver_dzaff02, cartpole_QP_solver_ubIdx02, cartpole_QP_solver_dsubaff02);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub02, cartpole_QP_solver_dsubaff02, cartpole_QP_solver_lub02, cartpole_QP_solver_dlubaff02);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_rilb03, cartpole_QP_solver_dslbaff03);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb03, cartpole_QP_solver_dslbaff03, cartpole_QP_solver_llb03, cartpole_QP_solver_dllbaff03);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub03, cartpole_QP_solver_dzaff03, cartpole_QP_solver_ubIdx03, cartpole_QP_solver_dsubaff03);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub03, cartpole_QP_solver_dsubaff03, cartpole_QP_solver_lub03, cartpole_QP_solver_dlubaff03);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_rilb04, cartpole_QP_solver_dslbaff04);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb04, cartpole_QP_solver_dslbaff04, cartpole_QP_solver_llb04, cartpole_QP_solver_dllbaff04);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub04, cartpole_QP_solver_dzaff04, cartpole_QP_solver_ubIdx04, cartpole_QP_solver_dsubaff04);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub04, cartpole_QP_solver_dsubaff04, cartpole_QP_solver_lub04, cartpole_QP_solver_dlubaff04);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_rilb05, cartpole_QP_solver_dslbaff05);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb05, cartpole_QP_solver_dslbaff05, cartpole_QP_solver_llb05, cartpole_QP_solver_dllbaff05);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub05, cartpole_QP_solver_dzaff05, cartpole_QP_solver_ubIdx05, cartpole_QP_solver_dsubaff05);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub05, cartpole_QP_solver_dsubaff05, cartpole_QP_solver_lub05, cartpole_QP_solver_dlubaff05);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_rilb06, cartpole_QP_solver_dslbaff06);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb06, cartpole_QP_solver_dslbaff06, cartpole_QP_solver_llb06, cartpole_QP_solver_dllbaff06);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub06, cartpole_QP_solver_dzaff06, cartpole_QP_solver_ubIdx06, cartpole_QP_solver_dsubaff06);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub06, cartpole_QP_solver_dsubaff06, cartpole_QP_solver_lub06, cartpole_QP_solver_dlubaff06);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_rilb07, cartpole_QP_solver_dslbaff07);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb07, cartpole_QP_solver_dslbaff07, cartpole_QP_solver_llb07, cartpole_QP_solver_dllbaff07);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub07, cartpole_QP_solver_dzaff07, cartpole_QP_solver_ubIdx07, cartpole_QP_solver_dsubaff07);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub07, cartpole_QP_solver_dsubaff07, cartpole_QP_solver_lub07, cartpole_QP_solver_dlubaff07);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_rilb08, cartpole_QP_solver_dslbaff08);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb08, cartpole_QP_solver_dslbaff08, cartpole_QP_solver_llb08, cartpole_QP_solver_dllbaff08);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub08, cartpole_QP_solver_dzaff08, cartpole_QP_solver_ubIdx08, cartpole_QP_solver_dsubaff08);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub08, cartpole_QP_solver_dsubaff08, cartpole_QP_solver_lub08, cartpole_QP_solver_dlubaff08);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_rilb09, cartpole_QP_solver_dslbaff09);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb09, cartpole_QP_solver_dslbaff09, cartpole_QP_solver_llb09, cartpole_QP_solver_dllbaff09);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub09, cartpole_QP_solver_dzaff09, cartpole_QP_solver_ubIdx09, cartpole_QP_solver_dsubaff09);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub09, cartpole_QP_solver_dsubaff09, cartpole_QP_solver_lub09, cartpole_QP_solver_dlubaff09);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_rilb10, cartpole_QP_solver_dslbaff10);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb10, cartpole_QP_solver_dslbaff10, cartpole_QP_solver_llb10, cartpole_QP_solver_dllbaff10);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub10, cartpole_QP_solver_dzaff10, cartpole_QP_solver_ubIdx10, cartpole_QP_solver_dsubaff10);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub10, cartpole_QP_solver_dsubaff10, cartpole_QP_solver_lub10, cartpole_QP_solver_dlubaff10);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_rilb11, cartpole_QP_solver_dslbaff11);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb11, cartpole_QP_solver_dslbaff11, cartpole_QP_solver_llb11, cartpole_QP_solver_dllbaff11);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub11, cartpole_QP_solver_dzaff11, cartpole_QP_solver_ubIdx11, cartpole_QP_solver_dsubaff11);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub11, cartpole_QP_solver_dsubaff11, cartpole_QP_solver_lub11, cartpole_QP_solver_dlubaff11);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_rilb12, cartpole_QP_solver_dslbaff12);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb12, cartpole_QP_solver_dslbaff12, cartpole_QP_solver_llb12, cartpole_QP_solver_dllbaff12);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub12, cartpole_QP_solver_dzaff12, cartpole_QP_solver_ubIdx12, cartpole_QP_solver_dsubaff12);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub12, cartpole_QP_solver_dsubaff12, cartpole_QP_solver_lub12, cartpole_QP_solver_dlubaff12);
cartpole_QP_solver_LA_VSUB_INDEXED_18(cartpole_QP_solver_dzaff13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_rilb13, cartpole_QP_solver_dslbaff13);
cartpole_QP_solver_LA_VSUB3_18(cartpole_QP_solver_llbbyslb13, cartpole_QP_solver_dslbaff13, cartpole_QP_solver_llb13, cartpole_QP_solver_dllbaff13);
cartpole_QP_solver_LA_VSUB2_INDEXED_10(cartpole_QP_solver_riub13, cartpole_QP_solver_dzaff13, cartpole_QP_solver_ubIdx13, cartpole_QP_solver_dsubaff13);
cartpole_QP_solver_LA_VSUB3_10(cartpole_QP_solver_lubbysub13, cartpole_QP_solver_dsubaff13, cartpole_QP_solver_lub13, cartpole_QP_solver_dlubaff13);
cartpole_QP_solver_LA_VSUB_INDEXED_5(cartpole_QP_solver_dzaff14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_rilb14, cartpole_QP_solver_dslbaff14);
cartpole_QP_solver_LA_VSUB3_5(cartpole_QP_solver_llbbyslb14, cartpole_QP_solver_dslbaff14, cartpole_QP_solver_llb14, cartpole_QP_solver_dllbaff14);
cartpole_QP_solver_LA_VSUB2_INDEXED_5(cartpole_QP_solver_riub14, cartpole_QP_solver_dzaff14, cartpole_QP_solver_ubIdx14, cartpole_QP_solver_dsubaff14);
cartpole_QP_solver_LA_VSUB3_5(cartpole_QP_solver_lubbysub14, cartpole_QP_solver_dsubaff14, cartpole_QP_solver_lub14, cartpole_QP_solver_dlubaff14);
info->lsit_aff = cartpole_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(cartpole_QP_solver_l, cartpole_QP_solver_s, cartpole_QP_solver_dl_aff, cartpole_QP_solver_ds_aff, &info->step_aff, &info->mu_aff);
if( info->lsit_aff == cartpole_QP_solver_NOPROGRESS ){
exitcode = cartpole_QP_solver_NOPROGRESS; break;
}
sigma_3rdroot = info->mu_aff / info->mu;
info->sigma = sigma_3rdroot*sigma_3rdroot*sigma_3rdroot;
musigma = info->mu * info->sigma;
cartpole_QP_solver_LA_VSUB5_402(cartpole_QP_solver_ds_aff, cartpole_QP_solver_dl_aff, info->mu, info->sigma, cartpole_QP_solver_ccrhs);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub00, cartpole_QP_solver_sub00, cartpole_QP_solver_ubIdx00, cartpole_QP_solver_ccrhsl00, cartpole_QP_solver_slb00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_rd00);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub01, cartpole_QP_solver_sub01, cartpole_QP_solver_ubIdx01, cartpole_QP_solver_ccrhsl01, cartpole_QP_solver_slb01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_rd01);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi00, cartpole_QP_solver_rd00, cartpole_QP_solver_Lbyrd00);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi01, cartpole_QP_solver_rd01, cartpole_QP_solver_Lbyrd01);
cartpole_QP_solver_LA_DENSE_2MVMADD_9_18_18(cartpole_QP_solver_V00, cartpole_QP_solver_Lbyrd00, cartpole_QP_solver_W01, cartpole_QP_solver_Lbyrd01, cartpole_QP_solver_beta00);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_9(cartpole_QP_solver_Ld00, cartpole_QP_solver_beta00, cartpole_QP_solver_yy00);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub02, cartpole_QP_solver_sub02, cartpole_QP_solver_ubIdx02, cartpole_QP_solver_ccrhsl02, cartpole_QP_solver_slb02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_rd02);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi02, cartpole_QP_solver_rd02, cartpole_QP_solver_Lbyrd02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V01, cartpole_QP_solver_Lbyrd01, cartpole_QP_solver_W02, cartpole_QP_solver_Lbyrd02, cartpole_QP_solver_beta01);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_9(cartpole_QP_solver_Lsd01, cartpole_QP_solver_yy00, cartpole_QP_solver_beta01, cartpole_QP_solver_bmy01);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld01, cartpole_QP_solver_bmy01, cartpole_QP_solver_yy01);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub03, cartpole_QP_solver_sub03, cartpole_QP_solver_ubIdx03, cartpole_QP_solver_ccrhsl03, cartpole_QP_solver_slb03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_rd03);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi03, cartpole_QP_solver_rd03, cartpole_QP_solver_Lbyrd03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V02, cartpole_QP_solver_Lbyrd02, cartpole_QP_solver_W03, cartpole_QP_solver_Lbyrd03, cartpole_QP_solver_beta02);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd02, cartpole_QP_solver_yy01, cartpole_QP_solver_beta02, cartpole_QP_solver_bmy02);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld02, cartpole_QP_solver_bmy02, cartpole_QP_solver_yy02);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub04, cartpole_QP_solver_sub04, cartpole_QP_solver_ubIdx04, cartpole_QP_solver_ccrhsl04, cartpole_QP_solver_slb04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_rd04);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi04, cartpole_QP_solver_rd04, cartpole_QP_solver_Lbyrd04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V03, cartpole_QP_solver_Lbyrd03, cartpole_QP_solver_W04, cartpole_QP_solver_Lbyrd04, cartpole_QP_solver_beta03);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd03, cartpole_QP_solver_yy02, cartpole_QP_solver_beta03, cartpole_QP_solver_bmy03);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld03, cartpole_QP_solver_bmy03, cartpole_QP_solver_yy03);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub05, cartpole_QP_solver_sub05, cartpole_QP_solver_ubIdx05, cartpole_QP_solver_ccrhsl05, cartpole_QP_solver_slb05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_rd05);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi05, cartpole_QP_solver_rd05, cartpole_QP_solver_Lbyrd05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V04, cartpole_QP_solver_Lbyrd04, cartpole_QP_solver_W05, cartpole_QP_solver_Lbyrd05, cartpole_QP_solver_beta04);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd04, cartpole_QP_solver_yy03, cartpole_QP_solver_beta04, cartpole_QP_solver_bmy04);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld04, cartpole_QP_solver_bmy04, cartpole_QP_solver_yy04);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub06, cartpole_QP_solver_sub06, cartpole_QP_solver_ubIdx06, cartpole_QP_solver_ccrhsl06, cartpole_QP_solver_slb06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_rd06);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi06, cartpole_QP_solver_rd06, cartpole_QP_solver_Lbyrd06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V05, cartpole_QP_solver_Lbyrd05, cartpole_QP_solver_W06, cartpole_QP_solver_Lbyrd06, cartpole_QP_solver_beta05);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd05, cartpole_QP_solver_yy04, cartpole_QP_solver_beta05, cartpole_QP_solver_bmy05);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld05, cartpole_QP_solver_bmy05, cartpole_QP_solver_yy05);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub07, cartpole_QP_solver_sub07, cartpole_QP_solver_ubIdx07, cartpole_QP_solver_ccrhsl07, cartpole_QP_solver_slb07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_rd07);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi07, cartpole_QP_solver_rd07, cartpole_QP_solver_Lbyrd07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V06, cartpole_QP_solver_Lbyrd06, cartpole_QP_solver_W07, cartpole_QP_solver_Lbyrd07, cartpole_QP_solver_beta06);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd06, cartpole_QP_solver_yy05, cartpole_QP_solver_beta06, cartpole_QP_solver_bmy06);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld06, cartpole_QP_solver_bmy06, cartpole_QP_solver_yy06);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub08, cartpole_QP_solver_sub08, cartpole_QP_solver_ubIdx08, cartpole_QP_solver_ccrhsl08, cartpole_QP_solver_slb08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_rd08);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi08, cartpole_QP_solver_rd08, cartpole_QP_solver_Lbyrd08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V07, cartpole_QP_solver_Lbyrd07, cartpole_QP_solver_W08, cartpole_QP_solver_Lbyrd08, cartpole_QP_solver_beta07);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd07, cartpole_QP_solver_yy06, cartpole_QP_solver_beta07, cartpole_QP_solver_bmy07);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld07, cartpole_QP_solver_bmy07, cartpole_QP_solver_yy07);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub09, cartpole_QP_solver_sub09, cartpole_QP_solver_ubIdx09, cartpole_QP_solver_ccrhsl09, cartpole_QP_solver_slb09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_rd09);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi09, cartpole_QP_solver_rd09, cartpole_QP_solver_Lbyrd09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V08, cartpole_QP_solver_Lbyrd08, cartpole_QP_solver_W09, cartpole_QP_solver_Lbyrd09, cartpole_QP_solver_beta08);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd08, cartpole_QP_solver_yy07, cartpole_QP_solver_beta08, cartpole_QP_solver_bmy08);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld08, cartpole_QP_solver_bmy08, cartpole_QP_solver_yy08);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub10, cartpole_QP_solver_sub10, cartpole_QP_solver_ubIdx10, cartpole_QP_solver_ccrhsl10, cartpole_QP_solver_slb10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_rd10);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi10, cartpole_QP_solver_rd10, cartpole_QP_solver_Lbyrd10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V09, cartpole_QP_solver_Lbyrd09, cartpole_QP_solver_W10, cartpole_QP_solver_Lbyrd10, cartpole_QP_solver_beta09);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd09, cartpole_QP_solver_yy08, cartpole_QP_solver_beta09, cartpole_QP_solver_bmy09);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld09, cartpole_QP_solver_bmy09, cartpole_QP_solver_yy09);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub11, cartpole_QP_solver_sub11, cartpole_QP_solver_ubIdx11, cartpole_QP_solver_ccrhsl11, cartpole_QP_solver_slb11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_rd11);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi11, cartpole_QP_solver_rd11, cartpole_QP_solver_Lbyrd11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V10, cartpole_QP_solver_Lbyrd10, cartpole_QP_solver_W11, cartpole_QP_solver_Lbyrd11, cartpole_QP_solver_beta10);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd10, cartpole_QP_solver_yy09, cartpole_QP_solver_beta10, cartpole_QP_solver_bmy10);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld10, cartpole_QP_solver_bmy10, cartpole_QP_solver_yy10);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub12, cartpole_QP_solver_sub12, cartpole_QP_solver_ubIdx12, cartpole_QP_solver_ccrhsl12, cartpole_QP_solver_slb12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_rd12);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi12, cartpole_QP_solver_rd12, cartpole_QP_solver_Lbyrd12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V11, cartpole_QP_solver_Lbyrd11, cartpole_QP_solver_W12, cartpole_QP_solver_Lbyrd12, cartpole_QP_solver_beta11);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd11, cartpole_QP_solver_yy10, cartpole_QP_solver_beta11, cartpole_QP_solver_bmy11);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld11, cartpole_QP_solver_bmy11, cartpole_QP_solver_yy11);
cartpole_QP_solver_LA_VSUB6_INDEXED_18_10_18(cartpole_QP_solver_ccrhsub13, cartpole_QP_solver_sub13, cartpole_QP_solver_ubIdx13, cartpole_QP_solver_ccrhsl13, cartpole_QP_solver_slb13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_rd13);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_18(cartpole_QP_solver_Phi13, cartpole_QP_solver_rd13, cartpole_QP_solver_Lbyrd13);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_18(cartpole_QP_solver_V12, cartpole_QP_solver_Lbyrd12, cartpole_QP_solver_W13, cartpole_QP_solver_Lbyrd13, cartpole_QP_solver_beta12);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd12, cartpole_QP_solver_yy11, cartpole_QP_solver_beta12, cartpole_QP_solver_bmy12);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld12, cartpole_QP_solver_bmy12, cartpole_QP_solver_yy12);
cartpole_QP_solver_LA_VSUB6_INDEXED_5_5_5(cartpole_QP_solver_ccrhsub14, cartpole_QP_solver_sub14, cartpole_QP_solver_ubIdx14, cartpole_QP_solver_ccrhsl14, cartpole_QP_solver_slb14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_rd14);
cartpole_QP_solver_LA_DIAG_FORWARDSUB_5(cartpole_QP_solver_Phi14, cartpole_QP_solver_rd14, cartpole_QP_solver_Lbyrd14);
cartpole_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_5_18_5(cartpole_QP_solver_V13, cartpole_QP_solver_Lbyrd13, cartpole_QP_solver_W14, cartpole_QP_solver_Lbyrd14, cartpole_QP_solver_beta13);
cartpole_QP_solver_LA_DENSE_MVMSUB1_5_5(cartpole_QP_solver_Lsd13, cartpole_QP_solver_yy12, cartpole_QP_solver_beta13, cartpole_QP_solver_bmy13);
cartpole_QP_solver_LA_DENSE_FORWARDSUB_5(cartpole_QP_solver_Ld13, cartpole_QP_solver_bmy13, cartpole_QP_solver_yy13);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld13, cartpole_QP_solver_yy13, cartpole_QP_solver_dvcc13);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd13, cartpole_QP_solver_dvcc13, cartpole_QP_solver_yy12, cartpole_QP_solver_bmy12);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld12, cartpole_QP_solver_bmy12, cartpole_QP_solver_dvcc12);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd12, cartpole_QP_solver_dvcc12, cartpole_QP_solver_yy11, cartpole_QP_solver_bmy11);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld11, cartpole_QP_solver_bmy11, cartpole_QP_solver_dvcc11);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd11, cartpole_QP_solver_dvcc11, cartpole_QP_solver_yy10, cartpole_QP_solver_bmy10);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld10, cartpole_QP_solver_bmy10, cartpole_QP_solver_dvcc10);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd10, cartpole_QP_solver_dvcc10, cartpole_QP_solver_yy09, cartpole_QP_solver_bmy09);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld09, cartpole_QP_solver_bmy09, cartpole_QP_solver_dvcc09);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd09, cartpole_QP_solver_dvcc09, cartpole_QP_solver_yy08, cartpole_QP_solver_bmy08);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld08, cartpole_QP_solver_bmy08, cartpole_QP_solver_dvcc08);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd08, cartpole_QP_solver_dvcc08, cartpole_QP_solver_yy07, cartpole_QP_solver_bmy07);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld07, cartpole_QP_solver_bmy07, cartpole_QP_solver_dvcc07);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd07, cartpole_QP_solver_dvcc07, cartpole_QP_solver_yy06, cartpole_QP_solver_bmy06);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld06, cartpole_QP_solver_bmy06, cartpole_QP_solver_dvcc06);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd06, cartpole_QP_solver_dvcc06, cartpole_QP_solver_yy05, cartpole_QP_solver_bmy05);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld05, cartpole_QP_solver_bmy05, cartpole_QP_solver_dvcc05);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd05, cartpole_QP_solver_dvcc05, cartpole_QP_solver_yy04, cartpole_QP_solver_bmy04);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld04, cartpole_QP_solver_bmy04, cartpole_QP_solver_dvcc04);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd04, cartpole_QP_solver_dvcc04, cartpole_QP_solver_yy03, cartpole_QP_solver_bmy03);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld03, cartpole_QP_solver_bmy03, cartpole_QP_solver_dvcc03);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd03, cartpole_QP_solver_dvcc03, cartpole_QP_solver_yy02, cartpole_QP_solver_bmy02);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld02, cartpole_QP_solver_bmy02, cartpole_QP_solver_dvcc02);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_5(cartpole_QP_solver_Lsd02, cartpole_QP_solver_dvcc02, cartpole_QP_solver_yy01, cartpole_QP_solver_bmy01);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_5(cartpole_QP_solver_Ld01, cartpole_QP_solver_bmy01, cartpole_QP_solver_dvcc01);
cartpole_QP_solver_LA_DENSE_MTVMSUB_5_9(cartpole_QP_solver_Lsd01, cartpole_QP_solver_dvcc01, cartpole_QP_solver_yy00, cartpole_QP_solver_bmy00);
cartpole_QP_solver_LA_DENSE_BACKWARDSUB_9(cartpole_QP_solver_Ld00, cartpole_QP_solver_bmy00, cartpole_QP_solver_dvcc00);
cartpole_QP_solver_LA_DENSE_MTVM_9_18(params->C1, cartpole_QP_solver_dvcc00, cartpole_QP_solver_grad_eq00);
cartpole_QP_solver_LA_DENSE_MTVM2_5_18_9(params->C2, cartpole_QP_solver_dvcc01, cartpole_QP_solver_D01, cartpole_QP_solver_dvcc00, cartpole_QP_solver_grad_eq01);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C3, cartpole_QP_solver_dvcc02, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc01, cartpole_QP_solver_grad_eq02);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C4, cartpole_QP_solver_dvcc03, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc02, cartpole_QP_solver_grad_eq03);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C5, cartpole_QP_solver_dvcc04, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc03, cartpole_QP_solver_grad_eq04);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C6, cartpole_QP_solver_dvcc05, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc04, cartpole_QP_solver_grad_eq05);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C7, cartpole_QP_solver_dvcc06, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc05, cartpole_QP_solver_grad_eq06);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C8, cartpole_QP_solver_dvcc07, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc06, cartpole_QP_solver_grad_eq07);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C9, cartpole_QP_solver_dvcc08, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc07, cartpole_QP_solver_grad_eq08);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C10, cartpole_QP_solver_dvcc09, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc08, cartpole_QP_solver_grad_eq09);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C11, cartpole_QP_solver_dvcc10, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc09, cartpole_QP_solver_grad_eq10);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C12, cartpole_QP_solver_dvcc11, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc10, cartpole_QP_solver_grad_eq11);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C13, cartpole_QP_solver_dvcc12, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc11, cartpole_QP_solver_grad_eq12);
cartpole_QP_solver_LA_DENSE_DIAGZERO_MTVM2_5_18_5(params->C14, cartpole_QP_solver_dvcc13, cartpole_QP_solver_D02, cartpole_QP_solver_dvcc12, cartpole_QP_solver_grad_eq13);
cartpole_QP_solver_LA_DIAGZERO_MTVM_5_5(cartpole_QP_solver_D14, cartpole_QP_solver_dvcc13, cartpole_QP_solver_grad_eq14);
cartpole_QP_solver_LA_VSUB_257(cartpole_QP_solver_rd, cartpole_QP_solver_grad_eq, cartpole_QP_solver_rd);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi00, cartpole_QP_solver_rd00, cartpole_QP_solver_dzcc00);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi01, cartpole_QP_solver_rd01, cartpole_QP_solver_dzcc01);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi02, cartpole_QP_solver_rd02, cartpole_QP_solver_dzcc02);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi03, cartpole_QP_solver_rd03, cartpole_QP_solver_dzcc03);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi04, cartpole_QP_solver_rd04, cartpole_QP_solver_dzcc04);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi05, cartpole_QP_solver_rd05, cartpole_QP_solver_dzcc05);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi06, cartpole_QP_solver_rd06, cartpole_QP_solver_dzcc06);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi07, cartpole_QP_solver_rd07, cartpole_QP_solver_dzcc07);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi08, cartpole_QP_solver_rd08, cartpole_QP_solver_dzcc08);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi09, cartpole_QP_solver_rd09, cartpole_QP_solver_dzcc09);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi10, cartpole_QP_solver_rd10, cartpole_QP_solver_dzcc10);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi11, cartpole_QP_solver_rd11, cartpole_QP_solver_dzcc11);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi12, cartpole_QP_solver_rd12, cartpole_QP_solver_dzcc12);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_18(cartpole_QP_solver_Phi13, cartpole_QP_solver_rd13, cartpole_QP_solver_dzcc13);
cartpole_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_5(cartpole_QP_solver_Phi14, cartpole_QP_solver_rd14, cartpole_QP_solver_dzcc14);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl00, cartpole_QP_solver_slb00, cartpole_QP_solver_llbbyslb00, cartpole_QP_solver_dzcc00, cartpole_QP_solver_lbIdx00, cartpole_QP_solver_dllbcc00);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub00, cartpole_QP_solver_sub00, cartpole_QP_solver_lubbysub00, cartpole_QP_solver_dzcc00, cartpole_QP_solver_ubIdx00, cartpole_QP_solver_dlubcc00);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl01, cartpole_QP_solver_slb01, cartpole_QP_solver_llbbyslb01, cartpole_QP_solver_dzcc01, cartpole_QP_solver_lbIdx01, cartpole_QP_solver_dllbcc01);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub01, cartpole_QP_solver_sub01, cartpole_QP_solver_lubbysub01, cartpole_QP_solver_dzcc01, cartpole_QP_solver_ubIdx01, cartpole_QP_solver_dlubcc01);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl02, cartpole_QP_solver_slb02, cartpole_QP_solver_llbbyslb02, cartpole_QP_solver_dzcc02, cartpole_QP_solver_lbIdx02, cartpole_QP_solver_dllbcc02);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub02, cartpole_QP_solver_sub02, cartpole_QP_solver_lubbysub02, cartpole_QP_solver_dzcc02, cartpole_QP_solver_ubIdx02, cartpole_QP_solver_dlubcc02);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl03, cartpole_QP_solver_slb03, cartpole_QP_solver_llbbyslb03, cartpole_QP_solver_dzcc03, cartpole_QP_solver_lbIdx03, cartpole_QP_solver_dllbcc03);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub03, cartpole_QP_solver_sub03, cartpole_QP_solver_lubbysub03, cartpole_QP_solver_dzcc03, cartpole_QP_solver_ubIdx03, cartpole_QP_solver_dlubcc03);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl04, cartpole_QP_solver_slb04, cartpole_QP_solver_llbbyslb04, cartpole_QP_solver_dzcc04, cartpole_QP_solver_lbIdx04, cartpole_QP_solver_dllbcc04);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub04, cartpole_QP_solver_sub04, cartpole_QP_solver_lubbysub04, cartpole_QP_solver_dzcc04, cartpole_QP_solver_ubIdx04, cartpole_QP_solver_dlubcc04);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl05, cartpole_QP_solver_slb05, cartpole_QP_solver_llbbyslb05, cartpole_QP_solver_dzcc05, cartpole_QP_solver_lbIdx05, cartpole_QP_solver_dllbcc05);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub05, cartpole_QP_solver_sub05, cartpole_QP_solver_lubbysub05, cartpole_QP_solver_dzcc05, cartpole_QP_solver_ubIdx05, cartpole_QP_solver_dlubcc05);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl06, cartpole_QP_solver_slb06, cartpole_QP_solver_llbbyslb06, cartpole_QP_solver_dzcc06, cartpole_QP_solver_lbIdx06, cartpole_QP_solver_dllbcc06);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub06, cartpole_QP_solver_sub06, cartpole_QP_solver_lubbysub06, cartpole_QP_solver_dzcc06, cartpole_QP_solver_ubIdx06, cartpole_QP_solver_dlubcc06);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl07, cartpole_QP_solver_slb07, cartpole_QP_solver_llbbyslb07, cartpole_QP_solver_dzcc07, cartpole_QP_solver_lbIdx07, cartpole_QP_solver_dllbcc07);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub07, cartpole_QP_solver_sub07, cartpole_QP_solver_lubbysub07, cartpole_QP_solver_dzcc07, cartpole_QP_solver_ubIdx07, cartpole_QP_solver_dlubcc07);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl08, cartpole_QP_solver_slb08, cartpole_QP_solver_llbbyslb08, cartpole_QP_solver_dzcc08, cartpole_QP_solver_lbIdx08, cartpole_QP_solver_dllbcc08);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub08, cartpole_QP_solver_sub08, cartpole_QP_solver_lubbysub08, cartpole_QP_solver_dzcc08, cartpole_QP_solver_ubIdx08, cartpole_QP_solver_dlubcc08);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl09, cartpole_QP_solver_slb09, cartpole_QP_solver_llbbyslb09, cartpole_QP_solver_dzcc09, cartpole_QP_solver_lbIdx09, cartpole_QP_solver_dllbcc09);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub09, cartpole_QP_solver_sub09, cartpole_QP_solver_lubbysub09, cartpole_QP_solver_dzcc09, cartpole_QP_solver_ubIdx09, cartpole_QP_solver_dlubcc09);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl10, cartpole_QP_solver_slb10, cartpole_QP_solver_llbbyslb10, cartpole_QP_solver_dzcc10, cartpole_QP_solver_lbIdx10, cartpole_QP_solver_dllbcc10);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub10, cartpole_QP_solver_sub10, cartpole_QP_solver_lubbysub10, cartpole_QP_solver_dzcc10, cartpole_QP_solver_ubIdx10, cartpole_QP_solver_dlubcc10);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl11, cartpole_QP_solver_slb11, cartpole_QP_solver_llbbyslb11, cartpole_QP_solver_dzcc11, cartpole_QP_solver_lbIdx11, cartpole_QP_solver_dllbcc11);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub11, cartpole_QP_solver_sub11, cartpole_QP_solver_lubbysub11, cartpole_QP_solver_dzcc11, cartpole_QP_solver_ubIdx11, cartpole_QP_solver_dlubcc11);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl12, cartpole_QP_solver_slb12, cartpole_QP_solver_llbbyslb12, cartpole_QP_solver_dzcc12, cartpole_QP_solver_lbIdx12, cartpole_QP_solver_dllbcc12);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub12, cartpole_QP_solver_sub12, cartpole_QP_solver_lubbysub12, cartpole_QP_solver_dzcc12, cartpole_QP_solver_ubIdx12, cartpole_QP_solver_dlubcc12);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_18(cartpole_QP_solver_ccrhsl13, cartpole_QP_solver_slb13, cartpole_QP_solver_llbbyslb13, cartpole_QP_solver_dzcc13, cartpole_QP_solver_lbIdx13, cartpole_QP_solver_dllbcc13);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_10(cartpole_QP_solver_ccrhsub13, cartpole_QP_solver_sub13, cartpole_QP_solver_lubbysub13, cartpole_QP_solver_dzcc13, cartpole_QP_solver_ubIdx13, cartpole_QP_solver_dlubcc13);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_5(cartpole_QP_solver_ccrhsl14, cartpole_QP_solver_slb14, cartpole_QP_solver_llbbyslb14, cartpole_QP_solver_dzcc14, cartpole_QP_solver_lbIdx14, cartpole_QP_solver_dllbcc14);
cartpole_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_5(cartpole_QP_solver_ccrhsub14, cartpole_QP_solver_sub14, cartpole_QP_solver_lubbysub14, cartpole_QP_solver_dzcc14, cartpole_QP_solver_ubIdx14, cartpole_QP_solver_dlubcc14);
cartpole_QP_solver_LA_VSUB7_402(cartpole_QP_solver_l, cartpole_QP_solver_ccrhs, cartpole_QP_solver_s, cartpole_QP_solver_dl_cc, cartpole_QP_solver_ds_cc);
cartpole_QP_solver_LA_VADD_257(cartpole_QP_solver_dz_cc, cartpole_QP_solver_dz_aff);
cartpole_QP_solver_LA_VADD_74(cartpole_QP_solver_dv_cc, cartpole_QP_solver_dv_aff);
cartpole_QP_solver_LA_VADD_402(cartpole_QP_solver_dl_cc, cartpole_QP_solver_dl_aff);
cartpole_QP_solver_LA_VADD_402(cartpole_QP_solver_ds_cc, cartpole_QP_solver_ds_aff);
info->lsit_cc = cartpole_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(cartpole_QP_solver_z, cartpole_QP_solver_v, cartpole_QP_solver_l, cartpole_QP_solver_s, cartpole_QP_solver_dz_cc, cartpole_QP_solver_dv_cc, cartpole_QP_solver_dl_cc, cartpole_QP_solver_ds_cc, &info->step_cc, &info->mu);
if( info->lsit_cc == cartpole_QP_solver_NOPROGRESS ){
exitcode = cartpole_QP_solver_NOPROGRESS; break;
}
info->it++;
}
output->z1[0] = cartpole_QP_solver_z00[0];
output->z1[1] = cartpole_QP_solver_z00[1];
output->z1[2] = cartpole_QP_solver_z00[2];
output->z1[3] = cartpole_QP_solver_z00[3];
output->z1[4] = cartpole_QP_solver_z00[4];
output->z1[5] = cartpole_QP_solver_z00[5];
output->z1[6] = cartpole_QP_solver_z00[6];
output->z1[7] = cartpole_QP_solver_z00[7];
output->z1[8] = cartpole_QP_solver_z00[8];
output->z1[9] = cartpole_QP_solver_z00[9];
output->z2[0] = cartpole_QP_solver_z01[0];
output->z2[1] = cartpole_QP_solver_z01[1];
output->z2[2] = cartpole_QP_solver_z01[2];
output->z2[3] = cartpole_QP_solver_z01[3];
output->z2[4] = cartpole_QP_solver_z01[4];
output->z2[5] = cartpole_QP_solver_z01[5];
output->z2[6] = cartpole_QP_solver_z01[6];
output->z2[7] = cartpole_QP_solver_z01[7];
output->z2[8] = cartpole_QP_solver_z01[8];
output->z2[9] = cartpole_QP_solver_z01[9];
output->z3[0] = cartpole_QP_solver_z02[0];
output->z3[1] = cartpole_QP_solver_z02[1];
output->z3[2] = cartpole_QP_solver_z02[2];
output->z3[3] = cartpole_QP_solver_z02[3];
output->z3[4] = cartpole_QP_solver_z02[4];
output->z3[5] = cartpole_QP_solver_z02[5];
output->z3[6] = cartpole_QP_solver_z02[6];
output->z3[7] = cartpole_QP_solver_z02[7];
output->z3[8] = cartpole_QP_solver_z02[8];
output->z3[9] = cartpole_QP_solver_z02[9];
output->z4[0] = cartpole_QP_solver_z03[0];
output->z4[1] = cartpole_QP_solver_z03[1];
output->z4[2] = cartpole_QP_solver_z03[2];
output->z4[3] = cartpole_QP_solver_z03[3];
output->z4[4] = cartpole_QP_solver_z03[4];
output->z4[5] = cartpole_QP_solver_z03[5];
output->z4[6] = cartpole_QP_solver_z03[6];
output->z4[7] = cartpole_QP_solver_z03[7];
output->z4[8] = cartpole_QP_solver_z03[8];
output->z4[9] = cartpole_QP_solver_z03[9];
output->z5[0] = cartpole_QP_solver_z04[0];
output->z5[1] = cartpole_QP_solver_z04[1];
output->z5[2] = cartpole_QP_solver_z04[2];
output->z5[3] = cartpole_QP_solver_z04[3];
output->z5[4] = cartpole_QP_solver_z04[4];
output->z5[5] = cartpole_QP_solver_z04[5];
output->z5[6] = cartpole_QP_solver_z04[6];
output->z5[7] = cartpole_QP_solver_z04[7];
output->z5[8] = cartpole_QP_solver_z04[8];
output->z5[9] = cartpole_QP_solver_z04[9];
output->z6[0] = cartpole_QP_solver_z05[0];
output->z6[1] = cartpole_QP_solver_z05[1];
output->z6[2] = cartpole_QP_solver_z05[2];
output->z6[3] = cartpole_QP_solver_z05[3];
output->z6[4] = cartpole_QP_solver_z05[4];
output->z6[5] = cartpole_QP_solver_z05[5];
output->z6[6] = cartpole_QP_solver_z05[6];
output->z6[7] = cartpole_QP_solver_z05[7];
output->z6[8] = cartpole_QP_solver_z05[8];
output->z6[9] = cartpole_QP_solver_z05[9];
output->z7[0] = cartpole_QP_solver_z06[0];
output->z7[1] = cartpole_QP_solver_z06[1];
output->z7[2] = cartpole_QP_solver_z06[2];
output->z7[3] = cartpole_QP_solver_z06[3];
output->z7[4] = cartpole_QP_solver_z06[4];
output->z7[5] = cartpole_QP_solver_z06[5];
output->z7[6] = cartpole_QP_solver_z06[6];
output->z7[7] = cartpole_QP_solver_z06[7];
output->z7[8] = cartpole_QP_solver_z06[8];
output->z7[9] = cartpole_QP_solver_z06[9];
output->z8[0] = cartpole_QP_solver_z07[0];
output->z8[1] = cartpole_QP_solver_z07[1];
output->z8[2] = cartpole_QP_solver_z07[2];
output->z8[3] = cartpole_QP_solver_z07[3];
output->z8[4] = cartpole_QP_solver_z07[4];
output->z8[5] = cartpole_QP_solver_z07[5];
output->z8[6] = cartpole_QP_solver_z07[6];
output->z8[7] = cartpole_QP_solver_z07[7];
output->z8[8] = cartpole_QP_solver_z07[8];
output->z8[9] = cartpole_QP_solver_z07[9];
output->z9[0] = cartpole_QP_solver_z08[0];
output->z9[1] = cartpole_QP_solver_z08[1];
output->z9[2] = cartpole_QP_solver_z08[2];
output->z9[3] = cartpole_QP_solver_z08[3];
output->z9[4] = cartpole_QP_solver_z08[4];
output->z9[5] = cartpole_QP_solver_z08[5];
output->z9[6] = cartpole_QP_solver_z08[6];
output->z9[7] = cartpole_QP_solver_z08[7];
output->z9[8] = cartpole_QP_solver_z08[8];
output->z9[9] = cartpole_QP_solver_z08[9];
output->z10[0] = cartpole_QP_solver_z09[0];
output->z10[1] = cartpole_QP_solver_z09[1];
output->z10[2] = cartpole_QP_solver_z09[2];
output->z10[3] = cartpole_QP_solver_z09[3];
output->z10[4] = cartpole_QP_solver_z09[4];
output->z10[5] = cartpole_QP_solver_z09[5];
output->z10[6] = cartpole_QP_solver_z09[6];
output->z10[7] = cartpole_QP_solver_z09[7];
output->z10[8] = cartpole_QP_solver_z09[8];
output->z10[9] = cartpole_QP_solver_z09[9];
output->z11[0] = cartpole_QP_solver_z10[0];
output->z11[1] = cartpole_QP_solver_z10[1];
output->z11[2] = cartpole_QP_solver_z10[2];
output->z11[3] = cartpole_QP_solver_z10[3];
output->z11[4] = cartpole_QP_solver_z10[4];
output->z11[5] = cartpole_QP_solver_z10[5];
output->z11[6] = cartpole_QP_solver_z10[6];
output->z11[7] = cartpole_QP_solver_z10[7];
output->z11[8] = cartpole_QP_solver_z10[8];
output->z11[9] = cartpole_QP_solver_z10[9];
output->z12[0] = cartpole_QP_solver_z11[0];
output->z12[1] = cartpole_QP_solver_z11[1];
output->z12[2] = cartpole_QP_solver_z11[2];
output->z12[3] = cartpole_QP_solver_z11[3];
output->z12[4] = cartpole_QP_solver_z11[4];
output->z12[5] = cartpole_QP_solver_z11[5];
output->z12[6] = cartpole_QP_solver_z11[6];
output->z12[7] = cartpole_QP_solver_z11[7];
output->z12[8] = cartpole_QP_solver_z11[8];
output->z12[9] = cartpole_QP_solver_z11[9];
output->z13[0] = cartpole_QP_solver_z12[0];
output->z13[1] = cartpole_QP_solver_z12[1];
output->z13[2] = cartpole_QP_solver_z12[2];
output->z13[3] = cartpole_QP_solver_z12[3];
output->z13[4] = cartpole_QP_solver_z12[4];
output->z13[5] = cartpole_QP_solver_z12[5];
output->z13[6] = cartpole_QP_solver_z12[6];
output->z13[7] = cartpole_QP_solver_z12[7];
output->z13[8] = cartpole_QP_solver_z12[8];
output->z13[9] = cartpole_QP_solver_z12[9];
output->z14[0] = cartpole_QP_solver_z13[0];
output->z14[1] = cartpole_QP_solver_z13[1];
output->z14[2] = cartpole_QP_solver_z13[2];
output->z14[3] = cartpole_QP_solver_z13[3];
output->z14[4] = cartpole_QP_solver_z13[4];
output->z14[5] = cartpole_QP_solver_z13[5];
output->z14[6] = cartpole_QP_solver_z13[6];
output->z14[7] = cartpole_QP_solver_z13[7];
output->z14[8] = cartpole_QP_solver_z13[8];
output->z14[9] = cartpole_QP_solver_z13[9];
output->z15[0] = cartpole_QP_solver_z14[0];
output->z15[1] = cartpole_QP_solver_z14[1];
output->z15[2] = cartpole_QP_solver_z14[2];
output->z15[3] = cartpole_QP_solver_z14[3];
output->z15[4] = cartpole_QP_solver_z14[4];

#if cartpole_QP_solver_SET_TIMING == 1
info->solvetime = cartpole_QP_solver_toc(&solvertimer);
#if cartpole_QP_solver_SET_PRINTLEVEL > 0 && cartpole_QP_solver_SET_TIMING == 1
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
