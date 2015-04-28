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

#include "wam7dofarm_QP_solver.h"

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
#ifndef USEMEXPRINTS
#include <stdio.h>
#define PRINTTEXT printf
#else
#include "mex.h"
#define PRINTTEXT mexPrintf
#endif



/* LINEAR ALGEBRA LIBRARY ---------------------------------------------- */
/*
 * Initializes a vector of length 603 with a value.
 */
void wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_603(wam7dofarm_QP_solver_FLOAT* vec, wam7dofarm_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<603; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 149 with a value.
 */
void wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_149(wam7dofarm_QP_solver_FLOAT* vec, wam7dofarm_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<149; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 954 with a value.
 */
void wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_954(wam7dofarm_QP_solver_FLOAT* vec, wam7dofarm_QP_solver_FLOAT value)
{
	int i;
	for( i=0; i<954; i++ )
	{
		vec[i] = value;
	}
}


/* 
 * Calculates a dot product and adds it to a variable: z += x'*y; 
 * This function is for vectors of length 954.
 */
void wam7dofarm_QP_solver_LA_DOTACC_954(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<954; i++ ){
		*z += x[i]*y[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [64 x 64]
 *             f  - column vector of size 64
 *             z  - column vector of size 64
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 64
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_FLOAT* H, wam7dofarm_QP_solver_FLOAT* f, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* grad, wam7dofarm_QP_solver_FLOAT* value)
{
	int i;
	wam7dofarm_QP_solver_FLOAT hz;	
	for( i=0; i<64; i++){
		hz = H[i]*z[i];
		grad[i] = hz + f[i];
		*value += 0.5*hz*z[i] + f[i]*z[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [27 x 27]
 *             f  - column vector of size 27
 *             z  - column vector of size 27
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 27
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void wam7dofarm_QP_solver_LA_DIAG_QUADFCN_27(wam7dofarm_QP_solver_FLOAT* H, wam7dofarm_QP_solver_FLOAT* f, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* grad, wam7dofarm_QP_solver_FLOAT* value)
{
	int i;
	wam7dofarm_QP_solver_FLOAT hz;	
	for( i=0; i<27; i++){
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
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB3_29_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *z, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;
	wam7dofarm_QP_solver_FLOAT AxBu[29];
	wam7dofarm_QP_solver_FLOAT norm = *y;
	wam7dofarm_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<29; i++ ){
		AxBu[i] = A[k++]*x[0] + B[m++]*u[0];
	}	
	for( j=1; j<64; j++ ){		
		for( i=0; i<29; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<64; n++ ){
		for( i=0; i<29; i++ ){
			AxBu[i] += B[m++]*u[n];
		}		
	}

	for( i=0; i<29; i++ ){
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
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *z, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	wam7dofarm_QP_solver_FLOAT AxBu[15];
	wam7dofarm_QP_solver_FLOAT norm = *y;
	wam7dofarm_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<15; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<15; i++ ){
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
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *z, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	wam7dofarm_QP_solver_FLOAT AxBu[15];
	wam7dofarm_QP_solver_FLOAT norm = *y;
	wam7dofarm_QP_solver_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<15; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<15; i++ ){
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
 * Matrix vector multiplication y = M'*x where M is of size [29 x 64]
 * and stored in column major format. Note the transpose of M!
 */
void wam7dofarm_QP_solver_LA_DENSE_MTVM_29_64(wam7dofarm_QP_solver_FLOAT *M, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<64; i++ ){
		y[i] = 0;
		for( j=0; j<29; j++ ){
			y[i] += M[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [15 x 64]
 * and B is of size [29 x 64]
 * and stored in column major format. Note the transposes of A and B!
 */
void wam7dofarm_QP_solver_LA_DENSE_MTVM2_15_64_29(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	int n;
	int m = 0;
	for( i=0; i<64; i++ ){
		z[i] = 0;
		for( j=0; j<15; j++ ){
			z[i] += A[k++]*x[j];
		}
		for( n=0; n<29; n++ ){
			z[i] += B[m++]*y[n];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [15 x 64] and stored in column major format.
 * and B is of size [15 x 64] and stored in diagzero format
 * Note the transposes of A and B!
 */
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	for( i=0; i<15; i++ ){
		z[i] = 0;
		for( j=0; j<15; j++ ){
			z[i] += A[k++]*x[j];
		}
		z[i] += B[i]*y[i];
	}
	for( i=15 ;i<64; i++ ){
		z[i] = 0;
		for( j=0; j<15; j++ ){
			z[i] += A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = M'*x where M is of size [15 x 27]
 * and stored in diagzero format. Note the transpose of M!
 */
void wam7dofarm_QP_solver_LA_DIAGZERO_MTVM_15_27(wam7dofarm_QP_solver_FLOAT *M, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<27; i++ ){
		y[i] = M[i]*x[i];
	}
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 64. Output z is of course scalar.
 */
void wam7dofarm_QP_solver_LA_VSUBADD3_64(wam7dofarm_QP_solver_FLOAT* t, wam7dofarm_QP_solver_FLOAT* u, int* uidx, wam7dofarm_QP_solver_FLOAT* v, wam7dofarm_QP_solver_FLOAT* w, wam7dofarm_QP_solver_FLOAT* y, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* r)
{
	int i;
	wam7dofarm_QP_solver_FLOAT norm = *r;
	wam7dofarm_QP_solver_FLOAT vx = 0;
	wam7dofarm_QP_solver_FLOAT x;
	for( i=0; i<64; i++){
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
 * for vectors of length 36. Output z is of course scalar.
 */
void wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_FLOAT* t, int* tidx, wam7dofarm_QP_solver_FLOAT* u, wam7dofarm_QP_solver_FLOAT* v, wam7dofarm_QP_solver_FLOAT* w, wam7dofarm_QP_solver_FLOAT* y, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* r)
{
	int i;
	wam7dofarm_QP_solver_FLOAT norm = *r;
	wam7dofarm_QP_solver_FLOAT vx = 0;
	wam7dofarm_QP_solver_FLOAT x;
	for( i=0; i<36; i++){
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
 * for vectors of length 27. Output z is of course scalar.
 */
void wam7dofarm_QP_solver_LA_VSUBADD3_27(wam7dofarm_QP_solver_FLOAT* t, wam7dofarm_QP_solver_FLOAT* u, int* uidx, wam7dofarm_QP_solver_FLOAT* v, wam7dofarm_QP_solver_FLOAT* w, wam7dofarm_QP_solver_FLOAT* y, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* r)
{
	int i;
	wam7dofarm_QP_solver_FLOAT norm = *r;
	wam7dofarm_QP_solver_FLOAT vx = 0;
	wam7dofarm_QP_solver_FLOAT x;
	for( i=0; i<27; i++){
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
 * for vectors of length 15. Output z is of course scalar.
 */
void wam7dofarm_QP_solver_LA_VSUBADD2_15(wam7dofarm_QP_solver_FLOAT* t, int* tidx, wam7dofarm_QP_solver_FLOAT* u, wam7dofarm_QP_solver_FLOAT* v, wam7dofarm_QP_solver_FLOAT* w, wam7dofarm_QP_solver_FLOAT* y, wam7dofarm_QP_solver_FLOAT* z, wam7dofarm_QP_solver_FLOAT* r)
{
	int i;
	wam7dofarm_QP_solver_FLOAT norm = *r;
	wam7dofarm_QP_solver_FLOAT vx = 0;
	wam7dofarm_QP_solver_FLOAT x;
	for( i=0; i<15; i++){
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
 * Computes r = A*x - b + s
 * and      y = max([norm(r,inf), y])
 * and      z -= l'*(Ax-b)
 * where A is stored in column major format
 */
void wam7dofarm_QP_solver_LA_MVSUBADD_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *z, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	wam7dofarm_QP_solver_FLOAT Ax[12];
	wam7dofarm_QP_solver_FLOAT Axlessb;
	wam7dofarm_QP_solver_FLOAT norm = *y;
	wam7dofarm_QP_solver_FLOAT lAxlessb = 0;

	/* do A*x first */
	for( i=0; i<12; i++ ){
		Ax[i] = A[k++]*x[0];				
	}	
	for( j=1; j<27; j++ ){		
		for( i=0; i<12; i++ ){
			Ax[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<12; i++ ){
		Axlessb = Ax[i] - b[i];
		r[i] = Axlessb + s[i];
		lAxlessb += l[i]*Axlessb;
		if( r[i] > norm ){
			norm = r[i];
		}
		if( -r[i] > norm ){
			norm = -r[i];
		}
	}
	*y = norm;
	*z -= lAxlessb;
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 64
 * Returns also L/S, a value that is often used elsewhere.
 */
void wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_FLOAT *lu, wam7dofarm_QP_solver_FLOAT *su, wam7dofarm_QP_solver_FLOAT *ru, wam7dofarm_QP_solver_FLOAT *ll, wam7dofarm_QP_solver_FLOAT *sl, wam7dofarm_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, wam7dofarm_QP_solver_FLOAT *grad, wam7dofarm_QP_solver_FLOAT *lubysu, wam7dofarm_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<64; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<64; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<36; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Computes inequality constraints gradient-
 * Special function for box constraints of length 27
 * Returns also L/S, a value that is often used elsewhere.
 */
void wam7dofarm_QP_solver_LA_INEQ_B_GRAD_27_27_15(wam7dofarm_QP_solver_FLOAT *lu, wam7dofarm_QP_solver_FLOAT *su, wam7dofarm_QP_solver_FLOAT *ru, wam7dofarm_QP_solver_FLOAT *ll, wam7dofarm_QP_solver_FLOAT *sl, wam7dofarm_QP_solver_FLOAT *rl, int* lbIdx, int* ubIdx, wam7dofarm_QP_solver_FLOAT *grad, wam7dofarm_QP_solver_FLOAT *lubysu, wam7dofarm_QP_solver_FLOAT *llbysl)
{
	int i;
	for( i=0; i<27; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<27; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<15; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Special function for gradient of inequality constraints
 * Calculates grad += A'*(L/S)*rI
 */
void wam7dofarm_QP_solver_LA_INEQ_P_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *lp, wam7dofarm_QP_solver_FLOAT *sp, wam7dofarm_QP_solver_FLOAT *rip, wam7dofarm_QP_solver_FLOAT *grad, wam7dofarm_QP_solver_FLOAT *lpbysp)
{
	int i;
	int j;
	int k = 0;

	wam7dofarm_QP_solver_FLOAT lsr[12];
	
	/* do (L/S)*ri first */
	for( j=0; j<12; j++ ){
		lpbysp[j] = lp[j] / sp[j];
		lsr[j] = lpbysp[j]*rip[j];
	}

	for( i=0; i<27; i++ ){		
		for( j=0; j<12; j++ ){
			grad[i] += A[k++]*lsr[j];
		}
	}
}


/*
 * Addition of three vectors  z = u + w + v
 * of length 603.
 */
void wam7dofarm_QP_solver_LA_VVADD3_603(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<603; i++ ){
		z[i] = u[i] + v[i] + w[i];
	}
}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 64.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_FLOAT *H, wam7dofarm_QP_solver_FLOAT *llbysl, int* lbIdx, wam7dofarm_QP_solver_FLOAT *lubysu, int* ubIdx, wam7dofarm_QP_solver_FLOAT *Phi)


{
	int i;
	
	/* copy  H into PHI */
	for( i=0; i<64; i++ ){
		Phi[i] = H[i];
	}

	/* add llbysl onto Phi where necessary */
	for( i=0; i<64; i++ ){
		Phi[lbIdx[i]] += llbysl[i];
	}

	/* add lubysu onto Phi where necessary */
	for( i=0; i<36; i++){
		Phi[ubIdx[i]] +=  lubysu[i];
	}
	
	/* compute cholesky */
	for(i=0; i<64; i++)
	{
#if wam7dofarm_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
 * where A is to be computed and is of size [29 x 64],
 * B is given and of size [29 x 64], L is a diagonal
 * matrix of size 29 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_29_64(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<64; j++){
		for( i=0; i<29; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 64.
 */
void wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *y)
{
    int i;

    for( i=0; i<64; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [15 x 64],
 * B is given and of size [15 x 64], L is a diagonal
 * matrix of size 15 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<64; j++){
		for( i=0; i<15; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [29 x 64]
 *  size(B) = [15 x 64]
 * 
 * and all matrices are stored in column major format.
 *
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE.  
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_MMTM_29_64_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *C)
{
    int i, j, k;
    wam7dofarm_QP_solver_FLOAT temp;
    
    for( i=0; i<29; i++ ){        
        for( j=0; j<15; j++ ){
            temp = 0; 
            for( k=0; k<64; k++ ){
                temp += A[k*29+i]*B[k*15+j];
            }						
            C[j*29+i] = temp;
        }
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [15 x 64],
 * B is given and of size [15 x 64], L is a diagonal
 *  matrix of size 64 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
	int j;
    for( j=0; j<64; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [15 x 64]
 *  size(B) = [15 x 64] in diagzero format
 * 
 * A and C matrices are stored in column major format.
 * 
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *C)
{
    int i, j;
	
	for( i=0; i<15; i++ ){
		for( j=0; j<15; j++){
			C[j*15+i] = B[i*15+j]*A[i];
		}
	}

}


/*
 * Special function to compute the Dense positive definite 
 * augmented Hessian for block size 27.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = H + diag(llbysl) + diag(lubysu)
 * where Phi is stored in lower triangular row major format
 */
void wam7dofarm_QP_solver_LA_INEQ_DENSE_DIAG_HESS_27_27_15(wam7dofarm_QP_solver_FLOAT *H, wam7dofarm_QP_solver_FLOAT *llbysl, int* lbIdx, wam7dofarm_QP_solver_FLOAT *lubysu, int* ubIdx, wam7dofarm_QP_solver_FLOAT *Phi)
{
	int i;
	int j;
	int k = 0;
	
	/* copy diagonal of H into PHI and set lower part of PHI = 0*/
	for( i=0; i<27; i++ ){
		for( j=0; j<i; j++ ){
			Phi[k++] = 0;
		}		
		/* we are on the diagonal */
		Phi[k++] = H[i];
	}

	/* add llbysl onto Phi where necessary */
	for( i=0; i<27; i++ ){
		j = lbIdx[i];
		Phi[((j+1)*(j+2))/2-1] += llbysl[i];
	}

	/* add lubysu onto Phi where necessary */
	for( i=0; i<15; i++){
		j = ubIdx[i];
		Phi[((j+1)*(j+2))/2-1] +=  lubysu[i];
	}

}


/**
 * Compute X = X + A'*D*A, where A is a general full matrix, D is
 * is a diagonal matrix stored in the vector d and X is a symmetric
 * positive definite matrix in lower triangular storage format. 
 * A is stored in column major format and is of size [12 x 27]
 * Phi is of size [27 x 27].
 */
void wam7dofarm_QP_solver_LA_DENSE_ADDMTDM_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *d, wam7dofarm_QP_solver_FLOAT *X)
{    
    int i,j,k,ii,di;
    wam7dofarm_QP_solver_FLOAT x;
    
    di = 0; ii = 0;
    for( i=0; i<27; i++ ){        
        for( j=0; j<=i; j++ ){
            x = 0;
            for( k=0; k<12; k++ ){
                x += A[i*12+k]*A[j*12+k]*d[k];
            }
            X[ii+j] += x;
        }
        ii += ++di;
    }
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 27.
 */
void wam7dofarm_QP_solver_LA_DENSE_CHOL2_27(wam7dofarm_QP_solver_FLOAT *A)
{
    int i, j, k, di, dj;
	 int ii, jj;
    wam7dofarm_QP_solver_FLOAT l;
    wam7dofarm_QP_solver_FLOAT Mii;
    
	ii=0; di=0;
    for( i=0; i<27; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += A[ii+k]*A[ii+k];
        }        
        
        Mii = A[ii+i] - l;
        
#if wam7dofarm_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
        if( Mii < 1.0000000000000000E-013 ){
             PRINTTEXT("WARNING (CHOL2): small %d-th pivot in Cholesky fact. (=%3.1e < eps=%3.1e), regularizing to %3.1e\n",i,Mii,1.0000000000000000E-013,4.0000000000000002E-004);
			 A[ii+i] = 2.0000000000000000E-002;
		} else
		{
			A[ii+i] = sqrt(Mii);
		}
#else
		A[ii+i] = Mii < 1.0000000000000000E-013 ? 2.0000000000000000E-002 : sqrt(Mii);
#endif
                    
		jj = ((i+1)*(i+2))/2; dj = i+1;
        for( j=i+1; j<27; j++ ){
            l = 0;            
            for( k=0; k<i; k++ ){
                l += A[jj+k]*A[ii+k];
            }

			/* saturate values for numerical stability */
			l = MIN(l,  BIGMM);
			l = MAX(l, -BIGMM);

            A[jj+i] = (A[jj+i] - l)/A[ii+i];            
			jj += ++dj;
        }
		ii += ++di;
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [15 x 27],
 * B is given and of size [15 x 27] stored in 
 * diagzero storage format, L is a lower tri-
 * angular matrix of size 27 stored in lower triangular 
 * storage format. Note the transpose of L!
 *
 * Result: A in column major storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MATRIXFORWARDSUB_15_27(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
    int i,j,k,di;
	 int ii;
    wam7dofarm_QP_solver_FLOAT a;
	
	/*
	* The matrix A has the form
	*
	* d u u u r r r r r 
	* 0 d u u r r r r r 
	* 0 0 d u r r r r r 
	* 0 0 0 d r r r r r
	*
	* |Part1|| Part 2 |
	* 
	* d: diagonal
	* u: upper
	* r: right
	*/
	
	
    /* Part 1 */
    ii=0; di=0;
    for( j=0; j<15; j++ ){        
        for( i=0; i<j; i++ ){
            /* Calculate part of A which is non-zero and not diagonal "u"
             * i < j */
            a = 0;
			
            for( k=i; k<j; k++ ){
                a -= A[k*15+i]*L[ii+k];
            }
            A[j*15+i] = a/L[ii+j];
        }
        /* do the diagonal "d"
         * i = j */
        A[j*15+j] = B[i]/L[ii+j];
        
        /* fill lower triangular part with zeros "0"
         * n > i > j */
        for( i=j+1     ; i < 15; i++ ){
            A[j*15+i] = 0;
        }
        
        /* increment index of L */
        ii += ++di;	
    }
	
	/* Part 2 */ 
	for( j=15; j<27; j++ ){        
        for( i=0; i<15; i++ ){
            /* Calculate part of A which is non-zero and not diagonal "r" */
            a = 0;
			
            for( k=i; k<j; k++ ){
                a -= A[k*15+i]*L[ii+k];
            }
            A[j*15+i] = a/L[ii+j];
        }
        
        /* increment index of L */
        ii += ++di;	
    }
	
	
	
}


/**
 * Forward substitution to solve L*y = b where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * The dimensions involved are 27.
 */
void wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_27(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    wam7dofarm_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<27; i++ ){
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
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [29 x 64] in column
 * storage format, and B is of size [29 x 64] also in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_MMT2_29_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    wam7dofarm_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<29; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<64; k++ ){
                ltemp += A[k*29+i]*A[k*29+j];
            }			
			for( k=0; k<64; k++ ){
                ltemp += B[k*29+i]*B[k*29+j];
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
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB2_29_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<29; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[m++]*u[0];
	}	
	for( j=1; j<64; j++ ){		
		for( i=0; i<29; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
	for( n=1; n<64; n++ ){
		for( i=0; i<29; i++ ){
			r[i] -= B[m++]*u[n];
		}		
	}
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [15 x 64] in column
 * storage format, and B is of size [15 x 64] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    wam7dofarm_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<15; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<64; k++ ){
                ltemp += A[k*15+i]*A[k*15+j];
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
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<15; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [15 x 64] in column
 * storage format, and B is of size [15 x 27] also in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_MMT2_15_64_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    wam7dofarm_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<15; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<64; k++ ){
                ltemp += A[k*15+i]*A[k*15+j];
            }			
			for( k=0; k<27; k++ ){
                ltemp += B[k*15+i]*B[k*15+j];
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
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB2_15_64_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<15; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[m++]*u[0];
	}	
	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
	for( n=1; n<27; n++ ){
		for( i=0; i<15; i++ ){
			r[i] -= B[m++]*u[n];
		}		
	}
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 29 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void wam7dofarm_QP_solver_LA_DENSE_CHOL_29(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    wam7dofarm_QP_solver_FLOAT l;
    wam7dofarm_QP_solver_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<29; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<29; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if wam7dofarm_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
        for( j=i+1; j<29; j++ ){
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
 * The dimensions involved are 29.
 */
void wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_29(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    wam7dofarm_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<29; i++ ){
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
 * where A is to be computed and is of size [15 x 29],
 * B is given and of size [15 x 29], L is a lower tri-
 * angular matrix of size 29 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_29(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    wam7dofarm_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<29; j++ ){        
        for( i=0; i<15; i++ ){
            a = B[i*29+j];
            for( k=0; k<j; k++ ){
                a -= A[k*15+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*15+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 15
 * and A is a dense matrix of size [15 x 29] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_29(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    wam7dofarm_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<15; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<29; k++ ){
                ltemp += A[k*15+i]*A[k*15+j];
            }						
            L[ii+j] -= ltemp;
        }
        ii += ++di;
    }
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 15 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    wam7dofarm_QP_solver_FLOAT l;
    wam7dofarm_QP_solver_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<15; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<15; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if wam7dofarm_QP_solver_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
        for( j=i+1; j<15; j++ ){
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
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_29(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<15; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<29; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Forward substitution to solve L*y = b where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * The dimensions involved are 15.
 */
void wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *y)
{
    int i,j,ii,di;
    wam7dofarm_QP_solver_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<15; i++ ){
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
 * where A is to be computed and is of size [15 x 15],
 * B is given and of size [15 x 15], L is a lower tri-
 * angular matrix of size 15 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *A)
{
    int i,j,k,ii,di;
    wam7dofarm_QP_solver_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<15; j++ ){        
        for( i=0; i<15; i++ ){
            a = B[i*15+j];
            for( k=0; k<j; k++ ){
                a -= A[k*15+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*15+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 15
 * and A is a dense matrix of size [15 x 15] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *L)
{
    int i, j, k, ii, di;
    wam7dofarm_QP_solver_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<15; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<15; k++ ){
                ltemp += A[k*15+i]*A[k*15+j];
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
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<15; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<15; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 15.
 */
void wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    wam7dofarm_QP_solver_FLOAT xel;    
	int start = 105;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 14;
    for( i=14; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 14;
        for( j=14; j>i; j-- ){
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
 * Matrix vector multiplication y = b - M'*x where M is of size [15 x 15]
 * and stored in column major format. Note the transpose of M!
 */
void wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<15; i++ ){
		r[i] = b[i];
		for( j=0; j<15; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = b - M'*x where M is of size [15 x 29]
 * and stored in column major format. Note the transpose of M!
 */
void wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_29(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<29; i++ ){
		r[i] = b[i];
		for( j=0; j<15; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 29.
 */
void wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_29(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    wam7dofarm_QP_solver_FLOAT xel;    
	int start = 406;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 28;
    for( i=28; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 28;
        for( j=28; j>i; j-- ){
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
 * Vector subtraction z = -x - y for vectors of length 603.
 */
void wam7dofarm_QP_solver_LA_VSUB2_603(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<603; i++){
		z[i] = -x[i] - y[i];
	}
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 64 in vector
 * storage format.
 */
void wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<64; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * lower triangular matrix of size 27 in lower triangular
 * storage format.
 */
void wam7dofarm_QP_solver_LA_DENSE_FORWARDBACKWARDSUB_27(wam7dofarm_QP_solver_FLOAT *L, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    wam7dofarm_QP_solver_FLOAT y[27];
    wam7dofarm_QP_solver_FLOAT yel,xel;
	int start = 351;
            
    /* first solve Ly = b by forward substitution */
     ii = 0; di = 0;
    for( i=0; i<27; i++ ){
        yel = b[i];        
        for( j=0; j<i; j++ ){
            yel -= y[j]*L[ii+j];
        }

		/* saturate for numerical stability */
		yel = MIN(yel, BIGM);
		yel = MAX(yel, -BIGM); 

        y[i] = yel / L[ii+i];
        ii += ++di;
    }
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 26;
    for( i=26; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 26;
        for( j=26; j>i; j-- ){
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
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 64,
 * and x has length 64 and is indexed through yidx.
 */
void wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_FLOAT *x, int* xidx, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<64; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 64.
 */
void wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<64; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 64
 * and z, x and yidx are of length 36.
 */
void wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<36; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 36.
 */
void wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<36; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 27,
 * and x has length 27 and is indexed through yidx.
 */
void wam7dofarm_QP_solver_LA_VSUB_INDEXED_27(wam7dofarm_QP_solver_FLOAT *x, int* xidx, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<27; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 27.
 */
void wam7dofarm_QP_solver_LA_VSUB3_27(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<27; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 27
 * and z, x and yidx are of length 15.
 */
void wam7dofarm_QP_solver_LA_VSUB2_INDEXED_15(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<15; i++){
		z[i] = -x[i] - y[yidx[i]];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 15.
 */
void wam7dofarm_QP_solver_LA_VSUB3_15(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<15; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/* 
 * Computes r = -b - A*x
 * where A is stored in column major format
 */
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB4_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<12; i++ ){
		r[i] = -b[i] - A[k++]*x[0];
	}	
	for( j=1; j<27; j++ ){		
		for( i=0; i<12; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 12.
 */
void wam7dofarm_QP_solver_LA_VSUB3_12(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *w, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<12; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/**
 * Backtracking line search.
 * 
 * First determine the maximum line length by a feasibility line
 * search, i.e. a ~= argmax{ a \in [0...1] s.t. l+a*dl >= 0 and s+a*ds >= 0}.
 *
 * The function returns either the number of iterations or exits the error code
 * wam7dofarm_QP_solver_NOPROGRESS (should be negative).
 */
int wam7dofarm_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *dl, wam7dofarm_QP_solver_FLOAT *ds, wam7dofarm_QP_solver_FLOAT *a, wam7dofarm_QP_solver_FLOAT *mu_aff)
{
    int i;
	int lsIt=1;    
    wam7dofarm_QP_solver_FLOAT dltemp;
    wam7dofarm_QP_solver_FLOAT dstemp;
    wam7dofarm_QP_solver_FLOAT mya = 1.0;
    wam7dofarm_QP_solver_FLOAT mymu;
        
    while( 1 ){                        

        /* 
         * Compute both snew and wnew together.
         * We compute also mu_affine along the way here, as the
         * values might be in registers, so it should be cheaper.
         */
        mymu = 0;
        for( i=0; i<954; i++ ){
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
        if( i == 954 ){
            break;
        } else {
            mya *= wam7dofarm_QP_solver_SET_LS_SCALE_AFF;
            if( mya < wam7dofarm_QP_solver_SET_LS_MINSTEP ){
                return wam7dofarm_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* return new values and iteration counter */
    *a = mya;
    *mu_aff = mymu / (wam7dofarm_QP_solver_FLOAT)954;
    return lsIt;
}


/*
 * Vector subtraction x = (u.*v - mu)*sigma where a is a scalar
*  and x,u,v are vectors of length 954.
 */
void wam7dofarm_QP_solver_LA_VSUB5_954(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT mu,  wam7dofarm_QP_solver_FLOAT sigma, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<954; i++){
		x[i] = u[i]*v[i] - mu;
		x[i] *= sigma;
	}
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 64,
 * u, su, uidx are of length 36 and v, sv, vidx are of length 64.
 */
void wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *su, int* uidx, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *sv, int* vidx, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<64; i++ ){
		x[i] = 0;
	}
	for( i=0; i<36; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<64; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A an B are stored in column major format
 */
void wam7dofarm_QP_solver_LA_DENSE_2MVMADD_29_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<29; i++ ){
		r[i] = A[k++]*x[0] + B[m++]*u[0];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<29; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<64; n++ ){
		for( i=0; i<29; i++ ){
			r[i] += B[m++]*u[n];
		}		
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<15; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 27,
 * u, su, uidx are of length 15 and v, sv, vidx are of length 27.
 */
void wam7dofarm_QP_solver_LA_VSUB6_INDEXED_27_15_27(wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *su, int* uidx, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *sv, int* vidx, wam7dofarm_QP_solver_FLOAT *x)
{
	int i;
	for( i=0; i<27; i++ ){
		x[i] = 0;
	}
	for( i=0; i<15; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<27; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/*
 * Matrix vector multiplication z = z + A'*(x./s) where A is of size [12 x 27]
 * and stored in column major format. Note the transpose of M!
 */
void wam7dofarm_QP_solver_LA_DENSE_MTVMADD2_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	int j;
	int k = 0; 
	wam7dofarm_QP_solver_FLOAT temp[12];

	for( j=0; j<12; j++ ){
		temp[j] = x[j] / s[j];
	}

	for( i=0; i<27; i++ ){
		for( j=0; j<12; j++ ){
			z[i] += A[k++]*temp[j];
		}
	}
}


/* 
 * Computes r = A*x + B*u
 * where A an B are stored in column major format
 */
void wam7dofarm_QP_solver_LA_DENSE_2MVMADD_15_64_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *B, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;
	int m = 0;
	int n;

	for( i=0; i<15; i++ ){
		r[i] = A[k++]*x[0] + B[m++]*u[0];
	}	

	for( j=1; j<64; j++ ){		
		for( i=0; i<15; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
	for( n=1; n<27; n++ ){
		for( i=0; i<15; i++ ){
			r[i] += B[m++]*u[n];
		}		
	}
}


/*
 * Vector subtraction z = x - y for vectors of length 603.
 */
void wam7dofarm_QP_solver_LA_VSUB_603(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<603; i++){
		z[i] = x[i] - y[i];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 64 (length of y >= 64).
 */
void wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<64; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 36 (length of y >= 36).
 */
void wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<36; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 27 (length of y >= 27).
 */
void wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_27(wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<27; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 15 (length of y >= 15).
 */
void wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_15(wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *u, wam7dofarm_QP_solver_FLOAT *y, int* yidx, wam7dofarm_QP_solver_FLOAT *z)
{
	int i;
	for( i=0; i<15; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/* 
 * Computes r = (-b + l.*(A*x))./s
 * where A is stored in column major format
 */
void wam7dofarm_QP_solver_LA_DENSE_MVMSUB5_12_27(wam7dofarm_QP_solver_FLOAT *A, wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *b, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	wam7dofarm_QP_solver_FLOAT temp[12];

	
	for( i=0; i<12; i++ ){
		temp[i] = A[k++]*x[0];
	}
	

	for( j=1; j<27; j++ ){		
		for( i=0; i<12; i++ ){
			temp[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<12; i++ ){
		r[i] = (-b[i] + l[i]*temp[i])/s[i]; 
	}	
	
}


/*
 * Computes ds = -l.\(r + s.*dl) for vectors of length 954.
 */
void wam7dofarm_QP_solver_LA_VSUB7_954(wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *r, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *dl, wam7dofarm_QP_solver_FLOAT *ds)
{
	int i;
	for( i=0; i<954; i++){
		ds[i] = -(r[i] + s[i]*dl[i])/l[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 603.
 */
void wam7dofarm_QP_solver_LA_VADD_603(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<603; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 149.
 */
void wam7dofarm_QP_solver_LA_VADD_149(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<149; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 954.
 */
void wam7dofarm_QP_solver_LA_VADD_954(wam7dofarm_QP_solver_FLOAT *x, wam7dofarm_QP_solver_FLOAT *y)
{
	int i;
	for( i=0; i<954; i++){
		x[i] += y[i];
	}
}


/**
 * Backtracking line search for combined predictor/corrector step.
 * Update on variables with safety factor gamma (to keep us away from
 * boundary).
 */
int wam7dofarm_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(wam7dofarm_QP_solver_FLOAT *z, wam7dofarm_QP_solver_FLOAT *v, wam7dofarm_QP_solver_FLOAT *l, wam7dofarm_QP_solver_FLOAT *s, wam7dofarm_QP_solver_FLOAT *dz, wam7dofarm_QP_solver_FLOAT *dv, wam7dofarm_QP_solver_FLOAT *dl, wam7dofarm_QP_solver_FLOAT *ds, wam7dofarm_QP_solver_FLOAT *a, wam7dofarm_QP_solver_FLOAT *mu)
{
    int i, lsIt=1;       
    wam7dofarm_QP_solver_FLOAT dltemp;
    wam7dofarm_QP_solver_FLOAT dstemp;    
    wam7dofarm_QP_solver_FLOAT a_gamma;
            
    *a = 1.0;
    while( 1 ){                        

        /* check whether search criterion is fulfilled */
        for( i=0; i<954; i++ ){
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
        if( i == 954 ){
            break;
        } else {
            *a *= wam7dofarm_QP_solver_SET_LS_SCALE;
            if( *a < wam7dofarm_QP_solver_SET_LS_MINSTEP ){
                return wam7dofarm_QP_solver_NOPROGRESS;
            }
        }
    }
    
    /* update variables with safety margin */
    a_gamma = (*a)*wam7dofarm_QP_solver_SET_LS_MAXSTEP;
    
    /* primal variables */
    for( i=0; i<603; i++ ){
        z[i] += a_gamma*dz[i];
    }
    
    /* equality constraint multipliers */
    for( i=0; i<149; i++ ){
        v[i] += a_gamma*dv[i];
    }
    
    /* inequality constraint multipliers & slacks, also update mu */
    *mu = 0;
    for( i=0; i<954; i++ ){
        dltemp = l[i] + a_gamma*dl[i]; l[i] = dltemp;
        dstemp = s[i] + a_gamma*ds[i]; s[i] = dstemp;
        *mu += dltemp*dstemp;
    }
    
    *a = a_gamma;
    *mu /= (wam7dofarm_QP_solver_FLOAT)954;
    return lsIt;
}




/* VARIABLE DEFINITIONS ------------------------------------------------ */
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_z[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_v[149];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dz_aff[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dv_aff[149];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_grad_cost[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_grad_eq[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rd[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_l[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_s[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_lbys[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dl_aff[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ds_aff[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dz_cc[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dv_cc[149];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_dl_cc[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ds_cc[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ccrhs[954];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_grad_ineq[603];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_H0[64] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z0 = wam7dofarm_QP_solver_z + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff0 = wam7dofarm_QP_solver_dz_aff + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc0 = wam7dofarm_QP_solver_dz_cc + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd0 = wam7dofarm_QP_solver_rd + 0;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd0[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost0 = wam7dofarm_QP_solver_grad_cost + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq0 = wam7dofarm_QP_solver_grad_eq + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq0 = wam7dofarm_QP_solver_grad_ineq + 0;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv0[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v0 = wam7dofarm_QP_solver_v + 0;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re0[29];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta0[29];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc0[29];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff0 = wam7dofarm_QP_solver_dv_aff + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc0 = wam7dofarm_QP_solver_dv_cc + 0;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V0[1856];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd0[435];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld0[435];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy0[29];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy0[29];
int wam7dofarm_QP_solver_lbIdx0[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb0 = wam7dofarm_QP_solver_l + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb0 = wam7dofarm_QP_solver_s + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb0 = wam7dofarm_QP_solver_lbys + 0;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb0[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff0 = wam7dofarm_QP_solver_dl_aff + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff0 = wam7dofarm_QP_solver_ds_aff + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc0 = wam7dofarm_QP_solver_dl_cc + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc0 = wam7dofarm_QP_solver_ds_cc + 0;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl0 = wam7dofarm_QP_solver_ccrhs + 0;
int wam7dofarm_QP_solver_ubIdx0[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub0 = wam7dofarm_QP_solver_l + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub0 = wam7dofarm_QP_solver_s + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub0 = wam7dofarm_QP_solver_lbys + 64;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub0[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff0 = wam7dofarm_QP_solver_dl_aff + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff0 = wam7dofarm_QP_solver_ds_aff + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc0 = wam7dofarm_QP_solver_dl_cc + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc0 = wam7dofarm_QP_solver_ds_cc + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub0 = wam7dofarm_QP_solver_ccrhs + 64;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi0[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z1 = wam7dofarm_QP_solver_z + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff1 = wam7dofarm_QP_solver_dz_aff + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc1 = wam7dofarm_QP_solver_dz_cc + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd1 = wam7dofarm_QP_solver_rd + 64;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd1[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost1 = wam7dofarm_QP_solver_grad_cost + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq1 = wam7dofarm_QP_solver_grad_eq + 64;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq1 = wam7dofarm_QP_solver_grad_ineq + 64;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv1[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v1 = wam7dofarm_QP_solver_v + 29;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re1[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta1[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc1[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff1 = wam7dofarm_QP_solver_dv_aff + 29;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc1 = wam7dofarm_QP_solver_dv_cc + 29;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V1[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd1[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld1[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy1[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy1[15];
int wam7dofarm_QP_solver_lbIdx1[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb1 = wam7dofarm_QP_solver_l + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb1 = wam7dofarm_QP_solver_s + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb1 = wam7dofarm_QP_solver_lbys + 100;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb1[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff1 = wam7dofarm_QP_solver_dl_aff + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff1 = wam7dofarm_QP_solver_ds_aff + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc1 = wam7dofarm_QP_solver_dl_cc + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc1 = wam7dofarm_QP_solver_ds_cc + 100;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl1 = wam7dofarm_QP_solver_ccrhs + 100;
int wam7dofarm_QP_solver_ubIdx1[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub1 = wam7dofarm_QP_solver_l + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub1 = wam7dofarm_QP_solver_s + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub1 = wam7dofarm_QP_solver_lbys + 164;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub1[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff1 = wam7dofarm_QP_solver_dl_aff + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff1 = wam7dofarm_QP_solver_ds_aff + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc1 = wam7dofarm_QP_solver_dl_cc + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc1 = wam7dofarm_QP_solver_ds_cc + 164;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub1 = wam7dofarm_QP_solver_ccrhs + 164;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi1[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_D1[1856] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, -1.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 
0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W1[1856];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd1[435];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd1[435];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z2 = wam7dofarm_QP_solver_z + 128;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff2 = wam7dofarm_QP_solver_dz_aff + 128;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc2 = wam7dofarm_QP_solver_dz_cc + 128;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd2 = wam7dofarm_QP_solver_rd + 128;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd2[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost2 = wam7dofarm_QP_solver_grad_cost + 128;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq2 = wam7dofarm_QP_solver_grad_eq + 128;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq2 = wam7dofarm_QP_solver_grad_ineq + 128;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv2[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v2 = wam7dofarm_QP_solver_v + 44;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re2[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta2[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc2[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff2 = wam7dofarm_QP_solver_dv_aff + 44;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc2 = wam7dofarm_QP_solver_dv_cc + 44;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V2[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd2[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld2[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy2[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy2[15];
int wam7dofarm_QP_solver_lbIdx2[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb2 = wam7dofarm_QP_solver_l + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb2 = wam7dofarm_QP_solver_s + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb2 = wam7dofarm_QP_solver_lbys + 200;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb2[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff2 = wam7dofarm_QP_solver_dl_aff + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff2 = wam7dofarm_QP_solver_ds_aff + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc2 = wam7dofarm_QP_solver_dl_cc + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc2 = wam7dofarm_QP_solver_ds_cc + 200;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl2 = wam7dofarm_QP_solver_ccrhs + 200;
int wam7dofarm_QP_solver_ubIdx2[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub2 = wam7dofarm_QP_solver_l + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub2 = wam7dofarm_QP_solver_s + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub2 = wam7dofarm_QP_solver_lbys + 264;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub2[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff2 = wam7dofarm_QP_solver_dl_aff + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff2 = wam7dofarm_QP_solver_ds_aff + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc2 = wam7dofarm_QP_solver_dl_cc + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc2 = wam7dofarm_QP_solver_ds_cc + 264;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub2 = wam7dofarm_QP_solver_ccrhs + 264;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi2[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_D2[64] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W2[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd2[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd2[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z3 = wam7dofarm_QP_solver_z + 192;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff3 = wam7dofarm_QP_solver_dz_aff + 192;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc3 = wam7dofarm_QP_solver_dz_cc + 192;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd3 = wam7dofarm_QP_solver_rd + 192;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd3[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost3 = wam7dofarm_QP_solver_grad_cost + 192;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq3 = wam7dofarm_QP_solver_grad_eq + 192;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq3 = wam7dofarm_QP_solver_grad_ineq + 192;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv3[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v3 = wam7dofarm_QP_solver_v + 59;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re3[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta3[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc3[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff3 = wam7dofarm_QP_solver_dv_aff + 59;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc3 = wam7dofarm_QP_solver_dv_cc + 59;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V3[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd3[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld3[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy3[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy3[15];
int wam7dofarm_QP_solver_lbIdx3[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb3 = wam7dofarm_QP_solver_l + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb3 = wam7dofarm_QP_solver_s + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb3 = wam7dofarm_QP_solver_lbys + 300;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb3[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff3 = wam7dofarm_QP_solver_dl_aff + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff3 = wam7dofarm_QP_solver_ds_aff + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc3 = wam7dofarm_QP_solver_dl_cc + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc3 = wam7dofarm_QP_solver_ds_cc + 300;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl3 = wam7dofarm_QP_solver_ccrhs + 300;
int wam7dofarm_QP_solver_ubIdx3[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub3 = wam7dofarm_QP_solver_l + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub3 = wam7dofarm_QP_solver_s + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub3 = wam7dofarm_QP_solver_lbys + 364;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub3[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff3 = wam7dofarm_QP_solver_dl_aff + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff3 = wam7dofarm_QP_solver_ds_aff + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc3 = wam7dofarm_QP_solver_dl_cc + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc3 = wam7dofarm_QP_solver_ds_cc + 364;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub3 = wam7dofarm_QP_solver_ccrhs + 364;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi3[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W3[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd3[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd3[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z4 = wam7dofarm_QP_solver_z + 256;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff4 = wam7dofarm_QP_solver_dz_aff + 256;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc4 = wam7dofarm_QP_solver_dz_cc + 256;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd4 = wam7dofarm_QP_solver_rd + 256;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd4[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost4 = wam7dofarm_QP_solver_grad_cost + 256;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq4 = wam7dofarm_QP_solver_grad_eq + 256;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq4 = wam7dofarm_QP_solver_grad_ineq + 256;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv4[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v4 = wam7dofarm_QP_solver_v + 74;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re4[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta4[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc4[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff4 = wam7dofarm_QP_solver_dv_aff + 74;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc4 = wam7dofarm_QP_solver_dv_cc + 74;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V4[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd4[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld4[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy4[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy4[15];
int wam7dofarm_QP_solver_lbIdx4[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb4 = wam7dofarm_QP_solver_l + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb4 = wam7dofarm_QP_solver_s + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb4 = wam7dofarm_QP_solver_lbys + 400;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb4[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff4 = wam7dofarm_QP_solver_dl_aff + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff4 = wam7dofarm_QP_solver_ds_aff + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc4 = wam7dofarm_QP_solver_dl_cc + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc4 = wam7dofarm_QP_solver_ds_cc + 400;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl4 = wam7dofarm_QP_solver_ccrhs + 400;
int wam7dofarm_QP_solver_ubIdx4[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub4 = wam7dofarm_QP_solver_l + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub4 = wam7dofarm_QP_solver_s + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub4 = wam7dofarm_QP_solver_lbys + 464;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub4[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff4 = wam7dofarm_QP_solver_dl_aff + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff4 = wam7dofarm_QP_solver_ds_aff + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc4 = wam7dofarm_QP_solver_dl_cc + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc4 = wam7dofarm_QP_solver_ds_cc + 464;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub4 = wam7dofarm_QP_solver_ccrhs + 464;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi4[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W4[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd4[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd4[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z5 = wam7dofarm_QP_solver_z + 320;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff5 = wam7dofarm_QP_solver_dz_aff + 320;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc5 = wam7dofarm_QP_solver_dz_cc + 320;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd5 = wam7dofarm_QP_solver_rd + 320;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd5[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost5 = wam7dofarm_QP_solver_grad_cost + 320;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq5 = wam7dofarm_QP_solver_grad_eq + 320;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq5 = wam7dofarm_QP_solver_grad_ineq + 320;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv5[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v5 = wam7dofarm_QP_solver_v + 89;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re5[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta5[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc5[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff5 = wam7dofarm_QP_solver_dv_aff + 89;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc5 = wam7dofarm_QP_solver_dv_cc + 89;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V5[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd5[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld5[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy5[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy5[15];
int wam7dofarm_QP_solver_lbIdx5[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb5 = wam7dofarm_QP_solver_l + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb5 = wam7dofarm_QP_solver_s + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb5 = wam7dofarm_QP_solver_lbys + 500;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb5[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff5 = wam7dofarm_QP_solver_dl_aff + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff5 = wam7dofarm_QP_solver_ds_aff + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc5 = wam7dofarm_QP_solver_dl_cc + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc5 = wam7dofarm_QP_solver_ds_cc + 500;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl5 = wam7dofarm_QP_solver_ccrhs + 500;
int wam7dofarm_QP_solver_ubIdx5[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub5 = wam7dofarm_QP_solver_l + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub5 = wam7dofarm_QP_solver_s + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub5 = wam7dofarm_QP_solver_lbys + 564;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub5[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff5 = wam7dofarm_QP_solver_dl_aff + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff5 = wam7dofarm_QP_solver_ds_aff + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc5 = wam7dofarm_QP_solver_dl_cc + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc5 = wam7dofarm_QP_solver_ds_cc + 564;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub5 = wam7dofarm_QP_solver_ccrhs + 564;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi5[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W5[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd5[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd5[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z6 = wam7dofarm_QP_solver_z + 384;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff6 = wam7dofarm_QP_solver_dz_aff + 384;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc6 = wam7dofarm_QP_solver_dz_cc + 384;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd6 = wam7dofarm_QP_solver_rd + 384;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd6[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost6 = wam7dofarm_QP_solver_grad_cost + 384;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq6 = wam7dofarm_QP_solver_grad_eq + 384;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq6 = wam7dofarm_QP_solver_grad_ineq + 384;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv6[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v6 = wam7dofarm_QP_solver_v + 104;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re6[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta6[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc6[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff6 = wam7dofarm_QP_solver_dv_aff + 104;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc6 = wam7dofarm_QP_solver_dv_cc + 104;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V6[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd6[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld6[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy6[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy6[15];
int wam7dofarm_QP_solver_lbIdx6[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb6 = wam7dofarm_QP_solver_l + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb6 = wam7dofarm_QP_solver_s + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb6 = wam7dofarm_QP_solver_lbys + 600;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb6[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff6 = wam7dofarm_QP_solver_dl_aff + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff6 = wam7dofarm_QP_solver_ds_aff + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc6 = wam7dofarm_QP_solver_dl_cc + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc6 = wam7dofarm_QP_solver_ds_cc + 600;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl6 = wam7dofarm_QP_solver_ccrhs + 600;
int wam7dofarm_QP_solver_ubIdx6[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub6 = wam7dofarm_QP_solver_l + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub6 = wam7dofarm_QP_solver_s + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub6 = wam7dofarm_QP_solver_lbys + 664;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub6[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff6 = wam7dofarm_QP_solver_dl_aff + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff6 = wam7dofarm_QP_solver_ds_aff + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc6 = wam7dofarm_QP_solver_dl_cc + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc6 = wam7dofarm_QP_solver_ds_cc + 664;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub6 = wam7dofarm_QP_solver_ccrhs + 664;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi6[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W6[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd6[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd6[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z7 = wam7dofarm_QP_solver_z + 448;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff7 = wam7dofarm_QP_solver_dz_aff + 448;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc7 = wam7dofarm_QP_solver_dz_cc + 448;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd7 = wam7dofarm_QP_solver_rd + 448;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd7[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost7 = wam7dofarm_QP_solver_grad_cost + 448;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq7 = wam7dofarm_QP_solver_grad_eq + 448;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq7 = wam7dofarm_QP_solver_grad_ineq + 448;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv7[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v7 = wam7dofarm_QP_solver_v + 119;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re7[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta7[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc7[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff7 = wam7dofarm_QP_solver_dv_aff + 119;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc7 = wam7dofarm_QP_solver_dv_cc + 119;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V7[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd7[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld7[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy7[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy7[15];
int wam7dofarm_QP_solver_lbIdx7[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb7 = wam7dofarm_QP_solver_l + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb7 = wam7dofarm_QP_solver_s + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb7 = wam7dofarm_QP_solver_lbys + 700;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb7[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff7 = wam7dofarm_QP_solver_dl_aff + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff7 = wam7dofarm_QP_solver_ds_aff + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc7 = wam7dofarm_QP_solver_dl_cc + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc7 = wam7dofarm_QP_solver_ds_cc + 700;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl7 = wam7dofarm_QP_solver_ccrhs + 700;
int wam7dofarm_QP_solver_ubIdx7[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub7 = wam7dofarm_QP_solver_l + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub7 = wam7dofarm_QP_solver_s + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub7 = wam7dofarm_QP_solver_lbys + 764;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub7[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff7 = wam7dofarm_QP_solver_dl_aff + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff7 = wam7dofarm_QP_solver_ds_aff + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc7 = wam7dofarm_QP_solver_dl_cc + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc7 = wam7dofarm_QP_solver_ds_cc + 764;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub7 = wam7dofarm_QP_solver_ccrhs + 764;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi7[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W7[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd7[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd7[225];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z8 = wam7dofarm_QP_solver_z + 512;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff8 = wam7dofarm_QP_solver_dz_aff + 512;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc8 = wam7dofarm_QP_solver_dz_cc + 512;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd8 = wam7dofarm_QP_solver_rd + 512;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd8[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost8 = wam7dofarm_QP_solver_grad_cost + 512;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq8 = wam7dofarm_QP_solver_grad_eq + 512;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq8 = wam7dofarm_QP_solver_grad_ineq + 512;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv8[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_v8 = wam7dofarm_QP_solver_v + 134;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_re8[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_beta8[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_betacc8[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvaff8 = wam7dofarm_QP_solver_dv_aff + 134;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dvcc8 = wam7dofarm_QP_solver_dv_cc + 134;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_V8[960];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Yd8[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ld8[120];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_yy8[15];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_bmy8[15];
int wam7dofarm_QP_solver_lbIdx8[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb8 = wam7dofarm_QP_solver_l + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb8 = wam7dofarm_QP_solver_s + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb8 = wam7dofarm_QP_solver_lbys + 800;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb8[64];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff8 = wam7dofarm_QP_solver_dl_aff + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff8 = wam7dofarm_QP_solver_ds_aff + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc8 = wam7dofarm_QP_solver_dl_cc + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc8 = wam7dofarm_QP_solver_ds_cc + 800;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl8 = wam7dofarm_QP_solver_ccrhs + 800;
int wam7dofarm_QP_solver_ubIdx8[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub8 = wam7dofarm_QP_solver_l + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub8 = wam7dofarm_QP_solver_s + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub8 = wam7dofarm_QP_solver_lbys + 864;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub8[36];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff8 = wam7dofarm_QP_solver_dl_aff + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff8 = wam7dofarm_QP_solver_ds_aff + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc8 = wam7dofarm_QP_solver_dl_cc + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc8 = wam7dofarm_QP_solver_ds_cc + 864;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub8 = wam7dofarm_QP_solver_ccrhs + 864;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi8[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W8[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Ysd8[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lsd8[225];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_H9[27] = {0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000, 0.0000000000000000E+000};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_z9 = wam7dofarm_QP_solver_z + 576;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzaff9 = wam7dofarm_QP_solver_dz_aff + 576;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dzcc9 = wam7dofarm_QP_solver_dz_cc + 576;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_rd9 = wam7dofarm_QP_solver_rd + 576;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Lbyrd9[27];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_cost9 = wam7dofarm_QP_solver_grad_cost + 576;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_eq9 = wam7dofarm_QP_solver_grad_eq + 576;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_grad_ineq9 = wam7dofarm_QP_solver_grad_ineq + 576;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_ctv9[27];
int wam7dofarm_QP_solver_lbIdx9[27] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llb9 = wam7dofarm_QP_solver_l + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_slb9 = wam7dofarm_QP_solver_s + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_llbbyslb9 = wam7dofarm_QP_solver_lbys + 900;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rilb9[27];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbaff9 = wam7dofarm_QP_solver_dl_aff + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbaff9 = wam7dofarm_QP_solver_ds_aff + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dllbcc9 = wam7dofarm_QP_solver_dl_cc + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dslbcc9 = wam7dofarm_QP_solver_ds_cc + 900;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsl9 = wam7dofarm_QP_solver_ccrhs + 900;
int wam7dofarm_QP_solver_ubIdx9[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lub9 = wam7dofarm_QP_solver_l + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sub9 = wam7dofarm_QP_solver_s + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lubbysub9 = wam7dofarm_QP_solver_lbys + 927;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_riub9[15];
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubaff9 = wam7dofarm_QP_solver_dl_aff + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubaff9 = wam7dofarm_QP_solver_ds_aff + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlubcc9 = wam7dofarm_QP_solver_dl_cc + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsubcc9 = wam7dofarm_QP_solver_ds_cc + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsub9 = wam7dofarm_QP_solver_ccrhs + 927;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_sp9 = wam7dofarm_QP_solver_s + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lp9 = wam7dofarm_QP_solver_l + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_lpbysp9 = wam7dofarm_QP_solver_lbys + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlp_aff9 = wam7dofarm_QP_solver_dl_aff + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsp_aff9 = wam7dofarm_QP_solver_ds_aff + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dlp_cc9 = wam7dofarm_QP_solver_dl_cc + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_dsp_cc9 = wam7dofarm_QP_solver_ds_cc + 942;
wam7dofarm_QP_solver_FLOAT* wam7dofarm_QP_solver_ccrhsp9 = wam7dofarm_QP_solver_ccrhs + 942;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_rip9[12];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Phi9[378];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_D9[27] = {-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000, 
-1.0000000000000000E+000};
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_W9[405];
wam7dofarm_QP_solver_FLOAT musigma;
wam7dofarm_QP_solver_FLOAT sigma_3rdroot;
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Diag1_0[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_Diag2_0[64];
wam7dofarm_QP_solver_FLOAT wam7dofarm_QP_solver_L_0[2016];




/* SOLVER CODE --------------------------------------------------------- */
int wam7dofarm_QP_solver_solve(wam7dofarm_QP_solver_params* params, wam7dofarm_QP_solver_output* output, wam7dofarm_QP_solver_info* info)
{	
int exitcode;

#if wam7dofarm_QP_solver_SET_TIMING == 1
	wam7dofarm_QP_solver_timer solvertimer;
	wam7dofarm_QP_solver_tic(&solvertimer);
#endif
/* FUNCTION CALLS INTO LA LIBRARY -------------------------------------- */
info->it = 0;
wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_603(wam7dofarm_QP_solver_z, 0);
wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_149(wam7dofarm_QP_solver_v, 1);
wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_954(wam7dofarm_QP_solver_l, 10);
wam7dofarm_QP_solver_LA_INITIALIZEVECTOR_954(wam7dofarm_QP_solver_s, 10);
info->mu = 0;
wam7dofarm_QP_solver_LA_DOTACC_954(wam7dofarm_QP_solver_l, wam7dofarm_QP_solver_s, &info->mu);
info->mu /= 954;
PRINTTEXT("This is wam7dofarm_QP_solver, a solver generated by FORCES (forces.ethz.ch).\n");
PRINTTEXT("(c) Alexander Domahidi, Automatic Control Laboratory, ETH Zurich, 2011-2014.\n");
PRINTTEXT("\n  #it  res_eq   res_ineq     pobj         dobj       dgap     rdgap     mu\n");
PRINTTEXT("  ---------------------------------------------------------------------------\n");
while( 1 ){
info->pobj = 0;
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f1, wam7dofarm_QP_solver_z0, wam7dofarm_QP_solver_grad_cost0, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f2, wam7dofarm_QP_solver_z1, wam7dofarm_QP_solver_grad_cost1, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f3, wam7dofarm_QP_solver_z2, wam7dofarm_QP_solver_grad_cost2, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f4, wam7dofarm_QP_solver_z3, wam7dofarm_QP_solver_grad_cost3, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f5, wam7dofarm_QP_solver_z4, wam7dofarm_QP_solver_grad_cost4, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f6, wam7dofarm_QP_solver_z5, wam7dofarm_QP_solver_grad_cost5, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f7, wam7dofarm_QP_solver_z6, wam7dofarm_QP_solver_grad_cost6, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f8, wam7dofarm_QP_solver_z7, wam7dofarm_QP_solver_grad_cost7, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_64(wam7dofarm_QP_solver_H0, params->f9, wam7dofarm_QP_solver_z8, wam7dofarm_QP_solver_grad_cost8, &info->pobj);
wam7dofarm_QP_solver_LA_DIAG_QUADFCN_27(wam7dofarm_QP_solver_H9, params->f10, wam7dofarm_QP_solver_z9, wam7dofarm_QP_solver_grad_cost9, &info->pobj);
info->res_eq = 0;
info->dgap = 0;
wam7dofarm_QP_solver_LA_DENSE_MVMSUB3_29_64_64(params->C1, wam7dofarm_QP_solver_z0, wam7dofarm_QP_solver_D1, wam7dofarm_QP_solver_z1, params->e1, wam7dofarm_QP_solver_v0, wam7dofarm_QP_solver_re0, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C2, wam7dofarm_QP_solver_z1, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z2, params->e2, wam7dofarm_QP_solver_v1, wam7dofarm_QP_solver_re1, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C3, wam7dofarm_QP_solver_z2, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z3, params->e3, wam7dofarm_QP_solver_v2, wam7dofarm_QP_solver_re2, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C4, wam7dofarm_QP_solver_z3, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z4, params->e4, wam7dofarm_QP_solver_v3, wam7dofarm_QP_solver_re3, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C5, wam7dofarm_QP_solver_z4, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z5, params->e5, wam7dofarm_QP_solver_v4, wam7dofarm_QP_solver_re4, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C6, wam7dofarm_QP_solver_z5, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z6, params->e6, wam7dofarm_QP_solver_v5, wam7dofarm_QP_solver_re5, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C7, wam7dofarm_QP_solver_z6, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z7, params->e7, wam7dofarm_QP_solver_v6, wam7dofarm_QP_solver_re6, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_64(params->C8, wam7dofarm_QP_solver_z7, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_z8, params->e8, wam7dofarm_QP_solver_v7, wam7dofarm_QP_solver_re7, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MVMSUB3_15_64_27(params->C9, wam7dofarm_QP_solver_z8, wam7dofarm_QP_solver_D9, wam7dofarm_QP_solver_z9, params->e9, wam7dofarm_QP_solver_v8, wam7dofarm_QP_solver_re8, &info->dgap, &info->res_eq);
wam7dofarm_QP_solver_LA_DENSE_MTVM_29_64(params->C1, wam7dofarm_QP_solver_v0, wam7dofarm_QP_solver_grad_eq0);
wam7dofarm_QP_solver_LA_DENSE_MTVM2_15_64_29(params->C2, wam7dofarm_QP_solver_v1, wam7dofarm_QP_solver_D1, wam7dofarm_QP_solver_v0, wam7dofarm_QP_solver_grad_eq1);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C3, wam7dofarm_QP_solver_v2, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v1, wam7dofarm_QP_solver_grad_eq2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C4, wam7dofarm_QP_solver_v3, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v2, wam7dofarm_QP_solver_grad_eq3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C5, wam7dofarm_QP_solver_v4, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v3, wam7dofarm_QP_solver_grad_eq4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C6, wam7dofarm_QP_solver_v5, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v4, wam7dofarm_QP_solver_grad_eq5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C7, wam7dofarm_QP_solver_v6, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v5, wam7dofarm_QP_solver_grad_eq6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C8, wam7dofarm_QP_solver_v7, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v6, wam7dofarm_QP_solver_grad_eq7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C9, wam7dofarm_QP_solver_v8, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_v7, wam7dofarm_QP_solver_grad_eq8);
wam7dofarm_QP_solver_LA_DIAGZERO_MTVM_15_27(wam7dofarm_QP_solver_D9, wam7dofarm_QP_solver_v8, wam7dofarm_QP_solver_grad_eq9);
info->res_ineq = 0;
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb1, wam7dofarm_QP_solver_z0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_llb0, wam7dofarm_QP_solver_slb0, wam7dofarm_QP_solver_rilb0, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z0, wam7dofarm_QP_solver_ubIdx0, params->ub1, wam7dofarm_QP_solver_lub0, wam7dofarm_QP_solver_sub0, wam7dofarm_QP_solver_riub0, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb2, wam7dofarm_QP_solver_z1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_llb1, wam7dofarm_QP_solver_slb1, wam7dofarm_QP_solver_rilb1, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z1, wam7dofarm_QP_solver_ubIdx1, params->ub2, wam7dofarm_QP_solver_lub1, wam7dofarm_QP_solver_sub1, wam7dofarm_QP_solver_riub1, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb3, wam7dofarm_QP_solver_z2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_llb2, wam7dofarm_QP_solver_slb2, wam7dofarm_QP_solver_rilb2, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z2, wam7dofarm_QP_solver_ubIdx2, params->ub3, wam7dofarm_QP_solver_lub2, wam7dofarm_QP_solver_sub2, wam7dofarm_QP_solver_riub2, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb4, wam7dofarm_QP_solver_z3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_llb3, wam7dofarm_QP_solver_slb3, wam7dofarm_QP_solver_rilb3, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z3, wam7dofarm_QP_solver_ubIdx3, params->ub4, wam7dofarm_QP_solver_lub3, wam7dofarm_QP_solver_sub3, wam7dofarm_QP_solver_riub3, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb5, wam7dofarm_QP_solver_z4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_llb4, wam7dofarm_QP_solver_slb4, wam7dofarm_QP_solver_rilb4, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z4, wam7dofarm_QP_solver_ubIdx4, params->ub5, wam7dofarm_QP_solver_lub4, wam7dofarm_QP_solver_sub4, wam7dofarm_QP_solver_riub4, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb6, wam7dofarm_QP_solver_z5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_llb5, wam7dofarm_QP_solver_slb5, wam7dofarm_QP_solver_rilb5, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z5, wam7dofarm_QP_solver_ubIdx5, params->ub6, wam7dofarm_QP_solver_lub5, wam7dofarm_QP_solver_sub5, wam7dofarm_QP_solver_riub5, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb7, wam7dofarm_QP_solver_z6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_llb6, wam7dofarm_QP_solver_slb6, wam7dofarm_QP_solver_rilb6, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z6, wam7dofarm_QP_solver_ubIdx6, params->ub7, wam7dofarm_QP_solver_lub6, wam7dofarm_QP_solver_sub6, wam7dofarm_QP_solver_riub6, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb8, wam7dofarm_QP_solver_z7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_llb7, wam7dofarm_QP_solver_slb7, wam7dofarm_QP_solver_rilb7, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z7, wam7dofarm_QP_solver_ubIdx7, params->ub8, wam7dofarm_QP_solver_lub7, wam7dofarm_QP_solver_sub7, wam7dofarm_QP_solver_riub7, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_64(params->lb9, wam7dofarm_QP_solver_z8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_llb8, wam7dofarm_QP_solver_slb8, wam7dofarm_QP_solver_rilb8, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_36(wam7dofarm_QP_solver_z8, wam7dofarm_QP_solver_ubIdx8, params->ub9, wam7dofarm_QP_solver_lub8, wam7dofarm_QP_solver_sub8, wam7dofarm_QP_solver_riub8, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD3_27(params->lb10, wam7dofarm_QP_solver_z9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_llb9, wam7dofarm_QP_solver_slb9, wam7dofarm_QP_solver_rilb9, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_VSUBADD2_15(wam7dofarm_QP_solver_z9, wam7dofarm_QP_solver_ubIdx9, params->ub10, wam7dofarm_QP_solver_lub9, wam7dofarm_QP_solver_sub9, wam7dofarm_QP_solver_riub9, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_MVSUBADD_12_27(params->A10, wam7dofarm_QP_solver_z9, params->b10, wam7dofarm_QP_solver_sp9, wam7dofarm_QP_solver_lp9, wam7dofarm_QP_solver_rip9, &info->dgap, &info->res_ineq);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub0, wam7dofarm_QP_solver_sub0, wam7dofarm_QP_solver_riub0, wam7dofarm_QP_solver_llb0, wam7dofarm_QP_solver_slb0, wam7dofarm_QP_solver_rilb0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_ubIdx0, wam7dofarm_QP_solver_grad_ineq0, wam7dofarm_QP_solver_lubbysub0, wam7dofarm_QP_solver_llbbyslb0);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub1, wam7dofarm_QP_solver_sub1, wam7dofarm_QP_solver_riub1, wam7dofarm_QP_solver_llb1, wam7dofarm_QP_solver_slb1, wam7dofarm_QP_solver_rilb1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_ubIdx1, wam7dofarm_QP_solver_grad_ineq1, wam7dofarm_QP_solver_lubbysub1, wam7dofarm_QP_solver_llbbyslb1);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub2, wam7dofarm_QP_solver_sub2, wam7dofarm_QP_solver_riub2, wam7dofarm_QP_solver_llb2, wam7dofarm_QP_solver_slb2, wam7dofarm_QP_solver_rilb2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_ubIdx2, wam7dofarm_QP_solver_grad_ineq2, wam7dofarm_QP_solver_lubbysub2, wam7dofarm_QP_solver_llbbyslb2);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub3, wam7dofarm_QP_solver_sub3, wam7dofarm_QP_solver_riub3, wam7dofarm_QP_solver_llb3, wam7dofarm_QP_solver_slb3, wam7dofarm_QP_solver_rilb3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_ubIdx3, wam7dofarm_QP_solver_grad_ineq3, wam7dofarm_QP_solver_lubbysub3, wam7dofarm_QP_solver_llbbyslb3);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub4, wam7dofarm_QP_solver_sub4, wam7dofarm_QP_solver_riub4, wam7dofarm_QP_solver_llb4, wam7dofarm_QP_solver_slb4, wam7dofarm_QP_solver_rilb4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_ubIdx4, wam7dofarm_QP_solver_grad_ineq4, wam7dofarm_QP_solver_lubbysub4, wam7dofarm_QP_solver_llbbyslb4);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub5, wam7dofarm_QP_solver_sub5, wam7dofarm_QP_solver_riub5, wam7dofarm_QP_solver_llb5, wam7dofarm_QP_solver_slb5, wam7dofarm_QP_solver_rilb5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_ubIdx5, wam7dofarm_QP_solver_grad_ineq5, wam7dofarm_QP_solver_lubbysub5, wam7dofarm_QP_solver_llbbyslb5);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub6, wam7dofarm_QP_solver_sub6, wam7dofarm_QP_solver_riub6, wam7dofarm_QP_solver_llb6, wam7dofarm_QP_solver_slb6, wam7dofarm_QP_solver_rilb6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_ubIdx6, wam7dofarm_QP_solver_grad_ineq6, wam7dofarm_QP_solver_lubbysub6, wam7dofarm_QP_solver_llbbyslb6);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub7, wam7dofarm_QP_solver_sub7, wam7dofarm_QP_solver_riub7, wam7dofarm_QP_solver_llb7, wam7dofarm_QP_solver_slb7, wam7dofarm_QP_solver_rilb7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_ubIdx7, wam7dofarm_QP_solver_grad_ineq7, wam7dofarm_QP_solver_lubbysub7, wam7dofarm_QP_solver_llbbyslb7);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_64_64_36(wam7dofarm_QP_solver_lub8, wam7dofarm_QP_solver_sub8, wam7dofarm_QP_solver_riub8, wam7dofarm_QP_solver_llb8, wam7dofarm_QP_solver_slb8, wam7dofarm_QP_solver_rilb8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_ubIdx8, wam7dofarm_QP_solver_grad_ineq8, wam7dofarm_QP_solver_lubbysub8, wam7dofarm_QP_solver_llbbyslb8);
wam7dofarm_QP_solver_LA_INEQ_B_GRAD_27_27_15(wam7dofarm_QP_solver_lub9, wam7dofarm_QP_solver_sub9, wam7dofarm_QP_solver_riub9, wam7dofarm_QP_solver_llb9, wam7dofarm_QP_solver_slb9, wam7dofarm_QP_solver_rilb9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_ubIdx9, wam7dofarm_QP_solver_grad_ineq9, wam7dofarm_QP_solver_lubbysub9, wam7dofarm_QP_solver_llbbyslb9);
wam7dofarm_QP_solver_LA_INEQ_P_12_27(params->A10, wam7dofarm_QP_solver_lp9, wam7dofarm_QP_solver_sp9, wam7dofarm_QP_solver_rip9, wam7dofarm_QP_solver_grad_ineq9, wam7dofarm_QP_solver_lpbysp9);
info->dobj = info->pobj - info->dgap;
info->rdgap = info->pobj ? info->dgap / info->pobj : 1e6;
if( info->rdgap < 0 ) info->rdgap = -info->rdgap;
PRINTTEXT("  %3d  %3.1e  %3.1e  %+6.4e  %+6.4e  %+3.1e  %3.1e  %3.1e\n",info->it, info->res_eq, info->res_ineq, info->pobj, info->dobj, info->dgap, info->rdgap, info->mu);
if( info->mu < wam7dofarm_QP_solver_SET_ACC_KKTCOMPL
    && (info->rdgap < wam7dofarm_QP_solver_SET_ACC_RDGAP || info->dgap < wam7dofarm_QP_solver_SET_ACC_KKTCOMPL)
    && info->res_eq < wam7dofarm_QP_solver_SET_ACC_RESEQ
    && info->res_ineq < wam7dofarm_QP_solver_SET_ACC_RESINEQ ){
PRINTTEXT("OPTIMAL (within RESEQ=%2.1e, RESINEQ=%2.1e, (R)DGAP=(%2.1e)%2.1e, MU=%2.1e).\n",wam7dofarm_QP_solver_SET_ACC_RESEQ, wam7dofarm_QP_solver_SET_ACC_RESINEQ,wam7dofarm_QP_solver_SET_ACC_KKTCOMPL,wam7dofarm_QP_solver_SET_ACC_RDGAP,wam7dofarm_QP_solver_SET_ACC_KKTCOMPL);
exitcode = wam7dofarm_QP_solver_OPTIMAL; break; }
if( info->it == wam7dofarm_QP_solver_SET_MAXIT ){
PRINTTEXT("Maximum number of iterations reached, exiting.\n");
exitcode = wam7dofarm_QP_solver_MAXITREACHED; break; }
wam7dofarm_QP_solver_LA_VVADD3_603(wam7dofarm_QP_solver_grad_cost, wam7dofarm_QP_solver_grad_eq, wam7dofarm_QP_solver_grad_ineq, wam7dofarm_QP_solver_rd);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_lubbysub0, wam7dofarm_QP_solver_ubIdx0, wam7dofarm_QP_solver_Phi0);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_29_64(wam7dofarm_QP_solver_Phi0, params->C1, wam7dofarm_QP_solver_V0);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi0, wam7dofarm_QP_solver_rd0, wam7dofarm_QP_solver_Lbyrd0);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_lubbysub1, wam7dofarm_QP_solver_ubIdx1, wam7dofarm_QP_solver_Phi1);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi1, params->C2, wam7dofarm_QP_solver_V1);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_29_64(wam7dofarm_QP_solver_Phi1, wam7dofarm_QP_solver_D1, wam7dofarm_QP_solver_W1);
wam7dofarm_QP_solver_LA_DENSE_MMTM_29_64_15(wam7dofarm_QP_solver_W1, wam7dofarm_QP_solver_V1, wam7dofarm_QP_solver_Ysd1);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi1, wam7dofarm_QP_solver_rd1, wam7dofarm_QP_solver_Lbyrd1);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_lubbysub2, wam7dofarm_QP_solver_ubIdx2, wam7dofarm_QP_solver_Phi2);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi2, params->C3, wam7dofarm_QP_solver_V2);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi2, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W2, wam7dofarm_QP_solver_V2, wam7dofarm_QP_solver_Ysd2);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi2, wam7dofarm_QP_solver_rd2, wam7dofarm_QP_solver_Lbyrd2);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_lubbysub3, wam7dofarm_QP_solver_ubIdx3, wam7dofarm_QP_solver_Phi3);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi3, params->C4, wam7dofarm_QP_solver_V3);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi3, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W3, wam7dofarm_QP_solver_V3, wam7dofarm_QP_solver_Ysd3);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi3, wam7dofarm_QP_solver_rd3, wam7dofarm_QP_solver_Lbyrd3);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_lubbysub4, wam7dofarm_QP_solver_ubIdx4, wam7dofarm_QP_solver_Phi4);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi4, params->C5, wam7dofarm_QP_solver_V4);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi4, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W4, wam7dofarm_QP_solver_V4, wam7dofarm_QP_solver_Ysd4);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi4, wam7dofarm_QP_solver_rd4, wam7dofarm_QP_solver_Lbyrd4);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_lubbysub5, wam7dofarm_QP_solver_ubIdx5, wam7dofarm_QP_solver_Phi5);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi5, params->C6, wam7dofarm_QP_solver_V5);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi5, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W5, wam7dofarm_QP_solver_V5, wam7dofarm_QP_solver_Ysd5);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi5, wam7dofarm_QP_solver_rd5, wam7dofarm_QP_solver_Lbyrd5);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_lubbysub6, wam7dofarm_QP_solver_ubIdx6, wam7dofarm_QP_solver_Phi6);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi6, params->C7, wam7dofarm_QP_solver_V6);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi6, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W6, wam7dofarm_QP_solver_V6, wam7dofarm_QP_solver_Ysd6);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi6, wam7dofarm_QP_solver_rd6, wam7dofarm_QP_solver_Lbyrd6);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_lubbysub7, wam7dofarm_QP_solver_ubIdx7, wam7dofarm_QP_solver_Phi7);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi7, params->C8, wam7dofarm_QP_solver_V7);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi7, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W7, wam7dofarm_QP_solver_V7, wam7dofarm_QP_solver_Ysd7);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi7, wam7dofarm_QP_solver_rd7, wam7dofarm_QP_solver_Lbyrd7);
wam7dofarm_QP_solver_LA_DIAG_CHOL_LBUB_64_64_36(wam7dofarm_QP_solver_H0, wam7dofarm_QP_solver_llbbyslb8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_lubbysub8, wam7dofarm_QP_solver_ubIdx8, wam7dofarm_QP_solver_Phi8);
wam7dofarm_QP_solver_LA_DIAG_MATRIXFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi8, params->C9, wam7dofarm_QP_solver_V8);
wam7dofarm_QP_solver_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_15_64(wam7dofarm_QP_solver_Phi8, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_W8);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMTM_15_64_15(wam7dofarm_QP_solver_W8, wam7dofarm_QP_solver_V8, wam7dofarm_QP_solver_Ysd8);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi8, wam7dofarm_QP_solver_rd8, wam7dofarm_QP_solver_Lbyrd8);
wam7dofarm_QP_solver_LA_INEQ_DENSE_DIAG_HESS_27_27_15(wam7dofarm_QP_solver_H9, wam7dofarm_QP_solver_llbbyslb9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_lubbysub9, wam7dofarm_QP_solver_ubIdx9, wam7dofarm_QP_solver_Phi9);
wam7dofarm_QP_solver_LA_DENSE_ADDMTDM_12_27(params->A10, wam7dofarm_QP_solver_lpbysp9, wam7dofarm_QP_solver_Phi9);
wam7dofarm_QP_solver_LA_DENSE_CHOL2_27(wam7dofarm_QP_solver_Phi9);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MATRIXFORWARDSUB_15_27(wam7dofarm_QP_solver_Phi9, wam7dofarm_QP_solver_D9, wam7dofarm_QP_solver_W9);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_27(wam7dofarm_QP_solver_Phi9, wam7dofarm_QP_solver_rd9, wam7dofarm_QP_solver_Lbyrd9);
wam7dofarm_QP_solver_LA_DENSE_MMT2_29_64_64(wam7dofarm_QP_solver_V0, wam7dofarm_QP_solver_W1, wam7dofarm_QP_solver_Yd0);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB2_29_64_64(wam7dofarm_QP_solver_V0, wam7dofarm_QP_solver_Lbyrd0, wam7dofarm_QP_solver_W1, wam7dofarm_QP_solver_Lbyrd1, wam7dofarm_QP_solver_re0, wam7dofarm_QP_solver_beta0);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V1, wam7dofarm_QP_solver_W2, wam7dofarm_QP_solver_Yd1);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V1, wam7dofarm_QP_solver_Lbyrd1, wam7dofarm_QP_solver_W2, wam7dofarm_QP_solver_Lbyrd2, wam7dofarm_QP_solver_re1, wam7dofarm_QP_solver_beta1);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V2, wam7dofarm_QP_solver_W3, wam7dofarm_QP_solver_Yd2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V2, wam7dofarm_QP_solver_Lbyrd2, wam7dofarm_QP_solver_W3, wam7dofarm_QP_solver_Lbyrd3, wam7dofarm_QP_solver_re2, wam7dofarm_QP_solver_beta2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V3, wam7dofarm_QP_solver_W4, wam7dofarm_QP_solver_Yd3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V3, wam7dofarm_QP_solver_Lbyrd3, wam7dofarm_QP_solver_W4, wam7dofarm_QP_solver_Lbyrd4, wam7dofarm_QP_solver_re3, wam7dofarm_QP_solver_beta3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V4, wam7dofarm_QP_solver_W5, wam7dofarm_QP_solver_Yd4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V4, wam7dofarm_QP_solver_Lbyrd4, wam7dofarm_QP_solver_W5, wam7dofarm_QP_solver_Lbyrd5, wam7dofarm_QP_solver_re4, wam7dofarm_QP_solver_beta4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V5, wam7dofarm_QP_solver_W6, wam7dofarm_QP_solver_Yd5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V5, wam7dofarm_QP_solver_Lbyrd5, wam7dofarm_QP_solver_W6, wam7dofarm_QP_solver_Lbyrd6, wam7dofarm_QP_solver_re5, wam7dofarm_QP_solver_beta5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V6, wam7dofarm_QP_solver_W7, wam7dofarm_QP_solver_Yd6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V6, wam7dofarm_QP_solver_Lbyrd6, wam7dofarm_QP_solver_W7, wam7dofarm_QP_solver_Lbyrd7, wam7dofarm_QP_solver_re6, wam7dofarm_QP_solver_beta6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MMT2_15_64_64(wam7dofarm_QP_solver_V7, wam7dofarm_QP_solver_W8, wam7dofarm_QP_solver_Yd7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMSUB2_15_64_64(wam7dofarm_QP_solver_V7, wam7dofarm_QP_solver_Lbyrd7, wam7dofarm_QP_solver_W8, wam7dofarm_QP_solver_Lbyrd8, wam7dofarm_QP_solver_re7, wam7dofarm_QP_solver_beta7);
wam7dofarm_QP_solver_LA_DENSE_MMT2_15_64_27(wam7dofarm_QP_solver_V8, wam7dofarm_QP_solver_W9, wam7dofarm_QP_solver_Yd8);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB2_15_64_27(wam7dofarm_QP_solver_V8, wam7dofarm_QP_solver_Lbyrd8, wam7dofarm_QP_solver_W9, wam7dofarm_QP_solver_Lbyrd9, wam7dofarm_QP_solver_re8, wam7dofarm_QP_solver_beta8);
wam7dofarm_QP_solver_LA_DENSE_CHOL_29(wam7dofarm_QP_solver_Yd0, wam7dofarm_QP_solver_Ld0);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_29(wam7dofarm_QP_solver_Ld0, wam7dofarm_QP_solver_beta0, wam7dofarm_QP_solver_yy0);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_29(wam7dofarm_QP_solver_Ld0, wam7dofarm_QP_solver_Ysd1, wam7dofarm_QP_solver_Lsd1);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_29(wam7dofarm_QP_solver_Lsd1, wam7dofarm_QP_solver_Yd1);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd1, wam7dofarm_QP_solver_Ld1);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_29(wam7dofarm_QP_solver_Lsd1, wam7dofarm_QP_solver_yy0, wam7dofarm_QP_solver_beta1, wam7dofarm_QP_solver_bmy1);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld1, wam7dofarm_QP_solver_bmy1, wam7dofarm_QP_solver_yy1);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld1, wam7dofarm_QP_solver_Ysd2, wam7dofarm_QP_solver_Lsd2);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd2, wam7dofarm_QP_solver_Yd2);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd2, wam7dofarm_QP_solver_Ld2);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd2, wam7dofarm_QP_solver_yy1, wam7dofarm_QP_solver_beta2, wam7dofarm_QP_solver_bmy2);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld2, wam7dofarm_QP_solver_bmy2, wam7dofarm_QP_solver_yy2);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld2, wam7dofarm_QP_solver_Ysd3, wam7dofarm_QP_solver_Lsd3);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd3, wam7dofarm_QP_solver_Yd3);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd3, wam7dofarm_QP_solver_Ld3);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd3, wam7dofarm_QP_solver_yy2, wam7dofarm_QP_solver_beta3, wam7dofarm_QP_solver_bmy3);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld3, wam7dofarm_QP_solver_bmy3, wam7dofarm_QP_solver_yy3);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld3, wam7dofarm_QP_solver_Ysd4, wam7dofarm_QP_solver_Lsd4);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd4, wam7dofarm_QP_solver_Yd4);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd4, wam7dofarm_QP_solver_Ld4);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd4, wam7dofarm_QP_solver_yy3, wam7dofarm_QP_solver_beta4, wam7dofarm_QP_solver_bmy4);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld4, wam7dofarm_QP_solver_bmy4, wam7dofarm_QP_solver_yy4);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld4, wam7dofarm_QP_solver_Ysd5, wam7dofarm_QP_solver_Lsd5);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd5, wam7dofarm_QP_solver_Yd5);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd5, wam7dofarm_QP_solver_Ld5);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd5, wam7dofarm_QP_solver_yy4, wam7dofarm_QP_solver_beta5, wam7dofarm_QP_solver_bmy5);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld5, wam7dofarm_QP_solver_bmy5, wam7dofarm_QP_solver_yy5);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld5, wam7dofarm_QP_solver_Ysd6, wam7dofarm_QP_solver_Lsd6);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd6, wam7dofarm_QP_solver_Yd6);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd6, wam7dofarm_QP_solver_Ld6);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd6, wam7dofarm_QP_solver_yy5, wam7dofarm_QP_solver_beta6, wam7dofarm_QP_solver_bmy6);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld6, wam7dofarm_QP_solver_bmy6, wam7dofarm_QP_solver_yy6);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld6, wam7dofarm_QP_solver_Ysd7, wam7dofarm_QP_solver_Lsd7);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd7, wam7dofarm_QP_solver_Yd7);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd7, wam7dofarm_QP_solver_Ld7);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd7, wam7dofarm_QP_solver_yy6, wam7dofarm_QP_solver_beta7, wam7dofarm_QP_solver_bmy7);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld7, wam7dofarm_QP_solver_bmy7, wam7dofarm_QP_solver_yy7);
wam7dofarm_QP_solver_LA_DENSE_MATRIXTFORWARDSUB_15_15(wam7dofarm_QP_solver_Ld7, wam7dofarm_QP_solver_Ysd8, wam7dofarm_QP_solver_Lsd8);
wam7dofarm_QP_solver_LA_DENSE_MMTSUB_15_15(wam7dofarm_QP_solver_Lsd8, wam7dofarm_QP_solver_Yd8);
wam7dofarm_QP_solver_LA_DENSE_CHOL_15(wam7dofarm_QP_solver_Yd8, wam7dofarm_QP_solver_Ld8);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd8, wam7dofarm_QP_solver_yy7, wam7dofarm_QP_solver_beta8, wam7dofarm_QP_solver_bmy8);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld8, wam7dofarm_QP_solver_bmy8, wam7dofarm_QP_solver_yy8);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld8, wam7dofarm_QP_solver_yy8, wam7dofarm_QP_solver_dvaff8);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd8, wam7dofarm_QP_solver_dvaff8, wam7dofarm_QP_solver_yy7, wam7dofarm_QP_solver_bmy7);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld7, wam7dofarm_QP_solver_bmy7, wam7dofarm_QP_solver_dvaff7);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd7, wam7dofarm_QP_solver_dvaff7, wam7dofarm_QP_solver_yy6, wam7dofarm_QP_solver_bmy6);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld6, wam7dofarm_QP_solver_bmy6, wam7dofarm_QP_solver_dvaff6);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd6, wam7dofarm_QP_solver_dvaff6, wam7dofarm_QP_solver_yy5, wam7dofarm_QP_solver_bmy5);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld5, wam7dofarm_QP_solver_bmy5, wam7dofarm_QP_solver_dvaff5);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd5, wam7dofarm_QP_solver_dvaff5, wam7dofarm_QP_solver_yy4, wam7dofarm_QP_solver_bmy4);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld4, wam7dofarm_QP_solver_bmy4, wam7dofarm_QP_solver_dvaff4);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd4, wam7dofarm_QP_solver_dvaff4, wam7dofarm_QP_solver_yy3, wam7dofarm_QP_solver_bmy3);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld3, wam7dofarm_QP_solver_bmy3, wam7dofarm_QP_solver_dvaff3);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd3, wam7dofarm_QP_solver_dvaff3, wam7dofarm_QP_solver_yy2, wam7dofarm_QP_solver_bmy2);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld2, wam7dofarm_QP_solver_bmy2, wam7dofarm_QP_solver_dvaff2);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd2, wam7dofarm_QP_solver_dvaff2, wam7dofarm_QP_solver_yy1, wam7dofarm_QP_solver_bmy1);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld1, wam7dofarm_QP_solver_bmy1, wam7dofarm_QP_solver_dvaff1);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_29(wam7dofarm_QP_solver_Lsd1, wam7dofarm_QP_solver_dvaff1, wam7dofarm_QP_solver_yy0, wam7dofarm_QP_solver_bmy0);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_29(wam7dofarm_QP_solver_Ld0, wam7dofarm_QP_solver_bmy0, wam7dofarm_QP_solver_dvaff0);
wam7dofarm_QP_solver_LA_DENSE_MTVM_29_64(params->C1, wam7dofarm_QP_solver_dvaff0, wam7dofarm_QP_solver_grad_eq0);
wam7dofarm_QP_solver_LA_DENSE_MTVM2_15_64_29(params->C2, wam7dofarm_QP_solver_dvaff1, wam7dofarm_QP_solver_D1, wam7dofarm_QP_solver_dvaff0, wam7dofarm_QP_solver_grad_eq1);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C3, wam7dofarm_QP_solver_dvaff2, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff1, wam7dofarm_QP_solver_grad_eq2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C4, wam7dofarm_QP_solver_dvaff3, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff2, wam7dofarm_QP_solver_grad_eq3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C5, wam7dofarm_QP_solver_dvaff4, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff3, wam7dofarm_QP_solver_grad_eq4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C6, wam7dofarm_QP_solver_dvaff5, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff4, wam7dofarm_QP_solver_grad_eq5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C7, wam7dofarm_QP_solver_dvaff6, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff5, wam7dofarm_QP_solver_grad_eq6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C8, wam7dofarm_QP_solver_dvaff7, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff6, wam7dofarm_QP_solver_grad_eq7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C9, wam7dofarm_QP_solver_dvaff8, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvaff7, wam7dofarm_QP_solver_grad_eq8);
wam7dofarm_QP_solver_LA_DIAGZERO_MTVM_15_27(wam7dofarm_QP_solver_D9, wam7dofarm_QP_solver_dvaff8, wam7dofarm_QP_solver_grad_eq9);
wam7dofarm_QP_solver_LA_VSUB2_603(wam7dofarm_QP_solver_rd, wam7dofarm_QP_solver_grad_eq, wam7dofarm_QP_solver_rd);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi0, wam7dofarm_QP_solver_rd0, wam7dofarm_QP_solver_dzaff0);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi1, wam7dofarm_QP_solver_rd1, wam7dofarm_QP_solver_dzaff1);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi2, wam7dofarm_QP_solver_rd2, wam7dofarm_QP_solver_dzaff2);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi3, wam7dofarm_QP_solver_rd3, wam7dofarm_QP_solver_dzaff3);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi4, wam7dofarm_QP_solver_rd4, wam7dofarm_QP_solver_dzaff4);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi5, wam7dofarm_QP_solver_rd5, wam7dofarm_QP_solver_dzaff5);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi6, wam7dofarm_QP_solver_rd6, wam7dofarm_QP_solver_dzaff6);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi7, wam7dofarm_QP_solver_rd7, wam7dofarm_QP_solver_dzaff7);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi8, wam7dofarm_QP_solver_rd8, wam7dofarm_QP_solver_dzaff8);
wam7dofarm_QP_solver_LA_DENSE_FORWARDBACKWARDSUB_27(wam7dofarm_QP_solver_Phi9, wam7dofarm_QP_solver_rd9, wam7dofarm_QP_solver_dzaff9);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_rilb0, wam7dofarm_QP_solver_dslbaff0);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb0, wam7dofarm_QP_solver_dslbaff0, wam7dofarm_QP_solver_llb0, wam7dofarm_QP_solver_dllbaff0);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub0, wam7dofarm_QP_solver_dzaff0, wam7dofarm_QP_solver_ubIdx0, wam7dofarm_QP_solver_dsubaff0);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub0, wam7dofarm_QP_solver_dsubaff0, wam7dofarm_QP_solver_lub0, wam7dofarm_QP_solver_dlubaff0);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_rilb1, wam7dofarm_QP_solver_dslbaff1);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb1, wam7dofarm_QP_solver_dslbaff1, wam7dofarm_QP_solver_llb1, wam7dofarm_QP_solver_dllbaff1);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub1, wam7dofarm_QP_solver_dzaff1, wam7dofarm_QP_solver_ubIdx1, wam7dofarm_QP_solver_dsubaff1);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub1, wam7dofarm_QP_solver_dsubaff1, wam7dofarm_QP_solver_lub1, wam7dofarm_QP_solver_dlubaff1);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_rilb2, wam7dofarm_QP_solver_dslbaff2);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb2, wam7dofarm_QP_solver_dslbaff2, wam7dofarm_QP_solver_llb2, wam7dofarm_QP_solver_dllbaff2);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub2, wam7dofarm_QP_solver_dzaff2, wam7dofarm_QP_solver_ubIdx2, wam7dofarm_QP_solver_dsubaff2);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub2, wam7dofarm_QP_solver_dsubaff2, wam7dofarm_QP_solver_lub2, wam7dofarm_QP_solver_dlubaff2);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_rilb3, wam7dofarm_QP_solver_dslbaff3);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb3, wam7dofarm_QP_solver_dslbaff3, wam7dofarm_QP_solver_llb3, wam7dofarm_QP_solver_dllbaff3);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub3, wam7dofarm_QP_solver_dzaff3, wam7dofarm_QP_solver_ubIdx3, wam7dofarm_QP_solver_dsubaff3);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub3, wam7dofarm_QP_solver_dsubaff3, wam7dofarm_QP_solver_lub3, wam7dofarm_QP_solver_dlubaff3);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_rilb4, wam7dofarm_QP_solver_dslbaff4);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb4, wam7dofarm_QP_solver_dslbaff4, wam7dofarm_QP_solver_llb4, wam7dofarm_QP_solver_dllbaff4);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub4, wam7dofarm_QP_solver_dzaff4, wam7dofarm_QP_solver_ubIdx4, wam7dofarm_QP_solver_dsubaff4);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub4, wam7dofarm_QP_solver_dsubaff4, wam7dofarm_QP_solver_lub4, wam7dofarm_QP_solver_dlubaff4);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_rilb5, wam7dofarm_QP_solver_dslbaff5);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb5, wam7dofarm_QP_solver_dslbaff5, wam7dofarm_QP_solver_llb5, wam7dofarm_QP_solver_dllbaff5);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub5, wam7dofarm_QP_solver_dzaff5, wam7dofarm_QP_solver_ubIdx5, wam7dofarm_QP_solver_dsubaff5);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub5, wam7dofarm_QP_solver_dsubaff5, wam7dofarm_QP_solver_lub5, wam7dofarm_QP_solver_dlubaff5);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_rilb6, wam7dofarm_QP_solver_dslbaff6);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb6, wam7dofarm_QP_solver_dslbaff6, wam7dofarm_QP_solver_llb6, wam7dofarm_QP_solver_dllbaff6);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub6, wam7dofarm_QP_solver_dzaff6, wam7dofarm_QP_solver_ubIdx6, wam7dofarm_QP_solver_dsubaff6);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub6, wam7dofarm_QP_solver_dsubaff6, wam7dofarm_QP_solver_lub6, wam7dofarm_QP_solver_dlubaff6);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_rilb7, wam7dofarm_QP_solver_dslbaff7);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb7, wam7dofarm_QP_solver_dslbaff7, wam7dofarm_QP_solver_llb7, wam7dofarm_QP_solver_dllbaff7);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub7, wam7dofarm_QP_solver_dzaff7, wam7dofarm_QP_solver_ubIdx7, wam7dofarm_QP_solver_dsubaff7);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub7, wam7dofarm_QP_solver_dsubaff7, wam7dofarm_QP_solver_lub7, wam7dofarm_QP_solver_dlubaff7);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_64(wam7dofarm_QP_solver_dzaff8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_rilb8, wam7dofarm_QP_solver_dslbaff8);
wam7dofarm_QP_solver_LA_VSUB3_64(wam7dofarm_QP_solver_llbbyslb8, wam7dofarm_QP_solver_dslbaff8, wam7dofarm_QP_solver_llb8, wam7dofarm_QP_solver_dllbaff8);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_36(wam7dofarm_QP_solver_riub8, wam7dofarm_QP_solver_dzaff8, wam7dofarm_QP_solver_ubIdx8, wam7dofarm_QP_solver_dsubaff8);
wam7dofarm_QP_solver_LA_VSUB3_36(wam7dofarm_QP_solver_lubbysub8, wam7dofarm_QP_solver_dsubaff8, wam7dofarm_QP_solver_lub8, wam7dofarm_QP_solver_dlubaff8);
wam7dofarm_QP_solver_LA_VSUB_INDEXED_27(wam7dofarm_QP_solver_dzaff9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_rilb9, wam7dofarm_QP_solver_dslbaff9);
wam7dofarm_QP_solver_LA_VSUB3_27(wam7dofarm_QP_solver_llbbyslb9, wam7dofarm_QP_solver_dslbaff9, wam7dofarm_QP_solver_llb9, wam7dofarm_QP_solver_dllbaff9);
wam7dofarm_QP_solver_LA_VSUB2_INDEXED_15(wam7dofarm_QP_solver_riub9, wam7dofarm_QP_solver_dzaff9, wam7dofarm_QP_solver_ubIdx9, wam7dofarm_QP_solver_dsubaff9);
wam7dofarm_QP_solver_LA_VSUB3_15(wam7dofarm_QP_solver_lubbysub9, wam7dofarm_QP_solver_dsubaff9, wam7dofarm_QP_solver_lub9, wam7dofarm_QP_solver_dlubaff9);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB4_12_27(params->A10, wam7dofarm_QP_solver_dzaff9, wam7dofarm_QP_solver_rip9, wam7dofarm_QP_solver_dsp_aff9);
wam7dofarm_QP_solver_LA_VSUB3_12(wam7dofarm_QP_solver_lpbysp9, wam7dofarm_QP_solver_dsp_aff9, wam7dofarm_QP_solver_lp9, wam7dofarm_QP_solver_dlp_aff9);
info->lsit_aff = wam7dofarm_QP_solver_LINESEARCH_BACKTRACKING_AFFINE(wam7dofarm_QP_solver_l, wam7dofarm_QP_solver_s, wam7dofarm_QP_solver_dl_aff, wam7dofarm_QP_solver_ds_aff, &info->step_aff, &info->mu_aff);
if( info->lsit_aff == wam7dofarm_QP_solver_NOPROGRESS ){
PRINTTEXT("Affine line search could not proceed at iteration %d.\nThe problem might be infeasible -- exiting.\n",info->it+1);
exitcode = wam7dofarm_QP_solver_NOPROGRESS; break;
}
sigma_3rdroot = info->mu_aff / info->mu;
info->sigma = sigma_3rdroot*sigma_3rdroot*sigma_3rdroot;
musigma = info->mu * info->sigma;
wam7dofarm_QP_solver_LA_VSUB5_954(wam7dofarm_QP_solver_ds_aff, wam7dofarm_QP_solver_dl_aff, info->mu, info->sigma, wam7dofarm_QP_solver_ccrhs);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub0, wam7dofarm_QP_solver_sub0, wam7dofarm_QP_solver_ubIdx0, wam7dofarm_QP_solver_ccrhsl0, wam7dofarm_QP_solver_slb0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_rd0);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub1, wam7dofarm_QP_solver_sub1, wam7dofarm_QP_solver_ubIdx1, wam7dofarm_QP_solver_ccrhsl1, wam7dofarm_QP_solver_slb1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_rd1);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi0, wam7dofarm_QP_solver_rd0, wam7dofarm_QP_solver_Lbyrd0);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi1, wam7dofarm_QP_solver_rd1, wam7dofarm_QP_solver_Lbyrd1);
wam7dofarm_QP_solver_LA_DENSE_2MVMADD_29_64_64(wam7dofarm_QP_solver_V0, wam7dofarm_QP_solver_Lbyrd0, wam7dofarm_QP_solver_W1, wam7dofarm_QP_solver_Lbyrd1, wam7dofarm_QP_solver_beta0);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_29(wam7dofarm_QP_solver_Ld0, wam7dofarm_QP_solver_beta0, wam7dofarm_QP_solver_yy0);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub2, wam7dofarm_QP_solver_sub2, wam7dofarm_QP_solver_ubIdx2, wam7dofarm_QP_solver_ccrhsl2, wam7dofarm_QP_solver_slb2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_rd2);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi2, wam7dofarm_QP_solver_rd2, wam7dofarm_QP_solver_Lbyrd2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V1, wam7dofarm_QP_solver_Lbyrd1, wam7dofarm_QP_solver_W2, wam7dofarm_QP_solver_Lbyrd2, wam7dofarm_QP_solver_beta1);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_29(wam7dofarm_QP_solver_Lsd1, wam7dofarm_QP_solver_yy0, wam7dofarm_QP_solver_beta1, wam7dofarm_QP_solver_bmy1);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld1, wam7dofarm_QP_solver_bmy1, wam7dofarm_QP_solver_yy1);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub3, wam7dofarm_QP_solver_sub3, wam7dofarm_QP_solver_ubIdx3, wam7dofarm_QP_solver_ccrhsl3, wam7dofarm_QP_solver_slb3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_rd3);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi3, wam7dofarm_QP_solver_rd3, wam7dofarm_QP_solver_Lbyrd3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V2, wam7dofarm_QP_solver_Lbyrd2, wam7dofarm_QP_solver_W3, wam7dofarm_QP_solver_Lbyrd3, wam7dofarm_QP_solver_beta2);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd2, wam7dofarm_QP_solver_yy1, wam7dofarm_QP_solver_beta2, wam7dofarm_QP_solver_bmy2);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld2, wam7dofarm_QP_solver_bmy2, wam7dofarm_QP_solver_yy2);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub4, wam7dofarm_QP_solver_sub4, wam7dofarm_QP_solver_ubIdx4, wam7dofarm_QP_solver_ccrhsl4, wam7dofarm_QP_solver_slb4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_rd4);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi4, wam7dofarm_QP_solver_rd4, wam7dofarm_QP_solver_Lbyrd4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V3, wam7dofarm_QP_solver_Lbyrd3, wam7dofarm_QP_solver_W4, wam7dofarm_QP_solver_Lbyrd4, wam7dofarm_QP_solver_beta3);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd3, wam7dofarm_QP_solver_yy2, wam7dofarm_QP_solver_beta3, wam7dofarm_QP_solver_bmy3);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld3, wam7dofarm_QP_solver_bmy3, wam7dofarm_QP_solver_yy3);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub5, wam7dofarm_QP_solver_sub5, wam7dofarm_QP_solver_ubIdx5, wam7dofarm_QP_solver_ccrhsl5, wam7dofarm_QP_solver_slb5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_rd5);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi5, wam7dofarm_QP_solver_rd5, wam7dofarm_QP_solver_Lbyrd5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V4, wam7dofarm_QP_solver_Lbyrd4, wam7dofarm_QP_solver_W5, wam7dofarm_QP_solver_Lbyrd5, wam7dofarm_QP_solver_beta4);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd4, wam7dofarm_QP_solver_yy3, wam7dofarm_QP_solver_beta4, wam7dofarm_QP_solver_bmy4);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld4, wam7dofarm_QP_solver_bmy4, wam7dofarm_QP_solver_yy4);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub6, wam7dofarm_QP_solver_sub6, wam7dofarm_QP_solver_ubIdx6, wam7dofarm_QP_solver_ccrhsl6, wam7dofarm_QP_solver_slb6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_rd6);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi6, wam7dofarm_QP_solver_rd6, wam7dofarm_QP_solver_Lbyrd6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V5, wam7dofarm_QP_solver_Lbyrd5, wam7dofarm_QP_solver_W6, wam7dofarm_QP_solver_Lbyrd6, wam7dofarm_QP_solver_beta5);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd5, wam7dofarm_QP_solver_yy4, wam7dofarm_QP_solver_beta5, wam7dofarm_QP_solver_bmy5);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld5, wam7dofarm_QP_solver_bmy5, wam7dofarm_QP_solver_yy5);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub7, wam7dofarm_QP_solver_sub7, wam7dofarm_QP_solver_ubIdx7, wam7dofarm_QP_solver_ccrhsl7, wam7dofarm_QP_solver_slb7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_rd7);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi7, wam7dofarm_QP_solver_rd7, wam7dofarm_QP_solver_Lbyrd7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V6, wam7dofarm_QP_solver_Lbyrd6, wam7dofarm_QP_solver_W7, wam7dofarm_QP_solver_Lbyrd7, wam7dofarm_QP_solver_beta6);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd6, wam7dofarm_QP_solver_yy5, wam7dofarm_QP_solver_beta6, wam7dofarm_QP_solver_bmy6);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld6, wam7dofarm_QP_solver_bmy6, wam7dofarm_QP_solver_yy6);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_64_36_64(wam7dofarm_QP_solver_ccrhsub8, wam7dofarm_QP_solver_sub8, wam7dofarm_QP_solver_ubIdx8, wam7dofarm_QP_solver_ccrhsl8, wam7dofarm_QP_solver_slb8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_rd8);
wam7dofarm_QP_solver_LA_DIAG_FORWARDSUB_64(wam7dofarm_QP_solver_Phi8, wam7dofarm_QP_solver_rd8, wam7dofarm_QP_solver_Lbyrd8);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_2MVMADD_15_64_64(wam7dofarm_QP_solver_V7, wam7dofarm_QP_solver_Lbyrd7, wam7dofarm_QP_solver_W8, wam7dofarm_QP_solver_Lbyrd8, wam7dofarm_QP_solver_beta7);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd7, wam7dofarm_QP_solver_yy6, wam7dofarm_QP_solver_beta7, wam7dofarm_QP_solver_bmy7);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld7, wam7dofarm_QP_solver_bmy7, wam7dofarm_QP_solver_yy7);
wam7dofarm_QP_solver_LA_VSUB6_INDEXED_27_15_27(wam7dofarm_QP_solver_ccrhsub9, wam7dofarm_QP_solver_sub9, wam7dofarm_QP_solver_ubIdx9, wam7dofarm_QP_solver_ccrhsl9, wam7dofarm_QP_solver_slb9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_rd9);
wam7dofarm_QP_solver_LA_DENSE_MTVMADD2_12_27(params->A10, wam7dofarm_QP_solver_ccrhsp9, wam7dofarm_QP_solver_sp9, wam7dofarm_QP_solver_rd9);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_27(wam7dofarm_QP_solver_Phi9, wam7dofarm_QP_solver_rd9, wam7dofarm_QP_solver_Lbyrd9);
wam7dofarm_QP_solver_LA_DENSE_2MVMADD_15_64_27(wam7dofarm_QP_solver_V8, wam7dofarm_QP_solver_Lbyrd8, wam7dofarm_QP_solver_W9, wam7dofarm_QP_solver_Lbyrd9, wam7dofarm_QP_solver_beta8);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB1_15_15(wam7dofarm_QP_solver_Lsd8, wam7dofarm_QP_solver_yy7, wam7dofarm_QP_solver_beta8, wam7dofarm_QP_solver_bmy8);
wam7dofarm_QP_solver_LA_DENSE_FORWARDSUB_15(wam7dofarm_QP_solver_Ld8, wam7dofarm_QP_solver_bmy8, wam7dofarm_QP_solver_yy8);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld8, wam7dofarm_QP_solver_yy8, wam7dofarm_QP_solver_dvcc8);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd8, wam7dofarm_QP_solver_dvcc8, wam7dofarm_QP_solver_yy7, wam7dofarm_QP_solver_bmy7);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld7, wam7dofarm_QP_solver_bmy7, wam7dofarm_QP_solver_dvcc7);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd7, wam7dofarm_QP_solver_dvcc7, wam7dofarm_QP_solver_yy6, wam7dofarm_QP_solver_bmy6);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld6, wam7dofarm_QP_solver_bmy6, wam7dofarm_QP_solver_dvcc6);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd6, wam7dofarm_QP_solver_dvcc6, wam7dofarm_QP_solver_yy5, wam7dofarm_QP_solver_bmy5);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld5, wam7dofarm_QP_solver_bmy5, wam7dofarm_QP_solver_dvcc5);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd5, wam7dofarm_QP_solver_dvcc5, wam7dofarm_QP_solver_yy4, wam7dofarm_QP_solver_bmy4);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld4, wam7dofarm_QP_solver_bmy4, wam7dofarm_QP_solver_dvcc4);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd4, wam7dofarm_QP_solver_dvcc4, wam7dofarm_QP_solver_yy3, wam7dofarm_QP_solver_bmy3);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld3, wam7dofarm_QP_solver_bmy3, wam7dofarm_QP_solver_dvcc3);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd3, wam7dofarm_QP_solver_dvcc3, wam7dofarm_QP_solver_yy2, wam7dofarm_QP_solver_bmy2);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld2, wam7dofarm_QP_solver_bmy2, wam7dofarm_QP_solver_dvcc2);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_15(wam7dofarm_QP_solver_Lsd2, wam7dofarm_QP_solver_dvcc2, wam7dofarm_QP_solver_yy1, wam7dofarm_QP_solver_bmy1);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_15(wam7dofarm_QP_solver_Ld1, wam7dofarm_QP_solver_bmy1, wam7dofarm_QP_solver_dvcc1);
wam7dofarm_QP_solver_LA_DENSE_MTVMSUB_15_29(wam7dofarm_QP_solver_Lsd1, wam7dofarm_QP_solver_dvcc1, wam7dofarm_QP_solver_yy0, wam7dofarm_QP_solver_bmy0);
wam7dofarm_QP_solver_LA_DENSE_BACKWARDSUB_29(wam7dofarm_QP_solver_Ld0, wam7dofarm_QP_solver_bmy0, wam7dofarm_QP_solver_dvcc0);
wam7dofarm_QP_solver_LA_DENSE_MTVM_29_64(params->C1, wam7dofarm_QP_solver_dvcc0, wam7dofarm_QP_solver_grad_eq0);
wam7dofarm_QP_solver_LA_DENSE_MTVM2_15_64_29(params->C2, wam7dofarm_QP_solver_dvcc1, wam7dofarm_QP_solver_D1, wam7dofarm_QP_solver_dvcc0, wam7dofarm_QP_solver_grad_eq1);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C3, wam7dofarm_QP_solver_dvcc2, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc1, wam7dofarm_QP_solver_grad_eq2);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C4, wam7dofarm_QP_solver_dvcc3, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc2, wam7dofarm_QP_solver_grad_eq3);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C5, wam7dofarm_QP_solver_dvcc4, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc3, wam7dofarm_QP_solver_grad_eq4);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C6, wam7dofarm_QP_solver_dvcc5, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc4, wam7dofarm_QP_solver_grad_eq5);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C7, wam7dofarm_QP_solver_dvcc6, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc5, wam7dofarm_QP_solver_grad_eq6);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C8, wam7dofarm_QP_solver_dvcc7, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc6, wam7dofarm_QP_solver_grad_eq7);
wam7dofarm_QP_solver_LA_DENSE_DIAGZERO_MTVM2_15_64_15(params->C9, wam7dofarm_QP_solver_dvcc8, wam7dofarm_QP_solver_D2, wam7dofarm_QP_solver_dvcc7, wam7dofarm_QP_solver_grad_eq8);
wam7dofarm_QP_solver_LA_DIAGZERO_MTVM_15_27(wam7dofarm_QP_solver_D9, wam7dofarm_QP_solver_dvcc8, wam7dofarm_QP_solver_grad_eq9);
wam7dofarm_QP_solver_LA_VSUB_603(wam7dofarm_QP_solver_rd, wam7dofarm_QP_solver_grad_eq, wam7dofarm_QP_solver_rd);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi0, wam7dofarm_QP_solver_rd0, wam7dofarm_QP_solver_dzcc0);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi1, wam7dofarm_QP_solver_rd1, wam7dofarm_QP_solver_dzcc1);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi2, wam7dofarm_QP_solver_rd2, wam7dofarm_QP_solver_dzcc2);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi3, wam7dofarm_QP_solver_rd3, wam7dofarm_QP_solver_dzcc3);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi4, wam7dofarm_QP_solver_rd4, wam7dofarm_QP_solver_dzcc4);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi5, wam7dofarm_QP_solver_rd5, wam7dofarm_QP_solver_dzcc5);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi6, wam7dofarm_QP_solver_rd6, wam7dofarm_QP_solver_dzcc6);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi7, wam7dofarm_QP_solver_rd7, wam7dofarm_QP_solver_dzcc7);
wam7dofarm_QP_solver_LA_DIAG_FORWARDBACKWARDSUB_64(wam7dofarm_QP_solver_Phi8, wam7dofarm_QP_solver_rd8, wam7dofarm_QP_solver_dzcc8);
wam7dofarm_QP_solver_LA_DENSE_FORWARDBACKWARDSUB_27(wam7dofarm_QP_solver_Phi9, wam7dofarm_QP_solver_rd9, wam7dofarm_QP_solver_dzcc9);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl0, wam7dofarm_QP_solver_slb0, wam7dofarm_QP_solver_llbbyslb0, wam7dofarm_QP_solver_dzcc0, wam7dofarm_QP_solver_lbIdx0, wam7dofarm_QP_solver_dllbcc0);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub0, wam7dofarm_QP_solver_sub0, wam7dofarm_QP_solver_lubbysub0, wam7dofarm_QP_solver_dzcc0, wam7dofarm_QP_solver_ubIdx0, wam7dofarm_QP_solver_dlubcc0);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl1, wam7dofarm_QP_solver_slb1, wam7dofarm_QP_solver_llbbyslb1, wam7dofarm_QP_solver_dzcc1, wam7dofarm_QP_solver_lbIdx1, wam7dofarm_QP_solver_dllbcc1);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub1, wam7dofarm_QP_solver_sub1, wam7dofarm_QP_solver_lubbysub1, wam7dofarm_QP_solver_dzcc1, wam7dofarm_QP_solver_ubIdx1, wam7dofarm_QP_solver_dlubcc1);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl2, wam7dofarm_QP_solver_slb2, wam7dofarm_QP_solver_llbbyslb2, wam7dofarm_QP_solver_dzcc2, wam7dofarm_QP_solver_lbIdx2, wam7dofarm_QP_solver_dllbcc2);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub2, wam7dofarm_QP_solver_sub2, wam7dofarm_QP_solver_lubbysub2, wam7dofarm_QP_solver_dzcc2, wam7dofarm_QP_solver_ubIdx2, wam7dofarm_QP_solver_dlubcc2);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl3, wam7dofarm_QP_solver_slb3, wam7dofarm_QP_solver_llbbyslb3, wam7dofarm_QP_solver_dzcc3, wam7dofarm_QP_solver_lbIdx3, wam7dofarm_QP_solver_dllbcc3);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub3, wam7dofarm_QP_solver_sub3, wam7dofarm_QP_solver_lubbysub3, wam7dofarm_QP_solver_dzcc3, wam7dofarm_QP_solver_ubIdx3, wam7dofarm_QP_solver_dlubcc3);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl4, wam7dofarm_QP_solver_slb4, wam7dofarm_QP_solver_llbbyslb4, wam7dofarm_QP_solver_dzcc4, wam7dofarm_QP_solver_lbIdx4, wam7dofarm_QP_solver_dllbcc4);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub4, wam7dofarm_QP_solver_sub4, wam7dofarm_QP_solver_lubbysub4, wam7dofarm_QP_solver_dzcc4, wam7dofarm_QP_solver_ubIdx4, wam7dofarm_QP_solver_dlubcc4);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl5, wam7dofarm_QP_solver_slb5, wam7dofarm_QP_solver_llbbyslb5, wam7dofarm_QP_solver_dzcc5, wam7dofarm_QP_solver_lbIdx5, wam7dofarm_QP_solver_dllbcc5);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub5, wam7dofarm_QP_solver_sub5, wam7dofarm_QP_solver_lubbysub5, wam7dofarm_QP_solver_dzcc5, wam7dofarm_QP_solver_ubIdx5, wam7dofarm_QP_solver_dlubcc5);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl6, wam7dofarm_QP_solver_slb6, wam7dofarm_QP_solver_llbbyslb6, wam7dofarm_QP_solver_dzcc6, wam7dofarm_QP_solver_lbIdx6, wam7dofarm_QP_solver_dllbcc6);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub6, wam7dofarm_QP_solver_sub6, wam7dofarm_QP_solver_lubbysub6, wam7dofarm_QP_solver_dzcc6, wam7dofarm_QP_solver_ubIdx6, wam7dofarm_QP_solver_dlubcc6);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl7, wam7dofarm_QP_solver_slb7, wam7dofarm_QP_solver_llbbyslb7, wam7dofarm_QP_solver_dzcc7, wam7dofarm_QP_solver_lbIdx7, wam7dofarm_QP_solver_dllbcc7);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub7, wam7dofarm_QP_solver_sub7, wam7dofarm_QP_solver_lubbysub7, wam7dofarm_QP_solver_dzcc7, wam7dofarm_QP_solver_ubIdx7, wam7dofarm_QP_solver_dlubcc7);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_64(wam7dofarm_QP_solver_ccrhsl8, wam7dofarm_QP_solver_slb8, wam7dofarm_QP_solver_llbbyslb8, wam7dofarm_QP_solver_dzcc8, wam7dofarm_QP_solver_lbIdx8, wam7dofarm_QP_solver_dllbcc8);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_36(wam7dofarm_QP_solver_ccrhsub8, wam7dofarm_QP_solver_sub8, wam7dofarm_QP_solver_lubbysub8, wam7dofarm_QP_solver_dzcc8, wam7dofarm_QP_solver_ubIdx8, wam7dofarm_QP_solver_dlubcc8);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTSUB_INDEXED_27(wam7dofarm_QP_solver_ccrhsl9, wam7dofarm_QP_solver_slb9, wam7dofarm_QP_solver_llbbyslb9, wam7dofarm_QP_solver_dzcc9, wam7dofarm_QP_solver_lbIdx9, wam7dofarm_QP_solver_dllbcc9);
wam7dofarm_QP_solver_LA_VEC_DIVSUB_MULTADD_INDEXED_15(wam7dofarm_QP_solver_ccrhsub9, wam7dofarm_QP_solver_sub9, wam7dofarm_QP_solver_lubbysub9, wam7dofarm_QP_solver_dzcc9, wam7dofarm_QP_solver_ubIdx9, wam7dofarm_QP_solver_dlubcc9);
wam7dofarm_QP_solver_LA_DENSE_MVMSUB5_12_27(params->A10, wam7dofarm_QP_solver_dzcc9, wam7dofarm_QP_solver_ccrhsp9, wam7dofarm_QP_solver_sp9, wam7dofarm_QP_solver_lp9, wam7dofarm_QP_solver_dlp_cc9);
wam7dofarm_QP_solver_LA_VSUB7_954(wam7dofarm_QP_solver_l, wam7dofarm_QP_solver_ccrhs, wam7dofarm_QP_solver_s, wam7dofarm_QP_solver_dl_cc, wam7dofarm_QP_solver_ds_cc);
wam7dofarm_QP_solver_LA_VADD_603(wam7dofarm_QP_solver_dz_cc, wam7dofarm_QP_solver_dz_aff);
wam7dofarm_QP_solver_LA_VADD_149(wam7dofarm_QP_solver_dv_cc, wam7dofarm_QP_solver_dv_aff);
wam7dofarm_QP_solver_LA_VADD_954(wam7dofarm_QP_solver_dl_cc, wam7dofarm_QP_solver_dl_aff);
wam7dofarm_QP_solver_LA_VADD_954(wam7dofarm_QP_solver_ds_cc, wam7dofarm_QP_solver_ds_aff);
info->lsit_cc = wam7dofarm_QP_solver_LINESEARCH_BACKTRACKING_COMBINED(wam7dofarm_QP_solver_z, wam7dofarm_QP_solver_v, wam7dofarm_QP_solver_l, wam7dofarm_QP_solver_s, wam7dofarm_QP_solver_dz_cc, wam7dofarm_QP_solver_dv_cc, wam7dofarm_QP_solver_dl_cc, wam7dofarm_QP_solver_ds_cc, &info->step_cc, &info->mu);
if( info->lsit_cc == wam7dofarm_QP_solver_NOPROGRESS ){
PRINTTEXT("Line search could not proceed at iteration %d, exiting.\n",info->it+1);
exitcode = wam7dofarm_QP_solver_NOPROGRESS; break;
}
info->it++;
}
output->z1[0] = wam7dofarm_QP_solver_z0[0];
output->z1[1] = wam7dofarm_QP_solver_z0[1];
output->z1[2] = wam7dofarm_QP_solver_z0[2];
output->z1[3] = wam7dofarm_QP_solver_z0[3];
output->z1[4] = wam7dofarm_QP_solver_z0[4];
output->z1[5] = wam7dofarm_QP_solver_z0[5];
output->z1[6] = wam7dofarm_QP_solver_z0[6];
output->z1[7] = wam7dofarm_QP_solver_z0[7];
output->z1[8] = wam7dofarm_QP_solver_z0[8];
output->z1[9] = wam7dofarm_QP_solver_z0[9];
output->z1[10] = wam7dofarm_QP_solver_z0[10];
output->z1[11] = wam7dofarm_QP_solver_z0[11];
output->z1[12] = wam7dofarm_QP_solver_z0[12];
output->z1[13] = wam7dofarm_QP_solver_z0[13];
output->z1[14] = wam7dofarm_QP_solver_z0[14];
output->z1[15] = wam7dofarm_QP_solver_z0[15];
output->z1[16] = wam7dofarm_QP_solver_z0[16];
output->z1[17] = wam7dofarm_QP_solver_z0[17];
output->z1[18] = wam7dofarm_QP_solver_z0[18];
output->z1[19] = wam7dofarm_QP_solver_z0[19];
output->z1[20] = wam7dofarm_QP_solver_z0[20];
output->z1[21] = wam7dofarm_QP_solver_z0[21];
output->z1[22] = wam7dofarm_QP_solver_z0[22];
output->z1[23] = wam7dofarm_QP_solver_z0[23];
output->z1[24] = wam7dofarm_QP_solver_z0[24];
output->z1[25] = wam7dofarm_QP_solver_z0[25];
output->z1[26] = wam7dofarm_QP_solver_z0[26];
output->z1[27] = wam7dofarm_QP_solver_z0[27];
output->z1[28] = wam7dofarm_QP_solver_z0[28];
output->z1[29] = wam7dofarm_QP_solver_z0[29];
output->z1[30] = wam7dofarm_QP_solver_z0[30];
output->z1[31] = wam7dofarm_QP_solver_z0[31];
output->z1[32] = wam7dofarm_QP_solver_z0[32];
output->z1[33] = wam7dofarm_QP_solver_z0[33];
output->z1[34] = wam7dofarm_QP_solver_z0[34];
output->z1[35] = wam7dofarm_QP_solver_z0[35];
output->z2[0] = wam7dofarm_QP_solver_z1[0];
output->z2[1] = wam7dofarm_QP_solver_z1[1];
output->z2[2] = wam7dofarm_QP_solver_z1[2];
output->z2[3] = wam7dofarm_QP_solver_z1[3];
output->z2[4] = wam7dofarm_QP_solver_z1[4];
output->z2[5] = wam7dofarm_QP_solver_z1[5];
output->z2[6] = wam7dofarm_QP_solver_z1[6];
output->z2[7] = wam7dofarm_QP_solver_z1[7];
output->z2[8] = wam7dofarm_QP_solver_z1[8];
output->z2[9] = wam7dofarm_QP_solver_z1[9];
output->z2[10] = wam7dofarm_QP_solver_z1[10];
output->z2[11] = wam7dofarm_QP_solver_z1[11];
output->z2[12] = wam7dofarm_QP_solver_z1[12];
output->z2[13] = wam7dofarm_QP_solver_z1[13];
output->z2[14] = wam7dofarm_QP_solver_z1[14];
output->z2[15] = wam7dofarm_QP_solver_z1[15];
output->z2[16] = wam7dofarm_QP_solver_z1[16];
output->z2[17] = wam7dofarm_QP_solver_z1[17];
output->z2[18] = wam7dofarm_QP_solver_z1[18];
output->z2[19] = wam7dofarm_QP_solver_z1[19];
output->z2[20] = wam7dofarm_QP_solver_z1[20];
output->z2[21] = wam7dofarm_QP_solver_z1[21];
output->z2[22] = wam7dofarm_QP_solver_z1[22];
output->z2[23] = wam7dofarm_QP_solver_z1[23];
output->z2[24] = wam7dofarm_QP_solver_z1[24];
output->z2[25] = wam7dofarm_QP_solver_z1[25];
output->z2[26] = wam7dofarm_QP_solver_z1[26];
output->z2[27] = wam7dofarm_QP_solver_z1[27];
output->z2[28] = wam7dofarm_QP_solver_z1[28];
output->z2[29] = wam7dofarm_QP_solver_z1[29];
output->z2[30] = wam7dofarm_QP_solver_z1[30];
output->z2[31] = wam7dofarm_QP_solver_z1[31];
output->z2[32] = wam7dofarm_QP_solver_z1[32];
output->z2[33] = wam7dofarm_QP_solver_z1[33];
output->z2[34] = wam7dofarm_QP_solver_z1[34];
output->z2[35] = wam7dofarm_QP_solver_z1[35];
output->z3[0] = wam7dofarm_QP_solver_z2[0];
output->z3[1] = wam7dofarm_QP_solver_z2[1];
output->z3[2] = wam7dofarm_QP_solver_z2[2];
output->z3[3] = wam7dofarm_QP_solver_z2[3];
output->z3[4] = wam7dofarm_QP_solver_z2[4];
output->z3[5] = wam7dofarm_QP_solver_z2[5];
output->z3[6] = wam7dofarm_QP_solver_z2[6];
output->z3[7] = wam7dofarm_QP_solver_z2[7];
output->z3[8] = wam7dofarm_QP_solver_z2[8];
output->z3[9] = wam7dofarm_QP_solver_z2[9];
output->z3[10] = wam7dofarm_QP_solver_z2[10];
output->z3[11] = wam7dofarm_QP_solver_z2[11];
output->z3[12] = wam7dofarm_QP_solver_z2[12];
output->z3[13] = wam7dofarm_QP_solver_z2[13];
output->z3[14] = wam7dofarm_QP_solver_z2[14];
output->z3[15] = wam7dofarm_QP_solver_z2[15];
output->z3[16] = wam7dofarm_QP_solver_z2[16];
output->z3[17] = wam7dofarm_QP_solver_z2[17];
output->z3[18] = wam7dofarm_QP_solver_z2[18];
output->z3[19] = wam7dofarm_QP_solver_z2[19];
output->z3[20] = wam7dofarm_QP_solver_z2[20];
output->z3[21] = wam7dofarm_QP_solver_z2[21];
output->z3[22] = wam7dofarm_QP_solver_z2[22];
output->z3[23] = wam7dofarm_QP_solver_z2[23];
output->z3[24] = wam7dofarm_QP_solver_z2[24];
output->z3[25] = wam7dofarm_QP_solver_z2[25];
output->z3[26] = wam7dofarm_QP_solver_z2[26];
output->z3[27] = wam7dofarm_QP_solver_z2[27];
output->z3[28] = wam7dofarm_QP_solver_z2[28];
output->z3[29] = wam7dofarm_QP_solver_z2[29];
output->z3[30] = wam7dofarm_QP_solver_z2[30];
output->z3[31] = wam7dofarm_QP_solver_z2[31];
output->z3[32] = wam7dofarm_QP_solver_z2[32];
output->z3[33] = wam7dofarm_QP_solver_z2[33];
output->z3[34] = wam7dofarm_QP_solver_z2[34];
output->z3[35] = wam7dofarm_QP_solver_z2[35];
output->z4[0] = wam7dofarm_QP_solver_z3[0];
output->z4[1] = wam7dofarm_QP_solver_z3[1];
output->z4[2] = wam7dofarm_QP_solver_z3[2];
output->z4[3] = wam7dofarm_QP_solver_z3[3];
output->z4[4] = wam7dofarm_QP_solver_z3[4];
output->z4[5] = wam7dofarm_QP_solver_z3[5];
output->z4[6] = wam7dofarm_QP_solver_z3[6];
output->z4[7] = wam7dofarm_QP_solver_z3[7];
output->z4[8] = wam7dofarm_QP_solver_z3[8];
output->z4[9] = wam7dofarm_QP_solver_z3[9];
output->z4[10] = wam7dofarm_QP_solver_z3[10];
output->z4[11] = wam7dofarm_QP_solver_z3[11];
output->z4[12] = wam7dofarm_QP_solver_z3[12];
output->z4[13] = wam7dofarm_QP_solver_z3[13];
output->z4[14] = wam7dofarm_QP_solver_z3[14];
output->z4[15] = wam7dofarm_QP_solver_z3[15];
output->z4[16] = wam7dofarm_QP_solver_z3[16];
output->z4[17] = wam7dofarm_QP_solver_z3[17];
output->z4[18] = wam7dofarm_QP_solver_z3[18];
output->z4[19] = wam7dofarm_QP_solver_z3[19];
output->z4[20] = wam7dofarm_QP_solver_z3[20];
output->z4[21] = wam7dofarm_QP_solver_z3[21];
output->z4[22] = wam7dofarm_QP_solver_z3[22];
output->z4[23] = wam7dofarm_QP_solver_z3[23];
output->z4[24] = wam7dofarm_QP_solver_z3[24];
output->z4[25] = wam7dofarm_QP_solver_z3[25];
output->z4[26] = wam7dofarm_QP_solver_z3[26];
output->z4[27] = wam7dofarm_QP_solver_z3[27];
output->z4[28] = wam7dofarm_QP_solver_z3[28];
output->z4[29] = wam7dofarm_QP_solver_z3[29];
output->z4[30] = wam7dofarm_QP_solver_z3[30];
output->z4[31] = wam7dofarm_QP_solver_z3[31];
output->z4[32] = wam7dofarm_QP_solver_z3[32];
output->z4[33] = wam7dofarm_QP_solver_z3[33];
output->z4[34] = wam7dofarm_QP_solver_z3[34];
output->z4[35] = wam7dofarm_QP_solver_z3[35];
output->z5[0] = wam7dofarm_QP_solver_z4[0];
output->z5[1] = wam7dofarm_QP_solver_z4[1];
output->z5[2] = wam7dofarm_QP_solver_z4[2];
output->z5[3] = wam7dofarm_QP_solver_z4[3];
output->z5[4] = wam7dofarm_QP_solver_z4[4];
output->z5[5] = wam7dofarm_QP_solver_z4[5];
output->z5[6] = wam7dofarm_QP_solver_z4[6];
output->z5[7] = wam7dofarm_QP_solver_z4[7];
output->z5[8] = wam7dofarm_QP_solver_z4[8];
output->z5[9] = wam7dofarm_QP_solver_z4[9];
output->z5[10] = wam7dofarm_QP_solver_z4[10];
output->z5[11] = wam7dofarm_QP_solver_z4[11];
output->z5[12] = wam7dofarm_QP_solver_z4[12];
output->z5[13] = wam7dofarm_QP_solver_z4[13];
output->z5[14] = wam7dofarm_QP_solver_z4[14];
output->z5[15] = wam7dofarm_QP_solver_z4[15];
output->z5[16] = wam7dofarm_QP_solver_z4[16];
output->z5[17] = wam7dofarm_QP_solver_z4[17];
output->z5[18] = wam7dofarm_QP_solver_z4[18];
output->z5[19] = wam7dofarm_QP_solver_z4[19];
output->z5[20] = wam7dofarm_QP_solver_z4[20];
output->z5[21] = wam7dofarm_QP_solver_z4[21];
output->z5[22] = wam7dofarm_QP_solver_z4[22];
output->z5[23] = wam7dofarm_QP_solver_z4[23];
output->z5[24] = wam7dofarm_QP_solver_z4[24];
output->z5[25] = wam7dofarm_QP_solver_z4[25];
output->z5[26] = wam7dofarm_QP_solver_z4[26];
output->z5[27] = wam7dofarm_QP_solver_z4[27];
output->z5[28] = wam7dofarm_QP_solver_z4[28];
output->z5[29] = wam7dofarm_QP_solver_z4[29];
output->z5[30] = wam7dofarm_QP_solver_z4[30];
output->z5[31] = wam7dofarm_QP_solver_z4[31];
output->z5[32] = wam7dofarm_QP_solver_z4[32];
output->z5[33] = wam7dofarm_QP_solver_z4[33];
output->z5[34] = wam7dofarm_QP_solver_z4[34];
output->z5[35] = wam7dofarm_QP_solver_z4[35];
output->z6[0] = wam7dofarm_QP_solver_z5[0];
output->z6[1] = wam7dofarm_QP_solver_z5[1];
output->z6[2] = wam7dofarm_QP_solver_z5[2];
output->z6[3] = wam7dofarm_QP_solver_z5[3];
output->z6[4] = wam7dofarm_QP_solver_z5[4];
output->z6[5] = wam7dofarm_QP_solver_z5[5];
output->z6[6] = wam7dofarm_QP_solver_z5[6];
output->z6[7] = wam7dofarm_QP_solver_z5[7];
output->z6[8] = wam7dofarm_QP_solver_z5[8];
output->z6[9] = wam7dofarm_QP_solver_z5[9];
output->z6[10] = wam7dofarm_QP_solver_z5[10];
output->z6[11] = wam7dofarm_QP_solver_z5[11];
output->z6[12] = wam7dofarm_QP_solver_z5[12];
output->z6[13] = wam7dofarm_QP_solver_z5[13];
output->z6[14] = wam7dofarm_QP_solver_z5[14];
output->z6[15] = wam7dofarm_QP_solver_z5[15];
output->z6[16] = wam7dofarm_QP_solver_z5[16];
output->z6[17] = wam7dofarm_QP_solver_z5[17];
output->z6[18] = wam7dofarm_QP_solver_z5[18];
output->z6[19] = wam7dofarm_QP_solver_z5[19];
output->z6[20] = wam7dofarm_QP_solver_z5[20];
output->z6[21] = wam7dofarm_QP_solver_z5[21];
output->z6[22] = wam7dofarm_QP_solver_z5[22];
output->z6[23] = wam7dofarm_QP_solver_z5[23];
output->z6[24] = wam7dofarm_QP_solver_z5[24];
output->z6[25] = wam7dofarm_QP_solver_z5[25];
output->z6[26] = wam7dofarm_QP_solver_z5[26];
output->z6[27] = wam7dofarm_QP_solver_z5[27];
output->z6[28] = wam7dofarm_QP_solver_z5[28];
output->z6[29] = wam7dofarm_QP_solver_z5[29];
output->z6[30] = wam7dofarm_QP_solver_z5[30];
output->z6[31] = wam7dofarm_QP_solver_z5[31];
output->z6[32] = wam7dofarm_QP_solver_z5[32];
output->z6[33] = wam7dofarm_QP_solver_z5[33];
output->z6[34] = wam7dofarm_QP_solver_z5[34];
output->z6[35] = wam7dofarm_QP_solver_z5[35];
output->z7[0] = wam7dofarm_QP_solver_z6[0];
output->z7[1] = wam7dofarm_QP_solver_z6[1];
output->z7[2] = wam7dofarm_QP_solver_z6[2];
output->z7[3] = wam7dofarm_QP_solver_z6[3];
output->z7[4] = wam7dofarm_QP_solver_z6[4];
output->z7[5] = wam7dofarm_QP_solver_z6[5];
output->z7[6] = wam7dofarm_QP_solver_z6[6];
output->z7[7] = wam7dofarm_QP_solver_z6[7];
output->z7[8] = wam7dofarm_QP_solver_z6[8];
output->z7[9] = wam7dofarm_QP_solver_z6[9];
output->z7[10] = wam7dofarm_QP_solver_z6[10];
output->z7[11] = wam7dofarm_QP_solver_z6[11];
output->z7[12] = wam7dofarm_QP_solver_z6[12];
output->z7[13] = wam7dofarm_QP_solver_z6[13];
output->z7[14] = wam7dofarm_QP_solver_z6[14];
output->z7[15] = wam7dofarm_QP_solver_z6[15];
output->z7[16] = wam7dofarm_QP_solver_z6[16];
output->z7[17] = wam7dofarm_QP_solver_z6[17];
output->z7[18] = wam7dofarm_QP_solver_z6[18];
output->z7[19] = wam7dofarm_QP_solver_z6[19];
output->z7[20] = wam7dofarm_QP_solver_z6[20];
output->z7[21] = wam7dofarm_QP_solver_z6[21];
output->z7[22] = wam7dofarm_QP_solver_z6[22];
output->z7[23] = wam7dofarm_QP_solver_z6[23];
output->z7[24] = wam7dofarm_QP_solver_z6[24];
output->z7[25] = wam7dofarm_QP_solver_z6[25];
output->z7[26] = wam7dofarm_QP_solver_z6[26];
output->z7[27] = wam7dofarm_QP_solver_z6[27];
output->z7[28] = wam7dofarm_QP_solver_z6[28];
output->z7[29] = wam7dofarm_QP_solver_z6[29];
output->z7[30] = wam7dofarm_QP_solver_z6[30];
output->z7[31] = wam7dofarm_QP_solver_z6[31];
output->z7[32] = wam7dofarm_QP_solver_z6[32];
output->z7[33] = wam7dofarm_QP_solver_z6[33];
output->z7[34] = wam7dofarm_QP_solver_z6[34];
output->z7[35] = wam7dofarm_QP_solver_z6[35];
output->z8[0] = wam7dofarm_QP_solver_z7[0];
output->z8[1] = wam7dofarm_QP_solver_z7[1];
output->z8[2] = wam7dofarm_QP_solver_z7[2];
output->z8[3] = wam7dofarm_QP_solver_z7[3];
output->z8[4] = wam7dofarm_QP_solver_z7[4];
output->z8[5] = wam7dofarm_QP_solver_z7[5];
output->z8[6] = wam7dofarm_QP_solver_z7[6];
output->z8[7] = wam7dofarm_QP_solver_z7[7];
output->z8[8] = wam7dofarm_QP_solver_z7[8];
output->z8[9] = wam7dofarm_QP_solver_z7[9];
output->z8[10] = wam7dofarm_QP_solver_z7[10];
output->z8[11] = wam7dofarm_QP_solver_z7[11];
output->z8[12] = wam7dofarm_QP_solver_z7[12];
output->z8[13] = wam7dofarm_QP_solver_z7[13];
output->z8[14] = wam7dofarm_QP_solver_z7[14];
output->z8[15] = wam7dofarm_QP_solver_z7[15];
output->z8[16] = wam7dofarm_QP_solver_z7[16];
output->z8[17] = wam7dofarm_QP_solver_z7[17];
output->z8[18] = wam7dofarm_QP_solver_z7[18];
output->z8[19] = wam7dofarm_QP_solver_z7[19];
output->z8[20] = wam7dofarm_QP_solver_z7[20];
output->z8[21] = wam7dofarm_QP_solver_z7[21];
output->z8[22] = wam7dofarm_QP_solver_z7[22];
output->z8[23] = wam7dofarm_QP_solver_z7[23];
output->z8[24] = wam7dofarm_QP_solver_z7[24];
output->z8[25] = wam7dofarm_QP_solver_z7[25];
output->z8[26] = wam7dofarm_QP_solver_z7[26];
output->z8[27] = wam7dofarm_QP_solver_z7[27];
output->z8[28] = wam7dofarm_QP_solver_z7[28];
output->z8[29] = wam7dofarm_QP_solver_z7[29];
output->z8[30] = wam7dofarm_QP_solver_z7[30];
output->z8[31] = wam7dofarm_QP_solver_z7[31];
output->z8[32] = wam7dofarm_QP_solver_z7[32];
output->z8[33] = wam7dofarm_QP_solver_z7[33];
output->z8[34] = wam7dofarm_QP_solver_z7[34];
output->z8[35] = wam7dofarm_QP_solver_z7[35];
output->z9[0] = wam7dofarm_QP_solver_z8[0];
output->z9[1] = wam7dofarm_QP_solver_z8[1];
output->z9[2] = wam7dofarm_QP_solver_z8[2];
output->z9[3] = wam7dofarm_QP_solver_z8[3];
output->z9[4] = wam7dofarm_QP_solver_z8[4];
output->z9[5] = wam7dofarm_QP_solver_z8[5];
output->z9[6] = wam7dofarm_QP_solver_z8[6];
output->z9[7] = wam7dofarm_QP_solver_z8[7];
output->z9[8] = wam7dofarm_QP_solver_z8[8];
output->z9[9] = wam7dofarm_QP_solver_z8[9];
output->z9[10] = wam7dofarm_QP_solver_z8[10];
output->z9[11] = wam7dofarm_QP_solver_z8[11];
output->z9[12] = wam7dofarm_QP_solver_z8[12];
output->z9[13] = wam7dofarm_QP_solver_z8[13];
output->z9[14] = wam7dofarm_QP_solver_z8[14];
output->z9[15] = wam7dofarm_QP_solver_z8[15];
output->z9[16] = wam7dofarm_QP_solver_z8[16];
output->z9[17] = wam7dofarm_QP_solver_z8[17];
output->z9[18] = wam7dofarm_QP_solver_z8[18];
output->z9[19] = wam7dofarm_QP_solver_z8[19];
output->z9[20] = wam7dofarm_QP_solver_z8[20];
output->z9[21] = wam7dofarm_QP_solver_z8[21];
output->z9[22] = wam7dofarm_QP_solver_z8[22];
output->z9[23] = wam7dofarm_QP_solver_z8[23];
output->z9[24] = wam7dofarm_QP_solver_z8[24];
output->z9[25] = wam7dofarm_QP_solver_z8[25];
output->z9[26] = wam7dofarm_QP_solver_z8[26];
output->z9[27] = wam7dofarm_QP_solver_z8[27];
output->z9[28] = wam7dofarm_QP_solver_z8[28];
output->z9[29] = wam7dofarm_QP_solver_z8[29];
output->z9[30] = wam7dofarm_QP_solver_z8[30];
output->z9[31] = wam7dofarm_QP_solver_z8[31];
output->z9[32] = wam7dofarm_QP_solver_z8[32];
output->z9[33] = wam7dofarm_QP_solver_z8[33];
output->z9[34] = wam7dofarm_QP_solver_z8[34];
output->z9[35] = wam7dofarm_QP_solver_z8[35];
output->z10[0] = wam7dofarm_QP_solver_z9[0];
output->z10[1] = wam7dofarm_QP_solver_z9[1];
output->z10[2] = wam7dofarm_QP_solver_z9[2];
output->z10[3] = wam7dofarm_QP_solver_z9[3];
output->z10[4] = wam7dofarm_QP_solver_z9[4];
output->z10[5] = wam7dofarm_QP_solver_z9[5];
output->z10[6] = wam7dofarm_QP_solver_z9[6];
output->z10[7] = wam7dofarm_QP_solver_z9[7];
output->z10[8] = wam7dofarm_QP_solver_z9[8];
output->z10[9] = wam7dofarm_QP_solver_z9[9];
output->z10[10] = wam7dofarm_QP_solver_z9[10];
output->z10[11] = wam7dofarm_QP_solver_z9[11];
output->z10[12] = wam7dofarm_QP_solver_z9[12];
output->z10[13] = wam7dofarm_QP_solver_z9[13];
output->z10[14] = wam7dofarm_QP_solver_z9[14];

#if wam7dofarm_QP_solver_SET_TIMING == 1
info->solvetime = wam7dofarm_QP_solver_toc(&solvertimer);
#if wam7dofarm_QP_solver_SET_PRINTLEVEL > 0 && wam7dofarm_QP_solver_SET_TIMING == 1
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