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

#include "doublependulum_QP_solver_shooting.h"

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
 * Initializes a vector of length 98 with a value.
 */
void doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_98(doublependulum_QP_solver_shooting_FLOAT* vec, doublependulum_QP_solver_shooting_FLOAT value)
{
	int i;
	for( i=0; i<98; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 13 with a value.
 */
void doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_13(doublependulum_QP_solver_shooting_FLOAT* vec, doublependulum_QP_solver_shooting_FLOAT value)
{
	int i;
	for( i=0; i<13; i++ )
	{
		vec[i] = value;
	}
}


/*
 * Initializes a vector of length 196 with a value.
 */
void doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_196(doublependulum_QP_solver_shooting_FLOAT* vec, doublependulum_QP_solver_shooting_FLOAT value)
{
	int i;
	for( i=0; i<196; i++ )
	{
		vec[i] = value;
	}
}


/* 
 * Calculates a dot product and adds it to a variable: z += x'*y; 
 * This function is for vectors of length 196.
 */
void doublependulum_QP_solver_shooting_LA_DOTACC_196(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<196; i++ ){
		*z += x[i]*y[i];
	}
}


/*
 * Calculates the gradient and the value for a quadratic function 0.5*z'*H*z + f'*z
 *
 * INPUTS:     H  - Symmetric Hessian, diag matrix of size [7 x 7]
 *             f  - column vector of size 7
 *             z  - column vector of size 7
 *
 * OUTPUTS: grad  - gradient at z (= H*z + f), column vector of size 7
 *          value <-- value + 0.5*z'*H*z + f'*z (value will be modified)
 */
void doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(doublependulum_QP_solver_shooting_FLOAT* H, doublependulum_QP_solver_shooting_FLOAT* f, doublependulum_QP_solver_shooting_FLOAT* z, doublependulum_QP_solver_shooting_FLOAT* grad, doublependulum_QP_solver_shooting_FLOAT* value)
{
	int i;
	doublependulum_QP_solver_shooting_FLOAT hz;	
	for( i=0; i<7; i++){
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
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *l, doublependulum_QP_solver_shooting_FLOAT *r, doublependulum_QP_solver_shooting_FLOAT *z, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	int j;
	int k = 0;
	doublependulum_QP_solver_shooting_FLOAT AxBu[1];
	doublependulum_QP_solver_shooting_FLOAT norm = *y;
	doublependulum_QP_solver_shooting_FLOAT lr = 0;

	/* do A*x + B*u first */
	for( i=0; i<1; i++ ){
		AxBu[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<7; j++ ){		
		for( i=0; i<1; i++ ){
			AxBu[i] += A[k++]*x[j];
		}
	}

	for( i=0; i<1; i++ ){
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
 * Matrix vector multiplication y = M'*x where M is of size [1 x 7]
 * and stored in column major format. Note the transpose of M!
 */
void doublependulum_QP_solver_shooting_LA_DENSE_MTVM_1_7(doublependulum_QP_solver_shooting_FLOAT *M, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<7; i++ ){
		y[i] = 0;
		for( j=0; j<1; j++ ){
			y[i] += M[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication z = A'*x + B'*y 
 * where A is of size [1 x 7] and stored in column major format.
 * and B is of size [1 x 7] and stored in diagzero format
 * Note the transposes of A and B!
 */
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	int j;
	int k = 0;
	for( i=0; i<1; i++ ){
		z[i] = 0;
		for( j=0; j<1; j++ ){
			z[i] += A[k++]*x[j];
		}
		z[i] += B[i]*y[i];
	}
	for( i=1 ;i<7; i++ ){
		z[i] = 0;
		for( j=0; j<1; j++ ){
			z[i] += A[k++]*x[j];
		}
	}
}


/*
 * Matrix vector multiplication y = M'*x where M is of size [1 x 7]
 * and stored in diagzero format. Note the transpose of M!
 */
void doublependulum_QP_solver_shooting_LA_DIAGZERO_MTVM_1_7(doublependulum_QP_solver_shooting_FLOAT *M, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	for( i=0; i<7; i++ ){
		y[i] = M[i]*x[i];
	}
}


/*
 * Vector subtraction and addition.
 *	 Input: five vectors t, tidx, u, v, w and two scalars z and r
 *	 Output: y = t(tidx) - u + w
 *           z = z - v'*x;
 *           r = max([norm(y,inf), z]);
 * for vectors of length 7. Output z is of course scalar.
 */
void doublependulum_QP_solver_shooting_LA_VSUBADD3_7(doublependulum_QP_solver_shooting_FLOAT* t, doublependulum_QP_solver_shooting_FLOAT* u, int* uidx, doublependulum_QP_solver_shooting_FLOAT* v, doublependulum_QP_solver_shooting_FLOAT* w, doublependulum_QP_solver_shooting_FLOAT* y, doublependulum_QP_solver_shooting_FLOAT* z, doublependulum_QP_solver_shooting_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_shooting_FLOAT norm = *r;
	doublependulum_QP_solver_shooting_FLOAT vx = 0;
	doublependulum_QP_solver_shooting_FLOAT x;
	for( i=0; i<7; i++){
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
 * for vectors of length 7. Output z is of course scalar.
 */
void doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_FLOAT* t, int* tidx, doublependulum_QP_solver_shooting_FLOAT* u, doublependulum_QP_solver_shooting_FLOAT* v, doublependulum_QP_solver_shooting_FLOAT* w, doublependulum_QP_solver_shooting_FLOAT* y, doublependulum_QP_solver_shooting_FLOAT* z, doublependulum_QP_solver_shooting_FLOAT* r)
{
	int i;
	doublependulum_QP_solver_shooting_FLOAT norm = *r;
	doublependulum_QP_solver_shooting_FLOAT vx = 0;
	doublependulum_QP_solver_shooting_FLOAT x;
	for( i=0; i<7; i++){
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
 * Special function for box constraints of length 7
 * Returns also L/S, a value that is often used elsewhere.
 */
void doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_FLOAT *lu, doublependulum_QP_solver_shooting_FLOAT *su, doublependulum_QP_solver_shooting_FLOAT *ru, doublependulum_QP_solver_shooting_FLOAT *ll, doublependulum_QP_solver_shooting_FLOAT *sl, doublependulum_QP_solver_shooting_FLOAT *rl, int* lbIdx, int* ubIdx, doublependulum_QP_solver_shooting_FLOAT *grad, doublependulum_QP_solver_shooting_FLOAT *lubysu, doublependulum_QP_solver_shooting_FLOAT *llbysl)
{
	int i;
	for( i=0; i<7; i++ ){
		grad[i] = 0;
	}
	for( i=0; i<7; i++ ){		
		llbysl[i] = ll[i] / sl[i];
		grad[lbIdx[i]] -= llbysl[i]*rl[i];
	}
	for( i=0; i<7; i++ ){
		lubysu[i] = lu[i] / su[i];
		grad[ubIdx[i]] += lubysu[i]*ru[i];
	}
}


/*
 * Addition of three vectors  z = u + w + v
 * of length 98.
 */
void doublependulum_QP_solver_shooting_LA_VVADD3_98(doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *v, doublependulum_QP_solver_shooting_FLOAT *w, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<98; i++ ){
		z[i] = u[i] + v[i] + w[i];
	}
}


/*
 * Special function to compute the diagonal cholesky factorization of the 
 * positive definite augmented Hessian for block size 7.
 *
 * Inputs: - H = diagonal cost Hessian in diagonal storage format
 *         - llbysl = L / S of lower bounds
 *         - lubysu = L / S of upper bounds
 *
 * Output: Phi = sqrt(H + diag(llbysl) + diag(lubysu))
 * where Phi is stored in diagonal storage format
 */
void doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(doublependulum_QP_solver_shooting_FLOAT *H, doublependulum_QP_solver_shooting_FLOAT *llbysl, int* lbIdx, doublependulum_QP_solver_shooting_FLOAT *lubysu, int* ubIdx, doublependulum_QP_solver_shooting_FLOAT *Phi)


{
	int i;
	
	/* compute cholesky */
	for( i=0; i<7; i++ ){
		Phi[i] = H[i] + llbysl[i] + lubysu[i];

#if doublependulum_QP_solver_shooting_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
 * where A is to be computed and is of size [1 x 7],
 * B is given and of size [1 x 7], L is a diagonal
 * matrix of size 1 stored in diagonal matrix 
 * storage format. Note the transpose of L has no impact!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *A)
{
    int i,j;
	 int k = 0;

	for( j=0; j<7; j++){
		for( i=0; i<1; i++){
			A[k] = B[k]/L[j];
			k++;
		}
	}

}


/**
 * Forward substitution to solve L*y = b where L is a
 * diagonal matrix in vector storage format.
 * 
 * The dimensions involved are 7.
 */
void doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *y)
{
    int i;

    for( i=0; i<7; i++ ){
		y[i] = b[i]/L[i];
    }
}


/**
 * Forward substitution for the matrix equation A*L' = B
 * where A is to be computed and is of size [1 x 7],
 * B is given and of size [1 x 7], L is a diagonal
 *  matrix of size 7 stored in diagonal 
 * storage format. Note the transpose of L!
 *
 * Result: A in diagonalzero storage format.
 *
 */
void doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *A)
{
	int j;
    for( j=0; j<7; j++ ){   
		A[j] = B[j]/L[j];
     }
}


/**
 * Compute C = A*B' where 
 *
 *	size(A) = [1 x 7]
 *  size(B) = [1 x 7] in diagzero format
 * 
 * A and C matrices are stored in column major format.
 * 
 * 
 */
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *C)
{
    int i, j;
	
	for( i=0; i<1; i++ ){
		for( j=0; j<1; j++){
			C[j*1+i] = B[i*1+j]*A[i];
		}
	}

}


/**
 * Compute L = A*A' + B*B', where L is lower triangular of size NXp1
 * and A is a dense matrix of size [1 x 7] in column
 * storage format, and B is of size [1 x 7] diagonalzero
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A AND B INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_shooting_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<1; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<7; k++ ){
                ltemp += A[k*1+i]*A[k*1+j];
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
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<1; i++ ){
		r[i] = b[i] - A[k++]*x[0] - B[i]*u[i];
	}	

	for( j=1; j<7; j++ ){		
		for( i=0; i<1; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
	
}


/**
 * Cholesky factorization as above, but working on a matrix in 
 * lower triangular storage format of size 1 and outputting
 * the Cholesky factor to matrix L in lower triangular format.
 */
void doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *L)
{
    int i, j, k, di, dj;
	 int ii, jj;

    doublependulum_QP_solver_shooting_FLOAT l;
    doublependulum_QP_solver_shooting_FLOAT Mii;

	/* copy A to L first and then operate on L */
	/* COULD BE OPTIMIZED */
	ii=0; di=0;
	for( i=0; i<1; i++ ){
		for( j=0; j<=i; j++ ){
			L[ii+j] = A[ii+j];
		}
		ii += ++di;
	}    
	
	/* factor L */
	ii=0; di=0;
    for( i=0; i<1; i++ ){
        l = 0;
        for( k=0; k<i; k++ ){
            l += L[ii+k]*L[ii+k];
        }        
        
        Mii = L[ii+i] - l;
        
#if doublependulum_QP_solver_shooting_SET_PRINTLEVEL > 0 && defined PRINTNUMERICALWARNINGS
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
        for( j=i+1; j<1; j++ ){
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
 * The dimensions involved are 1.
 */
void doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *y)
{
    int i,j,ii,di;
    doublependulum_QP_solver_shooting_FLOAT yel;
            
    ii = 0; di = 0;
    for( i=0; i<1; i++ ){
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
 * where A is to be computed and is of size [1 x 1],
 * B is given and of size [1 x 1], L is a lower tri-
 * angular matrix of size 1 stored in lower triangular 
 * storage format. Note the transpose of L AND B!
 *
 * Result: A in column major storage format.
 *
 */
void doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *A)
{
    int i,j,k,ii,di;
    doublependulum_QP_solver_shooting_FLOAT a;
    
    ii=0; di=0;
    for( j=0; j<1; j++ ){        
        for( i=0; i<1; i++ ){
            a = B[i*1+j];
            for( k=0; k<j; k++ ){
                a -= A[k*1+i]*L[ii+k];
            }    

			/* saturate for numerical stability */
			a = MIN(a, BIGM);
			a = MAX(a, -BIGM); 

			A[j*1+i] = a/L[ii+j];			
        }
        ii += ++di;
    }
}


/**
 * Compute L = L - A*A', where L is lower triangular of size 1
 * and A is a dense matrix of size [1 x 1] in column
 * storage format.
 * 
 * THIS ONE HAS THE WORST ACCES PATTERN POSSIBLE. 
 * POSSIBKE FIX: PUT A INTO ROW MAJOR FORMAT FIRST.
 * 
 */
void doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *L)
{
    int i, j, k, ii, di;
    doublependulum_QP_solver_shooting_FLOAT ltemp;
    
    ii = 0; di = 0;
    for( i=0; i<1; i++ ){        
        for( j=0; j<=i; j++ ){
            ltemp = 0; 
            for( k=0; k<1; k++ ){
                ltemp += A[k*1+i]*A[k*1+j];
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
void doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<1; i++ ){
		r[i] = b[i] - A[k++]*x[0];
	}	
	for( j=1; j<1; j++ ){		
		for( i=0; i<1; i++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/**
 * Backward Substitution to solve L^T*x = y where L is a
 * lower triangular matrix in triangular storage format.
 * 
 * All involved dimensions are 1.
 */
void doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *x)
{
    int i, ii, di, j, jj, dj;
    doublependulum_QP_solver_shooting_FLOAT xel;    
	int start = 0;
    
    /* now solve L^T*x = y by backward substitution */
    ii = start; di = 0;
    for( i=0; i>=0; i-- ){        
        xel = y[i];        
        jj = start; dj = 0;
        for( j=0; j>i; j-- ){
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
 * Matrix vector multiplication y = b - M'*x where M is of size [1 x 1]
 * and stored in column major format. Note the transpose of M!
 */
void doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *r)
{
	int i;
	int j;
	int k = 0; 
	for( i=0; i<1; i++ ){
		r[i] = b[i];
		for( j=0; j<1; j++ ){
			r[i] -= A[k++]*x[j];
		}
	}
}


/*
 * Vector subtraction z = -x - y for vectors of length 98.
 */
void doublependulum_QP_solver_shooting_LA_VSUB2_98(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<98; i++){
		z[i] = -x[i] - y[i];
	}
}


/**
 * Forward-Backward-Substitution to solve L*L^T*x = b where L is a
 * diagonal matrix of size 7 in vector
 * storage format.
 */
void doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_FLOAT *L, doublependulum_QP_solver_shooting_FLOAT *b, doublependulum_QP_solver_shooting_FLOAT *x)
{
    int i;
            
    /* solve Ly = b by forward and backward substitution */
    for( i=0; i<7; i++ ){
		x[i] = b[i]/(L[i]*L[i]);
    }
    
}


/*
 * Vector subtraction z = x(xidx) - y where y, z and xidx are of length 7,
 * and x has length 7 and is indexed through yidx.
 */
void doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_FLOAT *x, int* xidx, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<7; i++){
		z[i] = x[xidx[i]] - y[i];
	}
}


/*
 * Vector subtraction x = -u.*v - w for vectors of length 7.
 */
void doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *v, doublependulum_QP_solver_shooting_FLOAT *w, doublependulum_QP_solver_shooting_FLOAT *x)
{
	int i;
	for( i=0; i<7; i++){
		x[i] = -u[i]*v[i] - w[i];
	}
}


/*
 * Vector subtraction z = -x - y(yidx) where y is of length 7
 * and z, x and yidx are of length 7.
 */
void doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y, int* yidx, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<7; i++){
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
 * doublependulum_QP_solver_shooting_NOPROGRESS (should be negative).
 */
int doublependulum_QP_solver_shooting_LINESEARCH_BACKTRACKING_AFFINE(doublependulum_QP_solver_shooting_FLOAT *l, doublependulum_QP_solver_shooting_FLOAT *s, doublependulum_QP_solver_shooting_FLOAT *dl, doublependulum_QP_solver_shooting_FLOAT *ds, doublependulum_QP_solver_shooting_FLOAT *a, doublependulum_QP_solver_shooting_FLOAT *mu_aff)
{
    int i;
	int lsIt=1;    
    doublependulum_QP_solver_shooting_FLOAT dltemp;
    doublependulum_QP_solver_shooting_FLOAT dstemp;
    doublependulum_QP_solver_shooting_FLOAT mya = 1.0;
    doublependulum_QP_solver_shooting_FLOAT mymu;
        
    while( 1 ){                        

        /* 
         * Compute both snew and wnew together.
         * We compute also mu_affine along the way here, as the
         * values might be in registers, so it should be cheaper.
         */
        mymu = 0;
        for( i=0; i<196; i++ ){
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
        if( i == 196 ){
            break;
        } else {
            mya *= doublependulum_QP_solver_shooting_SET_LS_SCALE_AFF;
            if( mya < doublependulum_QP_solver_shooting_SET_LS_MINSTEP ){
                return doublependulum_QP_solver_shooting_NOPROGRESS;
            }
        }
    }
    
    /* return new values and iteration counter */
    *a = mya;
    *mu_aff = mymu / (doublependulum_QP_solver_shooting_FLOAT)196;
    return lsIt;
}


/*
 * Vector subtraction x = (u.*v - mu)*sigma where a is a scalar
*  and x,u,v are vectors of length 196.
 */
void doublependulum_QP_solver_shooting_LA_VSUB5_196(doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *v, doublependulum_QP_solver_shooting_FLOAT mu,  doublependulum_QP_solver_shooting_FLOAT sigma, doublependulum_QP_solver_shooting_FLOAT *x)
{
	int i;
	for( i=0; i<196; i++){
		x[i] = u[i]*v[i] - mu;
		x[i] *= sigma;
	}
}


/*
 * Computes x=0; x(uidx) += u/su; x(vidx) -= v/sv where x is of length 7,
 * u, su, uidx are of length 7 and v, sv, vidx are of length 7.
 */
void doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *su, int* uidx, doublependulum_QP_solver_shooting_FLOAT *v, doublependulum_QP_solver_shooting_FLOAT *sv, int* vidx, doublependulum_QP_solver_shooting_FLOAT *x)
{
	int i;
	for( i=0; i<7; i++ ){
		x[i] = 0;
	}
	for( i=0; i<7; i++){
		x[uidx[i]] += u[i]/su[i];
	}
	for( i=0; i<7; i++){
		x[vidx[i]] -= v[i]/sv[i];
	}
}


/* 
 * Computes r = A*x + B*u
 * where A is stored in column major format
 * and B is stored in diagzero format
 */
void doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_FLOAT *A, doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *B, doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *r)
{
	int i;
	int j;
	int k = 0;

	for( i=0; i<1; i++ ){
		r[i] = A[k++]*x[0] + B[i]*u[i];
	}	

	for( j=1; j<7; j++ ){		
		for( i=0; i<1; i++ ){
			r[i] += A[k++]*x[j];
		}
	}
	
}


/*
 * Vector subtraction z = x - y for vectors of length 98.
 */
void doublependulum_QP_solver_shooting_LA_VSUB_98(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<98; i++){
		z[i] = x[i] - y[i];
	}
}


/** 
 * Computes z = -r./s - u.*y(y)
 * where all vectors except of y are of length 7 (length of y >= 7).
 */
void doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_FLOAT *r, doublependulum_QP_solver_shooting_FLOAT *s, doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *y, int* yidx, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<7; i++ ){
		z[i] = -r[i]/s[i] - u[i]*y[yidx[i]];
	}
}


/** 
 * Computes z = -r./s + u.*y(y)
 * where all vectors except of y are of length 7 (length of y >= 7).
 */
void doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_FLOAT *r, doublependulum_QP_solver_shooting_FLOAT *s, doublependulum_QP_solver_shooting_FLOAT *u, doublependulum_QP_solver_shooting_FLOAT *y, int* yidx, doublependulum_QP_solver_shooting_FLOAT *z)
{
	int i;
	for( i=0; i<7; i++ ){
		z[i] = -r[i]/s[i] + u[i]*y[yidx[i]];
	}
}


/*
 * Computes ds = -l.\(r + s.*dl) for vectors of length 196.
 */
void doublependulum_QP_solver_shooting_LA_VSUB7_196(doublependulum_QP_solver_shooting_FLOAT *l, doublependulum_QP_solver_shooting_FLOAT *r, doublependulum_QP_solver_shooting_FLOAT *s, doublependulum_QP_solver_shooting_FLOAT *dl, doublependulum_QP_solver_shooting_FLOAT *ds)
{
	int i;
	for( i=0; i<196; i++){
		ds[i] = -(r[i] + s[i]*dl[i])/l[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 98.
 */
void doublependulum_QP_solver_shooting_LA_VADD_98(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	for( i=0; i<98; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 13.
 */
void doublependulum_QP_solver_shooting_LA_VADD_13(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	for( i=0; i<13; i++){
		x[i] += y[i];
	}
}


/*
 * Vector addition x = x + y for vectors of length 196.
 */
void doublependulum_QP_solver_shooting_LA_VADD_196(doublependulum_QP_solver_shooting_FLOAT *x, doublependulum_QP_solver_shooting_FLOAT *y)
{
	int i;
	for( i=0; i<196; i++){
		x[i] += y[i];
	}
}


/**
 * Backtracking line search for combined predictor/corrector step.
 * Update on variables with safety factor gamma (to keep us away from
 * boundary).
 */
int doublependulum_QP_solver_shooting_LINESEARCH_BACKTRACKING_COMBINED(doublependulum_QP_solver_shooting_FLOAT *z, doublependulum_QP_solver_shooting_FLOAT *v, doublependulum_QP_solver_shooting_FLOAT *l, doublependulum_QP_solver_shooting_FLOAT *s, doublependulum_QP_solver_shooting_FLOAT *dz, doublependulum_QP_solver_shooting_FLOAT *dv, doublependulum_QP_solver_shooting_FLOAT *dl, doublependulum_QP_solver_shooting_FLOAT *ds, doublependulum_QP_solver_shooting_FLOAT *a, doublependulum_QP_solver_shooting_FLOAT *mu)
{
    int i, lsIt=1;       
    doublependulum_QP_solver_shooting_FLOAT dltemp;
    doublependulum_QP_solver_shooting_FLOAT dstemp;    
    doublependulum_QP_solver_shooting_FLOAT a_gamma;
            
    *a = 1.0;
    while( 1 ){                        

        /* check whether search criterion is fulfilled */
        for( i=0; i<196; i++ ){
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
        if( i == 196 ){
            break;
        } else {
            *a *= doublependulum_QP_solver_shooting_SET_LS_SCALE;
            if( *a < doublependulum_QP_solver_shooting_SET_LS_MINSTEP ){
                return doublependulum_QP_solver_shooting_NOPROGRESS;
            }
        }
    }
    
    /* update variables with safety margin */
    a_gamma = (*a)*doublependulum_QP_solver_shooting_SET_LS_MAXSTEP;
    
    /* primal variables */
    for( i=0; i<98; i++ ){
        z[i] += a_gamma*dz[i];
    }
    
    /* equality constraint multipliers */
    for( i=0; i<13; i++ ){
        v[i] += a_gamma*dv[i];
    }
    
    /* inequality constraint multipliers & slacks, also update mu */
    *mu = 0;
    for( i=0; i<196; i++ ){
        dltemp = l[i] + a_gamma*dl[i]; l[i] = dltemp;
        dstemp = s[i] + a_gamma*ds[i]; s[i] = dstemp;
        *mu += dltemp*dstemp;
    }
    
    *a = a_gamma;
    *mu /= (doublependulum_QP_solver_shooting_FLOAT)196;
    return lsIt;
}




/* VARIABLE DEFINITIONS ------------------------------------------------ */
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_z[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_v[13];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dz_aff[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dv_aff[13];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_grad_cost[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_grad_eq[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rd[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_l[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_s[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_lbys[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dl_aff[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ds_aff[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dz_cc[98];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dv_cc[13];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_dl_cc[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ds_cc[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ccrhs[196];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_grad_ineq[98];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z00 = doublependulum_QP_solver_shooting_z + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff00 = doublependulum_QP_solver_shooting_dz_aff + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc00 = doublependulum_QP_solver_shooting_dz_cc + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd00 = doublependulum_QP_solver_shooting_rd + 0;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd00[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost00 = doublependulum_QP_solver_shooting_grad_cost + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq00 = doublependulum_QP_solver_shooting_grad_eq + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq00 = doublependulum_QP_solver_shooting_grad_ineq + 0;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv00[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_C00[7] = {1.0000000000000000E+000, 
0.0000000000000000E+000, 
0.0000000000000000E+000, 
0.0000000000000000E+000, 
0.0000000000000000E+000, 
0.0000000000000000E+000, 
0.0000000000000000E+000};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v00 = doublependulum_QP_solver_shooting_v + 0;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc00[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff00 = doublependulum_QP_solver_shooting_dv_aff + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc00 = doublependulum_QP_solver_shooting_dv_cc + 0;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V00[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy00[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c00[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx00[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb00 = doublependulum_QP_solver_shooting_l + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb00 = doublependulum_QP_solver_shooting_s + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb00 = doublependulum_QP_solver_shooting_lbys + 0;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb00[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff00 = doublependulum_QP_solver_shooting_dl_aff + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff00 = doublependulum_QP_solver_shooting_ds_aff + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc00 = doublependulum_QP_solver_shooting_dl_cc + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc00 = doublependulum_QP_solver_shooting_ds_cc + 0;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl00 = doublependulum_QP_solver_shooting_ccrhs + 0;
int doublependulum_QP_solver_shooting_ubIdx00[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub00 = doublependulum_QP_solver_shooting_l + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub00 = doublependulum_QP_solver_shooting_s + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub00 = doublependulum_QP_solver_shooting_lbys + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub00[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff00 = doublependulum_QP_solver_shooting_dl_aff + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff00 = doublependulum_QP_solver_shooting_ds_aff + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc00 = doublependulum_QP_solver_shooting_dl_cc + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc00 = doublependulum_QP_solver_shooting_ds_cc + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub00 = doublependulum_QP_solver_shooting_ccrhs + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi00[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z01 = doublependulum_QP_solver_shooting_z + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff01 = doublependulum_QP_solver_shooting_dz_aff + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc01 = doublependulum_QP_solver_shooting_dz_cc + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd01 = doublependulum_QP_solver_shooting_rd + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd01[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost01 = doublependulum_QP_solver_shooting_grad_cost + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq01 = doublependulum_QP_solver_shooting_grad_eq + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq01 = doublependulum_QP_solver_shooting_grad_ineq + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv01[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v01 = doublependulum_QP_solver_shooting_v + 1;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc01[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff01 = doublependulum_QP_solver_shooting_dv_aff + 1;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc01 = doublependulum_QP_solver_shooting_dv_cc + 1;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V01[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c01[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx01[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb01 = doublependulum_QP_solver_shooting_l + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb01 = doublependulum_QP_solver_shooting_s + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb01 = doublependulum_QP_solver_shooting_lbys + 14;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb01[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff01 = doublependulum_QP_solver_shooting_dl_aff + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff01 = doublependulum_QP_solver_shooting_ds_aff + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc01 = doublependulum_QP_solver_shooting_dl_cc + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc01 = doublependulum_QP_solver_shooting_ds_cc + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl01 = doublependulum_QP_solver_shooting_ccrhs + 14;
int doublependulum_QP_solver_shooting_ubIdx01[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub01 = doublependulum_QP_solver_shooting_l + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub01 = doublependulum_QP_solver_shooting_s + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub01 = doublependulum_QP_solver_shooting_lbys + 21;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub01[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff01 = doublependulum_QP_solver_shooting_dl_aff + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff01 = doublependulum_QP_solver_shooting_ds_aff + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc01 = doublependulum_QP_solver_shooting_dl_cc + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc01 = doublependulum_QP_solver_shooting_ds_cc + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub01 = doublependulum_QP_solver_shooting_ccrhs + 21;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi01[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_D01[7] = {-1.0000000000000000E+000};
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W01[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd01[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd01[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z02 = doublependulum_QP_solver_shooting_z + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff02 = doublependulum_QP_solver_shooting_dz_aff + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc02 = doublependulum_QP_solver_shooting_dz_cc + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd02 = doublependulum_QP_solver_shooting_rd + 14;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd02[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost02 = doublependulum_QP_solver_shooting_grad_cost + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq02 = doublependulum_QP_solver_shooting_grad_eq + 14;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq02 = doublependulum_QP_solver_shooting_grad_ineq + 14;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv02[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v02 = doublependulum_QP_solver_shooting_v + 2;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc02[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff02 = doublependulum_QP_solver_shooting_dv_aff + 2;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc02 = doublependulum_QP_solver_shooting_dv_cc + 2;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V02[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c02[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx02[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb02 = doublependulum_QP_solver_shooting_l + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb02 = doublependulum_QP_solver_shooting_s + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb02 = doublependulum_QP_solver_shooting_lbys + 28;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb02[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff02 = doublependulum_QP_solver_shooting_dl_aff + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff02 = doublependulum_QP_solver_shooting_ds_aff + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc02 = doublependulum_QP_solver_shooting_dl_cc + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc02 = doublependulum_QP_solver_shooting_ds_cc + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl02 = doublependulum_QP_solver_shooting_ccrhs + 28;
int doublependulum_QP_solver_shooting_ubIdx02[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub02 = doublependulum_QP_solver_shooting_l + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub02 = doublependulum_QP_solver_shooting_s + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub02 = doublependulum_QP_solver_shooting_lbys + 35;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub02[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff02 = doublependulum_QP_solver_shooting_dl_aff + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff02 = doublependulum_QP_solver_shooting_ds_aff + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc02 = doublependulum_QP_solver_shooting_dl_cc + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc02 = doublependulum_QP_solver_shooting_ds_cc + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub02 = doublependulum_QP_solver_shooting_ccrhs + 35;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi02[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W02[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd02[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd02[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z03 = doublependulum_QP_solver_shooting_z + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff03 = doublependulum_QP_solver_shooting_dz_aff + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc03 = doublependulum_QP_solver_shooting_dz_cc + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd03 = doublependulum_QP_solver_shooting_rd + 21;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd03[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost03 = doublependulum_QP_solver_shooting_grad_cost + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq03 = doublependulum_QP_solver_shooting_grad_eq + 21;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq03 = doublependulum_QP_solver_shooting_grad_ineq + 21;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv03[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v03 = doublependulum_QP_solver_shooting_v + 3;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc03[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff03 = doublependulum_QP_solver_shooting_dv_aff + 3;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc03 = doublependulum_QP_solver_shooting_dv_cc + 3;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V03[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c03[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx03[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb03 = doublependulum_QP_solver_shooting_l + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb03 = doublependulum_QP_solver_shooting_s + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb03 = doublependulum_QP_solver_shooting_lbys + 42;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb03[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff03 = doublependulum_QP_solver_shooting_dl_aff + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff03 = doublependulum_QP_solver_shooting_ds_aff + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc03 = doublependulum_QP_solver_shooting_dl_cc + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc03 = doublependulum_QP_solver_shooting_ds_cc + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl03 = doublependulum_QP_solver_shooting_ccrhs + 42;
int doublependulum_QP_solver_shooting_ubIdx03[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub03 = doublependulum_QP_solver_shooting_l + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub03 = doublependulum_QP_solver_shooting_s + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub03 = doublependulum_QP_solver_shooting_lbys + 49;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub03[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff03 = doublependulum_QP_solver_shooting_dl_aff + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff03 = doublependulum_QP_solver_shooting_ds_aff + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc03 = doublependulum_QP_solver_shooting_dl_cc + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc03 = doublependulum_QP_solver_shooting_ds_cc + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub03 = doublependulum_QP_solver_shooting_ccrhs + 49;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi03[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W03[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd03[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd03[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z04 = doublependulum_QP_solver_shooting_z + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff04 = doublependulum_QP_solver_shooting_dz_aff + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc04 = doublependulum_QP_solver_shooting_dz_cc + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd04 = doublependulum_QP_solver_shooting_rd + 28;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd04[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost04 = doublependulum_QP_solver_shooting_grad_cost + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq04 = doublependulum_QP_solver_shooting_grad_eq + 28;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq04 = doublependulum_QP_solver_shooting_grad_ineq + 28;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv04[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v04 = doublependulum_QP_solver_shooting_v + 4;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc04[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff04 = doublependulum_QP_solver_shooting_dv_aff + 4;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc04 = doublependulum_QP_solver_shooting_dv_cc + 4;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V04[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c04[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx04[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb04 = doublependulum_QP_solver_shooting_l + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb04 = doublependulum_QP_solver_shooting_s + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb04 = doublependulum_QP_solver_shooting_lbys + 56;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb04[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff04 = doublependulum_QP_solver_shooting_dl_aff + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff04 = doublependulum_QP_solver_shooting_ds_aff + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc04 = doublependulum_QP_solver_shooting_dl_cc + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc04 = doublependulum_QP_solver_shooting_ds_cc + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl04 = doublependulum_QP_solver_shooting_ccrhs + 56;
int doublependulum_QP_solver_shooting_ubIdx04[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub04 = doublependulum_QP_solver_shooting_l + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub04 = doublependulum_QP_solver_shooting_s + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub04 = doublependulum_QP_solver_shooting_lbys + 63;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub04[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff04 = doublependulum_QP_solver_shooting_dl_aff + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff04 = doublependulum_QP_solver_shooting_ds_aff + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc04 = doublependulum_QP_solver_shooting_dl_cc + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc04 = doublependulum_QP_solver_shooting_ds_cc + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub04 = doublependulum_QP_solver_shooting_ccrhs + 63;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi04[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W04[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd04[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd04[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z05 = doublependulum_QP_solver_shooting_z + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff05 = doublependulum_QP_solver_shooting_dz_aff + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc05 = doublependulum_QP_solver_shooting_dz_cc + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd05 = doublependulum_QP_solver_shooting_rd + 35;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd05[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost05 = doublependulum_QP_solver_shooting_grad_cost + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq05 = doublependulum_QP_solver_shooting_grad_eq + 35;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq05 = doublependulum_QP_solver_shooting_grad_ineq + 35;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv05[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v05 = doublependulum_QP_solver_shooting_v + 5;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc05[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff05 = doublependulum_QP_solver_shooting_dv_aff + 5;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc05 = doublependulum_QP_solver_shooting_dv_cc + 5;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V05[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c05[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx05[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb05 = doublependulum_QP_solver_shooting_l + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb05 = doublependulum_QP_solver_shooting_s + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb05 = doublependulum_QP_solver_shooting_lbys + 70;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb05[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff05 = doublependulum_QP_solver_shooting_dl_aff + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff05 = doublependulum_QP_solver_shooting_ds_aff + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc05 = doublependulum_QP_solver_shooting_dl_cc + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc05 = doublependulum_QP_solver_shooting_ds_cc + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl05 = doublependulum_QP_solver_shooting_ccrhs + 70;
int doublependulum_QP_solver_shooting_ubIdx05[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub05 = doublependulum_QP_solver_shooting_l + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub05 = doublependulum_QP_solver_shooting_s + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub05 = doublependulum_QP_solver_shooting_lbys + 77;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub05[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff05 = doublependulum_QP_solver_shooting_dl_aff + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff05 = doublependulum_QP_solver_shooting_ds_aff + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc05 = doublependulum_QP_solver_shooting_dl_cc + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc05 = doublependulum_QP_solver_shooting_ds_cc + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub05 = doublependulum_QP_solver_shooting_ccrhs + 77;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi05[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W05[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd05[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd05[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z06 = doublependulum_QP_solver_shooting_z + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff06 = doublependulum_QP_solver_shooting_dz_aff + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc06 = doublependulum_QP_solver_shooting_dz_cc + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd06 = doublependulum_QP_solver_shooting_rd + 42;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd06[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost06 = doublependulum_QP_solver_shooting_grad_cost + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq06 = doublependulum_QP_solver_shooting_grad_eq + 42;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq06 = doublependulum_QP_solver_shooting_grad_ineq + 42;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv06[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v06 = doublependulum_QP_solver_shooting_v + 6;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc06[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff06 = doublependulum_QP_solver_shooting_dv_aff + 6;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc06 = doublependulum_QP_solver_shooting_dv_cc + 6;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V06[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c06[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx06[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb06 = doublependulum_QP_solver_shooting_l + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb06 = doublependulum_QP_solver_shooting_s + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb06 = doublependulum_QP_solver_shooting_lbys + 84;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb06[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff06 = doublependulum_QP_solver_shooting_dl_aff + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff06 = doublependulum_QP_solver_shooting_ds_aff + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc06 = doublependulum_QP_solver_shooting_dl_cc + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc06 = doublependulum_QP_solver_shooting_ds_cc + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl06 = doublependulum_QP_solver_shooting_ccrhs + 84;
int doublependulum_QP_solver_shooting_ubIdx06[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub06 = doublependulum_QP_solver_shooting_l + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub06 = doublependulum_QP_solver_shooting_s + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub06 = doublependulum_QP_solver_shooting_lbys + 91;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub06[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff06 = doublependulum_QP_solver_shooting_dl_aff + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff06 = doublependulum_QP_solver_shooting_ds_aff + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc06 = doublependulum_QP_solver_shooting_dl_cc + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc06 = doublependulum_QP_solver_shooting_ds_cc + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub06 = doublependulum_QP_solver_shooting_ccrhs + 91;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi06[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W06[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd06[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd06[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z07 = doublependulum_QP_solver_shooting_z + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff07 = doublependulum_QP_solver_shooting_dz_aff + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc07 = doublependulum_QP_solver_shooting_dz_cc + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd07 = doublependulum_QP_solver_shooting_rd + 49;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd07[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost07 = doublependulum_QP_solver_shooting_grad_cost + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq07 = doublependulum_QP_solver_shooting_grad_eq + 49;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq07 = doublependulum_QP_solver_shooting_grad_ineq + 49;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv07[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v07 = doublependulum_QP_solver_shooting_v + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc07[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff07 = doublependulum_QP_solver_shooting_dv_aff + 7;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc07 = doublependulum_QP_solver_shooting_dv_cc + 7;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V07[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c07[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx07[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb07 = doublependulum_QP_solver_shooting_l + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb07 = doublependulum_QP_solver_shooting_s + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb07 = doublependulum_QP_solver_shooting_lbys + 98;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb07[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff07 = doublependulum_QP_solver_shooting_dl_aff + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff07 = doublependulum_QP_solver_shooting_ds_aff + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc07 = doublependulum_QP_solver_shooting_dl_cc + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc07 = doublependulum_QP_solver_shooting_ds_cc + 98;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl07 = doublependulum_QP_solver_shooting_ccrhs + 98;
int doublependulum_QP_solver_shooting_ubIdx07[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub07 = doublependulum_QP_solver_shooting_l + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub07 = doublependulum_QP_solver_shooting_s + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub07 = doublependulum_QP_solver_shooting_lbys + 105;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub07[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff07 = doublependulum_QP_solver_shooting_dl_aff + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff07 = doublependulum_QP_solver_shooting_ds_aff + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc07 = doublependulum_QP_solver_shooting_dl_cc + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc07 = doublependulum_QP_solver_shooting_ds_cc + 105;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub07 = doublependulum_QP_solver_shooting_ccrhs + 105;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi07[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W07[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd07[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd07[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z08 = doublependulum_QP_solver_shooting_z + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff08 = doublependulum_QP_solver_shooting_dz_aff + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc08 = doublependulum_QP_solver_shooting_dz_cc + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd08 = doublependulum_QP_solver_shooting_rd + 56;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd08[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost08 = doublependulum_QP_solver_shooting_grad_cost + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq08 = doublependulum_QP_solver_shooting_grad_eq + 56;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq08 = doublependulum_QP_solver_shooting_grad_ineq + 56;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv08[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v08 = doublependulum_QP_solver_shooting_v + 8;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc08[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff08 = doublependulum_QP_solver_shooting_dv_aff + 8;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc08 = doublependulum_QP_solver_shooting_dv_cc + 8;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V08[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c08[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx08[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb08 = doublependulum_QP_solver_shooting_l + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb08 = doublependulum_QP_solver_shooting_s + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb08 = doublependulum_QP_solver_shooting_lbys + 112;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb08[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff08 = doublependulum_QP_solver_shooting_dl_aff + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff08 = doublependulum_QP_solver_shooting_ds_aff + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc08 = doublependulum_QP_solver_shooting_dl_cc + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc08 = doublependulum_QP_solver_shooting_ds_cc + 112;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl08 = doublependulum_QP_solver_shooting_ccrhs + 112;
int doublependulum_QP_solver_shooting_ubIdx08[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub08 = doublependulum_QP_solver_shooting_l + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub08 = doublependulum_QP_solver_shooting_s + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub08 = doublependulum_QP_solver_shooting_lbys + 119;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub08[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff08 = doublependulum_QP_solver_shooting_dl_aff + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff08 = doublependulum_QP_solver_shooting_ds_aff + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc08 = doublependulum_QP_solver_shooting_dl_cc + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc08 = doublependulum_QP_solver_shooting_ds_cc + 119;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub08 = doublependulum_QP_solver_shooting_ccrhs + 119;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi08[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W08[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd08[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd08[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z09 = doublependulum_QP_solver_shooting_z + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff09 = doublependulum_QP_solver_shooting_dz_aff + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc09 = doublependulum_QP_solver_shooting_dz_cc + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd09 = doublependulum_QP_solver_shooting_rd + 63;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd09[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost09 = doublependulum_QP_solver_shooting_grad_cost + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq09 = doublependulum_QP_solver_shooting_grad_eq + 63;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq09 = doublependulum_QP_solver_shooting_grad_ineq + 63;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv09[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v09 = doublependulum_QP_solver_shooting_v + 9;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc09[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff09 = doublependulum_QP_solver_shooting_dv_aff + 9;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc09 = doublependulum_QP_solver_shooting_dv_cc + 9;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V09[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c09[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx09[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb09 = doublependulum_QP_solver_shooting_l + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb09 = doublependulum_QP_solver_shooting_s + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb09 = doublependulum_QP_solver_shooting_lbys + 126;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb09[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff09 = doublependulum_QP_solver_shooting_dl_aff + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff09 = doublependulum_QP_solver_shooting_ds_aff + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc09 = doublependulum_QP_solver_shooting_dl_cc + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc09 = doublependulum_QP_solver_shooting_ds_cc + 126;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl09 = doublependulum_QP_solver_shooting_ccrhs + 126;
int doublependulum_QP_solver_shooting_ubIdx09[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub09 = doublependulum_QP_solver_shooting_l + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub09 = doublependulum_QP_solver_shooting_s + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub09 = doublependulum_QP_solver_shooting_lbys + 133;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub09[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff09 = doublependulum_QP_solver_shooting_dl_aff + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff09 = doublependulum_QP_solver_shooting_ds_aff + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc09 = doublependulum_QP_solver_shooting_dl_cc + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc09 = doublependulum_QP_solver_shooting_ds_cc + 133;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub09 = doublependulum_QP_solver_shooting_ccrhs + 133;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi09[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W09[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd09[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd09[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z10 = doublependulum_QP_solver_shooting_z + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff10 = doublependulum_QP_solver_shooting_dz_aff + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc10 = doublependulum_QP_solver_shooting_dz_cc + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd10 = doublependulum_QP_solver_shooting_rd + 70;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd10[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost10 = doublependulum_QP_solver_shooting_grad_cost + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq10 = doublependulum_QP_solver_shooting_grad_eq + 70;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq10 = doublependulum_QP_solver_shooting_grad_ineq + 70;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv10[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v10 = doublependulum_QP_solver_shooting_v + 10;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc10[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff10 = doublependulum_QP_solver_shooting_dv_aff + 10;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc10 = doublependulum_QP_solver_shooting_dv_cc + 10;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V10[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c10[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx10[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb10 = doublependulum_QP_solver_shooting_l + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb10 = doublependulum_QP_solver_shooting_s + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb10 = doublependulum_QP_solver_shooting_lbys + 140;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb10[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff10 = doublependulum_QP_solver_shooting_dl_aff + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff10 = doublependulum_QP_solver_shooting_ds_aff + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc10 = doublependulum_QP_solver_shooting_dl_cc + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc10 = doublependulum_QP_solver_shooting_ds_cc + 140;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl10 = doublependulum_QP_solver_shooting_ccrhs + 140;
int doublependulum_QP_solver_shooting_ubIdx10[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub10 = doublependulum_QP_solver_shooting_l + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub10 = doublependulum_QP_solver_shooting_s + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub10 = doublependulum_QP_solver_shooting_lbys + 147;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub10[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff10 = doublependulum_QP_solver_shooting_dl_aff + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff10 = doublependulum_QP_solver_shooting_ds_aff + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc10 = doublependulum_QP_solver_shooting_dl_cc + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc10 = doublependulum_QP_solver_shooting_ds_cc + 147;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub10 = doublependulum_QP_solver_shooting_ccrhs + 147;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi10[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W10[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd10[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd10[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z11 = doublependulum_QP_solver_shooting_z + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff11 = doublependulum_QP_solver_shooting_dz_aff + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc11 = doublependulum_QP_solver_shooting_dz_cc + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd11 = doublependulum_QP_solver_shooting_rd + 77;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd11[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost11 = doublependulum_QP_solver_shooting_grad_cost + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq11 = doublependulum_QP_solver_shooting_grad_eq + 77;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq11 = doublependulum_QP_solver_shooting_grad_ineq + 77;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv11[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v11 = doublependulum_QP_solver_shooting_v + 11;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc11[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff11 = doublependulum_QP_solver_shooting_dv_aff + 11;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc11 = doublependulum_QP_solver_shooting_dv_cc + 11;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V11[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c11[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx11[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb11 = doublependulum_QP_solver_shooting_l + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb11 = doublependulum_QP_solver_shooting_s + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb11 = doublependulum_QP_solver_shooting_lbys + 154;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb11[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff11 = doublependulum_QP_solver_shooting_dl_aff + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff11 = doublependulum_QP_solver_shooting_ds_aff + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc11 = doublependulum_QP_solver_shooting_dl_cc + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc11 = doublependulum_QP_solver_shooting_ds_cc + 154;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl11 = doublependulum_QP_solver_shooting_ccrhs + 154;
int doublependulum_QP_solver_shooting_ubIdx11[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub11 = doublependulum_QP_solver_shooting_l + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub11 = doublependulum_QP_solver_shooting_s + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub11 = doublependulum_QP_solver_shooting_lbys + 161;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub11[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff11 = doublependulum_QP_solver_shooting_dl_aff + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff11 = doublependulum_QP_solver_shooting_ds_aff + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc11 = doublependulum_QP_solver_shooting_dl_cc + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc11 = doublependulum_QP_solver_shooting_ds_cc + 161;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub11 = doublependulum_QP_solver_shooting_ccrhs + 161;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi11[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W11[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd11[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd11[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z12 = doublependulum_QP_solver_shooting_z + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff12 = doublependulum_QP_solver_shooting_dz_aff + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc12 = doublependulum_QP_solver_shooting_dz_cc + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd12 = doublependulum_QP_solver_shooting_rd + 84;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd12[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost12 = doublependulum_QP_solver_shooting_grad_cost + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq12 = doublependulum_QP_solver_shooting_grad_eq + 84;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq12 = doublependulum_QP_solver_shooting_grad_ineq + 84;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv12[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_v12 = doublependulum_QP_solver_shooting_v + 12;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_re12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_beta12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_betacc12[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvaff12 = doublependulum_QP_solver_shooting_dv_aff + 12;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dvcc12 = doublependulum_QP_solver_shooting_dv_cc + 12;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_V12[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Yd12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ld12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_yy12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_bmy12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_c12[1] = {0.0000000000000000E+000};
int doublependulum_QP_solver_shooting_lbIdx12[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb12 = doublependulum_QP_solver_shooting_l + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb12 = doublependulum_QP_solver_shooting_s + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb12 = doublependulum_QP_solver_shooting_lbys + 168;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb12[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff12 = doublependulum_QP_solver_shooting_dl_aff + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff12 = doublependulum_QP_solver_shooting_ds_aff + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc12 = doublependulum_QP_solver_shooting_dl_cc + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc12 = doublependulum_QP_solver_shooting_ds_cc + 168;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl12 = doublependulum_QP_solver_shooting_ccrhs + 168;
int doublependulum_QP_solver_shooting_ubIdx12[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub12 = doublependulum_QP_solver_shooting_l + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub12 = doublependulum_QP_solver_shooting_s + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub12 = doublependulum_QP_solver_shooting_lbys + 175;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub12[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff12 = doublependulum_QP_solver_shooting_dl_aff + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff12 = doublependulum_QP_solver_shooting_ds_aff + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc12 = doublependulum_QP_solver_shooting_dl_cc + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc12 = doublependulum_QP_solver_shooting_ds_cc + 175;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub12 = doublependulum_QP_solver_shooting_ccrhs + 175;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi12[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W12[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Ysd12[1];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lsd12[1];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_z13 = doublependulum_QP_solver_shooting_z + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzaff13 = doublependulum_QP_solver_shooting_dz_aff + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dzcc13 = doublependulum_QP_solver_shooting_dz_cc + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_rd13 = doublependulum_QP_solver_shooting_rd + 91;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Lbyrd13[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_cost13 = doublependulum_QP_solver_shooting_grad_cost + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_eq13 = doublependulum_QP_solver_shooting_grad_eq + 91;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_grad_ineq13 = doublependulum_QP_solver_shooting_grad_ineq + 91;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_ctv13[7];
int doublependulum_QP_solver_shooting_lbIdx13[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llb13 = doublependulum_QP_solver_shooting_l + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_slb13 = doublependulum_QP_solver_shooting_s + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_llbbyslb13 = doublependulum_QP_solver_shooting_lbys + 182;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_rilb13[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbaff13 = doublependulum_QP_solver_shooting_dl_aff + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbaff13 = doublependulum_QP_solver_shooting_ds_aff + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dllbcc13 = doublependulum_QP_solver_shooting_dl_cc + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dslbcc13 = doublependulum_QP_solver_shooting_ds_cc + 182;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsl13 = doublependulum_QP_solver_shooting_ccrhs + 182;
int doublependulum_QP_solver_shooting_ubIdx13[7] = {0, 1, 2, 3, 4, 5, 6};
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lub13 = doublependulum_QP_solver_shooting_l + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_sub13 = doublependulum_QP_solver_shooting_s + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_lubbysub13 = doublependulum_QP_solver_shooting_lbys + 189;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_riub13[7];
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubaff13 = doublependulum_QP_solver_shooting_dl_aff + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubaff13 = doublependulum_QP_solver_shooting_ds_aff + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dlubcc13 = doublependulum_QP_solver_shooting_dl_cc + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_dsubcc13 = doublependulum_QP_solver_shooting_ds_cc + 189;
doublependulum_QP_solver_shooting_FLOAT* doublependulum_QP_solver_shooting_ccrhsub13 = doublependulum_QP_solver_shooting_ccrhs + 189;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Phi13[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_W13[7];
doublependulum_QP_solver_shooting_FLOAT musigma;
doublependulum_QP_solver_shooting_FLOAT sigma_3rdroot;
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Diag1_0[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_Diag2_0[7];
doublependulum_QP_solver_shooting_FLOAT doublependulum_QP_solver_shooting_L_0[21];




/* SOLVER CODE --------------------------------------------------------- */
int doublependulum_QP_solver_shooting_solve(doublependulum_QP_solver_shooting_params* params, doublependulum_QP_solver_shooting_output* output, doublependulum_QP_solver_shooting_info* info)
{	
int exitcode;

#if doublependulum_QP_solver_shooting_SET_TIMING == 1
	doublependulum_QP_solver_shooting_timer solvertimer;
	doublependulum_QP_solver_shooting_tic(&solvertimer);
#endif
/* FUNCTION CALLS INTO LA LIBRARY -------------------------------------- */
info->it = 0;
doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_98(doublependulum_QP_solver_shooting_z, 0);
doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_13(doublependulum_QP_solver_shooting_v, 1);
doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_196(doublependulum_QP_solver_shooting_l, 10);
doublependulum_QP_solver_shooting_LA_INITIALIZEVECTOR_196(doublependulum_QP_solver_shooting_s, 10);
info->mu = 0;
doublependulum_QP_solver_shooting_LA_DOTACC_196(doublependulum_QP_solver_shooting_l, doublependulum_QP_solver_shooting_s, &info->mu);
info->mu /= 196;
while( 1 ){
info->pobj = 0;
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H1, params->f1, doublependulum_QP_solver_shooting_z00, doublependulum_QP_solver_shooting_grad_cost00, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H2, params->f2, doublependulum_QP_solver_shooting_z01, doublependulum_QP_solver_shooting_grad_cost01, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H3, params->f3, doublependulum_QP_solver_shooting_z02, doublependulum_QP_solver_shooting_grad_cost02, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H4, params->f4, doublependulum_QP_solver_shooting_z03, doublependulum_QP_solver_shooting_grad_cost03, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H5, params->f5, doublependulum_QP_solver_shooting_z04, doublependulum_QP_solver_shooting_grad_cost04, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H6, params->f6, doublependulum_QP_solver_shooting_z05, doublependulum_QP_solver_shooting_grad_cost05, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H7, params->f7, doublependulum_QP_solver_shooting_z06, doublependulum_QP_solver_shooting_grad_cost06, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H8, params->f8, doublependulum_QP_solver_shooting_z07, doublependulum_QP_solver_shooting_grad_cost07, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H9, params->f9, doublependulum_QP_solver_shooting_z08, doublependulum_QP_solver_shooting_grad_cost08, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H10, params->f10, doublependulum_QP_solver_shooting_z09, doublependulum_QP_solver_shooting_grad_cost09, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H11, params->f11, doublependulum_QP_solver_shooting_z10, doublependulum_QP_solver_shooting_grad_cost10, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H12, params->f12, doublependulum_QP_solver_shooting_z11, doublependulum_QP_solver_shooting_grad_cost11, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H13, params->f13, doublependulum_QP_solver_shooting_z12, doublependulum_QP_solver_shooting_grad_cost12, &info->pobj);
doublependulum_QP_solver_shooting_LA_DIAG_QUADFCN_7(params->H14, params->f14, doublependulum_QP_solver_shooting_z13, doublependulum_QP_solver_shooting_grad_cost13, &info->pobj);
info->res_eq = 0;
info->dgap = 0;
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z00, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z01, doublependulum_QP_solver_shooting_c00, doublependulum_QP_solver_shooting_v00, doublependulum_QP_solver_shooting_re00, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z01, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z02, doublependulum_QP_solver_shooting_c01, doublependulum_QP_solver_shooting_v01, doublependulum_QP_solver_shooting_re01, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z02, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z03, doublependulum_QP_solver_shooting_c02, doublependulum_QP_solver_shooting_v02, doublependulum_QP_solver_shooting_re02, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z03, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z04, doublependulum_QP_solver_shooting_c03, doublependulum_QP_solver_shooting_v03, doublependulum_QP_solver_shooting_re03, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z04, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z05, doublependulum_QP_solver_shooting_c04, doublependulum_QP_solver_shooting_v04, doublependulum_QP_solver_shooting_re04, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z05, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z06, doublependulum_QP_solver_shooting_c05, doublependulum_QP_solver_shooting_v05, doublependulum_QP_solver_shooting_re05, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z06, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z07, doublependulum_QP_solver_shooting_c06, doublependulum_QP_solver_shooting_v06, doublependulum_QP_solver_shooting_re06, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z07, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z08, doublependulum_QP_solver_shooting_c07, doublependulum_QP_solver_shooting_v07, doublependulum_QP_solver_shooting_re07, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z08, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z09, doublependulum_QP_solver_shooting_c08, doublependulum_QP_solver_shooting_v08, doublependulum_QP_solver_shooting_re08, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z09, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z10, doublependulum_QP_solver_shooting_c09, doublependulum_QP_solver_shooting_v09, doublependulum_QP_solver_shooting_re09, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z10, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z11, doublependulum_QP_solver_shooting_c10, doublependulum_QP_solver_shooting_v10, doublependulum_QP_solver_shooting_re10, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z11, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z12, doublependulum_QP_solver_shooting_c11, doublependulum_QP_solver_shooting_v11, doublependulum_QP_solver_shooting_re11, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MVMSUB3_1_7_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_z12, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_z13, doublependulum_QP_solver_shooting_c12, doublependulum_QP_solver_shooting_v12, doublependulum_QP_solver_shooting_re12, &info->dgap, &info->res_eq);
doublependulum_QP_solver_shooting_LA_DENSE_MTVM_1_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v00, doublependulum_QP_solver_shooting_grad_eq00);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v01, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v00, doublependulum_QP_solver_shooting_grad_eq01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v02, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v01, doublependulum_QP_solver_shooting_grad_eq02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v03, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v02, doublependulum_QP_solver_shooting_grad_eq03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v04, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v03, doublependulum_QP_solver_shooting_grad_eq04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v05, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v04, doublependulum_QP_solver_shooting_grad_eq05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v06, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v05, doublependulum_QP_solver_shooting_grad_eq06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v07, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v06, doublependulum_QP_solver_shooting_grad_eq07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v08, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v07, doublependulum_QP_solver_shooting_grad_eq08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v09, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v08, doublependulum_QP_solver_shooting_grad_eq09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v10, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v09, doublependulum_QP_solver_shooting_grad_eq10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v11, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v10, doublependulum_QP_solver_shooting_grad_eq11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_v12, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v11, doublependulum_QP_solver_shooting_grad_eq12);
doublependulum_QP_solver_shooting_LA_DIAGZERO_MTVM_1_7(doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_v12, doublependulum_QP_solver_shooting_grad_eq13);
info->res_ineq = 0;
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb1, doublependulum_QP_solver_shooting_z00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_llb00, doublependulum_QP_solver_shooting_slb00, doublependulum_QP_solver_shooting_rilb00, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z00, doublependulum_QP_solver_shooting_ubIdx00, params->ub1, doublependulum_QP_solver_shooting_lub00, doublependulum_QP_solver_shooting_sub00, doublependulum_QP_solver_shooting_riub00, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb2, doublependulum_QP_solver_shooting_z01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_llb01, doublependulum_QP_solver_shooting_slb01, doublependulum_QP_solver_shooting_rilb01, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z01, doublependulum_QP_solver_shooting_ubIdx01, params->ub2, doublependulum_QP_solver_shooting_lub01, doublependulum_QP_solver_shooting_sub01, doublependulum_QP_solver_shooting_riub01, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb3, doublependulum_QP_solver_shooting_z02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_llb02, doublependulum_QP_solver_shooting_slb02, doublependulum_QP_solver_shooting_rilb02, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z02, doublependulum_QP_solver_shooting_ubIdx02, params->ub3, doublependulum_QP_solver_shooting_lub02, doublependulum_QP_solver_shooting_sub02, doublependulum_QP_solver_shooting_riub02, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb4, doublependulum_QP_solver_shooting_z03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_llb03, doublependulum_QP_solver_shooting_slb03, doublependulum_QP_solver_shooting_rilb03, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z03, doublependulum_QP_solver_shooting_ubIdx03, params->ub4, doublependulum_QP_solver_shooting_lub03, doublependulum_QP_solver_shooting_sub03, doublependulum_QP_solver_shooting_riub03, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb5, doublependulum_QP_solver_shooting_z04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_llb04, doublependulum_QP_solver_shooting_slb04, doublependulum_QP_solver_shooting_rilb04, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z04, doublependulum_QP_solver_shooting_ubIdx04, params->ub5, doublependulum_QP_solver_shooting_lub04, doublependulum_QP_solver_shooting_sub04, doublependulum_QP_solver_shooting_riub04, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb6, doublependulum_QP_solver_shooting_z05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_llb05, doublependulum_QP_solver_shooting_slb05, doublependulum_QP_solver_shooting_rilb05, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z05, doublependulum_QP_solver_shooting_ubIdx05, params->ub6, doublependulum_QP_solver_shooting_lub05, doublependulum_QP_solver_shooting_sub05, doublependulum_QP_solver_shooting_riub05, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb7, doublependulum_QP_solver_shooting_z06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_llb06, doublependulum_QP_solver_shooting_slb06, doublependulum_QP_solver_shooting_rilb06, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z06, doublependulum_QP_solver_shooting_ubIdx06, params->ub7, doublependulum_QP_solver_shooting_lub06, doublependulum_QP_solver_shooting_sub06, doublependulum_QP_solver_shooting_riub06, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb8, doublependulum_QP_solver_shooting_z07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_llb07, doublependulum_QP_solver_shooting_slb07, doublependulum_QP_solver_shooting_rilb07, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z07, doublependulum_QP_solver_shooting_ubIdx07, params->ub8, doublependulum_QP_solver_shooting_lub07, doublependulum_QP_solver_shooting_sub07, doublependulum_QP_solver_shooting_riub07, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb9, doublependulum_QP_solver_shooting_z08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_llb08, doublependulum_QP_solver_shooting_slb08, doublependulum_QP_solver_shooting_rilb08, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z08, doublependulum_QP_solver_shooting_ubIdx08, params->ub9, doublependulum_QP_solver_shooting_lub08, doublependulum_QP_solver_shooting_sub08, doublependulum_QP_solver_shooting_riub08, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb10, doublependulum_QP_solver_shooting_z09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_llb09, doublependulum_QP_solver_shooting_slb09, doublependulum_QP_solver_shooting_rilb09, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z09, doublependulum_QP_solver_shooting_ubIdx09, params->ub10, doublependulum_QP_solver_shooting_lub09, doublependulum_QP_solver_shooting_sub09, doublependulum_QP_solver_shooting_riub09, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb11, doublependulum_QP_solver_shooting_z10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_llb10, doublependulum_QP_solver_shooting_slb10, doublependulum_QP_solver_shooting_rilb10, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z10, doublependulum_QP_solver_shooting_ubIdx10, params->ub11, doublependulum_QP_solver_shooting_lub10, doublependulum_QP_solver_shooting_sub10, doublependulum_QP_solver_shooting_riub10, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb12, doublependulum_QP_solver_shooting_z11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_llb11, doublependulum_QP_solver_shooting_slb11, doublependulum_QP_solver_shooting_rilb11, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z11, doublependulum_QP_solver_shooting_ubIdx11, params->ub12, doublependulum_QP_solver_shooting_lub11, doublependulum_QP_solver_shooting_sub11, doublependulum_QP_solver_shooting_riub11, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb13, doublependulum_QP_solver_shooting_z12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_llb12, doublependulum_QP_solver_shooting_slb12, doublependulum_QP_solver_shooting_rilb12, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z12, doublependulum_QP_solver_shooting_ubIdx12, params->ub13, doublependulum_QP_solver_shooting_lub12, doublependulum_QP_solver_shooting_sub12, doublependulum_QP_solver_shooting_riub12, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD3_7(params->lb14, doublependulum_QP_solver_shooting_z13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_llb13, doublependulum_QP_solver_shooting_slb13, doublependulum_QP_solver_shooting_rilb13, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_VSUBADD2_7(doublependulum_QP_solver_shooting_z13, doublependulum_QP_solver_shooting_ubIdx13, params->ub14, doublependulum_QP_solver_shooting_lub13, doublependulum_QP_solver_shooting_sub13, doublependulum_QP_solver_shooting_riub13, &info->dgap, &info->res_ineq);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub00, doublependulum_QP_solver_shooting_sub00, doublependulum_QP_solver_shooting_riub00, doublependulum_QP_solver_shooting_llb00, doublependulum_QP_solver_shooting_slb00, doublependulum_QP_solver_shooting_rilb00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_ubIdx00, doublependulum_QP_solver_shooting_grad_ineq00, doublependulum_QP_solver_shooting_lubbysub00, doublependulum_QP_solver_shooting_llbbyslb00);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub01, doublependulum_QP_solver_shooting_sub01, doublependulum_QP_solver_shooting_riub01, doublependulum_QP_solver_shooting_llb01, doublependulum_QP_solver_shooting_slb01, doublependulum_QP_solver_shooting_rilb01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_ubIdx01, doublependulum_QP_solver_shooting_grad_ineq01, doublependulum_QP_solver_shooting_lubbysub01, doublependulum_QP_solver_shooting_llbbyslb01);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub02, doublependulum_QP_solver_shooting_sub02, doublependulum_QP_solver_shooting_riub02, doublependulum_QP_solver_shooting_llb02, doublependulum_QP_solver_shooting_slb02, doublependulum_QP_solver_shooting_rilb02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_ubIdx02, doublependulum_QP_solver_shooting_grad_ineq02, doublependulum_QP_solver_shooting_lubbysub02, doublependulum_QP_solver_shooting_llbbyslb02);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub03, doublependulum_QP_solver_shooting_sub03, doublependulum_QP_solver_shooting_riub03, doublependulum_QP_solver_shooting_llb03, doublependulum_QP_solver_shooting_slb03, doublependulum_QP_solver_shooting_rilb03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_ubIdx03, doublependulum_QP_solver_shooting_grad_ineq03, doublependulum_QP_solver_shooting_lubbysub03, doublependulum_QP_solver_shooting_llbbyslb03);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub04, doublependulum_QP_solver_shooting_sub04, doublependulum_QP_solver_shooting_riub04, doublependulum_QP_solver_shooting_llb04, doublependulum_QP_solver_shooting_slb04, doublependulum_QP_solver_shooting_rilb04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_ubIdx04, doublependulum_QP_solver_shooting_grad_ineq04, doublependulum_QP_solver_shooting_lubbysub04, doublependulum_QP_solver_shooting_llbbyslb04);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub05, doublependulum_QP_solver_shooting_sub05, doublependulum_QP_solver_shooting_riub05, doublependulum_QP_solver_shooting_llb05, doublependulum_QP_solver_shooting_slb05, doublependulum_QP_solver_shooting_rilb05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_ubIdx05, doublependulum_QP_solver_shooting_grad_ineq05, doublependulum_QP_solver_shooting_lubbysub05, doublependulum_QP_solver_shooting_llbbyslb05);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub06, doublependulum_QP_solver_shooting_sub06, doublependulum_QP_solver_shooting_riub06, doublependulum_QP_solver_shooting_llb06, doublependulum_QP_solver_shooting_slb06, doublependulum_QP_solver_shooting_rilb06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_ubIdx06, doublependulum_QP_solver_shooting_grad_ineq06, doublependulum_QP_solver_shooting_lubbysub06, doublependulum_QP_solver_shooting_llbbyslb06);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub07, doublependulum_QP_solver_shooting_sub07, doublependulum_QP_solver_shooting_riub07, doublependulum_QP_solver_shooting_llb07, doublependulum_QP_solver_shooting_slb07, doublependulum_QP_solver_shooting_rilb07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_ubIdx07, doublependulum_QP_solver_shooting_grad_ineq07, doublependulum_QP_solver_shooting_lubbysub07, doublependulum_QP_solver_shooting_llbbyslb07);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub08, doublependulum_QP_solver_shooting_sub08, doublependulum_QP_solver_shooting_riub08, doublependulum_QP_solver_shooting_llb08, doublependulum_QP_solver_shooting_slb08, doublependulum_QP_solver_shooting_rilb08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_ubIdx08, doublependulum_QP_solver_shooting_grad_ineq08, doublependulum_QP_solver_shooting_lubbysub08, doublependulum_QP_solver_shooting_llbbyslb08);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub09, doublependulum_QP_solver_shooting_sub09, doublependulum_QP_solver_shooting_riub09, doublependulum_QP_solver_shooting_llb09, doublependulum_QP_solver_shooting_slb09, doublependulum_QP_solver_shooting_rilb09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_ubIdx09, doublependulum_QP_solver_shooting_grad_ineq09, doublependulum_QP_solver_shooting_lubbysub09, doublependulum_QP_solver_shooting_llbbyslb09);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub10, doublependulum_QP_solver_shooting_sub10, doublependulum_QP_solver_shooting_riub10, doublependulum_QP_solver_shooting_llb10, doublependulum_QP_solver_shooting_slb10, doublependulum_QP_solver_shooting_rilb10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_ubIdx10, doublependulum_QP_solver_shooting_grad_ineq10, doublependulum_QP_solver_shooting_lubbysub10, doublependulum_QP_solver_shooting_llbbyslb10);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub11, doublependulum_QP_solver_shooting_sub11, doublependulum_QP_solver_shooting_riub11, doublependulum_QP_solver_shooting_llb11, doublependulum_QP_solver_shooting_slb11, doublependulum_QP_solver_shooting_rilb11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_ubIdx11, doublependulum_QP_solver_shooting_grad_ineq11, doublependulum_QP_solver_shooting_lubbysub11, doublependulum_QP_solver_shooting_llbbyslb11);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub12, doublependulum_QP_solver_shooting_sub12, doublependulum_QP_solver_shooting_riub12, doublependulum_QP_solver_shooting_llb12, doublependulum_QP_solver_shooting_slb12, doublependulum_QP_solver_shooting_rilb12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_ubIdx12, doublependulum_QP_solver_shooting_grad_ineq12, doublependulum_QP_solver_shooting_lubbysub12, doublependulum_QP_solver_shooting_llbbyslb12);
doublependulum_QP_solver_shooting_LA_INEQ_B_GRAD_7_7_7(doublependulum_QP_solver_shooting_lub13, doublependulum_QP_solver_shooting_sub13, doublependulum_QP_solver_shooting_riub13, doublependulum_QP_solver_shooting_llb13, doublependulum_QP_solver_shooting_slb13, doublependulum_QP_solver_shooting_rilb13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_ubIdx13, doublependulum_QP_solver_shooting_grad_ineq13, doublependulum_QP_solver_shooting_lubbysub13, doublependulum_QP_solver_shooting_llbbyslb13);
info->dobj = info->pobj - info->dgap;
info->rdgap = info->pobj ? info->dgap / info->pobj : 1e6;
if( info->rdgap < 0 ) info->rdgap = -info->rdgap;
if( info->mu < doublependulum_QP_solver_shooting_SET_ACC_KKTCOMPL
    && (info->rdgap < doublependulum_QP_solver_shooting_SET_ACC_RDGAP || info->dgap < doublependulum_QP_solver_shooting_SET_ACC_KKTCOMPL)
    && info->res_eq < doublependulum_QP_solver_shooting_SET_ACC_RESEQ
    && info->res_ineq < doublependulum_QP_solver_shooting_SET_ACC_RESINEQ ){
exitcode = doublependulum_QP_solver_shooting_OPTIMAL; break; }
if( info->it == doublependulum_QP_solver_shooting_SET_MAXIT ){
exitcode = doublependulum_QP_solver_shooting_MAXITREACHED; break; }
doublependulum_QP_solver_shooting_LA_VVADD3_98(doublependulum_QP_solver_shooting_grad_cost, doublependulum_QP_solver_shooting_grad_eq, doublependulum_QP_solver_shooting_grad_ineq, doublependulum_QP_solver_shooting_rd);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H1, doublependulum_QP_solver_shooting_llbbyslb00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_lubbysub00, doublependulum_QP_solver_shooting_ubIdx00, doublependulum_QP_solver_shooting_Phi00);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi00, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V00);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi00, doublependulum_QP_solver_shooting_rd00, doublependulum_QP_solver_shooting_Lbyrd00);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H2, doublependulum_QP_solver_shooting_llbbyslb01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_lubbysub01, doublependulum_QP_solver_shooting_ubIdx01, doublependulum_QP_solver_shooting_Phi01);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V01);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W01, doublependulum_QP_solver_shooting_V01, doublependulum_QP_solver_shooting_Ysd01);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_rd01, doublependulum_QP_solver_shooting_Lbyrd01);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H3, doublependulum_QP_solver_shooting_llbbyslb02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_lubbysub02, doublependulum_QP_solver_shooting_ubIdx02, doublependulum_QP_solver_shooting_Phi02);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V02);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W02, doublependulum_QP_solver_shooting_V02, doublependulum_QP_solver_shooting_Ysd02);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_rd02, doublependulum_QP_solver_shooting_Lbyrd02);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H4, doublependulum_QP_solver_shooting_llbbyslb03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_lubbysub03, doublependulum_QP_solver_shooting_ubIdx03, doublependulum_QP_solver_shooting_Phi03);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V03);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W03, doublependulum_QP_solver_shooting_V03, doublependulum_QP_solver_shooting_Ysd03);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_rd03, doublependulum_QP_solver_shooting_Lbyrd03);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H5, doublependulum_QP_solver_shooting_llbbyslb04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_lubbysub04, doublependulum_QP_solver_shooting_ubIdx04, doublependulum_QP_solver_shooting_Phi04);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V04);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W04, doublependulum_QP_solver_shooting_V04, doublependulum_QP_solver_shooting_Ysd04);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_rd04, doublependulum_QP_solver_shooting_Lbyrd04);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H6, doublependulum_QP_solver_shooting_llbbyslb05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_lubbysub05, doublependulum_QP_solver_shooting_ubIdx05, doublependulum_QP_solver_shooting_Phi05);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V05);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W05, doublependulum_QP_solver_shooting_V05, doublependulum_QP_solver_shooting_Ysd05);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_rd05, doublependulum_QP_solver_shooting_Lbyrd05);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H7, doublependulum_QP_solver_shooting_llbbyslb06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_lubbysub06, doublependulum_QP_solver_shooting_ubIdx06, doublependulum_QP_solver_shooting_Phi06);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V06);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W06, doublependulum_QP_solver_shooting_V06, doublependulum_QP_solver_shooting_Ysd06);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_rd06, doublependulum_QP_solver_shooting_Lbyrd06);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H8, doublependulum_QP_solver_shooting_llbbyslb07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_lubbysub07, doublependulum_QP_solver_shooting_ubIdx07, doublependulum_QP_solver_shooting_Phi07);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V07);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W07, doublependulum_QP_solver_shooting_V07, doublependulum_QP_solver_shooting_Ysd07);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_rd07, doublependulum_QP_solver_shooting_Lbyrd07);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H9, doublependulum_QP_solver_shooting_llbbyslb08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_lubbysub08, doublependulum_QP_solver_shooting_ubIdx08, doublependulum_QP_solver_shooting_Phi08);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V08);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W08, doublependulum_QP_solver_shooting_V08, doublependulum_QP_solver_shooting_Ysd08);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_rd08, doublependulum_QP_solver_shooting_Lbyrd08);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H10, doublependulum_QP_solver_shooting_llbbyslb09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_lubbysub09, doublependulum_QP_solver_shooting_ubIdx09, doublependulum_QP_solver_shooting_Phi09);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V09);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W09, doublependulum_QP_solver_shooting_V09, doublependulum_QP_solver_shooting_Ysd09);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_rd09, doublependulum_QP_solver_shooting_Lbyrd09);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H11, doublependulum_QP_solver_shooting_llbbyslb10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_lubbysub10, doublependulum_QP_solver_shooting_ubIdx10, doublependulum_QP_solver_shooting_Phi10);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V10);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W10, doublependulum_QP_solver_shooting_V10, doublependulum_QP_solver_shooting_Ysd10);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_rd10, doublependulum_QP_solver_shooting_Lbyrd10);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H12, doublependulum_QP_solver_shooting_llbbyslb11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_lubbysub11, doublependulum_QP_solver_shooting_ubIdx11, doublependulum_QP_solver_shooting_Phi11);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V11);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W11, doublependulum_QP_solver_shooting_V11, doublependulum_QP_solver_shooting_Ysd11);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_rd11, doublependulum_QP_solver_shooting_Lbyrd11);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H13, doublependulum_QP_solver_shooting_llbbyslb12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_lubbysub12, doublependulum_QP_solver_shooting_ubIdx12, doublependulum_QP_solver_shooting_Phi12);
doublependulum_QP_solver_shooting_LA_DIAG_MATRIXFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_V12);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W12);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMTM_1_7_1(doublependulum_QP_solver_shooting_W12, doublependulum_QP_solver_shooting_V12, doublependulum_QP_solver_shooting_Ysd12);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_rd12, doublependulum_QP_solver_shooting_Lbyrd12);
doublependulum_QP_solver_shooting_LA_DIAG_CHOL_ONELOOP_LBUB_7_7_7(params->H14, doublependulum_QP_solver_shooting_llbbyslb13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_lubbysub13, doublependulum_QP_solver_shooting_ubIdx13, doublependulum_QP_solver_shooting_Phi13);
doublependulum_QP_solver_shooting_LA_DIAG_DIAGZERO_MATRIXTFORWARDSUB_1_7(doublependulum_QP_solver_shooting_Phi13, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_W13);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi13, doublependulum_QP_solver_shooting_rd13, doublependulum_QP_solver_shooting_Lbyrd13);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V00, doublependulum_QP_solver_shooting_W01, doublependulum_QP_solver_shooting_Yd00);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V00, doublependulum_QP_solver_shooting_Lbyrd00, doublependulum_QP_solver_shooting_W01, doublependulum_QP_solver_shooting_Lbyrd01, doublependulum_QP_solver_shooting_re00, doublependulum_QP_solver_shooting_beta00);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V01, doublependulum_QP_solver_shooting_W02, doublependulum_QP_solver_shooting_Yd01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V01, doublependulum_QP_solver_shooting_Lbyrd01, doublependulum_QP_solver_shooting_W02, doublependulum_QP_solver_shooting_Lbyrd02, doublependulum_QP_solver_shooting_re01, doublependulum_QP_solver_shooting_beta01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V02, doublependulum_QP_solver_shooting_W03, doublependulum_QP_solver_shooting_Yd02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V02, doublependulum_QP_solver_shooting_Lbyrd02, doublependulum_QP_solver_shooting_W03, doublependulum_QP_solver_shooting_Lbyrd03, doublependulum_QP_solver_shooting_re02, doublependulum_QP_solver_shooting_beta02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V03, doublependulum_QP_solver_shooting_W04, doublependulum_QP_solver_shooting_Yd03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V03, doublependulum_QP_solver_shooting_Lbyrd03, doublependulum_QP_solver_shooting_W04, doublependulum_QP_solver_shooting_Lbyrd04, doublependulum_QP_solver_shooting_re03, doublependulum_QP_solver_shooting_beta03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V04, doublependulum_QP_solver_shooting_W05, doublependulum_QP_solver_shooting_Yd04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V04, doublependulum_QP_solver_shooting_Lbyrd04, doublependulum_QP_solver_shooting_W05, doublependulum_QP_solver_shooting_Lbyrd05, doublependulum_QP_solver_shooting_re04, doublependulum_QP_solver_shooting_beta04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V05, doublependulum_QP_solver_shooting_W06, doublependulum_QP_solver_shooting_Yd05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V05, doublependulum_QP_solver_shooting_Lbyrd05, doublependulum_QP_solver_shooting_W06, doublependulum_QP_solver_shooting_Lbyrd06, doublependulum_QP_solver_shooting_re05, doublependulum_QP_solver_shooting_beta05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V06, doublependulum_QP_solver_shooting_W07, doublependulum_QP_solver_shooting_Yd06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V06, doublependulum_QP_solver_shooting_Lbyrd06, doublependulum_QP_solver_shooting_W07, doublependulum_QP_solver_shooting_Lbyrd07, doublependulum_QP_solver_shooting_re06, doublependulum_QP_solver_shooting_beta06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V07, doublependulum_QP_solver_shooting_W08, doublependulum_QP_solver_shooting_Yd07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V07, doublependulum_QP_solver_shooting_Lbyrd07, doublependulum_QP_solver_shooting_W08, doublependulum_QP_solver_shooting_Lbyrd08, doublependulum_QP_solver_shooting_re07, doublependulum_QP_solver_shooting_beta07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V08, doublependulum_QP_solver_shooting_W09, doublependulum_QP_solver_shooting_Yd08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V08, doublependulum_QP_solver_shooting_Lbyrd08, doublependulum_QP_solver_shooting_W09, doublependulum_QP_solver_shooting_Lbyrd09, doublependulum_QP_solver_shooting_re08, doublependulum_QP_solver_shooting_beta08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V09, doublependulum_QP_solver_shooting_W10, doublependulum_QP_solver_shooting_Yd09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V09, doublependulum_QP_solver_shooting_Lbyrd09, doublependulum_QP_solver_shooting_W10, doublependulum_QP_solver_shooting_Lbyrd10, doublependulum_QP_solver_shooting_re09, doublependulum_QP_solver_shooting_beta09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V10, doublependulum_QP_solver_shooting_W11, doublependulum_QP_solver_shooting_Yd10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V10, doublependulum_QP_solver_shooting_Lbyrd10, doublependulum_QP_solver_shooting_W11, doublependulum_QP_solver_shooting_Lbyrd11, doublependulum_QP_solver_shooting_re10, doublependulum_QP_solver_shooting_beta10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V11, doublependulum_QP_solver_shooting_W12, doublependulum_QP_solver_shooting_Yd11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V11, doublependulum_QP_solver_shooting_Lbyrd11, doublependulum_QP_solver_shooting_W12, doublependulum_QP_solver_shooting_Lbyrd12, doublependulum_QP_solver_shooting_re11, doublependulum_QP_solver_shooting_beta11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MMT2_1_7_7(doublependulum_QP_solver_shooting_V12, doublependulum_QP_solver_shooting_W13, doublependulum_QP_solver_shooting_Yd12);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMSUB2_1_7_7(doublependulum_QP_solver_shooting_V12, doublependulum_QP_solver_shooting_Lbyrd12, doublependulum_QP_solver_shooting_W13, doublependulum_QP_solver_shooting_Lbyrd13, doublependulum_QP_solver_shooting_re12, doublependulum_QP_solver_shooting_beta12);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd00, doublependulum_QP_solver_shooting_Ld00);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld00, doublependulum_QP_solver_shooting_beta00, doublependulum_QP_solver_shooting_yy00);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld00, doublependulum_QP_solver_shooting_Ysd01, doublependulum_QP_solver_shooting_Lsd01);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd01, doublependulum_QP_solver_shooting_Yd01);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd01, doublependulum_QP_solver_shooting_Ld01);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd01, doublependulum_QP_solver_shooting_yy00, doublependulum_QP_solver_shooting_beta01, doublependulum_QP_solver_shooting_bmy01);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld01, doublependulum_QP_solver_shooting_bmy01, doublependulum_QP_solver_shooting_yy01);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld01, doublependulum_QP_solver_shooting_Ysd02, doublependulum_QP_solver_shooting_Lsd02);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd02, doublependulum_QP_solver_shooting_Yd02);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd02, doublependulum_QP_solver_shooting_Ld02);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd02, doublependulum_QP_solver_shooting_yy01, doublependulum_QP_solver_shooting_beta02, doublependulum_QP_solver_shooting_bmy02);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld02, doublependulum_QP_solver_shooting_bmy02, doublependulum_QP_solver_shooting_yy02);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld02, doublependulum_QP_solver_shooting_Ysd03, doublependulum_QP_solver_shooting_Lsd03);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd03, doublependulum_QP_solver_shooting_Yd03);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd03, doublependulum_QP_solver_shooting_Ld03);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd03, doublependulum_QP_solver_shooting_yy02, doublependulum_QP_solver_shooting_beta03, doublependulum_QP_solver_shooting_bmy03);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld03, doublependulum_QP_solver_shooting_bmy03, doublependulum_QP_solver_shooting_yy03);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld03, doublependulum_QP_solver_shooting_Ysd04, doublependulum_QP_solver_shooting_Lsd04);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd04, doublependulum_QP_solver_shooting_Yd04);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd04, doublependulum_QP_solver_shooting_Ld04);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd04, doublependulum_QP_solver_shooting_yy03, doublependulum_QP_solver_shooting_beta04, doublependulum_QP_solver_shooting_bmy04);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld04, doublependulum_QP_solver_shooting_bmy04, doublependulum_QP_solver_shooting_yy04);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld04, doublependulum_QP_solver_shooting_Ysd05, doublependulum_QP_solver_shooting_Lsd05);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd05, doublependulum_QP_solver_shooting_Yd05);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd05, doublependulum_QP_solver_shooting_Ld05);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd05, doublependulum_QP_solver_shooting_yy04, doublependulum_QP_solver_shooting_beta05, doublependulum_QP_solver_shooting_bmy05);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld05, doublependulum_QP_solver_shooting_bmy05, doublependulum_QP_solver_shooting_yy05);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld05, doublependulum_QP_solver_shooting_Ysd06, doublependulum_QP_solver_shooting_Lsd06);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd06, doublependulum_QP_solver_shooting_Yd06);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd06, doublependulum_QP_solver_shooting_Ld06);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd06, doublependulum_QP_solver_shooting_yy05, doublependulum_QP_solver_shooting_beta06, doublependulum_QP_solver_shooting_bmy06);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld06, doublependulum_QP_solver_shooting_bmy06, doublependulum_QP_solver_shooting_yy06);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld06, doublependulum_QP_solver_shooting_Ysd07, doublependulum_QP_solver_shooting_Lsd07);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd07, doublependulum_QP_solver_shooting_Yd07);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd07, doublependulum_QP_solver_shooting_Ld07);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd07, doublependulum_QP_solver_shooting_yy06, doublependulum_QP_solver_shooting_beta07, doublependulum_QP_solver_shooting_bmy07);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld07, doublependulum_QP_solver_shooting_bmy07, doublependulum_QP_solver_shooting_yy07);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld07, doublependulum_QP_solver_shooting_Ysd08, doublependulum_QP_solver_shooting_Lsd08);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd08, doublependulum_QP_solver_shooting_Yd08);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd08, doublependulum_QP_solver_shooting_Ld08);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd08, doublependulum_QP_solver_shooting_yy07, doublependulum_QP_solver_shooting_beta08, doublependulum_QP_solver_shooting_bmy08);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld08, doublependulum_QP_solver_shooting_bmy08, doublependulum_QP_solver_shooting_yy08);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld08, doublependulum_QP_solver_shooting_Ysd09, doublependulum_QP_solver_shooting_Lsd09);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd09, doublependulum_QP_solver_shooting_Yd09);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd09, doublependulum_QP_solver_shooting_Ld09);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd09, doublependulum_QP_solver_shooting_yy08, doublependulum_QP_solver_shooting_beta09, doublependulum_QP_solver_shooting_bmy09);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld09, doublependulum_QP_solver_shooting_bmy09, doublependulum_QP_solver_shooting_yy09);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld09, doublependulum_QP_solver_shooting_Ysd10, doublependulum_QP_solver_shooting_Lsd10);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd10, doublependulum_QP_solver_shooting_Yd10);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd10, doublependulum_QP_solver_shooting_Ld10);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd10, doublependulum_QP_solver_shooting_yy09, doublependulum_QP_solver_shooting_beta10, doublependulum_QP_solver_shooting_bmy10);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld10, doublependulum_QP_solver_shooting_bmy10, doublependulum_QP_solver_shooting_yy10);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld10, doublependulum_QP_solver_shooting_Ysd11, doublependulum_QP_solver_shooting_Lsd11);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd11, doublependulum_QP_solver_shooting_Yd11);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd11, doublependulum_QP_solver_shooting_Ld11);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd11, doublependulum_QP_solver_shooting_yy10, doublependulum_QP_solver_shooting_beta11, doublependulum_QP_solver_shooting_bmy11);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld11, doublependulum_QP_solver_shooting_bmy11, doublependulum_QP_solver_shooting_yy11);
doublependulum_QP_solver_shooting_LA_DENSE_MATRIXTFORWARDSUB_1_1(doublependulum_QP_solver_shooting_Ld11, doublependulum_QP_solver_shooting_Ysd12, doublependulum_QP_solver_shooting_Lsd12);
doublependulum_QP_solver_shooting_LA_DENSE_MMTSUB_1_1(doublependulum_QP_solver_shooting_Lsd12, doublependulum_QP_solver_shooting_Yd12);
doublependulum_QP_solver_shooting_LA_DENSE_CHOL_1(doublependulum_QP_solver_shooting_Yd12, doublependulum_QP_solver_shooting_Ld12);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd12, doublependulum_QP_solver_shooting_yy11, doublependulum_QP_solver_shooting_beta12, doublependulum_QP_solver_shooting_bmy12);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld12, doublependulum_QP_solver_shooting_bmy12, doublependulum_QP_solver_shooting_yy12);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld12, doublependulum_QP_solver_shooting_yy12, doublependulum_QP_solver_shooting_dvaff12);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd12, doublependulum_QP_solver_shooting_dvaff12, doublependulum_QP_solver_shooting_yy11, doublependulum_QP_solver_shooting_bmy11);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld11, doublependulum_QP_solver_shooting_bmy11, doublependulum_QP_solver_shooting_dvaff11);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd11, doublependulum_QP_solver_shooting_dvaff11, doublependulum_QP_solver_shooting_yy10, doublependulum_QP_solver_shooting_bmy10);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld10, doublependulum_QP_solver_shooting_bmy10, doublependulum_QP_solver_shooting_dvaff10);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd10, doublependulum_QP_solver_shooting_dvaff10, doublependulum_QP_solver_shooting_yy09, doublependulum_QP_solver_shooting_bmy09);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld09, doublependulum_QP_solver_shooting_bmy09, doublependulum_QP_solver_shooting_dvaff09);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd09, doublependulum_QP_solver_shooting_dvaff09, doublependulum_QP_solver_shooting_yy08, doublependulum_QP_solver_shooting_bmy08);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld08, doublependulum_QP_solver_shooting_bmy08, doublependulum_QP_solver_shooting_dvaff08);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd08, doublependulum_QP_solver_shooting_dvaff08, doublependulum_QP_solver_shooting_yy07, doublependulum_QP_solver_shooting_bmy07);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld07, doublependulum_QP_solver_shooting_bmy07, doublependulum_QP_solver_shooting_dvaff07);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd07, doublependulum_QP_solver_shooting_dvaff07, doublependulum_QP_solver_shooting_yy06, doublependulum_QP_solver_shooting_bmy06);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld06, doublependulum_QP_solver_shooting_bmy06, doublependulum_QP_solver_shooting_dvaff06);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd06, doublependulum_QP_solver_shooting_dvaff06, doublependulum_QP_solver_shooting_yy05, doublependulum_QP_solver_shooting_bmy05);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld05, doublependulum_QP_solver_shooting_bmy05, doublependulum_QP_solver_shooting_dvaff05);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd05, doublependulum_QP_solver_shooting_dvaff05, doublependulum_QP_solver_shooting_yy04, doublependulum_QP_solver_shooting_bmy04);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld04, doublependulum_QP_solver_shooting_bmy04, doublependulum_QP_solver_shooting_dvaff04);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd04, doublependulum_QP_solver_shooting_dvaff04, doublependulum_QP_solver_shooting_yy03, doublependulum_QP_solver_shooting_bmy03);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld03, doublependulum_QP_solver_shooting_bmy03, doublependulum_QP_solver_shooting_dvaff03);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd03, doublependulum_QP_solver_shooting_dvaff03, doublependulum_QP_solver_shooting_yy02, doublependulum_QP_solver_shooting_bmy02);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld02, doublependulum_QP_solver_shooting_bmy02, doublependulum_QP_solver_shooting_dvaff02);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd02, doublependulum_QP_solver_shooting_dvaff02, doublependulum_QP_solver_shooting_yy01, doublependulum_QP_solver_shooting_bmy01);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld01, doublependulum_QP_solver_shooting_bmy01, doublependulum_QP_solver_shooting_dvaff01);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd01, doublependulum_QP_solver_shooting_dvaff01, doublependulum_QP_solver_shooting_yy00, doublependulum_QP_solver_shooting_bmy00);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld00, doublependulum_QP_solver_shooting_bmy00, doublependulum_QP_solver_shooting_dvaff00);
doublependulum_QP_solver_shooting_LA_DENSE_MTVM_1_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff00, doublependulum_QP_solver_shooting_grad_eq00);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff01, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff00, doublependulum_QP_solver_shooting_grad_eq01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff02, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff01, doublependulum_QP_solver_shooting_grad_eq02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff03, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff02, doublependulum_QP_solver_shooting_grad_eq03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff04, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff03, doublependulum_QP_solver_shooting_grad_eq04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff05, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff04, doublependulum_QP_solver_shooting_grad_eq05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff06, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff05, doublependulum_QP_solver_shooting_grad_eq06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff07, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff06, doublependulum_QP_solver_shooting_grad_eq07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff08, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff07, doublependulum_QP_solver_shooting_grad_eq08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff09, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff08, doublependulum_QP_solver_shooting_grad_eq09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff10, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff09, doublependulum_QP_solver_shooting_grad_eq10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff11, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff10, doublependulum_QP_solver_shooting_grad_eq11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvaff12, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff11, doublependulum_QP_solver_shooting_grad_eq12);
doublependulum_QP_solver_shooting_LA_DIAGZERO_MTVM_1_7(doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvaff12, doublependulum_QP_solver_shooting_grad_eq13);
doublependulum_QP_solver_shooting_LA_VSUB2_98(doublependulum_QP_solver_shooting_rd, doublependulum_QP_solver_shooting_grad_eq, doublependulum_QP_solver_shooting_rd);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi00, doublependulum_QP_solver_shooting_rd00, doublependulum_QP_solver_shooting_dzaff00);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_rd01, doublependulum_QP_solver_shooting_dzaff01);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_rd02, doublependulum_QP_solver_shooting_dzaff02);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_rd03, doublependulum_QP_solver_shooting_dzaff03);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_rd04, doublependulum_QP_solver_shooting_dzaff04);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_rd05, doublependulum_QP_solver_shooting_dzaff05);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_rd06, doublependulum_QP_solver_shooting_dzaff06);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_rd07, doublependulum_QP_solver_shooting_dzaff07);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_rd08, doublependulum_QP_solver_shooting_dzaff08);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_rd09, doublependulum_QP_solver_shooting_dzaff09);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_rd10, doublependulum_QP_solver_shooting_dzaff10);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_rd11, doublependulum_QP_solver_shooting_dzaff11);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_rd12, doublependulum_QP_solver_shooting_dzaff12);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi13, doublependulum_QP_solver_shooting_rd13, doublependulum_QP_solver_shooting_dzaff13);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_rilb00, doublependulum_QP_solver_shooting_dslbaff00);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb00, doublependulum_QP_solver_shooting_dslbaff00, doublependulum_QP_solver_shooting_llb00, doublependulum_QP_solver_shooting_dllbaff00);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub00, doublependulum_QP_solver_shooting_dzaff00, doublependulum_QP_solver_shooting_ubIdx00, doublependulum_QP_solver_shooting_dsubaff00);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub00, doublependulum_QP_solver_shooting_dsubaff00, doublependulum_QP_solver_shooting_lub00, doublependulum_QP_solver_shooting_dlubaff00);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_rilb01, doublependulum_QP_solver_shooting_dslbaff01);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb01, doublependulum_QP_solver_shooting_dslbaff01, doublependulum_QP_solver_shooting_llb01, doublependulum_QP_solver_shooting_dllbaff01);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub01, doublependulum_QP_solver_shooting_dzaff01, doublependulum_QP_solver_shooting_ubIdx01, doublependulum_QP_solver_shooting_dsubaff01);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub01, doublependulum_QP_solver_shooting_dsubaff01, doublependulum_QP_solver_shooting_lub01, doublependulum_QP_solver_shooting_dlubaff01);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_rilb02, doublependulum_QP_solver_shooting_dslbaff02);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb02, doublependulum_QP_solver_shooting_dslbaff02, doublependulum_QP_solver_shooting_llb02, doublependulum_QP_solver_shooting_dllbaff02);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub02, doublependulum_QP_solver_shooting_dzaff02, doublependulum_QP_solver_shooting_ubIdx02, doublependulum_QP_solver_shooting_dsubaff02);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub02, doublependulum_QP_solver_shooting_dsubaff02, doublependulum_QP_solver_shooting_lub02, doublependulum_QP_solver_shooting_dlubaff02);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_rilb03, doublependulum_QP_solver_shooting_dslbaff03);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb03, doublependulum_QP_solver_shooting_dslbaff03, doublependulum_QP_solver_shooting_llb03, doublependulum_QP_solver_shooting_dllbaff03);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub03, doublependulum_QP_solver_shooting_dzaff03, doublependulum_QP_solver_shooting_ubIdx03, doublependulum_QP_solver_shooting_dsubaff03);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub03, doublependulum_QP_solver_shooting_dsubaff03, doublependulum_QP_solver_shooting_lub03, doublependulum_QP_solver_shooting_dlubaff03);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_rilb04, doublependulum_QP_solver_shooting_dslbaff04);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb04, doublependulum_QP_solver_shooting_dslbaff04, doublependulum_QP_solver_shooting_llb04, doublependulum_QP_solver_shooting_dllbaff04);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub04, doublependulum_QP_solver_shooting_dzaff04, doublependulum_QP_solver_shooting_ubIdx04, doublependulum_QP_solver_shooting_dsubaff04);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub04, doublependulum_QP_solver_shooting_dsubaff04, doublependulum_QP_solver_shooting_lub04, doublependulum_QP_solver_shooting_dlubaff04);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_rilb05, doublependulum_QP_solver_shooting_dslbaff05);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb05, doublependulum_QP_solver_shooting_dslbaff05, doublependulum_QP_solver_shooting_llb05, doublependulum_QP_solver_shooting_dllbaff05);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub05, doublependulum_QP_solver_shooting_dzaff05, doublependulum_QP_solver_shooting_ubIdx05, doublependulum_QP_solver_shooting_dsubaff05);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub05, doublependulum_QP_solver_shooting_dsubaff05, doublependulum_QP_solver_shooting_lub05, doublependulum_QP_solver_shooting_dlubaff05);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_rilb06, doublependulum_QP_solver_shooting_dslbaff06);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb06, doublependulum_QP_solver_shooting_dslbaff06, doublependulum_QP_solver_shooting_llb06, doublependulum_QP_solver_shooting_dllbaff06);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub06, doublependulum_QP_solver_shooting_dzaff06, doublependulum_QP_solver_shooting_ubIdx06, doublependulum_QP_solver_shooting_dsubaff06);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub06, doublependulum_QP_solver_shooting_dsubaff06, doublependulum_QP_solver_shooting_lub06, doublependulum_QP_solver_shooting_dlubaff06);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_rilb07, doublependulum_QP_solver_shooting_dslbaff07);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb07, doublependulum_QP_solver_shooting_dslbaff07, doublependulum_QP_solver_shooting_llb07, doublependulum_QP_solver_shooting_dllbaff07);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub07, doublependulum_QP_solver_shooting_dzaff07, doublependulum_QP_solver_shooting_ubIdx07, doublependulum_QP_solver_shooting_dsubaff07);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub07, doublependulum_QP_solver_shooting_dsubaff07, doublependulum_QP_solver_shooting_lub07, doublependulum_QP_solver_shooting_dlubaff07);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_rilb08, doublependulum_QP_solver_shooting_dslbaff08);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb08, doublependulum_QP_solver_shooting_dslbaff08, doublependulum_QP_solver_shooting_llb08, doublependulum_QP_solver_shooting_dllbaff08);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub08, doublependulum_QP_solver_shooting_dzaff08, doublependulum_QP_solver_shooting_ubIdx08, doublependulum_QP_solver_shooting_dsubaff08);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub08, doublependulum_QP_solver_shooting_dsubaff08, doublependulum_QP_solver_shooting_lub08, doublependulum_QP_solver_shooting_dlubaff08);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_rilb09, doublependulum_QP_solver_shooting_dslbaff09);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb09, doublependulum_QP_solver_shooting_dslbaff09, doublependulum_QP_solver_shooting_llb09, doublependulum_QP_solver_shooting_dllbaff09);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub09, doublependulum_QP_solver_shooting_dzaff09, doublependulum_QP_solver_shooting_ubIdx09, doublependulum_QP_solver_shooting_dsubaff09);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub09, doublependulum_QP_solver_shooting_dsubaff09, doublependulum_QP_solver_shooting_lub09, doublependulum_QP_solver_shooting_dlubaff09);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_rilb10, doublependulum_QP_solver_shooting_dslbaff10);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb10, doublependulum_QP_solver_shooting_dslbaff10, doublependulum_QP_solver_shooting_llb10, doublependulum_QP_solver_shooting_dllbaff10);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub10, doublependulum_QP_solver_shooting_dzaff10, doublependulum_QP_solver_shooting_ubIdx10, doublependulum_QP_solver_shooting_dsubaff10);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub10, doublependulum_QP_solver_shooting_dsubaff10, doublependulum_QP_solver_shooting_lub10, doublependulum_QP_solver_shooting_dlubaff10);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_rilb11, doublependulum_QP_solver_shooting_dslbaff11);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb11, doublependulum_QP_solver_shooting_dslbaff11, doublependulum_QP_solver_shooting_llb11, doublependulum_QP_solver_shooting_dllbaff11);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub11, doublependulum_QP_solver_shooting_dzaff11, doublependulum_QP_solver_shooting_ubIdx11, doublependulum_QP_solver_shooting_dsubaff11);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub11, doublependulum_QP_solver_shooting_dsubaff11, doublependulum_QP_solver_shooting_lub11, doublependulum_QP_solver_shooting_dlubaff11);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_rilb12, doublependulum_QP_solver_shooting_dslbaff12);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb12, doublependulum_QP_solver_shooting_dslbaff12, doublependulum_QP_solver_shooting_llb12, doublependulum_QP_solver_shooting_dllbaff12);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub12, doublependulum_QP_solver_shooting_dzaff12, doublependulum_QP_solver_shooting_ubIdx12, doublependulum_QP_solver_shooting_dsubaff12);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub12, doublependulum_QP_solver_shooting_dsubaff12, doublependulum_QP_solver_shooting_lub12, doublependulum_QP_solver_shooting_dlubaff12);
doublependulum_QP_solver_shooting_LA_VSUB_INDEXED_7(doublependulum_QP_solver_shooting_dzaff13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_rilb13, doublependulum_QP_solver_shooting_dslbaff13);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_llbbyslb13, doublependulum_QP_solver_shooting_dslbaff13, doublependulum_QP_solver_shooting_llb13, doublependulum_QP_solver_shooting_dllbaff13);
doublependulum_QP_solver_shooting_LA_VSUB2_INDEXED_7(doublependulum_QP_solver_shooting_riub13, doublependulum_QP_solver_shooting_dzaff13, doublependulum_QP_solver_shooting_ubIdx13, doublependulum_QP_solver_shooting_dsubaff13);
doublependulum_QP_solver_shooting_LA_VSUB3_7(doublependulum_QP_solver_shooting_lubbysub13, doublependulum_QP_solver_shooting_dsubaff13, doublependulum_QP_solver_shooting_lub13, doublependulum_QP_solver_shooting_dlubaff13);
info->lsit_aff = doublependulum_QP_solver_shooting_LINESEARCH_BACKTRACKING_AFFINE(doublependulum_QP_solver_shooting_l, doublependulum_QP_solver_shooting_s, doublependulum_QP_solver_shooting_dl_aff, doublependulum_QP_solver_shooting_ds_aff, &info->step_aff, &info->mu_aff);
if( info->lsit_aff == doublependulum_QP_solver_shooting_NOPROGRESS ){
exitcode = doublependulum_QP_solver_shooting_NOPROGRESS; break;
}
sigma_3rdroot = info->mu_aff / info->mu;
info->sigma = sigma_3rdroot*sigma_3rdroot*sigma_3rdroot;
musigma = info->mu * info->sigma;
doublependulum_QP_solver_shooting_LA_VSUB5_196(doublependulum_QP_solver_shooting_ds_aff, doublependulum_QP_solver_shooting_dl_aff, info->mu, info->sigma, doublependulum_QP_solver_shooting_ccrhs);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub00, doublependulum_QP_solver_shooting_sub00, doublependulum_QP_solver_shooting_ubIdx00, doublependulum_QP_solver_shooting_ccrhsl00, doublependulum_QP_solver_shooting_slb00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_rd00);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub01, doublependulum_QP_solver_shooting_sub01, doublependulum_QP_solver_shooting_ubIdx01, doublependulum_QP_solver_shooting_ccrhsl01, doublependulum_QP_solver_shooting_slb01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_rd01);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi00, doublependulum_QP_solver_shooting_rd00, doublependulum_QP_solver_shooting_Lbyrd00);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_rd01, doublependulum_QP_solver_shooting_Lbyrd01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V00, doublependulum_QP_solver_shooting_Lbyrd00, doublependulum_QP_solver_shooting_W01, doublependulum_QP_solver_shooting_Lbyrd01, doublependulum_QP_solver_shooting_beta00);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld00, doublependulum_QP_solver_shooting_beta00, doublependulum_QP_solver_shooting_yy00);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub02, doublependulum_QP_solver_shooting_sub02, doublependulum_QP_solver_shooting_ubIdx02, doublependulum_QP_solver_shooting_ccrhsl02, doublependulum_QP_solver_shooting_slb02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_rd02);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_rd02, doublependulum_QP_solver_shooting_Lbyrd02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V01, doublependulum_QP_solver_shooting_Lbyrd01, doublependulum_QP_solver_shooting_W02, doublependulum_QP_solver_shooting_Lbyrd02, doublependulum_QP_solver_shooting_beta01);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd01, doublependulum_QP_solver_shooting_yy00, doublependulum_QP_solver_shooting_beta01, doublependulum_QP_solver_shooting_bmy01);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld01, doublependulum_QP_solver_shooting_bmy01, doublependulum_QP_solver_shooting_yy01);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub03, doublependulum_QP_solver_shooting_sub03, doublependulum_QP_solver_shooting_ubIdx03, doublependulum_QP_solver_shooting_ccrhsl03, doublependulum_QP_solver_shooting_slb03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_rd03);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_rd03, doublependulum_QP_solver_shooting_Lbyrd03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V02, doublependulum_QP_solver_shooting_Lbyrd02, doublependulum_QP_solver_shooting_W03, doublependulum_QP_solver_shooting_Lbyrd03, doublependulum_QP_solver_shooting_beta02);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd02, doublependulum_QP_solver_shooting_yy01, doublependulum_QP_solver_shooting_beta02, doublependulum_QP_solver_shooting_bmy02);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld02, doublependulum_QP_solver_shooting_bmy02, doublependulum_QP_solver_shooting_yy02);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub04, doublependulum_QP_solver_shooting_sub04, doublependulum_QP_solver_shooting_ubIdx04, doublependulum_QP_solver_shooting_ccrhsl04, doublependulum_QP_solver_shooting_slb04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_rd04);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_rd04, doublependulum_QP_solver_shooting_Lbyrd04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V03, doublependulum_QP_solver_shooting_Lbyrd03, doublependulum_QP_solver_shooting_W04, doublependulum_QP_solver_shooting_Lbyrd04, doublependulum_QP_solver_shooting_beta03);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd03, doublependulum_QP_solver_shooting_yy02, doublependulum_QP_solver_shooting_beta03, doublependulum_QP_solver_shooting_bmy03);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld03, doublependulum_QP_solver_shooting_bmy03, doublependulum_QP_solver_shooting_yy03);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub05, doublependulum_QP_solver_shooting_sub05, doublependulum_QP_solver_shooting_ubIdx05, doublependulum_QP_solver_shooting_ccrhsl05, doublependulum_QP_solver_shooting_slb05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_rd05);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_rd05, doublependulum_QP_solver_shooting_Lbyrd05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V04, doublependulum_QP_solver_shooting_Lbyrd04, doublependulum_QP_solver_shooting_W05, doublependulum_QP_solver_shooting_Lbyrd05, doublependulum_QP_solver_shooting_beta04);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd04, doublependulum_QP_solver_shooting_yy03, doublependulum_QP_solver_shooting_beta04, doublependulum_QP_solver_shooting_bmy04);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld04, doublependulum_QP_solver_shooting_bmy04, doublependulum_QP_solver_shooting_yy04);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub06, doublependulum_QP_solver_shooting_sub06, doublependulum_QP_solver_shooting_ubIdx06, doublependulum_QP_solver_shooting_ccrhsl06, doublependulum_QP_solver_shooting_slb06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_rd06);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_rd06, doublependulum_QP_solver_shooting_Lbyrd06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V05, doublependulum_QP_solver_shooting_Lbyrd05, doublependulum_QP_solver_shooting_W06, doublependulum_QP_solver_shooting_Lbyrd06, doublependulum_QP_solver_shooting_beta05);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd05, doublependulum_QP_solver_shooting_yy04, doublependulum_QP_solver_shooting_beta05, doublependulum_QP_solver_shooting_bmy05);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld05, doublependulum_QP_solver_shooting_bmy05, doublependulum_QP_solver_shooting_yy05);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub07, doublependulum_QP_solver_shooting_sub07, doublependulum_QP_solver_shooting_ubIdx07, doublependulum_QP_solver_shooting_ccrhsl07, doublependulum_QP_solver_shooting_slb07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_rd07);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_rd07, doublependulum_QP_solver_shooting_Lbyrd07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V06, doublependulum_QP_solver_shooting_Lbyrd06, doublependulum_QP_solver_shooting_W07, doublependulum_QP_solver_shooting_Lbyrd07, doublependulum_QP_solver_shooting_beta06);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd06, doublependulum_QP_solver_shooting_yy05, doublependulum_QP_solver_shooting_beta06, doublependulum_QP_solver_shooting_bmy06);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld06, doublependulum_QP_solver_shooting_bmy06, doublependulum_QP_solver_shooting_yy06);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub08, doublependulum_QP_solver_shooting_sub08, doublependulum_QP_solver_shooting_ubIdx08, doublependulum_QP_solver_shooting_ccrhsl08, doublependulum_QP_solver_shooting_slb08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_rd08);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_rd08, doublependulum_QP_solver_shooting_Lbyrd08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V07, doublependulum_QP_solver_shooting_Lbyrd07, doublependulum_QP_solver_shooting_W08, doublependulum_QP_solver_shooting_Lbyrd08, doublependulum_QP_solver_shooting_beta07);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd07, doublependulum_QP_solver_shooting_yy06, doublependulum_QP_solver_shooting_beta07, doublependulum_QP_solver_shooting_bmy07);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld07, doublependulum_QP_solver_shooting_bmy07, doublependulum_QP_solver_shooting_yy07);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub09, doublependulum_QP_solver_shooting_sub09, doublependulum_QP_solver_shooting_ubIdx09, doublependulum_QP_solver_shooting_ccrhsl09, doublependulum_QP_solver_shooting_slb09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_rd09);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_rd09, doublependulum_QP_solver_shooting_Lbyrd09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V08, doublependulum_QP_solver_shooting_Lbyrd08, doublependulum_QP_solver_shooting_W09, doublependulum_QP_solver_shooting_Lbyrd09, doublependulum_QP_solver_shooting_beta08);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd08, doublependulum_QP_solver_shooting_yy07, doublependulum_QP_solver_shooting_beta08, doublependulum_QP_solver_shooting_bmy08);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld08, doublependulum_QP_solver_shooting_bmy08, doublependulum_QP_solver_shooting_yy08);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub10, doublependulum_QP_solver_shooting_sub10, doublependulum_QP_solver_shooting_ubIdx10, doublependulum_QP_solver_shooting_ccrhsl10, doublependulum_QP_solver_shooting_slb10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_rd10);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_rd10, doublependulum_QP_solver_shooting_Lbyrd10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V09, doublependulum_QP_solver_shooting_Lbyrd09, doublependulum_QP_solver_shooting_W10, doublependulum_QP_solver_shooting_Lbyrd10, doublependulum_QP_solver_shooting_beta09);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd09, doublependulum_QP_solver_shooting_yy08, doublependulum_QP_solver_shooting_beta09, doublependulum_QP_solver_shooting_bmy09);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld09, doublependulum_QP_solver_shooting_bmy09, doublependulum_QP_solver_shooting_yy09);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub11, doublependulum_QP_solver_shooting_sub11, doublependulum_QP_solver_shooting_ubIdx11, doublependulum_QP_solver_shooting_ccrhsl11, doublependulum_QP_solver_shooting_slb11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_rd11);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_rd11, doublependulum_QP_solver_shooting_Lbyrd11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V10, doublependulum_QP_solver_shooting_Lbyrd10, doublependulum_QP_solver_shooting_W11, doublependulum_QP_solver_shooting_Lbyrd11, doublependulum_QP_solver_shooting_beta10);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd10, doublependulum_QP_solver_shooting_yy09, doublependulum_QP_solver_shooting_beta10, doublependulum_QP_solver_shooting_bmy10);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld10, doublependulum_QP_solver_shooting_bmy10, doublependulum_QP_solver_shooting_yy10);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub12, doublependulum_QP_solver_shooting_sub12, doublependulum_QP_solver_shooting_ubIdx12, doublependulum_QP_solver_shooting_ccrhsl12, doublependulum_QP_solver_shooting_slb12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_rd12);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_rd12, doublependulum_QP_solver_shooting_Lbyrd12);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V11, doublependulum_QP_solver_shooting_Lbyrd11, doublependulum_QP_solver_shooting_W12, doublependulum_QP_solver_shooting_Lbyrd12, doublependulum_QP_solver_shooting_beta11);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd11, doublependulum_QP_solver_shooting_yy10, doublependulum_QP_solver_shooting_beta11, doublependulum_QP_solver_shooting_bmy11);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld11, doublependulum_QP_solver_shooting_bmy11, doublependulum_QP_solver_shooting_yy11);
doublependulum_QP_solver_shooting_LA_VSUB6_INDEXED_7_7_7(doublependulum_QP_solver_shooting_ccrhsub13, doublependulum_QP_solver_shooting_sub13, doublependulum_QP_solver_shooting_ubIdx13, doublependulum_QP_solver_shooting_ccrhsl13, doublependulum_QP_solver_shooting_slb13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_rd13);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDSUB_7(doublependulum_QP_solver_shooting_Phi13, doublependulum_QP_solver_shooting_rd13, doublependulum_QP_solver_shooting_Lbyrd13);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_2MVMADD_1_7_7(doublependulum_QP_solver_shooting_V12, doublependulum_QP_solver_shooting_Lbyrd12, doublependulum_QP_solver_shooting_W13, doublependulum_QP_solver_shooting_Lbyrd13, doublependulum_QP_solver_shooting_beta12);
doublependulum_QP_solver_shooting_LA_DENSE_MVMSUB1_1_1(doublependulum_QP_solver_shooting_Lsd12, doublependulum_QP_solver_shooting_yy11, doublependulum_QP_solver_shooting_beta12, doublependulum_QP_solver_shooting_bmy12);
doublependulum_QP_solver_shooting_LA_DENSE_FORWARDSUB_1(doublependulum_QP_solver_shooting_Ld12, doublependulum_QP_solver_shooting_bmy12, doublependulum_QP_solver_shooting_yy12);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld12, doublependulum_QP_solver_shooting_yy12, doublependulum_QP_solver_shooting_dvcc12);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd12, doublependulum_QP_solver_shooting_dvcc12, doublependulum_QP_solver_shooting_yy11, doublependulum_QP_solver_shooting_bmy11);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld11, doublependulum_QP_solver_shooting_bmy11, doublependulum_QP_solver_shooting_dvcc11);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd11, doublependulum_QP_solver_shooting_dvcc11, doublependulum_QP_solver_shooting_yy10, doublependulum_QP_solver_shooting_bmy10);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld10, doublependulum_QP_solver_shooting_bmy10, doublependulum_QP_solver_shooting_dvcc10);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd10, doublependulum_QP_solver_shooting_dvcc10, doublependulum_QP_solver_shooting_yy09, doublependulum_QP_solver_shooting_bmy09);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld09, doublependulum_QP_solver_shooting_bmy09, doublependulum_QP_solver_shooting_dvcc09);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd09, doublependulum_QP_solver_shooting_dvcc09, doublependulum_QP_solver_shooting_yy08, doublependulum_QP_solver_shooting_bmy08);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld08, doublependulum_QP_solver_shooting_bmy08, doublependulum_QP_solver_shooting_dvcc08);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd08, doublependulum_QP_solver_shooting_dvcc08, doublependulum_QP_solver_shooting_yy07, doublependulum_QP_solver_shooting_bmy07);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld07, doublependulum_QP_solver_shooting_bmy07, doublependulum_QP_solver_shooting_dvcc07);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd07, doublependulum_QP_solver_shooting_dvcc07, doublependulum_QP_solver_shooting_yy06, doublependulum_QP_solver_shooting_bmy06);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld06, doublependulum_QP_solver_shooting_bmy06, doublependulum_QP_solver_shooting_dvcc06);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd06, doublependulum_QP_solver_shooting_dvcc06, doublependulum_QP_solver_shooting_yy05, doublependulum_QP_solver_shooting_bmy05);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld05, doublependulum_QP_solver_shooting_bmy05, doublependulum_QP_solver_shooting_dvcc05);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd05, doublependulum_QP_solver_shooting_dvcc05, doublependulum_QP_solver_shooting_yy04, doublependulum_QP_solver_shooting_bmy04);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld04, doublependulum_QP_solver_shooting_bmy04, doublependulum_QP_solver_shooting_dvcc04);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd04, doublependulum_QP_solver_shooting_dvcc04, doublependulum_QP_solver_shooting_yy03, doublependulum_QP_solver_shooting_bmy03);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld03, doublependulum_QP_solver_shooting_bmy03, doublependulum_QP_solver_shooting_dvcc03);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd03, doublependulum_QP_solver_shooting_dvcc03, doublependulum_QP_solver_shooting_yy02, doublependulum_QP_solver_shooting_bmy02);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld02, doublependulum_QP_solver_shooting_bmy02, doublependulum_QP_solver_shooting_dvcc02);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd02, doublependulum_QP_solver_shooting_dvcc02, doublependulum_QP_solver_shooting_yy01, doublependulum_QP_solver_shooting_bmy01);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld01, doublependulum_QP_solver_shooting_bmy01, doublependulum_QP_solver_shooting_dvcc01);
doublependulum_QP_solver_shooting_LA_DENSE_MTVMSUB_1_1(doublependulum_QP_solver_shooting_Lsd01, doublependulum_QP_solver_shooting_dvcc01, doublependulum_QP_solver_shooting_yy00, doublependulum_QP_solver_shooting_bmy00);
doublependulum_QP_solver_shooting_LA_DENSE_BACKWARDSUB_1(doublependulum_QP_solver_shooting_Ld00, doublependulum_QP_solver_shooting_bmy00, doublependulum_QP_solver_shooting_dvcc00);
doublependulum_QP_solver_shooting_LA_DENSE_MTVM_1_7(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc00, doublependulum_QP_solver_shooting_grad_eq00);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc01, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc00, doublependulum_QP_solver_shooting_grad_eq01);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc02, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc01, doublependulum_QP_solver_shooting_grad_eq02);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc03, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc02, doublependulum_QP_solver_shooting_grad_eq03);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc04, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc03, doublependulum_QP_solver_shooting_grad_eq04);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc05, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc04, doublependulum_QP_solver_shooting_grad_eq05);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc06, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc05, doublependulum_QP_solver_shooting_grad_eq06);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc07, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc06, doublependulum_QP_solver_shooting_grad_eq07);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc08, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc07, doublependulum_QP_solver_shooting_grad_eq08);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc09, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc08, doublependulum_QP_solver_shooting_grad_eq09);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc10, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc09, doublependulum_QP_solver_shooting_grad_eq10);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc11, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc10, doublependulum_QP_solver_shooting_grad_eq11);
doublependulum_QP_solver_shooting_LA_DENSE_DIAGZERO_MTVM2_1_7_1(doublependulum_QP_solver_shooting_C00, doublependulum_QP_solver_shooting_dvcc12, doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc11, doublependulum_QP_solver_shooting_grad_eq12);
doublependulum_QP_solver_shooting_LA_DIAGZERO_MTVM_1_7(doublependulum_QP_solver_shooting_D01, doublependulum_QP_solver_shooting_dvcc12, doublependulum_QP_solver_shooting_grad_eq13);
doublependulum_QP_solver_shooting_LA_VSUB_98(doublependulum_QP_solver_shooting_rd, doublependulum_QP_solver_shooting_grad_eq, doublependulum_QP_solver_shooting_rd);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi00, doublependulum_QP_solver_shooting_rd00, doublependulum_QP_solver_shooting_dzcc00);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi01, doublependulum_QP_solver_shooting_rd01, doublependulum_QP_solver_shooting_dzcc01);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi02, doublependulum_QP_solver_shooting_rd02, doublependulum_QP_solver_shooting_dzcc02);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi03, doublependulum_QP_solver_shooting_rd03, doublependulum_QP_solver_shooting_dzcc03);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi04, doublependulum_QP_solver_shooting_rd04, doublependulum_QP_solver_shooting_dzcc04);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi05, doublependulum_QP_solver_shooting_rd05, doublependulum_QP_solver_shooting_dzcc05);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi06, doublependulum_QP_solver_shooting_rd06, doublependulum_QP_solver_shooting_dzcc06);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi07, doublependulum_QP_solver_shooting_rd07, doublependulum_QP_solver_shooting_dzcc07);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi08, doublependulum_QP_solver_shooting_rd08, doublependulum_QP_solver_shooting_dzcc08);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi09, doublependulum_QP_solver_shooting_rd09, doublependulum_QP_solver_shooting_dzcc09);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi10, doublependulum_QP_solver_shooting_rd10, doublependulum_QP_solver_shooting_dzcc10);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi11, doublependulum_QP_solver_shooting_rd11, doublependulum_QP_solver_shooting_dzcc11);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi12, doublependulum_QP_solver_shooting_rd12, doublependulum_QP_solver_shooting_dzcc12);
doublependulum_QP_solver_shooting_LA_DIAG_FORWARDBACKWARDSUB_7(doublependulum_QP_solver_shooting_Phi13, doublependulum_QP_solver_shooting_rd13, doublependulum_QP_solver_shooting_dzcc13);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl00, doublependulum_QP_solver_shooting_slb00, doublependulum_QP_solver_shooting_llbbyslb00, doublependulum_QP_solver_shooting_dzcc00, doublependulum_QP_solver_shooting_lbIdx00, doublependulum_QP_solver_shooting_dllbcc00);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub00, doublependulum_QP_solver_shooting_sub00, doublependulum_QP_solver_shooting_lubbysub00, doublependulum_QP_solver_shooting_dzcc00, doublependulum_QP_solver_shooting_ubIdx00, doublependulum_QP_solver_shooting_dlubcc00);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl01, doublependulum_QP_solver_shooting_slb01, doublependulum_QP_solver_shooting_llbbyslb01, doublependulum_QP_solver_shooting_dzcc01, doublependulum_QP_solver_shooting_lbIdx01, doublependulum_QP_solver_shooting_dllbcc01);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub01, doublependulum_QP_solver_shooting_sub01, doublependulum_QP_solver_shooting_lubbysub01, doublependulum_QP_solver_shooting_dzcc01, doublependulum_QP_solver_shooting_ubIdx01, doublependulum_QP_solver_shooting_dlubcc01);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl02, doublependulum_QP_solver_shooting_slb02, doublependulum_QP_solver_shooting_llbbyslb02, doublependulum_QP_solver_shooting_dzcc02, doublependulum_QP_solver_shooting_lbIdx02, doublependulum_QP_solver_shooting_dllbcc02);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub02, doublependulum_QP_solver_shooting_sub02, doublependulum_QP_solver_shooting_lubbysub02, doublependulum_QP_solver_shooting_dzcc02, doublependulum_QP_solver_shooting_ubIdx02, doublependulum_QP_solver_shooting_dlubcc02);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl03, doublependulum_QP_solver_shooting_slb03, doublependulum_QP_solver_shooting_llbbyslb03, doublependulum_QP_solver_shooting_dzcc03, doublependulum_QP_solver_shooting_lbIdx03, doublependulum_QP_solver_shooting_dllbcc03);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub03, doublependulum_QP_solver_shooting_sub03, doublependulum_QP_solver_shooting_lubbysub03, doublependulum_QP_solver_shooting_dzcc03, doublependulum_QP_solver_shooting_ubIdx03, doublependulum_QP_solver_shooting_dlubcc03);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl04, doublependulum_QP_solver_shooting_slb04, doublependulum_QP_solver_shooting_llbbyslb04, doublependulum_QP_solver_shooting_dzcc04, doublependulum_QP_solver_shooting_lbIdx04, doublependulum_QP_solver_shooting_dllbcc04);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub04, doublependulum_QP_solver_shooting_sub04, doublependulum_QP_solver_shooting_lubbysub04, doublependulum_QP_solver_shooting_dzcc04, doublependulum_QP_solver_shooting_ubIdx04, doublependulum_QP_solver_shooting_dlubcc04);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl05, doublependulum_QP_solver_shooting_slb05, doublependulum_QP_solver_shooting_llbbyslb05, doublependulum_QP_solver_shooting_dzcc05, doublependulum_QP_solver_shooting_lbIdx05, doublependulum_QP_solver_shooting_dllbcc05);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub05, doublependulum_QP_solver_shooting_sub05, doublependulum_QP_solver_shooting_lubbysub05, doublependulum_QP_solver_shooting_dzcc05, doublependulum_QP_solver_shooting_ubIdx05, doublependulum_QP_solver_shooting_dlubcc05);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl06, doublependulum_QP_solver_shooting_slb06, doublependulum_QP_solver_shooting_llbbyslb06, doublependulum_QP_solver_shooting_dzcc06, doublependulum_QP_solver_shooting_lbIdx06, doublependulum_QP_solver_shooting_dllbcc06);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub06, doublependulum_QP_solver_shooting_sub06, doublependulum_QP_solver_shooting_lubbysub06, doublependulum_QP_solver_shooting_dzcc06, doublependulum_QP_solver_shooting_ubIdx06, doublependulum_QP_solver_shooting_dlubcc06);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl07, doublependulum_QP_solver_shooting_slb07, doublependulum_QP_solver_shooting_llbbyslb07, doublependulum_QP_solver_shooting_dzcc07, doublependulum_QP_solver_shooting_lbIdx07, doublependulum_QP_solver_shooting_dllbcc07);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub07, doublependulum_QP_solver_shooting_sub07, doublependulum_QP_solver_shooting_lubbysub07, doublependulum_QP_solver_shooting_dzcc07, doublependulum_QP_solver_shooting_ubIdx07, doublependulum_QP_solver_shooting_dlubcc07);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl08, doublependulum_QP_solver_shooting_slb08, doublependulum_QP_solver_shooting_llbbyslb08, doublependulum_QP_solver_shooting_dzcc08, doublependulum_QP_solver_shooting_lbIdx08, doublependulum_QP_solver_shooting_dllbcc08);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub08, doublependulum_QP_solver_shooting_sub08, doublependulum_QP_solver_shooting_lubbysub08, doublependulum_QP_solver_shooting_dzcc08, doublependulum_QP_solver_shooting_ubIdx08, doublependulum_QP_solver_shooting_dlubcc08);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl09, doublependulum_QP_solver_shooting_slb09, doublependulum_QP_solver_shooting_llbbyslb09, doublependulum_QP_solver_shooting_dzcc09, doublependulum_QP_solver_shooting_lbIdx09, doublependulum_QP_solver_shooting_dllbcc09);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub09, doublependulum_QP_solver_shooting_sub09, doublependulum_QP_solver_shooting_lubbysub09, doublependulum_QP_solver_shooting_dzcc09, doublependulum_QP_solver_shooting_ubIdx09, doublependulum_QP_solver_shooting_dlubcc09);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl10, doublependulum_QP_solver_shooting_slb10, doublependulum_QP_solver_shooting_llbbyslb10, doublependulum_QP_solver_shooting_dzcc10, doublependulum_QP_solver_shooting_lbIdx10, doublependulum_QP_solver_shooting_dllbcc10);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub10, doublependulum_QP_solver_shooting_sub10, doublependulum_QP_solver_shooting_lubbysub10, doublependulum_QP_solver_shooting_dzcc10, doublependulum_QP_solver_shooting_ubIdx10, doublependulum_QP_solver_shooting_dlubcc10);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl11, doublependulum_QP_solver_shooting_slb11, doublependulum_QP_solver_shooting_llbbyslb11, doublependulum_QP_solver_shooting_dzcc11, doublependulum_QP_solver_shooting_lbIdx11, doublependulum_QP_solver_shooting_dllbcc11);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub11, doublependulum_QP_solver_shooting_sub11, doublependulum_QP_solver_shooting_lubbysub11, doublependulum_QP_solver_shooting_dzcc11, doublependulum_QP_solver_shooting_ubIdx11, doublependulum_QP_solver_shooting_dlubcc11);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl12, doublependulum_QP_solver_shooting_slb12, doublependulum_QP_solver_shooting_llbbyslb12, doublependulum_QP_solver_shooting_dzcc12, doublependulum_QP_solver_shooting_lbIdx12, doublependulum_QP_solver_shooting_dllbcc12);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub12, doublependulum_QP_solver_shooting_sub12, doublependulum_QP_solver_shooting_lubbysub12, doublependulum_QP_solver_shooting_dzcc12, doublependulum_QP_solver_shooting_ubIdx12, doublependulum_QP_solver_shooting_dlubcc12);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTSUB_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsl13, doublependulum_QP_solver_shooting_slb13, doublependulum_QP_solver_shooting_llbbyslb13, doublependulum_QP_solver_shooting_dzcc13, doublependulum_QP_solver_shooting_lbIdx13, doublependulum_QP_solver_shooting_dllbcc13);
doublependulum_QP_solver_shooting_LA_VEC_DIVSUB_MULTADD_INDEXED_7(doublependulum_QP_solver_shooting_ccrhsub13, doublependulum_QP_solver_shooting_sub13, doublependulum_QP_solver_shooting_lubbysub13, doublependulum_QP_solver_shooting_dzcc13, doublependulum_QP_solver_shooting_ubIdx13, doublependulum_QP_solver_shooting_dlubcc13);
doublependulum_QP_solver_shooting_LA_VSUB7_196(doublependulum_QP_solver_shooting_l, doublependulum_QP_solver_shooting_ccrhs, doublependulum_QP_solver_shooting_s, doublependulum_QP_solver_shooting_dl_cc, doublependulum_QP_solver_shooting_ds_cc);
doublependulum_QP_solver_shooting_LA_VADD_98(doublependulum_QP_solver_shooting_dz_cc, doublependulum_QP_solver_shooting_dz_aff);
doublependulum_QP_solver_shooting_LA_VADD_13(doublependulum_QP_solver_shooting_dv_cc, doublependulum_QP_solver_shooting_dv_aff);
doublependulum_QP_solver_shooting_LA_VADD_196(doublependulum_QP_solver_shooting_dl_cc, doublependulum_QP_solver_shooting_dl_aff);
doublependulum_QP_solver_shooting_LA_VADD_196(doublependulum_QP_solver_shooting_ds_cc, doublependulum_QP_solver_shooting_ds_aff);
info->lsit_cc = doublependulum_QP_solver_shooting_LINESEARCH_BACKTRACKING_COMBINED(doublependulum_QP_solver_shooting_z, doublependulum_QP_solver_shooting_v, doublependulum_QP_solver_shooting_l, doublependulum_QP_solver_shooting_s, doublependulum_QP_solver_shooting_dz_cc, doublependulum_QP_solver_shooting_dv_cc, doublependulum_QP_solver_shooting_dl_cc, doublependulum_QP_solver_shooting_ds_cc, &info->step_cc, &info->mu);
if( info->lsit_cc == doublependulum_QP_solver_shooting_NOPROGRESS ){
exitcode = doublependulum_QP_solver_shooting_NOPROGRESS; break;
}
info->it++;
}
output->z1[0] = doublependulum_QP_solver_shooting_z00[0];
output->z1[1] = doublependulum_QP_solver_shooting_z00[1];
output->z1[2] = doublependulum_QP_solver_shooting_z00[2];
output->z1[3] = doublependulum_QP_solver_shooting_z00[3];
output->z1[4] = doublependulum_QP_solver_shooting_z00[4];
output->z1[5] = doublependulum_QP_solver_shooting_z00[5];
output->z1[6] = doublependulum_QP_solver_shooting_z00[6];
output->z2[0] = doublependulum_QP_solver_shooting_z01[0];
output->z2[1] = doublependulum_QP_solver_shooting_z01[1];
output->z2[2] = doublependulum_QP_solver_shooting_z01[2];
output->z2[3] = doublependulum_QP_solver_shooting_z01[3];
output->z2[4] = doublependulum_QP_solver_shooting_z01[4];
output->z2[5] = doublependulum_QP_solver_shooting_z01[5];
output->z2[6] = doublependulum_QP_solver_shooting_z01[6];
output->z3[0] = doublependulum_QP_solver_shooting_z02[0];
output->z3[1] = doublependulum_QP_solver_shooting_z02[1];
output->z3[2] = doublependulum_QP_solver_shooting_z02[2];
output->z3[3] = doublependulum_QP_solver_shooting_z02[3];
output->z3[4] = doublependulum_QP_solver_shooting_z02[4];
output->z3[5] = doublependulum_QP_solver_shooting_z02[5];
output->z3[6] = doublependulum_QP_solver_shooting_z02[6];
output->z4[0] = doublependulum_QP_solver_shooting_z03[0];
output->z4[1] = doublependulum_QP_solver_shooting_z03[1];
output->z4[2] = doublependulum_QP_solver_shooting_z03[2];
output->z4[3] = doublependulum_QP_solver_shooting_z03[3];
output->z4[4] = doublependulum_QP_solver_shooting_z03[4];
output->z4[5] = doublependulum_QP_solver_shooting_z03[5];
output->z4[6] = doublependulum_QP_solver_shooting_z03[6];
output->z5[0] = doublependulum_QP_solver_shooting_z04[0];
output->z5[1] = doublependulum_QP_solver_shooting_z04[1];
output->z5[2] = doublependulum_QP_solver_shooting_z04[2];
output->z5[3] = doublependulum_QP_solver_shooting_z04[3];
output->z5[4] = doublependulum_QP_solver_shooting_z04[4];
output->z5[5] = doublependulum_QP_solver_shooting_z04[5];
output->z5[6] = doublependulum_QP_solver_shooting_z04[6];
output->z6[0] = doublependulum_QP_solver_shooting_z05[0];
output->z6[1] = doublependulum_QP_solver_shooting_z05[1];
output->z6[2] = doublependulum_QP_solver_shooting_z05[2];
output->z6[3] = doublependulum_QP_solver_shooting_z05[3];
output->z6[4] = doublependulum_QP_solver_shooting_z05[4];
output->z6[5] = doublependulum_QP_solver_shooting_z05[5];
output->z6[6] = doublependulum_QP_solver_shooting_z05[6];
output->z7[0] = doublependulum_QP_solver_shooting_z06[0];
output->z7[1] = doublependulum_QP_solver_shooting_z06[1];
output->z7[2] = doublependulum_QP_solver_shooting_z06[2];
output->z7[3] = doublependulum_QP_solver_shooting_z06[3];
output->z7[4] = doublependulum_QP_solver_shooting_z06[4];
output->z7[5] = doublependulum_QP_solver_shooting_z06[5];
output->z7[6] = doublependulum_QP_solver_shooting_z06[6];
output->z8[0] = doublependulum_QP_solver_shooting_z07[0];
output->z8[1] = doublependulum_QP_solver_shooting_z07[1];
output->z8[2] = doublependulum_QP_solver_shooting_z07[2];
output->z8[3] = doublependulum_QP_solver_shooting_z07[3];
output->z8[4] = doublependulum_QP_solver_shooting_z07[4];
output->z8[5] = doublependulum_QP_solver_shooting_z07[5];
output->z8[6] = doublependulum_QP_solver_shooting_z07[6];
output->z9[0] = doublependulum_QP_solver_shooting_z08[0];
output->z9[1] = doublependulum_QP_solver_shooting_z08[1];
output->z9[2] = doublependulum_QP_solver_shooting_z08[2];
output->z9[3] = doublependulum_QP_solver_shooting_z08[3];
output->z9[4] = doublependulum_QP_solver_shooting_z08[4];
output->z9[5] = doublependulum_QP_solver_shooting_z08[5];
output->z9[6] = doublependulum_QP_solver_shooting_z08[6];
output->z10[0] = doublependulum_QP_solver_shooting_z09[0];
output->z10[1] = doublependulum_QP_solver_shooting_z09[1];
output->z10[2] = doublependulum_QP_solver_shooting_z09[2];
output->z10[3] = doublependulum_QP_solver_shooting_z09[3];
output->z10[4] = doublependulum_QP_solver_shooting_z09[4];
output->z10[5] = doublependulum_QP_solver_shooting_z09[5];
output->z10[6] = doublependulum_QP_solver_shooting_z09[6];
output->z11[0] = doublependulum_QP_solver_shooting_z10[0];
output->z11[1] = doublependulum_QP_solver_shooting_z10[1];
output->z11[2] = doublependulum_QP_solver_shooting_z10[2];
output->z11[3] = doublependulum_QP_solver_shooting_z10[3];
output->z11[4] = doublependulum_QP_solver_shooting_z10[4];
output->z11[5] = doublependulum_QP_solver_shooting_z10[5];
output->z11[6] = doublependulum_QP_solver_shooting_z10[6];
output->z12[0] = doublependulum_QP_solver_shooting_z11[0];
output->z12[1] = doublependulum_QP_solver_shooting_z11[1];
output->z12[2] = doublependulum_QP_solver_shooting_z11[2];
output->z12[3] = doublependulum_QP_solver_shooting_z11[3];
output->z12[4] = doublependulum_QP_solver_shooting_z11[4];
output->z12[5] = doublependulum_QP_solver_shooting_z11[5];
output->z12[6] = doublependulum_QP_solver_shooting_z11[6];
output->z13[0] = doublependulum_QP_solver_shooting_z12[0];
output->z13[1] = doublependulum_QP_solver_shooting_z12[1];
output->z13[2] = doublependulum_QP_solver_shooting_z12[2];
output->z13[3] = doublependulum_QP_solver_shooting_z12[3];
output->z13[4] = doublependulum_QP_solver_shooting_z12[4];
output->z13[5] = doublependulum_QP_solver_shooting_z12[5];
output->z13[6] = doublependulum_QP_solver_shooting_z12[6];
output->z14[0] = doublependulum_QP_solver_shooting_z13[0];
output->z14[1] = doublependulum_QP_solver_shooting_z13[1];
output->z14[2] = doublependulum_QP_solver_shooting_z13[2];
output->z14[3] = doublependulum_QP_solver_shooting_z13[3];
output->z14[4] = doublependulum_QP_solver_shooting_z13[4];
output->z14[5] = doublependulum_QP_solver_shooting_z13[5];
output->z14[6] = doublependulum_QP_solver_shooting_z13[6];

#if doublependulum_QP_solver_shooting_SET_TIMING == 1
info->solvetime = doublependulum_QP_solver_shooting_toc(&solvertimer);
#if doublependulum_QP_solver_shooting_SET_PRINTLEVEL > 0 && doublependulum_QP_solver_shooting_SET_TIMING == 1
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
