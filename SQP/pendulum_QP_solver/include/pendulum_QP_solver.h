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

#ifndef __pendulum_QP_solver_H__
#define __pendulum_QP_solver_H__


/* DATA TYPE ------------------------------------------------------------*/
typedef double pendulum_QP_solver_FLOAT;


/* SOLVER SETTINGS ------------------------------------------------------*/
/* print level */
#ifndef pendulum_QP_solver_SET_PRINTLEVEL
#define pendulum_QP_solver_SET_PRINTLEVEL    (0)
#endif

/* timing */
#ifndef pendulum_QP_solver_SET_TIMING
#define pendulum_QP_solver_SET_TIMING    (0)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define pendulum_QP_solver_SET_MAXIT         (50)	

/* scaling factor of line search (affine direction) */
#define pendulum_QP_solver_SET_LS_SCALE_AFF  (0.9)      

/* scaling factor of line search (combined direction) */
#define pendulum_QP_solver_SET_LS_SCALE      (0.95)  

/* minimum required step size in each iteration */
#define pendulum_QP_solver_SET_LS_MINSTEP    (1E-08)

/* maximum step size (combined direction) */
#define pendulum_QP_solver_SET_LS_MAXSTEP    (0.995)

/* desired relative duality gap */
#define pendulum_QP_solver_SET_ACC_RDGAP     (0.0001)

/* desired maximum residual on equality constraints */
#define pendulum_QP_solver_SET_ACC_RESEQ     (1E-06)

/* desired maximum residual on inequality constraints */
#define pendulum_QP_solver_SET_ACC_RESINEQ   (1E-06)

/* desired maximum violation of complementarity */
#define pendulum_QP_solver_SET_ACC_KKTCOMPL  (1E-06)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define pendulum_QP_solver_OPTIMAL      (1)

/* maximum number of iterations has been reached */
#define pendulum_QP_solver_MAXITREACHED (0)

/* no progress in line search possible */
#define pendulum_QP_solver_NOPROGRESS   (-7)




/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct pendulum_QP_solver_params
{
    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f1[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb1[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub1[6];

    /* matrix of size [5 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C1[50];

    /* vector of size 5 */
    pendulum_QP_solver_FLOAT e1[5];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f2[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb2[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub2[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C2[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e2[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f3[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb3[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub3[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C3[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e3[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f4[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb4[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub4[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C4[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e4[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f5[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb5[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub5[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C5[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e5[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f6[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb6[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub6[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C6[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e6[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f7[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb7[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub7[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C7[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e7[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f8[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb8[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub8[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C8[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e8[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f9[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb9[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub9[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C9[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e9[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f10[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb10[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub10[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C10[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e10[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f11[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb11[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub11[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C11[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e11[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f12[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb12[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub12[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C12[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e12[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f13[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb13[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub13[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C13[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e13[3];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT f14[10];

    /* vector of size 10 */
    pendulum_QP_solver_FLOAT lb14[10];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT ub14[6];

    /* matrix of size [3 x 10] (column major format) */
    pendulum_QP_solver_FLOAT C14[30];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT e14[3];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT lb15[3];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT ub15[3];

} pendulum_QP_solver_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct pendulum_QP_solver_output
{
    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z1[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z2[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z3[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z4[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z5[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z6[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z7[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z8[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z9[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z10[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z11[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z12[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z13[6];

    /* vector of size 6 */
    pendulum_QP_solver_FLOAT z14[6];

    /* vector of size 3 */
    pendulum_QP_solver_FLOAT z15[3];

} pendulum_QP_solver_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct pendulum_QP_solver_info
{
    /* iteration number */
    int it;
	
    /* inf-norm of equality constraint residuals */
    pendulum_QP_solver_FLOAT res_eq;
	
    /* inf-norm of inequality constraint residuals */
    pendulum_QP_solver_FLOAT res_ineq;

    /* primal objective */
    pendulum_QP_solver_FLOAT pobj;	
	
    /* dual objective */
    pendulum_QP_solver_FLOAT dobj;	

    /* duality gap := pobj - dobj */
    pendulum_QP_solver_FLOAT dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    pendulum_QP_solver_FLOAT rdgap;		

    /* duality measure */
    pendulum_QP_solver_FLOAT mu;

	/* duality measure (after affine step) */
    pendulum_QP_solver_FLOAT mu_aff;
	
    /* centering parameter */
    pendulum_QP_solver_FLOAT sigma;
	
    /* number of backtracking line search steps (affine direction) */
    int lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    int lsit_cc;
    
    /* step size (affine direction) */
    pendulum_QP_solver_FLOAT step_aff;
    
    /* step size (combined direction) */
    pendulum_QP_solver_FLOAT step_cc;    

	/* solvertime */
	pendulum_QP_solver_FLOAT solvetime;   

} pendulum_QP_solver_info;


/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* examine exitflag before using the result! */
int pendulum_QP_solver_solve(pendulum_QP_solver_params* params, pendulum_QP_solver_output* output, pendulum_QP_solver_info* info);


#endif