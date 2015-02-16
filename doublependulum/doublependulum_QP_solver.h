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

#ifndef __doublependulum_QP_solver_H__
#define __doublependulum_QP_solver_H__


/* DATA TYPE ------------------------------------------------------------*/
typedef double doublependulum_QP_solver_FLOAT;


/* SOLVER SETTINGS ------------------------------------------------------*/
/* print level */
#ifndef doublependulum_QP_solver_SET_PRINTLEVEL
#define doublependulum_QP_solver_SET_PRINTLEVEL    (0)
#endif

/* timing */
#ifndef doublependulum_QP_solver_SET_TIMING
#define doublependulum_QP_solver_SET_TIMING    (0)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define doublependulum_QP_solver_SET_MAXIT         (50)	

/* scaling factor of line search (affine direction) */
#define doublependulum_QP_solver_SET_LS_SCALE_AFF  (0.9)      

/* scaling factor of line search (combined direction) */
#define doublependulum_QP_solver_SET_LS_SCALE      (0.95)  

/* minimum required step size in each iteration */
#define doublependulum_QP_solver_SET_LS_MINSTEP    (1E-08)

/* maximum step size (combined direction) */
#define doublependulum_QP_solver_SET_LS_MAXSTEP    (0.995)

/* desired relative duality gap */
#define doublependulum_QP_solver_SET_ACC_RDGAP     (0.0001)

/* desired maximum residual on equality constraints */
#define doublependulum_QP_solver_SET_ACC_RESEQ     (1E-06)

/* desired maximum residual on inequality constraints */
#define doublependulum_QP_solver_SET_ACC_RESINEQ   (1E-06)

/* desired maximum violation of complementarity */
#define doublependulum_QP_solver_SET_ACC_KKTCOMPL  (1E-06)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define doublependulum_QP_solver_OPTIMAL      (1)

/* maximum number of iterations has been reached */
#define doublependulum_QP_solver_MAXITREACHED (0)

/* no progress in line search possible */
#define doublependulum_QP_solver_NOPROGRESS   (-7)




/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct doublependulum_QP_solver_params
{
    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f1[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb1[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub1[11];

    /* matrix of size [9 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C1[171];

    /* vector of size 9 */
    doublependulum_QP_solver_FLOAT e1[9];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f2[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb2[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub2[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C2[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e2[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f3[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb3[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub3[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C3[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e3[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f4[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb4[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub4[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C4[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e4[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f5[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb5[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub5[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C5[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e5[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f6[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb6[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub6[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C6[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e6[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f7[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb7[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub7[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C7[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e7[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f8[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb8[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub8[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C8[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e8[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f9[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb9[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub9[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C9[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e9[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f10[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb10[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub10[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C10[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e10[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f11[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb11[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub11[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C11[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e11[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f12[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb12[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub12[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C12[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e12[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f13[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb13[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub13[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C13[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e13[5];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT f14[19];

    /* vector of size 19 */
    doublependulum_QP_solver_FLOAT lb14[19];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT ub14[11];

    /* matrix of size [5 x 19] (column major format) */
    doublependulum_QP_solver_FLOAT C14[95];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT e14[5];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT lb15[5];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT ub15[5];

} doublependulum_QP_solver_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct doublependulum_QP_solver_output
{
    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z1[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z2[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z3[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z4[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z5[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z6[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z7[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z8[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z9[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z10[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z11[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z12[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z13[11];

    /* vector of size 11 */
    doublependulum_QP_solver_FLOAT z14[11];

    /* vector of size 5 */
    doublependulum_QP_solver_FLOAT z15[5];

} doublependulum_QP_solver_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct doublependulum_QP_solver_info
{
    /* iteration number */
    int it;
	
    /* inf-norm of equality constraint residuals */
    doublependulum_QP_solver_FLOAT res_eq;
	
    /* inf-norm of inequality constraint residuals */
    doublependulum_QP_solver_FLOAT res_ineq;

    /* primal objective */
    doublependulum_QP_solver_FLOAT pobj;	
	
    /* dual objective */
    doublependulum_QP_solver_FLOAT dobj;	

    /* duality gap := pobj - dobj */
    doublependulum_QP_solver_FLOAT dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    doublependulum_QP_solver_FLOAT rdgap;		

    /* duality measure */
    doublependulum_QP_solver_FLOAT mu;

	/* duality measure (after affine step) */
    doublependulum_QP_solver_FLOAT mu_aff;
	
    /* centering parameter */
    doublependulum_QP_solver_FLOAT sigma;
	
    /* number of backtracking line search steps (affine direction) */
    int lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    int lsit_cc;
    
    /* step size (affine direction) */
    doublependulum_QP_solver_FLOAT step_aff;
    
    /* step size (combined direction) */
    doublependulum_QP_solver_FLOAT step_cc;    

	/* solvertime */
	doublependulum_QP_solver_FLOAT solvetime;   

} doublependulum_QP_solver_info;


/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* examine exitflag before using the result! */
int doublependulum_QP_solver_solve(doublependulum_QP_solver_params* params, doublependulum_QP_solver_output* output, doublependulum_QP_solver_info* info);


#endif