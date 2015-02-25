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

#ifndef __wam7dofarm_QP_solver_H__
#define __wam7dofarm_QP_solver_H__


/* DATA TYPE ------------------------------------------------------------*/
typedef double wam7dofarm_QP_solver_FLOAT;


/* SOLVER SETTINGS ------------------------------------------------------*/
/* print level */
#ifndef wam7dofarm_QP_solver_SET_PRINTLEVEL
#define wam7dofarm_QP_solver_SET_PRINTLEVEL    (0)
#endif

/* timing */
#ifndef wam7dofarm_QP_solver_SET_TIMING
#define wam7dofarm_QP_solver_SET_TIMING    (0)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define wam7dofarm_QP_solver_SET_MAXIT         (50)	

/* scaling factor of line search (affine direction) */
#define wam7dofarm_QP_solver_SET_LS_SCALE_AFF  (0.9)      

/* scaling factor of line search (combined direction) */
#define wam7dofarm_QP_solver_SET_LS_SCALE      (0.95)  

/* minimum required step size in each iteration */
#define wam7dofarm_QP_solver_SET_LS_MINSTEP    (1E-08)

/* maximum step size (combined direction) */
#define wam7dofarm_QP_solver_SET_LS_MAXSTEP    (0.995)

/* desired relative duality gap */
#define wam7dofarm_QP_solver_SET_ACC_RDGAP     (0.0001)

/* desired maximum residual on equality constraints */
#define wam7dofarm_QP_solver_SET_ACC_RESEQ     (1E-06)

/* desired maximum residual on inequality constraints */
#define wam7dofarm_QP_solver_SET_ACC_RESINEQ   (1E-06)

/* desired maximum violation of complementarity */
#define wam7dofarm_QP_solver_SET_ACC_KKTCOMPL  (1E-06)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define wam7dofarm_QP_solver_OPTIMAL      (1)

/* maximum number of iterations has been reached */
#define wam7dofarm_QP_solver_MAXITREACHED (0)

/* no progress in line search possible */
#define wam7dofarm_QP_solver_NOPROGRESS   (-7)




/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct wam7dofarm_QP_solver_params
{
    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT f1[64];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT lb1[64];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT ub1[36];

    /* matrix of size [29 x 64] (column major format) */
    wam7dofarm_QP_solver_FLOAT C1[1856];

    /* vector of size 29 */
    wam7dofarm_QP_solver_FLOAT e1[29];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT f2[64];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT lb2[64];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT ub2[36];

    /* matrix of size [15 x 64] (column major format) */
    wam7dofarm_QP_solver_FLOAT C2[960];

    /* vector of size 15 */
    wam7dofarm_QP_solver_FLOAT e2[15];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT f3[64];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT lb3[64];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT ub3[36];

    /* matrix of size [15 x 64] (column major format) */
    wam7dofarm_QP_solver_FLOAT C3[960];

    /* vector of size 15 */
    wam7dofarm_QP_solver_FLOAT e3[15];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT f4[64];

    /* vector of size 64 */
    wam7dofarm_QP_solver_FLOAT lb4[64];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT ub4[36];

    /* matrix of size [15 x 64] (column major format) */
    wam7dofarm_QP_solver_FLOAT C4[960];

    /* vector of size 15 */
    wam7dofarm_QP_solver_FLOAT e4[15];

    /* vector of size 27 */
    wam7dofarm_QP_solver_FLOAT f5[27];

    /* vector of size 27 */
    wam7dofarm_QP_solver_FLOAT lb5[27];

    /* vector of size 15 */
    wam7dofarm_QP_solver_FLOAT ub5[15];

    /* matrix of size [12 x 27] (column major format) */
    wam7dofarm_QP_solver_FLOAT A5[324];

    /* vector of size 12 */
    wam7dofarm_QP_solver_FLOAT b5[12];

} wam7dofarm_QP_solver_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct wam7dofarm_QP_solver_output
{
    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT z1[36];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT z2[36];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT z3[36];

    /* vector of size 36 */
    wam7dofarm_QP_solver_FLOAT z4[36];

    /* vector of size 15 */
    wam7dofarm_QP_solver_FLOAT z5[15];

} wam7dofarm_QP_solver_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct wam7dofarm_QP_solver_info
{
    /* iteration number */
    int it;
	
    /* inf-norm of equality constraint residuals */
    wam7dofarm_QP_solver_FLOAT res_eq;
	
    /* inf-norm of inequality constraint residuals */
    wam7dofarm_QP_solver_FLOAT res_ineq;

    /* primal objective */
    wam7dofarm_QP_solver_FLOAT pobj;	
	
    /* dual objective */
    wam7dofarm_QP_solver_FLOAT dobj;	

    /* duality gap := pobj - dobj */
    wam7dofarm_QP_solver_FLOAT dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    wam7dofarm_QP_solver_FLOAT rdgap;		

    /* duality measure */
    wam7dofarm_QP_solver_FLOAT mu;

	/* duality measure (after affine step) */
    wam7dofarm_QP_solver_FLOAT mu_aff;
	
    /* centering parameter */
    wam7dofarm_QP_solver_FLOAT sigma;
	
    /* number of backtracking line search steps (affine direction) */
    int lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    int lsit_cc;
    
    /* step size (affine direction) */
    wam7dofarm_QP_solver_FLOAT step_aff;
    
    /* step size (combined direction) */
    wam7dofarm_QP_solver_FLOAT step_cc;    

	/* solvertime */
	wam7dofarm_QP_solver_FLOAT solvetime;   

} wam7dofarm_QP_solver_info;


/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* examine exitflag before using the result! */
int wam7dofarm_QP_solver_solve(wam7dofarm_QP_solver_params* params, wam7dofarm_QP_solver_output* output, wam7dofarm_QP_solver_info* info);


#endif