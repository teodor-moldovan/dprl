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

#ifndef __doublependulum_QP_solver_shooting_H__
#define __doublependulum_QP_solver_shooting_H__


/* DATA TYPE ------------------------------------------------------------*/
typedef double doublependulum_QP_solver_shooting_FLOAT;


/* SOLVER SETTINGS ------------------------------------------------------*/
/* print level */
#ifndef doublependulum_QP_solver_shooting_SET_PRINTLEVEL
#define doublependulum_QP_solver_shooting_SET_PRINTLEVEL    (0)
#endif

/* timing */
#ifndef doublependulum_QP_solver_shooting_SET_TIMING
#define doublependulum_QP_solver_shooting_SET_TIMING    (0)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define doublependulum_QP_solver_shooting_SET_MAXIT         (50)	

/* scaling factor of line search (affine direction) */
#define doublependulum_QP_solver_shooting_SET_LS_SCALE_AFF  (0.9)      

/* scaling factor of line search (combined direction) */
#define doublependulum_QP_solver_shooting_SET_LS_SCALE      (0.95)  

/* minimum required step size in each iteration */
#define doublependulum_QP_solver_shooting_SET_LS_MINSTEP    (1E-08)

/* maximum step size (combined direction) */
#define doublependulum_QP_solver_shooting_SET_LS_MAXSTEP    (0.995)

/* desired relative duality gap */
#define doublependulum_QP_solver_shooting_SET_ACC_RDGAP     (0.0001)

/* desired maximum residual on equality constraints */
#define doublependulum_QP_solver_shooting_SET_ACC_RESEQ     (1E-06)

/* desired maximum residual on inequality constraints */
#define doublependulum_QP_solver_shooting_SET_ACC_RESINEQ   (1E-06)

/* desired maximum violation of complementarity */
#define doublependulum_QP_solver_shooting_SET_ACC_KKTCOMPL  (1E-06)


/* RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define doublependulum_QP_solver_shooting_OPTIMAL      (1)

/* maximum number of iterations has been reached */
#define doublependulum_QP_solver_shooting_MAXITREACHED (0)

/* no progress in line search possible */
#define doublependulum_QP_solver_shooting_NOPROGRESS   (-7)




/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct doublependulum_QP_solver_shooting_params
{
    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H1[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f1[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb1[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub1[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H2[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f2[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb2[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub2[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H3[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f3[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb3[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub3[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H4[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f4[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb4[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub4[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H5[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f5[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb5[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub5[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H6[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f6[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb6[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub6[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H7[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f7[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb7[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub7[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H8[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f8[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb8[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub8[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H9[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f9[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb9[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub9[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H10[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f10[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb10[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub10[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H11[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f11[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb11[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub11[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H12[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f12[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb12[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub12[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H13[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f13[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb13[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub13[7];

    /* diagonal matrix of size [7 x 7] (only the diagonal is stored) */
    doublependulum_QP_solver_shooting_FLOAT H14[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT f14[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT lb14[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT ub14[7];

} doublependulum_QP_solver_shooting_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct doublependulum_QP_solver_shooting_output
{
    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z1[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z2[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z3[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z4[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z5[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z6[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z7[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z8[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z9[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z10[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z11[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z12[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z13[7];

    /* vector of size 7 */
    doublependulum_QP_solver_shooting_FLOAT z14[7];

} doublependulum_QP_solver_shooting_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct doublependulum_QP_solver_shooting_info
{
    /* iteration number */
    int it;
	
    /* inf-norm of equality constraint residuals */
    doublependulum_QP_solver_shooting_FLOAT res_eq;
	
    /* inf-norm of inequality constraint residuals */
    doublependulum_QP_solver_shooting_FLOAT res_ineq;

    /* primal objective */
    doublependulum_QP_solver_shooting_FLOAT pobj;	
	
    /* dual objective */
    doublependulum_QP_solver_shooting_FLOAT dobj;	

    /* duality gap := pobj - dobj */
    doublependulum_QP_solver_shooting_FLOAT dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    doublependulum_QP_solver_shooting_FLOAT rdgap;		

    /* duality measure */
    doublependulum_QP_solver_shooting_FLOAT mu;

	/* duality measure (after affine step) */
    doublependulum_QP_solver_shooting_FLOAT mu_aff;
	
    /* centering parameter */
    doublependulum_QP_solver_shooting_FLOAT sigma;
	
    /* number of backtracking line search steps (affine direction) */
    int lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    int lsit_cc;
    
    /* step size (affine direction) */
    doublependulum_QP_solver_shooting_FLOAT step_aff;
    
    /* step size (combined direction) */
    doublependulum_QP_solver_shooting_FLOAT step_cc;    

	/* solvertime */
	doublependulum_QP_solver_shooting_FLOAT solvetime;   

} doublependulum_QP_solver_shooting_info;


/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* examine exitflag before using the result! */
int doublependulum_QP_solver_shooting_solve(doublependulum_QP_solver_shooting_params* params, doublependulum_QP_solver_shooting_output* output, doublependulum_QP_solver_shooting_info* info);


#endif