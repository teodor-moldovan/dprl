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

#include "mex.h"
#include "../include/wam7dofarm_QP_solver.h"

/* copy functions */
void copyCArrayToM(wam7dofarm_QP_solver_FLOAT *src, double *dest, int dim) {
    while (dim--) {
        *dest++ = (double)*src++;
    }
}
void copyMArrayToC(double *src, wam7dofarm_QP_solver_FLOAT *dest, int dim) {
    while (dim--) {
        *dest++ = (wam7dofarm_QP_solver_FLOAT)*src++;
    }
}


/* THE mex-function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )  
{
	/* define variables */	
	mxArray *par;
	mxArray *outvar;
	const mxArray *PARAMS = prhs[0];
	double *pvalue;
	int i;
	int exitflag;
	const char *fname;
	const char *outputnames[8] = {"z1","z2","z3","z4","z5","z6","z7","z8"};
	const char *infofields[15] = { "it",
		                       "res_eq",
			                   "res_ineq",
		                       "pobj",
		                       "dobj",
		                       "dgap",
							   "rdgap",
							   "mu",
							   "mu_aff",
							   "sigma",
		                       "lsit_aff",
		                       "lsit_cc",
		                       "step_aff",
							   "step_cc",
							   "solvetime"};
	wam7dofarm_QP_solver_params params;
	wam7dofarm_QP_solver_output output;
	wam7dofarm_QP_solver_info info;
	
	/* Check for proper number of arguments */
    if (nrhs != 1) {
        mexErrMsgTxt("This function requires exactly 1 input: PARAMS struct.\nType 'help wam7dofarm_QP_solver_mex' for details.");
    }    
	if (nlhs > 3) {
        mexErrMsgTxt("This function returns at most 3 outputs.\nType 'help wam7dofarm_QP_solver_mex' for details.");
    }

	/* Check whether params is actually a structure */
	if( !mxIsStruct(PARAMS) ) {
		mexErrMsgTxt("PARAMS must be a structure.");
	}

	/* copy parameters into the right location */
	par = mxGetField(PARAMS, 0, "f1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f1 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f1 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f1, 64);

	par = mxGetField(PARAMS, 0, "lb1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb1 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb1 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb1, 64);

	par = mxGetField(PARAMS, 0, "ub1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub1 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub1 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub1, 36);

	par = mxGetField(PARAMS, 0, "C1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C1 must be a double.");
    }
    if( mxGetM(par) != 29 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C1 must be of size [29 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C1, 1856);

	par = mxGetField(PARAMS, 0, "e1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e1 must be a double.");
    }
    if( mxGetM(par) != 29 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e1 must be of size [29 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e1, 29);

	par = mxGetField(PARAMS, 0, "f2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f2 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f2 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f2, 64);

	par = mxGetField(PARAMS, 0, "lb2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb2 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb2 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb2, 64);

	par = mxGetField(PARAMS, 0, "ub2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub2 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub2 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub2, 36);

	par = mxGetField(PARAMS, 0, "C2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C2 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C2 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C2, 960);

	par = mxGetField(PARAMS, 0, "e2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e2 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e2 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e2, 15);

	par = mxGetField(PARAMS, 0, "f3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f3 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f3 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f3, 64);

	par = mxGetField(PARAMS, 0, "lb3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb3 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb3 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb3, 64);

	par = mxGetField(PARAMS, 0, "ub3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub3 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub3 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub3, 36);

	par = mxGetField(PARAMS, 0, "C3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C3 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C3 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C3, 960);

	par = mxGetField(PARAMS, 0, "e3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e3 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e3 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e3, 15);

	par = mxGetField(PARAMS, 0, "f4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f4 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f4 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f4, 64);

	par = mxGetField(PARAMS, 0, "lb4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb4 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb4 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb4, 64);

	par = mxGetField(PARAMS, 0, "ub4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub4 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub4 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub4, 36);

	par = mxGetField(PARAMS, 0, "C4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C4 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C4 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C4, 960);

	par = mxGetField(PARAMS, 0, "e4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e4 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e4 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e4, 15);

	par = mxGetField(PARAMS, 0, "f5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f5 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f5 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f5, 64);

	par = mxGetField(PARAMS, 0, "lb5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb5 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb5 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb5, 64);

	par = mxGetField(PARAMS, 0, "ub5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub5 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub5 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub5, 36);

	par = mxGetField(PARAMS, 0, "C5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C5 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C5 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C5, 960);

	par = mxGetField(PARAMS, 0, "e5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e5 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e5 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e5, 15);

	par = mxGetField(PARAMS, 0, "f6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f6 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f6 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f6, 64);

	par = mxGetField(PARAMS, 0, "lb6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb6 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb6 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb6, 64);

	par = mxGetField(PARAMS, 0, "ub6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub6 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub6 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub6, 36);

	par = mxGetField(PARAMS, 0, "C6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C6 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C6 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C6, 960);

	par = mxGetField(PARAMS, 0, "e6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e6 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e6 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e6, 15);

	par = mxGetField(PARAMS, 0, "f7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f7 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f7 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f7, 64);

	par = mxGetField(PARAMS, 0, "lb7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb7 must be a double.");
    }
    if( mxGetM(par) != 64 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb7 must be of size [64 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb7, 64);

	par = mxGetField(PARAMS, 0, "ub7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub7 must be a double.");
    }
    if( mxGetM(par) != 36 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub7 must be of size [36 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub7, 36);

	par = mxGetField(PARAMS, 0, "C7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.C7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.C7 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 64 ) {
    mexErrMsgTxt("PARAMS.C7 must be of size [15 x 64]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.C7, 960);

	par = mxGetField(PARAMS, 0, "e7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.e7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.e7 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.e7 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.e7, 15);

	par = mxGetField(PARAMS, 0, "f8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f8 must be a double.");
    }
    if( mxGetM(par) != 27 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f8 must be of size [27 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f8, 27);

	par = mxGetField(PARAMS, 0, "lb8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb8 must be a double.");
    }
    if( mxGetM(par) != 27 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb8 must be of size [27 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb8, 27);

	par = mxGetField(PARAMS, 0, "ub8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub8 must be a double.");
    }
    if( mxGetM(par) != 15 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub8 must be of size [15 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub8, 15);

	par = mxGetField(PARAMS, 0, "A8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.A8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.A8 must be a double.");
    }
    if( mxGetM(par) != 12 || mxGetN(par) != 27 ) {
    mexErrMsgTxt("PARAMS.A8 must be of size [12 x 27]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.A8, 324);

	par = mxGetField(PARAMS, 0, "b8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.b8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.b8 must be a double.");
    }
    if( mxGetM(par) != 12 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.b8 must be of size [12 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.b8, 12);

	/* call solver */
	exitflag = wam7dofarm_QP_solver_solve(&params, &output, &info);
	
	/* copy output to matlab arrays */
	plhs[0] = mxCreateStructMatrix(1, 1, 8, outputnames);
	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z1, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z1", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z2, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z2", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z3, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z3", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z4, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z4", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z5, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z5", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z6, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z6", outvar);

	outvar = mxCreateDoubleMatrix(36, 1, mxREAL);
	copyCArrayToM( output.z7, mxGetPr(outvar), 36);
	mxSetField(plhs[0], 0, "z7", outvar);

	outvar = mxCreateDoubleMatrix(15, 1, mxREAL);
	copyCArrayToM( output.z8, mxGetPr(outvar), 15);
	mxSetField(plhs[0], 0, "z8", outvar);	

	/* copy exitflag */
	if( nlhs > 1 )
	{
		plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(plhs[1]) = (double)exitflag;
	}

	/* copy info struct */
	if( nlhs > 2 )
	{
		plhs[2] = mxCreateStructMatrix(1, 1, 15, infofields);
		
		/* iterations */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = (double)info.it;
		mxSetField(plhs[2], 0, "it", outvar);
		
		/* res_eq */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.res_eq;
		mxSetField(plhs[2], 0, "res_eq", outvar);

		/* res_ineq */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.res_ineq;
		mxSetField(plhs[2], 0, "res_ineq", outvar);

		/* pobj */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.pobj;
		mxSetField(plhs[2], 0, "pobj", outvar);

		/* dobj */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.dobj;
		mxSetField(plhs[2], 0, "dobj", outvar);

		/* dgap */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.dgap;
		mxSetField(plhs[2], 0, "dgap", outvar);

		/* rdgap */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.rdgap;
		mxSetField(plhs[2], 0, "rdgap", outvar);

		/* mu */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.mu;
		mxSetField(plhs[2], 0, "mu", outvar);

		/* mu_aff */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.mu_aff;
		mxSetField(plhs[2], 0, "mu_aff", outvar);

		/* sigma */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.sigma;
		mxSetField(plhs[2], 0, "sigma", outvar);

		/* lsit_aff */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = (double)info.lsit_aff;
		mxSetField(plhs[2], 0, "lsit_aff", outvar);

		/* lsit_cc */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = (double)info.lsit_cc;
		mxSetField(plhs[2], 0, "lsit_cc", outvar);

		/* step_aff */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.step_aff;
		mxSetField(plhs[2], 0, "step_aff", outvar);

		/* step_cc */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.step_cc;
		mxSetField(plhs[2], 0, "step_cc", outvar);

		/* solver time */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(outvar) = info.solvetime;
		mxSetField(plhs[2], 0, "solvetime", outvar);
	}
}