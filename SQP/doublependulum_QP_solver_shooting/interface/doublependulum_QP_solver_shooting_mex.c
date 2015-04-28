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
#include "../include/doublependulum_QP_solver_shooting.h"

/* copy functions */
void copyCArrayToM(doublependulum_QP_solver_shooting_FLOAT *src, double *dest, int dim) {
    while (dim--) {
        *dest++ = (double)*src++;
    }
}
void copyMArrayToC(double *src, doublependulum_QP_solver_shooting_FLOAT *dest, int dim) {
    while (dim--) {
        *dest++ = (doublependulum_QP_solver_shooting_FLOAT)*src++;
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
	const char *outputnames[14] = {"z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14"};
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
	doublependulum_QP_solver_shooting_params params;
	doublependulum_QP_solver_shooting_output output;
	doublependulum_QP_solver_shooting_info info;
	
	/* Check for proper number of arguments */
    if (nrhs != 1) {
        mexErrMsgTxt("This function requires exactly 1 input: PARAMS struct.\nType 'help doublependulum_QP_solver_shooting_mex' for details.");
    }    
	if (nlhs > 3) {
        mexErrMsgTxt("This function returns at most 3 outputs.\nType 'help doublependulum_QP_solver_shooting_mex' for details.");
    }

	/* Check whether params is actually a structure */
	if( !mxIsStruct(PARAMS) ) {
		mexErrMsgTxt("PARAMS must be a structure.");
	}

	/* copy parameters into the right location */
	par = mxGetField(PARAMS, 0, "H1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H1 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H1 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H1, 7);

	par = mxGetField(PARAMS, 0, "f1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f1 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f1 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f1, 7);

	par = mxGetField(PARAMS, 0, "lb1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb1 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb1 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb1, 7);

	par = mxGetField(PARAMS, 0, "ub1");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub1 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub1 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub1 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub1, 7);

	par = mxGetField(PARAMS, 0, "H2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H2 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H2 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H2, 7);

	par = mxGetField(PARAMS, 0, "f2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f2 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f2 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f2, 7);

	par = mxGetField(PARAMS, 0, "lb2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb2 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb2 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb2, 7);

	par = mxGetField(PARAMS, 0, "ub2");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub2 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub2 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub2 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub2, 7);

	par = mxGetField(PARAMS, 0, "H3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H3 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H3 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H3, 7);

	par = mxGetField(PARAMS, 0, "f3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f3 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f3 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f3, 7);

	par = mxGetField(PARAMS, 0, "lb3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb3 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb3 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb3, 7);

	par = mxGetField(PARAMS, 0, "ub3");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub3 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub3 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub3 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub3, 7);

	par = mxGetField(PARAMS, 0, "H4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H4 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H4 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H4, 7);

	par = mxGetField(PARAMS, 0, "f4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f4 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f4 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f4, 7);

	par = mxGetField(PARAMS, 0, "lb4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb4 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb4 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb4, 7);

	par = mxGetField(PARAMS, 0, "ub4");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub4 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub4 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub4 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub4, 7);

	par = mxGetField(PARAMS, 0, "H5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H5 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H5 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H5, 7);

	par = mxGetField(PARAMS, 0, "f5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f5 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f5 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f5, 7);

	par = mxGetField(PARAMS, 0, "lb5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb5 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb5 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb5, 7);

	par = mxGetField(PARAMS, 0, "ub5");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub5 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub5 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub5 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub5, 7);

	par = mxGetField(PARAMS, 0, "H6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H6 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H6 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H6, 7);

	par = mxGetField(PARAMS, 0, "f6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f6 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f6 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f6, 7);

	par = mxGetField(PARAMS, 0, "lb6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb6 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb6 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb6, 7);

	par = mxGetField(PARAMS, 0, "ub6");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub6 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub6 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub6 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub6, 7);

	par = mxGetField(PARAMS, 0, "H7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H7 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H7 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H7, 7);

	par = mxGetField(PARAMS, 0, "f7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f7 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f7 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f7, 7);

	par = mxGetField(PARAMS, 0, "lb7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb7 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb7 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb7, 7);

	par = mxGetField(PARAMS, 0, "ub7");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub7 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub7 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub7 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub7, 7);

	par = mxGetField(PARAMS, 0, "H8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H8 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H8 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H8, 7);

	par = mxGetField(PARAMS, 0, "f8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f8 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f8 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f8, 7);

	par = mxGetField(PARAMS, 0, "lb8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb8 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb8 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb8, 7);

	par = mxGetField(PARAMS, 0, "ub8");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub8 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub8 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub8 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub8, 7);

	par = mxGetField(PARAMS, 0, "H9");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H9 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H9 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H9 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H9, 7);

	par = mxGetField(PARAMS, 0, "f9");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f9 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f9 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f9 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f9, 7);

	par = mxGetField(PARAMS, 0, "lb9");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb9 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb9 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb9 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb9, 7);

	par = mxGetField(PARAMS, 0, "ub9");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub9 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub9 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub9 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub9, 7);

	par = mxGetField(PARAMS, 0, "H10");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H10 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H10 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H10 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H10, 7);

	par = mxGetField(PARAMS, 0, "f10");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f10 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f10 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f10 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f10, 7);

	par = mxGetField(PARAMS, 0, "lb10");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb10 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb10 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb10 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb10, 7);

	par = mxGetField(PARAMS, 0, "ub10");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub10 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub10 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub10 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub10, 7);

	par = mxGetField(PARAMS, 0, "H11");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H11 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H11 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H11 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H11, 7);

	par = mxGetField(PARAMS, 0, "f11");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f11 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f11 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f11 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f11, 7);

	par = mxGetField(PARAMS, 0, "lb11");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb11 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb11 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb11 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb11, 7);

	par = mxGetField(PARAMS, 0, "ub11");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub11 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub11 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub11 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub11, 7);

	par = mxGetField(PARAMS, 0, "H12");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H12 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H12 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H12 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H12, 7);

	par = mxGetField(PARAMS, 0, "f12");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f12 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f12 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f12 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f12, 7);

	par = mxGetField(PARAMS, 0, "lb12");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb12 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb12 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb12 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb12, 7);

	par = mxGetField(PARAMS, 0, "ub12");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub12 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub12 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub12 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub12, 7);

	par = mxGetField(PARAMS, 0, "H13");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H13 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H13 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H13 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H13, 7);

	par = mxGetField(PARAMS, 0, "f13");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f13 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f13 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f13 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f13, 7);

	par = mxGetField(PARAMS, 0, "lb13");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb13 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb13 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb13 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb13, 7);

	par = mxGetField(PARAMS, 0, "ub13");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub13 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub13 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub13 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub13, 7);

	par = mxGetField(PARAMS, 0, "H14");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.H14 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.H14 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.H14 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.H14, 7);

	par = mxGetField(PARAMS, 0, "f14");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.f14 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.f14 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.f14 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.f14, 7);

	par = mxGetField(PARAMS, 0, "lb14");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.lb14 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.lb14 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.lb14 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.lb14, 7);

	par = mxGetField(PARAMS, 0, "ub14");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	{
        mexErrMsgTxt("PARAMS.ub14 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.ub14 must be a double.");
    }
    if( mxGetM(par) != 7 || mxGetN(par) != 1 ) {
    mexErrMsgTxt("PARAMS.ub14 must be of size [7 x 1]");
    }
#endif	 
    copyMArrayToC(mxGetPr(par), params.ub14, 7);

	/* call solver */
	exitflag = doublependulum_QP_solver_shooting_solve(&params, &output, &info);
	
	/* copy output to matlab arrays */
	plhs[0] = mxCreateStructMatrix(1, 1, 14, outputnames);
	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z1, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z1", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z2, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z2", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z3, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z3", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z4, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z4", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z5, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z5", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z6, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z6", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z7, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z7", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z8, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z8", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z9, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z9", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z10, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z10", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z11, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z11", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z12, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z12", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z13, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z13", outvar);

	outvar = mxCreateDoubleMatrix(7, 1, mxREAL);
	copyCArrayToM( output.z14, mxGetPr(outvar), 7);
	mxSetField(plhs[0], 0, "z14", outvar);	

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