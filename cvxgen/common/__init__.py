from solver import *

import os 
import numpy as np
import pyparsing as pp
import ctypes
import warnings

def array_sizes():
    this_dir, this_filename = os.path.split(__file__)
    sf = os.path.join(this_dir, "cvxgen", "solver.h")

    #header = open(sf).read()

    nums = pp.Word('0123456789')
    idn = pp.Word( pp.alphas+"_", pp.alphanums+"_" )
    rng = pp.Group(pp.Literal('[') + nums + pp.Literal(']'))
    comm = pp.Group(pp.Literal('/*') +pp.Optional(nums + pp.Literal('rows')) + pp.SkipTo('*/') + pp.Literal('*/'))
    decl = (idn + pp.Optional('*') + idn + pp.Optional(rng) 
            + pp.Literal(';') + pp.Optional(comm) )

    blk = ( pp.Group(pp.Literal('typedef') + 
        pp.Literal('struct') + idn + 
        pp.Literal('{')) + pp.Group(pp.OneOrMore(pp.Group(decl))) +
         pp.Group(pp.Literal('}') + idn + pp.Literal(';')) 
            )

    extrn = pp.Group(pp.Literal('extern') + idn + idn + pp.Literal(';'))


    ex = (pp.SkipTo(pp.Literal('typedef')) + pp.Group(pp.ZeroOrMore(pp.Group(blk))) 
            + pp.Group(pp.ZeroOrMore(extrn)))
    rt = ex.parseFile(sf)

    evd = {}
    for ev in rt[-1]:
        evd[ev[1]]=ev[2]

    dct = {}
    for bl in rt[1]:
        stt = evd[bl[-1][-2]]
        dct[stt] = {}
        curd = dct[stt]
        for it in bl[1]:
            if it[1]=='*':
                if  it[-1]!=';' and it[-1][2]=='rows':
                    curd[it[2]]= int(it[-1][1])
            elif it[2]!=';':
                curd[it[1]]= int(it[2][1])
    return dct


array_sizes = array_sizes()

def npwrap(it,sz=None):
    ptr =  eval(it)
    _,s,i = it.split('.')
    if sz is None:
        sz = array_sizes[s][i]
    pn =  eval('ctypes.c_'+str(ptr).split('_')[3])* sz
    pn = pn.from_address(int(ptr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.ctypeslib.as_array(pn)
    
    
#print npwrap('cvar.params.x')

