
#ifndef SWIMMER_DYNAMICS_H_
#define SWIMMER_DYNAMICS_H_


#include <math.h>
#define NX 11    // size of state space 
#define NU 3    // number of controls
#define NV 11    // number of virtual controls
#define NT 5    // number of constrained target state components
#define NW 3740    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

double control_min[NU] = { -1.0, -1.0, -1.0 };
double control_max[NU] = { 1.0, 1.0, 1.0 };

#define EPS 1e-5
        
// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { 2, 7, 8, 9, 10 };
// Constraint values
double target_state[NT] = { -0.5, 0.0, 0.0, 0.0, 0.0 };

                 

namespace swimmer {

    /* Function specifying dynamics.
    Format assumed: M(x) * \dot_x + g(x,u) = 0
    where x is state and u are controls

    Input:
        z : [x,u,xi] concatenated; z is assumed to have dimension NX + NU + NV
        weights : weight parameter vector as provided by learning component 
    Output: 
        M : output array equal to [M,g] concatenated,
              an NX x (NX + 1) matrix flattened in row major order.
              (note this output array needs to be pre-allocated)
        
    */

    void feval(double z[], double weights[], double out[])
    { 
        
    
    
    double tmp0;
    double tmp1;
    double tmp2;
    double tmp3;
    double tmp4;
    double tmp5;
    double tmp6;
    double tmp7;
    double tmp8;
    double tmp9;
    double tmp10;
    double tmp11;
    double tmp12;
    double tmp13;
    double tmp14;
    double tmp15;
    double tmp16;
    double tmp17;
    double tmp18;
    double tmp19;
    double tmp20;
    double tmp21;
    double tmp22;
    double tmp23;
    double tmp24;
    double tmp25;
    double tmp26;
    double tmp27;
    double tmp28;
    double tmp29;
    double tmp30;
    double tmp31;
    double tmp32;
    double tmp33;
    double tmp34;
    double tmp35;
    double tmp36;
    double tmp37;
    double tmp38;
    double tmp39;
    double tmp40;
    double tmp41;
    double tmp42;
    double tmp43;
    double tmp44;
    double tmp45;
    double tmp46;
    double tmp47;
    double tmp48;
    double tmp49;
    double tmp50;
    double tmp51;
    double tmp52;
    double tmp53;
    double tmp54;
    double tmp55;
    double tmp56;
    double tmp57;
    double tmp58;
    double tmp59;
    double tmp60;
    double tmp61;
    double tmp62;
    double tmp63;
    double tmp64;
    double tmp65;
    double tmp66;
    double tmp67;
    double tmp68;
    double tmp69;
    double tmp70;
    double tmp71;
    double tmp72;
    double tmp73;
    double tmp74;
    double tmp75;
    double tmp76;
    double tmp77;
    double tmp78;
    double tmp79;
    double tmp80;
    double tmp81;
    double tmp82;
    double tmp83;
    double tmp84;
    double tmp85;
    double tmp86;
    double tmp87;
    double tmp88;
    double tmp89;
    double tmp90;
    double tmp91;
    double tmp92;
    double tmp93;
    double tmp94;
    double tmp95;
    double tmp96;
    double tmp97;
    double tmp98;
    double tmp99;
    double tmp100;
    double tmp101;
    double tmp102;
    double tmp103;
    double tmp104;
    double tmp105;
    double tmp106;
    double tmp107;
    double tmp108;
    double tmp109;
    double tmp110;
    double tmp111;
    double tmp112;
    double tmp113;
    double tmp114;
    double tmp115;
    double tmp116;
    double tmp117;
    double tmp118;
    double tmp119;
    double tmp120;
    double tmp121;
    double tmp122;
    double tmp123;
    double tmp124;
    double tmp125;
    double tmp126;
    double tmp127;
    double tmp128;
    double tmp129;
    double tmp130;
    double tmp131;
    double tmp132;
    double tmp133;
    double tmp134;
    double tmp135;
    double tmp136;
    double tmp137;
    double tmp138;
    double tmp139;
    double tmp140;
    double tmp141;
    double tmp142;
    double tmp143;
    double tmp144;
    double tmp145;
    double tmp146;
    double tmp147;
    double tmp148;
    double tmp149;
    double tmp150;
    double tmp151;
    double tmp152;
    double tmp153;
    double tmp154;
    double tmp155;
    double tmp156;
    double tmp157;
    double tmp158;
    double tmp159;
    double tmp160;
    double tmp161;
    double tmp162;
    double tmp163;
    double tmp164;
    double tmp165;
    double tmp166;
    double tmp167;
    double tmp168;
    double tmp169;
    double tmp170;
    double tmp171;
    double tmp172;
    double tmp173;
    double tmp174;
    double tmp175;
    double tmp176;
    double tmp177;
    double tmp178;
    double tmp179;
    double tmp180;
    double tmp181;
    double tmp182;
    double tmp183;
    double tmp184;
    double tmp185;
    double tmp186;
    double tmp187;
    double tmp188;
    double tmp189;
    double tmp190;
    double tmp191;
    double tmp192;
    double tmp193;
    double tmp194;
    double tmp195;
    double tmp196;
    double tmp197;
    double tmp198;
    double tmp199;
    double tmp200;
    double tmp201;
    double tmp202;
    double tmp203;
    double tmp204;
    double tmp205;
    double tmp206;
    double tmp207;
    double tmp208;
    double tmp209;
    double tmp210;
    double tmp211;
    double tmp212;
    double tmp213;
    double tmp214;
    double tmp215;
    double tmp216;
    double tmp217;
    double tmp218;
    double tmp219;
    double tmp220;
    double tmp221;
    double tmp222;
    double tmp223;
    double tmp224;
    double tmp225;
    double tmp226;
    double tmp227;
    double tmp228;
    double tmp229;
    double tmp230;
    double tmp231;
    double tmp232;
    double tmp233;
    double tmp234;
    double tmp235;
    double tmp236;
    double tmp237;
    double tmp238;
    double tmp239;
    double tmp240;
    double tmp241;
    double tmp242;
    double tmp243;
    double tmp244;
    double tmp245;
    double tmp246;
    double tmp247;
    double tmp248;
    double tmp249;
    double tmp250;
    double tmp251;
    double tmp252;
    double tmp253;
    double tmp254;
    double tmp255;
    double tmp256;
    double tmp257;
    double tmp258;
    double tmp259;
    double tmp260;
    double tmp261;
    double tmp262;
    double tmp263;
    double tmp264;
    double tmp265;
    double tmp266;
    double tmp267;
    double tmp268;
    double tmp269;
    double tmp270;
    double tmp271;
    double tmp272;
    double tmp273;
    double tmp274;
    double tmp275;
    double tmp276;
    double tmp277;
    double tmp278;
    double tmp279;
    double tmp280;
    double tmp281;
    double tmp282;
    double tmp283;
    double tmp284;
    double tmp285;
    double tmp286;
    double tmp287;
    double tmp288;
    double tmp289;
    double tmp290;
    double tmp291;
    double tmp292;
    double tmp293;
    double tmp294;
    double tmp295;
    double tmp296;
    double tmp297;
    double tmp298;
    double tmp299;
    double tmp300;
    double tmp301;
    double tmp302;
    double tmp303;
    double tmp304;
    double tmp305;
    double tmp306;
    double tmp307;
    double tmp308;
    double tmp309;
    double tmp310;
    double tmp311;
    double tmp312;
    double tmp313;
    double tmp314;
    double tmp315;
    double tmp316;
    double tmp317;
    double tmp318;
    double tmp319;
    double tmp320;
    double tmp321;
    double tmp322;
    double tmp323;
    double tmp324;
    double tmp325;
    double tmp326;
    double tmp327;
    double tmp328;
    double tmp329;
    double tmp330;
    double tmp331;
    
    tmp0 =  sin(z[7]);
    tmp1 =  sin(z[9]);
    tmp2 =  tmp0*tmp1;
    tmp3 =  cos(z[7]);
    tmp4 =  cos(z[8]);
    tmp5 =  tmp3*tmp4;
    tmp6 =  cos(z[9]);
    tmp7 =  tmp3*tmp6;
    tmp8 =  sin(z[10]);
    tmp9 =  tmp0*tmp8;
    tmp10 =  sin(z[8]);
    tmp11 =  tmp0*tmp10;
    tmp12 =  cos(z[10]);
    tmp13 =  tmp12*tmp3;
    tmp14 =  tmp12*tmp4;
    tmp15 =  tmp1*tmp10;
    tmp16 =  tmp10*tmp8;
    tmp17 =  tmp4*tmp6;
    tmp18 =  tmp12*tmp6;
    tmp19 =  tmp1*tmp8;
    tmp20 =  tmp10*z[0];
    tmp21 =  tmp1*z[5];
    tmp22 =  tmp8*z[6];
    tmp23 =  tmp0*z[3];
    tmp24 =  tmp3*z[1];
    tmp25 =  tmp8*z[0];
    tmp26 =  tmp0*z[0];
    tmp27 =  tmp12*z[6];
    tmp28 =  tmp6*z[5];
    tmp29 =  tmp3*z[3];
    tmp30 =  tmp4*z[4];
    tmp31 =  tmp4*z[1];
    tmp32 =  tmp12*z[1];
    tmp33 =  tmp10*z[4];
    tmp34 =  tmp6*z[1];
    tmp35 =  tmp1*z[0];
    tmp36 =  2*z[8];
    tmp37 =  cos(tmp36);
    tmp38 =  tmp37*z[1];
    tmp39 =  2*z[7];
    tmp40 =  sin(tmp39);
    tmp41 =  tmp40*z[0];
    tmp42 =  2*z[10];
    tmp43 =  sin(tmp42);
    tmp44 =  tmp43*z[0];
    tmp45 =  cos(tmp39);
    tmp46 =  tmp45*z[0];
    tmp47 =  sin(tmp36);
    tmp48 =  tmp47*z[0];
    tmp49 =  2*z[9];
    tmp50 =  sin(tmp49);
    tmp51 =  tmp50*z[1];
    tmp52 =  tmp40*z[1];
    tmp53 =  cos(tmp49);
    tmp54 =  tmp53*z[1];
    tmp55 =  tmp43*z[1];
    tmp56 =  tmp45*z[1];
    tmp57 =  tmp37*z[0];
    tmp58 =  cos(tmp42);
    tmp59 =  tmp58*z[1];
    tmp60 =  tmp58*z[0];
    tmp61 =  tmp47*z[1];
    tmp62 =  tmp50*z[0];
    tmp63 =  tmp53*z[0];
    tmp64 =  tmp12*tmp4*z[4];
    tmp65 =  tmp12*tmp6*z[5];
    tmp66 =  tmp0*tmp1*z[3];
    tmp67 =  tmp1*tmp8*z[6];
    tmp68 =  tmp3*tmp4*z[3];
    tmp69 =  tmp12*tmp4*z[6];
    tmp70 =  tmp4*tmp6*z[5];
    tmp71 =  tmp12*tmp3*z[6];
    tmp72 =  tmp1*tmp10*z[5];
    tmp73 =  tmp4*tmp6*z[4];
    tmp74 =  tmp12*tmp3*z[3];
    tmp75 =  tmp0*tmp10*z[3];
    tmp76 =  tmp0*tmp8*z[6];
    tmp77 =  tmp0*tmp8*z[3];
    tmp78 =  tmp3*tmp6*z[5];
    tmp79 =  tmp3*tmp4*z[4];
    tmp80 =  tmp0*tmp1*z[5];
    tmp81 =  tmp0*tmp10*z[4];
    tmp82 =  tmp3*tmp6*z[3];
    tmp83 =  tmp10*tmp8*z[6];
    tmp84 =  tmp10*tmp8*z[4];
    tmp85 =  tmp1*tmp8*z[5];
    tmp86 =  tmp1*tmp10*z[4];
    tmp87 =  tmp12*tmp6*z[6];
    tmp88 =  tmp0*tmp53*z[0];
    tmp89 =  tmp10*tmp40*z[4];
    tmp90 =  tmp0*tmp50*z[1];
    tmp91 =  tmp4*tmp53*z[1];
    tmp92 =  tmp3*tmp37*z[1];
    tmp93 =  tmp53*tmp8*z[0];
    tmp94 =  tmp4*tmp45*z[4];
    tmp95 =  tmp4*tmp58*z[1];
    tmp96 =  pow(z[6], 2);
    tmp97 =  tmp3*tmp8*tmp96;
    tmp98 =  pow(z[3], 2);
    tmp99 =  tmp0*tmp12*tmp98;
    tmp100 =  tmp1*tmp43*z[5];
    tmp101 =  tmp10*tmp53*z[0];
    tmp102 =  pow(z[5], 2);
    tmp103 =  tmp1*tmp102*tmp4;
    tmp104 =  tmp0*tmp58*z[3];
    tmp105 =  tmp0*tmp43*z[1];
    tmp106 =  tmp3*tmp58*z[1];
    tmp107 =  tmp10*tmp50*z[1];
    tmp108 =  tmp10*tmp58*z[4];
    tmp109 =  tmp4*tmp50*z[0];
    tmp110 =  tmp0*tmp37*z[3];
    tmp111 =  tmp3*tmp43*z[0];
    tmp112 =  pow(z[4], 2);
    tmp113 =  tmp0*tmp112*tmp4;
    tmp114 =  tmp40*tmp8*z[6];
    tmp115 =  tmp58*tmp6*z[1];
    tmp116 =  tmp3*tmp53*z[3];
    tmp117 =  tmp12*tmp50*z[0];
    tmp118 =  tmp12*tmp37*z[1];
    tmp119 =  tmp10*tmp45*z[4];
    tmp120 =  tmp4*tmp8*tmp96;
    tmp121 =  tmp0*tmp47*z[1];
    tmp122 =  tmp10*tmp58*z[0];
    tmp123 =  tmp40*tmp6*z[0];
    tmp124 =  tmp10*tmp45*z[0];
    tmp125 =  tmp3*tmp43*z[3];
    tmp126 =  tmp1*tmp47*z[5];
    tmp127 =  tmp47*tmp6*z[0];
    tmp128 =  tmp1*tmp58*z[0];
    tmp129 =  tmp45*tmp8*z[6];
    tmp130 =  tmp12*tmp47*z[6];
    tmp131 =  tmp4*tmp43*z[4];
    tmp132 =  tmp10*tmp53*z[4];
    tmp133 =  tmp3*tmp8*tmp98;
    tmp134 =  tmp47*tmp8*z[6];
    tmp135 =  tmp40*tmp6*z[5];
    tmp136 =  tmp0*tmp12*tmp96;
    tmp137 =  tmp12*tmp47*z[0];
    tmp138 =  tmp1*tmp37*z[5];
    tmp139 =  tmp12*tmp37*z[6];
    tmp140 =  tmp0*tmp102*tmp6;
    tmp141 =  tmp6*tmp8*tmp96;
    tmp142 =  tmp102*tmp6*tmp8;
    tmp143 =  tmp1*tmp47*z[1];
    tmp144 =  tmp3*tmp58*z[3];
    tmp145 =  tmp12*tmp40*z[0];
    tmp146 =  tmp37*tmp6*z[5];
    tmp147 =  tmp10*tmp43*z[1];
    tmp148 =  tmp37*tmp8*z[6];
    tmp149 =  tmp3*tmp37*z[3];
    tmp150 =  tmp45*tmp6*z[5];
    tmp151 =  tmp45*tmp6*z[1];
    tmp152 =  tmp4*tmp53*z[4];
    tmp153 =  tmp12*tmp53*z[6];
    tmp154 =  tmp0*tmp43*z[3];
    tmp155 =  tmp10*tmp12*tmp96;
    tmp156 =  tmp0*tmp50*z[3];
    tmp157 =  tmp37*tmp8*z[0];
    tmp158 =  tmp0*tmp4*tmp98;
    tmp159 =  tmp0*tmp6*tmp98;
    tmp160 =  tmp0*tmp53*z[3];
    tmp161 =  tmp50*tmp8*z[6];
    tmp162 =  tmp3*tmp50*z[3];
    tmp163 =  tmp12*tmp53*z[1];
    tmp164 =  tmp58*tmp6*z[5];
    tmp165 =  tmp1*tmp58*z[5];
    tmp166 =  tmp10*tmp3*tmp98;
    tmp167 =  tmp50*tmp8*z[1];
    tmp168 =  tmp53*tmp8*z[6];
    tmp169 =  tmp47*tmp8*z[1];
    tmp170 =  tmp4*tmp40*z[0];
    tmp171 =  tmp43*tmp6*z[0];
    tmp172 =  tmp10*tmp112*tmp3;
    tmp173 =  tmp10*tmp112*tmp12;
    tmp174 =  tmp4*tmp43*z[0];
    tmp175 =  tmp3*tmp53*z[1];
    tmp176 =  tmp45*tmp8*z[0];
    tmp177 =  tmp10*tmp43*z[4];
    tmp178 =  tmp3*tmp47*z[3];
    tmp179 =  tmp10*tmp102*tmp6;
    tmp180 =  tmp3*tmp47*z[0];
    tmp181 =  tmp1*tmp45*z[5];
    tmp182 =  tmp1*tmp3*tmp98;
    tmp183 =  tmp1*tmp102*tmp12;
    tmp184 =  tmp1*tmp45*z[0];
    tmp185 =  tmp4*tmp50*z[4];
    tmp186 =  tmp1*tmp112*tmp4;
    tmp187 =  tmp4*tmp58*z[4];
    tmp188 =  tmp12*tmp45*z[1];
    tmp189 =  tmp47*tmp6*z[5];
    tmp190 =  tmp112*tmp4*tmp8;
    tmp191 =  tmp10*tmp40*z[1];
    tmp192 =  tmp1*tmp37*z[0];
    tmp193 =  tmp1*tmp40*z[5];
    tmp194 =  tmp40*tmp8*z[1];
    tmp195 =  tmp4*tmp45*z[1];
    tmp196 =  tmp1*tmp40*z[1];
    tmp197 =  tmp0*tmp37*z[0];
    tmp198 =  tmp10*tmp50*z[4];
    tmp199 =  tmp12*tmp50*z[6];
    tmp200 =  tmp0*tmp47*z[3];
    tmp201 =  tmp0*tmp58*z[0];
    tmp202 =  tmp1*tmp12*tmp96;
    tmp203 =  tmp10*tmp112*tmp6;
    tmp204 =  tmp4*tmp40*z[4];
    tmp205 =  tmp1*tmp43*z[1];
    tmp206 =  tmp43*tmp6*z[5];
    tmp207 =  tmp1*tmp102*tmp3;
    tmp208 =  tmp12*tmp40*z[6];
    tmp209 =  tmp12*tmp45*z[6];
    tmp210 =  tmp3*tmp50*z[0];
    tmp211 =  tmp37*tmp6*z[1];
    tmp212 =  tmp40*tmp47*z[3];
    tmp213 =  tmp45*tmp58*z[6];
    tmp214 =  tmp45*tmp53*z[3];
    tmp215 =  tmp40*tmp50*z[3];
    tmp216 =  tmp53*tmp58*z[6];
    tmp217 =  tmp37*tmp45*z[4];
    tmp218 =  tmp43*tmp47*z[6];
    tmp219 =  tmp40*tmp50*z[5];
    tmp220 =  tmp37*tmp53*z[4];
    tmp221 =  tmp40*tmp43*z[3];
    tmp222 =  tmp40*tmp43*z[6];
    tmp223 =  tmp37*tmp58*z[6];
    tmp224 =  tmp43*tmp50*z[6];
    tmp225 =  tmp43*tmp47*z[4];
    tmp226 =  tmp40*tmp47*z[4];
    tmp227 =  tmp37*tmp45*z[3];
    tmp228 =  tmp47*tmp50*z[4];
    tmp229 =  tmp45*tmp58*z[3];
    tmp230 =  tmp45*tmp53*z[5];
    tmp231 =  tmp53*tmp58*z[5];
    tmp232 =  tmp47*tmp50*z[5];
    tmp233 =  tmp37*tmp58*z[4];
    tmp234 =  tmp43*tmp50*z[5];
    tmp235 =  tmp37*tmp53*z[5];
    tmp236 =  tmp1*tmp37*tmp8*z[5];
    tmp237 =  tmp1*tmp3*tmp47*z[5];
    tmp238 =  tmp1*tmp10*tmp58*z[4];
    tmp239 =  tmp10*tmp12*tmp40*z[6];
    tmp240 =  tmp10*tmp43*tmp6*z[4];
    tmp241 =  tmp12*tmp4*tmp53*z[4];
    tmp242 =  tmp12*tmp3*tmp53*z[3];
    tmp243 =  tmp0*tmp43*tmp6*z[3];
    tmp244 =  tmp1*tmp10*tmp45*z[4];
    tmp245 =  tmp10*tmp12*tmp50*z[6];
    tmp246 =  tmp0*tmp12*tmp50*z[6];
    tmp247 =  tmp12*tmp4*tmp45*z[4];
    tmp248 =  tmp3*tmp47*tmp8*z[3];
    tmp249 =  tmp12*tmp3*tmp53*z[6];
    tmp250 =  tmp0*tmp10*tmp58*z[4];
    tmp251 =  tmp0*tmp10*tmp53*z[3];
    tmp252 =  tmp4*tmp45*tmp6*z[4];
    tmp253 =  tmp0*tmp37*tmp8*z[6];
    tmp254 =  tmp1*tmp10*tmp45*z[5];
    tmp255 =  tmp0*tmp4*tmp50*z[4];
    tmp256 =  tmp47*tmp6*tmp8*z[5];
    tmp257 =  tmp1*tmp12*tmp40*z[5];
    tmp258 =  tmp40*tmp6*tmp8*z[6];
    tmp259 =  tmp10*tmp3*tmp43*z[4];
    tmp260 =  tmp3*tmp4*tmp53*z[4];
    tmp261 =  tmp10*tmp40*tmp6*z[4];
    tmp262 =  tmp0*tmp1*tmp58*z[3];
    tmp263 =  tmp12*tmp45*tmp6*z[6];
    tmp264 =  tmp12*tmp37*tmp6*z[6];
    tmp265 =  tmp10*tmp3*tmp50*z[3];
    tmp266 =  tmp3*tmp4*tmp58*z[3];
    tmp267 =  tmp1*tmp4*tmp43*z[5];
    tmp268 =  tmp1*tmp37*tmp8*z[6];
    tmp269 =  tmp4*tmp40*tmp8*z[4];
    tmp270 =  tmp10*tmp12*tmp40*z[4];
    tmp271 =  tmp10*tmp45*tmp8*z[6];
    tmp272 =  tmp12*tmp3*tmp37*z[6];
    tmp273 =  tmp0*tmp12*tmp50*z[3];
    tmp274 =  tmp0*tmp47*tmp6*z[5];
    tmp275 =  tmp10*tmp43*tmp6*z[5];
    tmp276 =  tmp1*tmp10*tmp58*z[5];
    tmp277 =  tmp3*tmp47*tmp8*z[6];
    tmp278 =  tmp3*tmp37*tmp6*z[5];
    tmp279 =  tmp4*tmp50*tmp8*z[6];
    tmp280 =  tmp0*tmp12*tmp47*z[6];
    tmp281 =  tmp4*tmp58*tmp6*z[5];
    tmp282 =  tmp1*tmp4*tmp40*z[4];
    tmp283 =  tmp47*tmp6*tmp8*z[6];
    tmp284 =  tmp3*tmp4*tmp53*z[3];
    tmp285 =  tmp10*tmp3*tmp50*z[4];
    tmp286 =  tmp0*tmp37*tmp8*z[3];
    tmp287 =  tmp10*tmp53*tmp8*z[6];
    tmp288 =  tmp10*tmp53*tmp8*z[4];
    tmp289 =  tmp0*tmp53*tmp8*z[3];
    tmp290 =  tmp1*tmp4*tmp40*z[5];
    tmp291 =  tmp1*tmp45*tmp8*z[6];
    tmp292 =  tmp1*tmp12*tmp40*z[6];
    tmp293 =  tmp0*tmp1*tmp37*z[3];
    tmp294 =  tmp0*tmp1*tmp37*z[5];
    tmp295 =  tmp0*tmp1*tmp58*z[5];
    tmp296 =  tmp3*tmp50*tmp8*z[3];
    tmp297 =  tmp3*tmp58*tmp6*z[5];
    tmp298 =  tmp1*tmp4*tmp43*z[4];
    tmp299 =  tmp12*tmp45*tmp6*z[5];
    tmp300 =  tmp0*tmp4*tmp50*z[3];
    tmp301 =  tmp0*tmp4*tmp43*z[4];
    tmp302 =  tmp10*tmp12*tmp50*z[4];
    tmp303 =  tmp0*tmp10*tmp58*z[3];
    tmp304 =  tmp0*tmp47*tmp6*z[3];
    tmp305 =  tmp1*tmp45*tmp8*z[5];
    tmp306 =  tmp10*tmp3*tmp43*z[3];
    tmp307 =  tmp1*tmp12*tmp47*z[6];
    tmp308 =  tmp1*tmp12*tmp47*z[5];
    tmp309 =  tmp0*tmp10*tmp53*z[4];
    tmp310 =  tmp4*tmp58*tmp6*z[4];
    tmp311 =  tmp0*tmp4*tmp43*z[3];
    tmp312 =  tmp1*tmp3*tmp43*z[5];
    tmp313 =  tmp4*tmp50*tmp8*z[4];
    tmp314 =  tmp0*tmp53*tmp8*z[6];
    tmp315 =  tmp3*tmp37*tmp6*z[3];
    tmp316 =  tmp12*tmp37*tmp6*z[5];
    tmp317 =  tmp4*tmp40*tmp8*z[6];
    tmp318 =  tmp12*tmp4*tmp53*z[6];
    tmp319 =  tmp12*tmp4*tmp45*z[6];
    tmp320 =  tmp1*tmp3*tmp43*z[3];
    tmp321 =  tmp3*tmp4*tmp58*z[4];
    tmp322 =  tmp3*tmp50*tmp8*z[6];
    tmp323 =  tmp10*tmp40*tmp6*z[5];
    tmp324 =  tmp0*tmp12*tmp47*z[3];
    tmp325 =  tmp0*tmp43*tmp6*z[5];
    tmp326 =  tmp1*tmp3*tmp47*z[3];
    tmp327 =  tmp3*tmp58*tmp6*z[3];
    tmp328 =  tmp10*tmp45*tmp8*z[4];
    tmp329 =  tmp40*tmp6*tmp8*z[5];
    tmp330 =  tmp12*tmp3*tmp37*z[3];
    tmp331 =  tmp4*tmp45*tmp6*z[5];
    out[0] =  weights[10];
    out[1] =  weights[3];
    out[2] =  weights[5];
    out[3] =  tmp11*weights[32] + tmp13*weights[34] + tmp2*weights[16] + tmp5*weights[19] + tmp7*weights[20] + tmp9*weights[25] + weights[2];
    out[4] =  tmp11*weights[17] + tmp14*weights[22] + tmp15*weights[23] + tmp16*weights[30] + tmp17*weights[31] + tmp5*weights[12] + weights[9];
    out[5] =  tmp15*weights[24] + tmp17*weights[29] + tmp18*weights[11] + tmp19*weights[18] + tmp2*weights[26] + tmp7*weights[28] + weights[8];
    out[6] =  tmp13*weights[15] + tmp14*weights[14] + tmp16*weights[27] + tmp18*weights[21] + tmp19*weights[13] + tmp9*weights[33] + weights[4];
    out[7] =  weights[7];
    out[8] =  weights[0];
    out[9] =  weights[1];
    out[10] =  weights[6];
    out[11] =  tmp100*weights[137] + tmp101*weights[141] + tmp103*weights[142] + tmp104*weights[143] + tmp105*weights[149] + tmp106*weights[154] + tmp107*weights[157] + tmp108*weights[158] + tmp109*weights[160] + tmp110*weights[161] + tmp111*weights[171] + tmp113*weights[174] + tmp114*weights[175] + tmp115*weights[179] + tmp116*weights[180] + tmp117*weights[181] + tmp118*weights[183] + tmp119*weights[185] + tmp120*weights[186] + tmp121*weights[187] + tmp122*weights[189] + tmp123*weights[191] + tmp124*weights[194] + tmp125*weights[199] + tmp126*weights[201] + tmp127*weights[204] + tmp128*weights[209] + tmp129*weights[210] + tmp130*weights[211] + tmp131*weights[213] + tmp132*weights[216] + tmp133*weights[217] + tmp134*weights[223] + tmp135*weights[228] + tmp136*weights[230] + tmp137*weights[231] + tmp138*weights[232] + tmp139*weights[234] + tmp140*weights[236] + tmp141*weights[239] + tmp142*weights[244] + tmp143*weights[245] + tmp144*weights[247] + tmp145*weights[248] + tmp146*weights[250] + tmp147*weights[251] + tmp148*weights[252] + tmp149*weights[253] + tmp150*weights[255] + tmp151*weights[256] + tmp152*weights[264] + tmp153*weights[265] + tmp154*weights[267] + tmp155*weights[269] + tmp156*weights[271] + tmp157*weights[272] + tmp158*weights[278] + tmp159*weights[279] + tmp160*weights[281] + tmp161*weights[286] + tmp162*weights[288] + tmp163*weights[291] + tmp164*weights[292] + tmp165*weights[293] + tmp166*weights[300] + tmp167*weights[302] + tmp168*weights[303] + tmp169*weights[305] + tmp170*weights[306] + tmp171*weights[307] + tmp172*weights[310] + tmp173*weights[312] + tmp174*weights[313] + tmp175*weights[316] + tmp176*weights[317] + tmp177*weights[319] + tmp178*weights[320] + tmp179*weights[332] + tmp180*weights[334] + tmp181*weights[335] + tmp182*weights[337] + tmp183*weights[339] + tmp184*weights[41] + tmp185*weights[43] + tmp186*weights[45] + tmp187*weights[47] + tmp188*weights[48] + tmp189*weights[49] + tmp190*weights[50] + tmp191*weights[51] + tmp192*weights[55] + tmp193*weights[56] + tmp194*weights[59] + tmp195*weights[63] + tmp196*weights[66] + tmp197*weights[67] + tmp198*weights[68] + tmp199*weights[70] + tmp20*weights[114] + tmp200*weights[71] + tmp201*weights[76] + tmp202*weights[79] + tmp203*weights[82] + tmp204*weights[84] + tmp205*weights[86] + tmp206*weights[88] + tmp207*weights[92] + tmp208*weights[94] + tmp209*weights[95] + tmp21*weights[117] + tmp210*weights[96] + tmp211*weights[98] + tmp212*weights[102] + tmp213*weights[105] + tmp214*weights[123] + tmp215*weights[126] + tmp216*weights[163] + tmp217*weights[164] + tmp218*weights[167] + tmp219*weights[168] + tmp22*weights[147] + tmp220*weights[182] + tmp221*weights[184] + tmp222*weights[195] + tmp223*weights[197] + tmp224*weights[200] + tmp225*weights[208] + tmp226*weights[219] + tmp227*weights[235] + tmp228*weights[266] + tmp229*weights[290] + tmp23*weights[226] + tmp230*weights[327] + tmp231*weights[336] + tmp232*weights[37] + tmp233*weights[52] + tmp234*weights[54] + tmp235*weights[85] + tmp236*weights[101] + tmp237*weights[103] + tmp238*weights[111] + tmp239*weights[113] + tmp24*weights[227] + tmp240*weights[115] + tmp241*weights[116] + tmp242*weights[118] + tmp243*weights[119] + tmp244*weights[121] + tmp245*weights[122] + tmp246*weights[125] + tmp247*weights[127] + tmp248*weights[129] + tmp249*weights[135] + tmp25*weights[257] + tmp250*weights[136] + tmp251*weights[139] + tmp252*weights[145] + tmp253*weights[151] + tmp254*weights[152] + tmp255*weights[159] + tmp256*weights[162] + tmp257*weights[165] + tmp258*weights[166] + tmp259*weights[169] + tmp26*weights[258] + tmp260*weights[170] + tmp261*weights[172] + tmp262*weights[173] + tmp263*weights[176] + tmp264*weights[177] + tmp265*weights[178] + tmp266*weights[188] + tmp267*weights[190] + tmp268*weights[192] + tmp269*weights[196] + tmp27*weights[268] + tmp270*weights[198] + tmp271*weights[202] + tmp272*weights[203] + tmp273*weights[206] + tmp274*weights[207] + tmp275*weights[212] + tmp276*weights[215] + tmp277*weights[218] + tmp278*weights[222] + tmp279*weights[224] + tmp28*weights[276] + tmp280*weights[233] + tmp281*weights[238] + tmp282*weights[241] + tmp283*weights[246] + tmp284*weights[249] + tmp285*weights[259] + tmp286*weights[260] + tmp287*weights[262] + tmp288*weights[263] + tmp289*weights[274] + tmp29*weights[297] + tmp290*weights[277] + tmp291*weights[280] + tmp292*weights[282] + tmp293*weights[284] + tmp294*weights[285] + tmp295*weights[295] + tmp296*weights[296] + tmp297*weights[298] + tmp298*weights[301] + tmp299*weights[304] + tmp30*weights[299] + tmp300*weights[309] + tmp301*weights[311] + tmp302*weights[318] + tmp303*weights[321] + tmp304*weights[322] + tmp305*weights[324] + tmp306*weights[325] + tmp307*weights[326] + tmp308*weights[328] + tmp309*weights[329] + tmp31*weights[57] + tmp310*weights[330] + tmp311*weights[331] + tmp312*weights[35] + tmp313*weights[36] + tmp314*weights[38] + tmp315*weights[39] + tmp316*weights[42] + tmp317*weights[44] + tmp318*weights[46] + tmp319*weights[58] + tmp32*weights[61] + tmp320*weights[60] + tmp321*weights[64] + tmp322*weights[65] + tmp323*weights[69] + tmp324*weights[72] + tmp325*weights[73] + tmp326*weights[75] + tmp327*weights[77] + tmp328*weights[80] + tmp329*weights[87] + tmp33*weights[62] + tmp330*weights[89] + tmp331*weights[99] + tmp34*weights[83] + tmp35*weights[93] + tmp38*weights[106] + tmp41*weights[110] + tmp44*weights[138] + tmp46*weights[150] + tmp48*weights[153] + tmp51*weights[237] + tmp52*weights[254] + tmp54*weights[273] + tmp55*weights[275] + tmp56*weights[287] + tmp57*weights[294] + tmp59*weights[314] + tmp60*weights[323] + tmp61*weights[53] + tmp62*weights[78] + tmp63*weights[97] + tmp64*weights[108] + tmp65*weights[124] + tmp66*weights[131] + tmp67*weights[140] + tmp68*weights[146] + tmp69*weights[148] + tmp70*weights[156] + tmp71*weights[205] + tmp72*weights[214] + tmp73*weights[220] + tmp74*weights[221] + tmp75*weights[229] + tmp76*weights[242] + tmp77*weights[243] + tmp78*weights[261] + tmp79*weights[270] + tmp80*weights[283] + tmp81*weights[289] + tmp82*weights[308] + tmp83*weights[315] + tmp84*weights[333] + tmp85*weights[74] + tmp86*weights[90] + tmp87*weights[91] + tmp88*weights[100] + tmp89*weights[104] + tmp90*weights[107] + tmp91*weights[109] + tmp92*weights[120] + tmp93*weights[128] + tmp94*weights[130] + tmp95*weights[132] + tmp97*weights[133] + tmp99*weights[134] + weights[112]*z[1] + weights[144]*z[5] + weights[155]*z[4] + weights[193]*z[0] + weights[225]*z[3] + weights[240]*z[12] + weights[338]*z[6] + weights[40]*z[13] + weights[81]*z[11] + z[14];
    out[12] =  weights[350];
    out[13] =  weights[343];
    out[14] =  weights[345];
    out[15] =  tmp11*weights[372] + tmp13*weights[374] + tmp2*weights[356] + tmp5*weights[359] + tmp7*weights[360] + tmp9*weights[365] + weights[342];
    out[16] =  tmp11*weights[357] + tmp14*weights[362] + tmp15*weights[363] + tmp16*weights[370] + tmp17*weights[371] + tmp5*weights[352] + weights[349];
    out[17] =  tmp15*weights[364] + tmp17*weights[369] + tmp18*weights[351] + tmp19*weights[358] + tmp2*weights[366] + tmp7*weights[368] + weights[348];
    out[18] =  tmp13*weights[355] + tmp14*weights[354] + tmp16*weights[367] + tmp18*weights[361] + tmp19*weights[353] + tmp9*weights[373] + weights[344];
    out[19] =  weights[347];
    out[20] =  weights[340];
    out[21] =  weights[341];
    out[22] =  weights[346];
    out[23] =  tmp100*weights[477] + tmp101*weights[481] + tmp103*weights[482] + tmp104*weights[483] + tmp105*weights[489] + tmp106*weights[494] + tmp107*weights[497] + tmp108*weights[498] + tmp109*weights[500] + tmp110*weights[501] + tmp111*weights[511] + tmp113*weights[514] + tmp114*weights[515] + tmp115*weights[519] + tmp116*weights[520] + tmp117*weights[521] + tmp118*weights[523] + tmp119*weights[525] + tmp120*weights[526] + tmp121*weights[527] + tmp122*weights[529] + tmp123*weights[531] + tmp124*weights[534] + tmp125*weights[539] + tmp126*weights[541] + tmp127*weights[544] + tmp128*weights[549] + tmp129*weights[550] + tmp130*weights[551] + tmp131*weights[553] + tmp132*weights[556] + tmp133*weights[557] + tmp134*weights[563] + tmp135*weights[568] + tmp136*weights[570] + tmp137*weights[571] + tmp138*weights[572] + tmp139*weights[574] + tmp140*weights[576] + tmp141*weights[579] + tmp142*weights[584] + tmp143*weights[585] + tmp144*weights[587] + tmp145*weights[588] + tmp146*weights[590] + tmp147*weights[591] + tmp148*weights[592] + tmp149*weights[593] + tmp150*weights[595] + tmp151*weights[596] + tmp152*weights[604] + tmp153*weights[605] + tmp154*weights[607] + tmp155*weights[609] + tmp156*weights[611] + tmp157*weights[612] + tmp158*weights[618] + tmp159*weights[619] + tmp160*weights[621] + tmp161*weights[626] + tmp162*weights[628] + tmp163*weights[631] + tmp164*weights[632] + tmp165*weights[633] + tmp166*weights[640] + tmp167*weights[642] + tmp168*weights[643] + tmp169*weights[645] + tmp170*weights[646] + tmp171*weights[647] + tmp172*weights[650] + tmp173*weights[652] + tmp174*weights[653] + tmp175*weights[656] + tmp176*weights[657] + tmp177*weights[659] + tmp178*weights[660] + tmp179*weights[672] + tmp180*weights[674] + tmp181*weights[675] + tmp182*weights[677] + tmp183*weights[679] + tmp184*weights[381] + tmp185*weights[383] + tmp186*weights[385] + tmp187*weights[387] + tmp188*weights[388] + tmp189*weights[389] + tmp190*weights[390] + tmp191*weights[391] + tmp192*weights[395] + tmp193*weights[396] + tmp194*weights[399] + tmp195*weights[403] + tmp196*weights[406] + tmp197*weights[407] + tmp198*weights[408] + tmp199*weights[410] + tmp20*weights[454] + tmp200*weights[411] + tmp201*weights[416] + tmp202*weights[419] + tmp203*weights[422] + tmp204*weights[424] + tmp205*weights[426] + tmp206*weights[428] + tmp207*weights[432] + tmp208*weights[434] + tmp209*weights[435] + tmp21*weights[457] + tmp210*weights[436] + tmp211*weights[438] + tmp212*weights[442] + tmp213*weights[445] + tmp214*weights[463] + tmp215*weights[466] + tmp216*weights[503] + tmp217*weights[504] + tmp218*weights[507] + tmp219*weights[508] + tmp22*weights[487] + tmp220*weights[522] + tmp221*weights[524] + tmp222*weights[535] + tmp223*weights[537] + tmp224*weights[540] + tmp225*weights[548] + tmp226*weights[559] + tmp227*weights[575] + tmp228*weights[606] + tmp229*weights[630] + tmp23*weights[566] + tmp230*weights[667] + tmp231*weights[676] + tmp232*weights[377] + tmp233*weights[392] + tmp234*weights[394] + tmp235*weights[425] + tmp236*weights[441] + tmp237*weights[443] + tmp238*weights[451] + tmp239*weights[453] + tmp24*weights[567] + tmp240*weights[455] + tmp241*weights[456] + tmp242*weights[458] + tmp243*weights[459] + tmp244*weights[461] + tmp245*weights[462] + tmp246*weights[465] + tmp247*weights[467] + tmp248*weights[469] + tmp249*weights[475] + tmp25*weights[597] + tmp250*weights[476] + tmp251*weights[479] + tmp252*weights[485] + tmp253*weights[491] + tmp254*weights[492] + tmp255*weights[499] + tmp256*weights[502] + tmp257*weights[505] + tmp258*weights[506] + tmp259*weights[509] + tmp26*weights[598] + tmp260*weights[510] + tmp261*weights[512] + tmp262*weights[513] + tmp263*weights[516] + tmp264*weights[517] + tmp265*weights[518] + tmp266*weights[528] + tmp267*weights[530] + tmp268*weights[532] + tmp269*weights[536] + tmp27*weights[608] + tmp270*weights[538] + tmp271*weights[542] + tmp272*weights[543] + tmp273*weights[546] + tmp274*weights[547] + tmp275*weights[552] + tmp276*weights[555] + tmp277*weights[558] + tmp278*weights[562] + tmp279*weights[564] + tmp28*weights[616] + tmp280*weights[573] + tmp281*weights[578] + tmp282*weights[581] + tmp283*weights[586] + tmp284*weights[589] + tmp285*weights[599] + tmp286*weights[600] + tmp287*weights[602] + tmp288*weights[603] + tmp289*weights[614] + tmp29*weights[637] + tmp290*weights[617] + tmp291*weights[620] + tmp292*weights[622] + tmp293*weights[624] + tmp294*weights[625] + tmp295*weights[635] + tmp296*weights[636] + tmp297*weights[638] + tmp298*weights[641] + tmp299*weights[644] + tmp30*weights[639] + tmp300*weights[649] + tmp301*weights[651] + tmp302*weights[658] + tmp303*weights[661] + tmp304*weights[662] + tmp305*weights[664] + tmp306*weights[665] + tmp307*weights[666] + tmp308*weights[668] + tmp309*weights[669] + tmp31*weights[397] + tmp310*weights[670] + tmp311*weights[671] + tmp312*weights[375] + tmp313*weights[376] + tmp314*weights[378] + tmp315*weights[379] + tmp316*weights[382] + tmp317*weights[384] + tmp318*weights[386] + tmp319*weights[398] + tmp32*weights[401] + tmp320*weights[400] + tmp321*weights[404] + tmp322*weights[405] + tmp323*weights[409] + tmp324*weights[412] + tmp325*weights[413] + tmp326*weights[415] + tmp327*weights[417] + tmp328*weights[420] + tmp329*weights[427] + tmp33*weights[402] + tmp330*weights[429] + tmp331*weights[439] + tmp34*weights[423] + tmp35*weights[433] + tmp38*weights[446] + tmp41*weights[450] + tmp44*weights[478] + tmp46*weights[490] + tmp48*weights[493] + tmp51*weights[577] + tmp52*weights[594] + tmp54*weights[613] + tmp55*weights[615] + tmp56*weights[627] + tmp57*weights[634] + tmp59*weights[654] + tmp60*weights[663] + tmp61*weights[393] + tmp62*weights[418] + tmp63*weights[437] + tmp64*weights[448] + tmp65*weights[464] + tmp66*weights[471] + tmp67*weights[480] + tmp68*weights[486] + tmp69*weights[488] + tmp70*weights[496] + tmp71*weights[545] + tmp72*weights[554] + tmp73*weights[560] + tmp74*weights[561] + tmp75*weights[569] + tmp76*weights[582] + tmp77*weights[583] + tmp78*weights[601] + tmp79*weights[610] + tmp80*weights[623] + tmp81*weights[629] + tmp82*weights[648] + tmp83*weights[655] + tmp84*weights[673] + tmp85*weights[414] + tmp86*weights[430] + tmp87*weights[431] + tmp88*weights[440] + tmp89*weights[444] + tmp90*weights[447] + tmp91*weights[449] + tmp92*weights[460] + tmp93*weights[468] + tmp94*weights[470] + tmp95*weights[472] + tmp97*weights[473] + tmp99*weights[474] + weights[380]*z[13] + weights[421]*z[11] + weights[452]*z[1] + weights[484]*z[5] + weights[495]*z[4] + weights[533]*z[0] + weights[565]*z[3] + weights[580]*z[12] + weights[678]*z[6] + z[15];
    out[24] =  weights[690];
    out[25] =  weights[683];
    out[26] =  weights[685];
    out[27] =  tmp11*weights[712] + tmp13*weights[714] + tmp2*weights[696] + tmp5*weights[699] + tmp7*weights[700] + tmp9*weights[705] + weights[682];
    out[28] =  tmp11*weights[697] + tmp14*weights[702] + tmp15*weights[703] + tmp16*weights[710] + tmp17*weights[711] + tmp5*weights[692] + weights[689];
    out[29] =  tmp15*weights[704] + tmp17*weights[709] + tmp18*weights[691] + tmp19*weights[698] + tmp2*weights[706] + tmp7*weights[708] + weights[688];
    out[30] =  tmp13*weights[695] + tmp14*weights[694] + tmp16*weights[707] + tmp18*weights[701] + tmp19*weights[693] + tmp9*weights[713] + weights[684];
    out[31] =  weights[687];
    out[32] =  weights[680];
    out[33] =  weights[681];
    out[34] =  weights[686];
    out[35] =  tmp100*weights[817] + tmp101*weights[821] + tmp103*weights[822] + tmp104*weights[823] + tmp105*weights[829] + tmp106*weights[834] + tmp107*weights[837] + tmp108*weights[838] + tmp109*weights[840] + tmp110*weights[841] + tmp111*weights[851] + tmp113*weights[854] + tmp114*weights[855] + tmp115*weights[859] + tmp116*weights[860] + tmp117*weights[861] + tmp118*weights[863] + tmp119*weights[865] + tmp120*weights[866] + tmp121*weights[867] + tmp122*weights[869] + tmp123*weights[871] + tmp124*weights[874] + tmp125*weights[879] + tmp126*weights[881] + tmp127*weights[884] + tmp128*weights[889] + tmp129*weights[890] + tmp130*weights[891] + tmp131*weights[893] + tmp132*weights[896] + tmp133*weights[897] + tmp134*weights[903] + tmp135*weights[908] + tmp136*weights[910] + tmp137*weights[911] + tmp138*weights[912] + tmp139*weights[914] + tmp140*weights[916] + tmp141*weights[919] + tmp142*weights[924] + tmp143*weights[925] + tmp144*weights[927] + tmp145*weights[928] + tmp146*weights[930] + tmp147*weights[931] + tmp148*weights[932] + tmp149*weights[933] + tmp150*weights[935] + tmp151*weights[936] + tmp152*weights[944] + tmp153*weights[945] + tmp154*weights[947] + tmp155*weights[949] + tmp156*weights[951] + tmp157*weights[952] + tmp158*weights[958] + tmp159*weights[959] + tmp160*weights[961] + tmp161*weights[966] + tmp162*weights[968] + tmp163*weights[971] + tmp164*weights[972] + tmp165*weights[973] + tmp166*weights[980] + tmp167*weights[982] + tmp168*weights[983] + tmp169*weights[985] + tmp170*weights[986] + tmp171*weights[987] + tmp172*weights[990] + tmp173*weights[992] + tmp174*weights[993] + tmp175*weights[996] + tmp176*weights[997] + tmp177*weights[999] + tmp178*weights[1000] + tmp179*weights[1012] + tmp180*weights[1014] + tmp181*weights[1015] + tmp182*weights[1017] + tmp183*weights[1019] + tmp184*weights[721] + tmp185*weights[723] + tmp186*weights[725] + tmp187*weights[727] + tmp188*weights[728] + tmp189*weights[729] + tmp190*weights[730] + tmp191*weights[731] + tmp192*weights[735] + tmp193*weights[736] + tmp194*weights[739] + tmp195*weights[743] + tmp196*weights[746] + tmp197*weights[747] + tmp198*weights[748] + tmp199*weights[750] + tmp20*weights[794] + tmp200*weights[751] + tmp201*weights[756] + tmp202*weights[759] + tmp203*weights[762] + tmp204*weights[764] + tmp205*weights[766] + tmp206*weights[768] + tmp207*weights[772] + tmp208*weights[774] + tmp209*weights[775] + tmp21*weights[797] + tmp210*weights[776] + tmp211*weights[778] + tmp212*weights[782] + tmp213*weights[785] + tmp214*weights[803] + tmp215*weights[806] + tmp216*weights[843] + tmp217*weights[844] + tmp218*weights[847] + tmp219*weights[848] + tmp22*weights[827] + tmp220*weights[862] + tmp221*weights[864] + tmp222*weights[875] + tmp223*weights[877] + tmp224*weights[880] + tmp225*weights[888] + tmp226*weights[899] + tmp227*weights[915] + tmp228*weights[946] + tmp229*weights[970] + tmp23*weights[906] + tmp230*weights[1007] + tmp231*weights[1016] + tmp232*weights[717] + tmp233*weights[732] + tmp234*weights[734] + tmp235*weights[765] + tmp236*weights[781] + tmp237*weights[783] + tmp238*weights[791] + tmp239*weights[793] + tmp24*weights[907] + tmp240*weights[795] + tmp241*weights[796] + tmp242*weights[798] + tmp243*weights[799] + tmp244*weights[801] + tmp245*weights[802] + tmp246*weights[805] + tmp247*weights[807] + tmp248*weights[809] + tmp249*weights[815] + tmp25*weights[937] + tmp250*weights[816] + tmp251*weights[819] + tmp252*weights[825] + tmp253*weights[831] + tmp254*weights[832] + tmp255*weights[839] + tmp256*weights[842] + tmp257*weights[845] + tmp258*weights[846] + tmp259*weights[849] + tmp26*weights[938] + tmp260*weights[850] + tmp261*weights[852] + tmp262*weights[853] + tmp263*weights[856] + tmp264*weights[857] + tmp265*weights[858] + tmp266*weights[868] + tmp267*weights[870] + tmp268*weights[872] + tmp269*weights[876] + tmp27*weights[948] + tmp270*weights[878] + tmp271*weights[882] + tmp272*weights[883] + tmp273*weights[886] + tmp274*weights[887] + tmp275*weights[892] + tmp276*weights[895] + tmp277*weights[898] + tmp278*weights[902] + tmp279*weights[904] + tmp28*weights[956] + tmp280*weights[913] + tmp281*weights[918] + tmp282*weights[921] + tmp283*weights[926] + tmp284*weights[929] + tmp285*weights[939] + tmp286*weights[940] + tmp287*weights[942] + tmp288*weights[943] + tmp289*weights[954] + tmp29*weights[977] + tmp290*weights[957] + tmp291*weights[960] + tmp292*weights[962] + tmp293*weights[964] + tmp294*weights[965] + tmp295*weights[975] + tmp296*weights[976] + tmp297*weights[978] + tmp298*weights[981] + tmp299*weights[984] + tmp30*weights[979] + tmp300*weights[989] + tmp301*weights[991] + tmp302*weights[998] + tmp303*weights[1001] + tmp304*weights[1002] + tmp305*weights[1004] + tmp306*weights[1005] + tmp307*weights[1006] + tmp308*weights[1008] + tmp309*weights[1009] + tmp31*weights[737] + tmp310*weights[1010] + tmp311*weights[1011] + tmp312*weights[715] + tmp313*weights[716] + tmp314*weights[718] + tmp315*weights[719] + tmp316*weights[722] + tmp317*weights[724] + tmp318*weights[726] + tmp319*weights[738] + tmp32*weights[741] + tmp320*weights[740] + tmp321*weights[744] + tmp322*weights[745] + tmp323*weights[749] + tmp324*weights[752] + tmp325*weights[753] + tmp326*weights[755] + tmp327*weights[757] + tmp328*weights[760] + tmp329*weights[767] + tmp33*weights[742] + tmp330*weights[769] + tmp331*weights[779] + tmp34*weights[763] + tmp35*weights[773] + tmp38*weights[786] + tmp41*weights[790] + tmp44*weights[818] + tmp46*weights[830] + tmp48*weights[833] + tmp51*weights[917] + tmp52*weights[934] + tmp54*weights[953] + tmp55*weights[955] + tmp56*weights[967] + tmp57*weights[974] + tmp59*weights[994] + tmp60*weights[1003] + tmp61*weights[733] + tmp62*weights[758] + tmp63*weights[777] + tmp64*weights[788] + tmp65*weights[804] + tmp66*weights[811] + tmp67*weights[820] + tmp68*weights[826] + tmp69*weights[828] + tmp70*weights[836] + tmp71*weights[885] + tmp72*weights[894] + tmp73*weights[900] + tmp74*weights[901] + tmp75*weights[909] + tmp76*weights[922] + tmp77*weights[923] + tmp78*weights[941] + tmp79*weights[950] + tmp80*weights[963] + tmp81*weights[969] + tmp82*weights[988] + tmp83*weights[995] + tmp84*weights[1013] + tmp85*weights[754] + tmp86*weights[770] + tmp87*weights[771] + tmp88*weights[780] + tmp89*weights[784] + tmp90*weights[787] + tmp91*weights[789] + tmp92*weights[800] + tmp93*weights[808] + tmp94*weights[810] + tmp95*weights[812] + tmp97*weights[813] + tmp99*weights[814] + weights[1018]*z[6] + weights[720]*z[13] + weights[761]*z[11] + weights[792]*z[1] + weights[824]*z[5] + weights[835]*z[4] + weights[873]*z[0] + weights[905]*z[3] + weights[920]*z[12] + z[16];
    out[36] =  weights[1030];
    out[37] =  weights[1023];
    out[38] =  weights[1025];
    out[39] =  tmp11*weights[1052] + tmp13*weights[1054] + tmp2*weights[1036] + tmp5*weights[1039] + tmp7*weights[1040] + tmp9*weights[1045] + weights[1022];
    out[40] =  tmp11*weights[1037] + tmp14*weights[1042] + tmp15*weights[1043] + tmp16*weights[1050] + tmp17*weights[1051] + tmp5*weights[1032] + weights[1029];
    out[41] =  tmp15*weights[1044] + tmp17*weights[1049] + tmp18*weights[1031] + tmp19*weights[1038] + tmp2*weights[1046] + tmp7*weights[1048] + weights[1028];
    out[42] =  tmp13*weights[1035] + tmp14*weights[1034] + tmp16*weights[1047] + tmp18*weights[1041] + tmp19*weights[1033] + tmp9*weights[1053] + weights[1024];
    out[43] =  weights[1027];
    out[44] =  weights[1020];
    out[45] =  weights[1021];
    out[46] =  weights[1026];
    out[47] =  tmp100*weights[1157] + tmp101*weights[1161] + tmp103*weights[1162] + tmp104*weights[1163] + tmp105*weights[1169] + tmp106*weights[1174] + tmp107*weights[1177] + tmp108*weights[1178] + tmp109*weights[1180] + tmp110*weights[1181] + tmp111*weights[1191] + tmp113*weights[1194] + tmp114*weights[1195] + tmp115*weights[1199] + tmp116*weights[1200] + tmp117*weights[1201] + tmp118*weights[1203] + tmp119*weights[1205] + tmp120*weights[1206] + tmp121*weights[1207] + tmp122*weights[1209] + tmp123*weights[1211] + tmp124*weights[1214] + tmp125*weights[1219] + tmp126*weights[1221] + tmp127*weights[1224] + tmp128*weights[1229] + tmp129*weights[1230] + tmp130*weights[1231] + tmp131*weights[1233] + tmp132*weights[1236] + tmp133*weights[1237] + tmp134*weights[1243] + tmp135*weights[1248] + tmp136*weights[1250] + tmp137*weights[1251] + tmp138*weights[1252] + tmp139*weights[1254] + tmp140*weights[1256] + tmp141*weights[1259] + tmp142*weights[1264] + tmp143*weights[1265] + tmp144*weights[1267] + tmp145*weights[1268] + tmp146*weights[1270] + tmp147*weights[1271] + tmp148*weights[1272] + tmp149*weights[1273] + tmp150*weights[1275] + tmp151*weights[1276] + tmp152*weights[1284] + tmp153*weights[1285] + tmp154*weights[1287] + tmp155*weights[1289] + tmp156*weights[1291] + tmp157*weights[1292] + tmp158*weights[1298] + tmp159*weights[1299] + tmp160*weights[1301] + tmp161*weights[1306] + tmp162*weights[1308] + tmp163*weights[1311] + tmp164*weights[1312] + tmp165*weights[1313] + tmp166*weights[1320] + tmp167*weights[1322] + tmp168*weights[1323] + tmp169*weights[1325] + tmp170*weights[1326] + tmp171*weights[1327] + tmp172*weights[1330] + tmp173*weights[1332] + tmp174*weights[1333] + tmp175*weights[1336] + tmp176*weights[1337] + tmp177*weights[1339] + tmp178*weights[1340] + tmp179*weights[1352] + tmp180*weights[1354] + tmp181*weights[1355] + tmp182*weights[1357] + tmp183*weights[1359] + tmp184*weights[1061] + tmp185*weights[1063] + tmp186*weights[1065] + tmp187*weights[1067] + tmp188*weights[1068] + tmp189*weights[1069] + tmp190*weights[1070] + tmp191*weights[1071] + tmp192*weights[1075] + tmp193*weights[1076] + tmp194*weights[1079] + tmp195*weights[1083] + tmp196*weights[1086] + tmp197*weights[1087] + tmp198*weights[1088] + tmp199*weights[1090] + tmp20*weights[1134] + tmp200*weights[1091] + tmp201*weights[1096] + tmp202*weights[1099] + tmp203*weights[1102] + tmp204*weights[1104] + tmp205*weights[1106] + tmp206*weights[1108] + tmp207*weights[1112] + tmp208*weights[1114] + tmp209*weights[1115] + tmp21*weights[1137] + tmp210*weights[1116] + tmp211*weights[1118] + tmp212*weights[1122] + tmp213*weights[1125] + tmp214*weights[1143] + tmp215*weights[1146] + tmp216*weights[1183] + tmp217*weights[1184] + tmp218*weights[1187] + tmp219*weights[1188] + tmp22*weights[1167] + tmp220*weights[1202] + tmp221*weights[1204] + tmp222*weights[1215] + tmp223*weights[1217] + tmp224*weights[1220] + tmp225*weights[1228] + tmp226*weights[1239] + tmp227*weights[1255] + tmp228*weights[1286] + tmp229*weights[1310] + tmp23*weights[1246] + tmp230*weights[1347] + tmp231*weights[1356] + tmp232*weights[1057] + tmp233*weights[1072] + tmp234*weights[1074] + tmp235*weights[1105] + tmp236*weights[1121] + tmp237*weights[1123] + tmp238*weights[1131] + tmp239*weights[1133] + tmp24*weights[1247] + tmp240*weights[1135] + tmp241*weights[1136] + tmp242*weights[1138] + tmp243*weights[1139] + tmp244*weights[1141] + tmp245*weights[1142] + tmp246*weights[1145] + tmp247*weights[1147] + tmp248*weights[1149] + tmp249*weights[1155] + tmp25*weights[1277] + tmp250*weights[1156] + tmp251*weights[1159] + tmp252*weights[1165] + tmp253*weights[1171] + tmp254*weights[1172] + tmp255*weights[1179] + tmp256*weights[1182] + tmp257*weights[1185] + tmp258*weights[1186] + tmp259*weights[1189] + tmp26*weights[1278] + tmp260*weights[1190] + tmp261*weights[1192] + tmp262*weights[1193] + tmp263*weights[1196] + tmp264*weights[1197] + tmp265*weights[1198] + tmp266*weights[1208] + tmp267*weights[1210] + tmp268*weights[1212] + tmp269*weights[1216] + tmp27*weights[1288] + tmp270*weights[1218] + tmp271*weights[1222] + tmp272*weights[1223] + tmp273*weights[1226] + tmp274*weights[1227] + tmp275*weights[1232] + tmp276*weights[1235] + tmp277*weights[1238] + tmp278*weights[1242] + tmp279*weights[1244] + tmp28*weights[1296] + tmp280*weights[1253] + tmp281*weights[1258] + tmp282*weights[1261] + tmp283*weights[1266] + tmp284*weights[1269] + tmp285*weights[1279] + tmp286*weights[1280] + tmp287*weights[1282] + tmp288*weights[1283] + tmp289*weights[1294] + tmp29*weights[1317] + tmp290*weights[1297] + tmp291*weights[1300] + tmp292*weights[1302] + tmp293*weights[1304] + tmp294*weights[1305] + tmp295*weights[1315] + tmp296*weights[1316] + tmp297*weights[1318] + tmp298*weights[1321] + tmp299*weights[1324] + tmp30*weights[1319] + tmp300*weights[1329] + tmp301*weights[1331] + tmp302*weights[1338] + tmp303*weights[1341] + tmp304*weights[1342] + tmp305*weights[1344] + tmp306*weights[1345] + tmp307*weights[1346] + tmp308*weights[1348] + tmp309*weights[1349] + tmp31*weights[1077] + tmp310*weights[1350] + tmp311*weights[1351] + tmp312*weights[1055] + tmp313*weights[1056] + tmp314*weights[1058] + tmp315*weights[1059] + tmp316*weights[1062] + tmp317*weights[1064] + tmp318*weights[1066] + tmp319*weights[1078] + tmp32*weights[1081] + tmp320*weights[1080] + tmp321*weights[1084] + tmp322*weights[1085] + tmp323*weights[1089] + tmp324*weights[1092] + tmp325*weights[1093] + tmp326*weights[1095] + tmp327*weights[1097] + tmp328*weights[1100] + tmp329*weights[1107] + tmp33*weights[1082] + tmp330*weights[1109] + tmp331*weights[1119] + tmp34*weights[1103] + tmp35*weights[1113] + tmp38*weights[1126] + tmp41*weights[1130] + tmp44*weights[1158] + tmp46*weights[1170] + tmp48*weights[1173] + tmp51*weights[1257] + tmp52*weights[1274] + tmp54*weights[1293] + tmp55*weights[1295] + tmp56*weights[1307] + tmp57*weights[1314] + tmp59*weights[1334] + tmp60*weights[1343] + tmp61*weights[1073] + tmp62*weights[1098] + tmp63*weights[1117] + tmp64*weights[1128] + tmp65*weights[1144] + tmp66*weights[1151] + tmp67*weights[1160] + tmp68*weights[1166] + tmp69*weights[1168] + tmp70*weights[1176] + tmp71*weights[1225] + tmp72*weights[1234] + tmp73*weights[1240] + tmp74*weights[1241] + tmp75*weights[1249] + tmp76*weights[1262] + tmp77*weights[1263] + tmp78*weights[1281] + tmp79*weights[1290] + tmp80*weights[1303] + tmp81*weights[1309] + tmp82*weights[1328] + tmp83*weights[1335] + tmp84*weights[1353] + tmp85*weights[1094] + tmp86*weights[1110] + tmp87*weights[1111] + tmp88*weights[1120] + tmp89*weights[1124] + tmp90*weights[1127] + tmp91*weights[1129] + tmp92*weights[1140] + tmp93*weights[1148] + tmp94*weights[1150] + tmp95*weights[1152] + tmp97*weights[1153] + tmp99*weights[1154] + weights[1060]*z[13] + weights[1101]*z[11] + weights[1132]*z[1] + weights[1164]*z[5] + weights[1175]*z[4] + weights[1213]*z[0] + weights[1245]*z[3] + weights[1260]*z[12] + weights[1358]*z[6] + z[17];
    out[48] =  weights[1370];
    out[49] =  weights[1363];
    out[50] =  weights[1365];
    out[51] =  tmp11*weights[1392] + tmp13*weights[1394] + tmp2*weights[1376] + tmp5*weights[1379] + tmp7*weights[1380] + tmp9*weights[1385] + weights[1362];
    out[52] =  tmp11*weights[1377] + tmp14*weights[1382] + tmp15*weights[1383] + tmp16*weights[1390] + tmp17*weights[1391] + tmp5*weights[1372] + weights[1369];
    out[53] =  tmp15*weights[1384] + tmp17*weights[1389] + tmp18*weights[1371] + tmp19*weights[1378] + tmp2*weights[1386] + tmp7*weights[1388] + weights[1368];
    out[54] =  tmp13*weights[1375] + tmp14*weights[1374] + tmp16*weights[1387] + tmp18*weights[1381] + tmp19*weights[1373] + tmp9*weights[1393] + weights[1364];
    out[55] =  weights[1367];
    out[56] =  weights[1360];
    out[57] =  weights[1361];
    out[58] =  weights[1366];
    out[59] =  tmp100*weights[1497] + tmp101*weights[1501] + tmp103*weights[1502] + tmp104*weights[1503] + tmp105*weights[1509] + tmp106*weights[1514] + tmp107*weights[1517] + tmp108*weights[1518] + tmp109*weights[1520] + tmp110*weights[1521] + tmp111*weights[1531] + tmp113*weights[1534] + tmp114*weights[1535] + tmp115*weights[1539] + tmp116*weights[1540] + tmp117*weights[1541] + tmp118*weights[1543] + tmp119*weights[1545] + tmp120*weights[1546] + tmp121*weights[1547] + tmp122*weights[1549] + tmp123*weights[1551] + tmp124*weights[1554] + tmp125*weights[1559] + tmp126*weights[1561] + tmp127*weights[1564] + tmp128*weights[1569] + tmp129*weights[1570] + tmp130*weights[1571] + tmp131*weights[1573] + tmp132*weights[1576] + tmp133*weights[1577] + tmp134*weights[1583] + tmp135*weights[1588] + tmp136*weights[1590] + tmp137*weights[1591] + tmp138*weights[1592] + tmp139*weights[1594] + tmp140*weights[1596] + tmp141*weights[1599] + tmp142*weights[1604] + tmp143*weights[1605] + tmp144*weights[1607] + tmp145*weights[1608] + tmp146*weights[1610] + tmp147*weights[1611] + tmp148*weights[1612] + tmp149*weights[1613] + tmp150*weights[1615] + tmp151*weights[1616] + tmp152*weights[1624] + tmp153*weights[1625] + tmp154*weights[1627] + tmp155*weights[1629] + tmp156*weights[1631] + tmp157*weights[1632] + tmp158*weights[1638] + tmp159*weights[1639] + tmp160*weights[1641] + tmp161*weights[1646] + tmp162*weights[1648] + tmp163*weights[1651] + tmp164*weights[1652] + tmp165*weights[1653] + tmp166*weights[1660] + tmp167*weights[1662] + tmp168*weights[1663] + tmp169*weights[1665] + tmp170*weights[1666] + tmp171*weights[1667] + tmp172*weights[1670] + tmp173*weights[1672] + tmp174*weights[1673] + tmp175*weights[1676] + tmp176*weights[1677] + tmp177*weights[1679] + tmp178*weights[1680] + tmp179*weights[1692] + tmp180*weights[1694] + tmp181*weights[1695] + tmp182*weights[1697] + tmp183*weights[1699] + tmp184*weights[1401] + tmp185*weights[1403] + tmp186*weights[1405] + tmp187*weights[1407] + tmp188*weights[1408] + tmp189*weights[1409] + tmp190*weights[1410] + tmp191*weights[1411] + tmp192*weights[1415] + tmp193*weights[1416] + tmp194*weights[1419] + tmp195*weights[1423] + tmp196*weights[1426] + tmp197*weights[1427] + tmp198*weights[1428] + tmp199*weights[1430] + tmp20*weights[1474] + tmp200*weights[1431] + tmp201*weights[1436] + tmp202*weights[1439] + tmp203*weights[1442] + tmp204*weights[1444] + tmp205*weights[1446] + tmp206*weights[1448] + tmp207*weights[1452] + tmp208*weights[1454] + tmp209*weights[1455] + tmp21*weights[1477] + tmp210*weights[1456] + tmp211*weights[1458] + tmp212*weights[1462] + tmp213*weights[1465] + tmp214*weights[1483] + tmp215*weights[1486] + tmp216*weights[1523] + tmp217*weights[1524] + tmp218*weights[1527] + tmp219*weights[1528] + tmp22*weights[1507] + tmp220*weights[1542] + tmp221*weights[1544] + tmp222*weights[1555] + tmp223*weights[1557] + tmp224*weights[1560] + tmp225*weights[1568] + tmp226*weights[1579] + tmp227*weights[1595] + tmp228*weights[1626] + tmp229*weights[1650] + tmp23*weights[1586] + tmp230*weights[1687] + tmp231*weights[1696] + tmp232*weights[1397] + tmp233*weights[1412] + tmp234*weights[1414] + tmp235*weights[1445] + tmp236*weights[1461] + tmp237*weights[1463] + tmp238*weights[1471] + tmp239*weights[1473] + tmp24*weights[1587] + tmp240*weights[1475] + tmp241*weights[1476] + tmp242*weights[1478] + tmp243*weights[1479] + tmp244*weights[1481] + tmp245*weights[1482] + tmp246*weights[1485] + tmp247*weights[1487] + tmp248*weights[1489] + tmp249*weights[1495] + tmp25*weights[1617] + tmp250*weights[1496] + tmp251*weights[1499] + tmp252*weights[1505] + tmp253*weights[1511] + tmp254*weights[1512] + tmp255*weights[1519] + tmp256*weights[1522] + tmp257*weights[1525] + tmp258*weights[1526] + tmp259*weights[1529] + tmp26*weights[1618] + tmp260*weights[1530] + tmp261*weights[1532] + tmp262*weights[1533] + tmp263*weights[1536] + tmp264*weights[1537] + tmp265*weights[1538] + tmp266*weights[1548] + tmp267*weights[1550] + tmp268*weights[1552] + tmp269*weights[1556] + tmp27*weights[1628] + tmp270*weights[1558] + tmp271*weights[1562] + tmp272*weights[1563] + tmp273*weights[1566] + tmp274*weights[1567] + tmp275*weights[1572] + tmp276*weights[1575] + tmp277*weights[1578] + tmp278*weights[1582] + tmp279*weights[1584] + tmp28*weights[1636] + tmp280*weights[1593] + tmp281*weights[1598] + tmp282*weights[1601] + tmp283*weights[1606] + tmp284*weights[1609] + tmp285*weights[1619] + tmp286*weights[1620] + tmp287*weights[1622] + tmp288*weights[1623] + tmp289*weights[1634] + tmp29*weights[1657] + tmp290*weights[1637] + tmp291*weights[1640] + tmp292*weights[1642] + tmp293*weights[1644] + tmp294*weights[1645] + tmp295*weights[1655] + tmp296*weights[1656] + tmp297*weights[1658] + tmp298*weights[1661] + tmp299*weights[1664] + tmp30*weights[1659] + tmp300*weights[1669] + tmp301*weights[1671] + tmp302*weights[1678] + tmp303*weights[1681] + tmp304*weights[1682] + tmp305*weights[1684] + tmp306*weights[1685] + tmp307*weights[1686] + tmp308*weights[1688] + tmp309*weights[1689] + tmp31*weights[1417] + tmp310*weights[1690] + tmp311*weights[1691] + tmp312*weights[1395] + tmp313*weights[1396] + tmp314*weights[1398] + tmp315*weights[1399] + tmp316*weights[1402] + tmp317*weights[1404] + tmp318*weights[1406] + tmp319*weights[1418] + tmp32*weights[1421] + tmp320*weights[1420] + tmp321*weights[1424] + tmp322*weights[1425] + tmp323*weights[1429] + tmp324*weights[1432] + tmp325*weights[1433] + tmp326*weights[1435] + tmp327*weights[1437] + tmp328*weights[1440] + tmp329*weights[1447] + tmp33*weights[1422] + tmp330*weights[1449] + tmp331*weights[1459] + tmp34*weights[1443] + tmp35*weights[1453] + tmp38*weights[1466] + tmp41*weights[1470] + tmp44*weights[1498] + tmp46*weights[1510] + tmp48*weights[1513] + tmp51*weights[1597] + tmp52*weights[1614] + tmp54*weights[1633] + tmp55*weights[1635] + tmp56*weights[1647] + tmp57*weights[1654] + tmp59*weights[1674] + tmp60*weights[1683] + tmp61*weights[1413] + tmp62*weights[1438] + tmp63*weights[1457] + tmp64*weights[1468] + tmp65*weights[1484] + tmp66*weights[1491] + tmp67*weights[1500] + tmp68*weights[1506] + tmp69*weights[1508] + tmp70*weights[1516] + tmp71*weights[1565] + tmp72*weights[1574] + tmp73*weights[1580] + tmp74*weights[1581] + tmp75*weights[1589] + tmp76*weights[1602] + tmp77*weights[1603] + tmp78*weights[1621] + tmp79*weights[1630] + tmp80*weights[1643] + tmp81*weights[1649] + tmp82*weights[1668] + tmp83*weights[1675] + tmp84*weights[1693] + tmp85*weights[1434] + tmp86*weights[1450] + tmp87*weights[1451] + tmp88*weights[1460] + tmp89*weights[1464] + tmp90*weights[1467] + tmp91*weights[1469] + tmp92*weights[1480] + tmp93*weights[1488] + tmp94*weights[1490] + tmp95*weights[1492] + tmp97*weights[1493] + tmp99*weights[1494] + weights[1400]*z[13] + weights[1441]*z[11] + weights[1472]*z[1] + weights[1504]*z[5] + weights[1515]*z[4] + weights[1553]*z[0] + weights[1585]*z[3] + weights[1600]*z[12] + weights[1698]*z[6] + z[18];
    out[60] =  weights[1710];
    out[61] =  weights[1703];
    out[62] =  weights[1705];
    out[63] =  tmp11*weights[1732] + tmp13*weights[1734] + tmp2*weights[1716] + tmp5*weights[1719] + tmp7*weights[1720] + tmp9*weights[1725] + weights[1702];
    out[64] =  tmp11*weights[1717] + tmp14*weights[1722] + tmp15*weights[1723] + tmp16*weights[1730] + tmp17*weights[1731] + tmp5*weights[1712] + weights[1709];
    out[65] =  tmp15*weights[1724] + tmp17*weights[1729] + tmp18*weights[1711] + tmp19*weights[1718] + tmp2*weights[1726] + tmp7*weights[1728] + weights[1708];
    out[66] =  tmp13*weights[1715] + tmp14*weights[1714] + tmp16*weights[1727] + tmp18*weights[1721] + tmp19*weights[1713] + tmp9*weights[1733] + weights[1704];
    out[67] =  weights[1707];
    out[68] =  weights[1700];
    out[69] =  weights[1701];
    out[70] =  weights[1706];
    out[71] =  tmp100*weights[1837] + tmp101*weights[1841] + tmp103*weights[1842] + tmp104*weights[1843] + tmp105*weights[1849] + tmp106*weights[1854] + tmp107*weights[1857] + tmp108*weights[1858] + tmp109*weights[1860] + tmp110*weights[1861] + tmp111*weights[1871] + tmp113*weights[1874] + tmp114*weights[1875] + tmp115*weights[1879] + tmp116*weights[1880] + tmp117*weights[1881] + tmp118*weights[1883] + tmp119*weights[1885] + tmp120*weights[1886] + tmp121*weights[1887] + tmp122*weights[1889] + tmp123*weights[1891] + tmp124*weights[1894] + tmp125*weights[1899] + tmp126*weights[1901] + tmp127*weights[1904] + tmp128*weights[1909] + tmp129*weights[1910] + tmp130*weights[1911] + tmp131*weights[1913] + tmp132*weights[1916] + tmp133*weights[1917] + tmp134*weights[1923] + tmp135*weights[1928] + tmp136*weights[1930] + tmp137*weights[1931] + tmp138*weights[1932] + tmp139*weights[1934] + tmp140*weights[1936] + tmp141*weights[1939] + tmp142*weights[1944] + tmp143*weights[1945] + tmp144*weights[1947] + tmp145*weights[1948] + tmp146*weights[1950] + tmp147*weights[1951] + tmp148*weights[1952] + tmp149*weights[1953] + tmp150*weights[1955] + tmp151*weights[1956] + tmp152*weights[1964] + tmp153*weights[1965] + tmp154*weights[1967] + tmp155*weights[1969] + tmp156*weights[1971] + tmp157*weights[1972] + tmp158*weights[1978] + tmp159*weights[1979] + tmp160*weights[1981] + tmp161*weights[1986] + tmp162*weights[1988] + tmp163*weights[1991] + tmp164*weights[1992] + tmp165*weights[1993] + tmp166*weights[2000] + tmp167*weights[2002] + tmp168*weights[2003] + tmp169*weights[2005] + tmp170*weights[2006] + tmp171*weights[2007] + tmp172*weights[2010] + tmp173*weights[2012] + tmp174*weights[2013] + tmp175*weights[2016] + tmp176*weights[2017] + tmp177*weights[2019] + tmp178*weights[2020] + tmp179*weights[2032] + tmp180*weights[2034] + tmp181*weights[2035] + tmp182*weights[2037] + tmp183*weights[2039] + tmp184*weights[1741] + tmp185*weights[1743] + tmp186*weights[1745] + tmp187*weights[1747] + tmp188*weights[1748] + tmp189*weights[1749] + tmp190*weights[1750] + tmp191*weights[1751] + tmp192*weights[1755] + tmp193*weights[1756] + tmp194*weights[1759] + tmp195*weights[1763] + tmp196*weights[1766] + tmp197*weights[1767] + tmp198*weights[1768] + tmp199*weights[1770] + tmp20*weights[1814] + tmp200*weights[1771] + tmp201*weights[1776] + tmp202*weights[1779] + tmp203*weights[1782] + tmp204*weights[1784] + tmp205*weights[1786] + tmp206*weights[1788] + tmp207*weights[1792] + tmp208*weights[1794] + tmp209*weights[1795] + tmp21*weights[1817] + tmp210*weights[1796] + tmp211*weights[1798] + tmp212*weights[1802] + tmp213*weights[1805] + tmp214*weights[1823] + tmp215*weights[1826] + tmp216*weights[1863] + tmp217*weights[1864] + tmp218*weights[1867] + tmp219*weights[1868] + tmp22*weights[1847] + tmp220*weights[1882] + tmp221*weights[1884] + tmp222*weights[1895] + tmp223*weights[1897] + tmp224*weights[1900] + tmp225*weights[1908] + tmp226*weights[1919] + tmp227*weights[1935] + tmp228*weights[1966] + tmp229*weights[1990] + tmp23*weights[1926] + tmp230*weights[2027] + tmp231*weights[2036] + tmp232*weights[1737] + tmp233*weights[1752] + tmp234*weights[1754] + tmp235*weights[1785] + tmp236*weights[1801] + tmp237*weights[1803] + tmp238*weights[1811] + tmp239*weights[1813] + tmp24*weights[1927] + tmp240*weights[1815] + tmp241*weights[1816] + tmp242*weights[1818] + tmp243*weights[1819] + tmp244*weights[1821] + tmp245*weights[1822] + tmp246*weights[1825] + tmp247*weights[1827] + tmp248*weights[1829] + tmp249*weights[1835] + tmp25*weights[1957] + tmp250*weights[1836] + tmp251*weights[1839] + tmp252*weights[1845] + tmp253*weights[1851] + tmp254*weights[1852] + tmp255*weights[1859] + tmp256*weights[1862] + tmp257*weights[1865] + tmp258*weights[1866] + tmp259*weights[1869] + tmp26*weights[1958] + tmp260*weights[1870] + tmp261*weights[1872] + tmp262*weights[1873] + tmp263*weights[1876] + tmp264*weights[1877] + tmp265*weights[1878] + tmp266*weights[1888] + tmp267*weights[1890] + tmp268*weights[1892] + tmp269*weights[1896] + tmp27*weights[1968] + tmp270*weights[1898] + tmp271*weights[1902] + tmp272*weights[1903] + tmp273*weights[1906] + tmp274*weights[1907] + tmp275*weights[1912] + tmp276*weights[1915] + tmp277*weights[1918] + tmp278*weights[1922] + tmp279*weights[1924] + tmp28*weights[1976] + tmp280*weights[1933] + tmp281*weights[1938] + tmp282*weights[1941] + tmp283*weights[1946] + tmp284*weights[1949] + tmp285*weights[1959] + tmp286*weights[1960] + tmp287*weights[1962] + tmp288*weights[1963] + tmp289*weights[1974] + tmp29*weights[1997] + tmp290*weights[1977] + tmp291*weights[1980] + tmp292*weights[1982] + tmp293*weights[1984] + tmp294*weights[1985] + tmp295*weights[1995] + tmp296*weights[1996] + tmp297*weights[1998] + tmp298*weights[2001] + tmp299*weights[2004] + tmp30*weights[1999] + tmp300*weights[2009] + tmp301*weights[2011] + tmp302*weights[2018] + tmp303*weights[2021] + tmp304*weights[2022] + tmp305*weights[2024] + tmp306*weights[2025] + tmp307*weights[2026] + tmp308*weights[2028] + tmp309*weights[2029] + tmp31*weights[1757] + tmp310*weights[2030] + tmp311*weights[2031] + tmp312*weights[1735] + tmp313*weights[1736] + tmp314*weights[1738] + tmp315*weights[1739] + tmp316*weights[1742] + tmp317*weights[1744] + tmp318*weights[1746] + tmp319*weights[1758] + tmp32*weights[1761] + tmp320*weights[1760] + tmp321*weights[1764] + tmp322*weights[1765] + tmp323*weights[1769] + tmp324*weights[1772] + tmp325*weights[1773] + tmp326*weights[1775] + tmp327*weights[1777] + tmp328*weights[1780] + tmp329*weights[1787] + tmp33*weights[1762] + tmp330*weights[1789] + tmp331*weights[1799] + tmp34*weights[1783] + tmp35*weights[1793] + tmp38*weights[1806] + tmp41*weights[1810] + tmp44*weights[1838] + tmp46*weights[1850] + tmp48*weights[1853] + tmp51*weights[1937] + tmp52*weights[1954] + tmp54*weights[1973] + tmp55*weights[1975] + tmp56*weights[1987] + tmp57*weights[1994] + tmp59*weights[2014] + tmp60*weights[2023] + tmp61*weights[1753] + tmp62*weights[1778] + tmp63*weights[1797] + tmp64*weights[1808] + tmp65*weights[1824] + tmp66*weights[1831] + tmp67*weights[1840] + tmp68*weights[1846] + tmp69*weights[1848] + tmp70*weights[1856] + tmp71*weights[1905] + tmp72*weights[1914] + tmp73*weights[1920] + tmp74*weights[1921] + tmp75*weights[1929] + tmp76*weights[1942] + tmp77*weights[1943] + tmp78*weights[1961] + tmp79*weights[1970] + tmp80*weights[1983] + tmp81*weights[1989] + tmp82*weights[2008] + tmp83*weights[2015] + tmp84*weights[2033] + tmp85*weights[1774] + tmp86*weights[1790] + tmp87*weights[1791] + tmp88*weights[1800] + tmp89*weights[1804] + tmp90*weights[1807] + tmp91*weights[1809] + tmp92*weights[1820] + tmp93*weights[1828] + tmp94*weights[1830] + tmp95*weights[1832] + tmp97*weights[1833] + tmp99*weights[1834] + weights[1740]*z[13] + weights[1781]*z[11] + weights[1812]*z[1] + weights[1844]*z[5] + weights[1855]*z[4] + weights[1893]*z[0] + weights[1925]*z[3] + weights[1940]*z[12] + weights[2038]*z[6] + z[19];
    out[72] =  weights[2050];
    out[73] =  weights[2043];
    out[74] =  weights[2045];
    out[75] =  tmp11*weights[2072] + tmp13*weights[2074] + tmp2*weights[2056] + tmp5*weights[2059] + tmp7*weights[2060] + tmp9*weights[2065] + weights[2042];
    out[76] =  tmp11*weights[2057] + tmp14*weights[2062] + tmp15*weights[2063] + tmp16*weights[2070] + tmp17*weights[2071] + tmp5*weights[2052] + weights[2049];
    out[77] =  tmp15*weights[2064] + tmp17*weights[2069] + tmp18*weights[2051] + tmp19*weights[2058] + tmp2*weights[2066] + tmp7*weights[2068] + weights[2048];
    out[78] =  tmp13*weights[2055] + tmp14*weights[2054] + tmp16*weights[2067] + tmp18*weights[2061] + tmp19*weights[2053] + tmp9*weights[2073] + weights[2044];
    out[79] =  weights[2047];
    out[80] =  weights[2040];
    out[81] =  weights[2041];
    out[82] =  weights[2046];
    out[83] =  tmp100*weights[2177] + tmp101*weights[2181] + tmp103*weights[2182] + tmp104*weights[2183] + tmp105*weights[2189] + tmp106*weights[2194] + tmp107*weights[2197] + tmp108*weights[2198] + tmp109*weights[2200] + tmp110*weights[2201] + tmp111*weights[2211] + tmp113*weights[2214] + tmp114*weights[2215] + tmp115*weights[2219] + tmp116*weights[2220] + tmp117*weights[2221] + tmp118*weights[2223] + tmp119*weights[2225] + tmp120*weights[2226] + tmp121*weights[2227] + tmp122*weights[2229] + tmp123*weights[2231] + tmp124*weights[2234] + tmp125*weights[2239] + tmp126*weights[2241] + tmp127*weights[2244] + tmp128*weights[2249] + tmp129*weights[2250] + tmp130*weights[2251] + tmp131*weights[2253] + tmp132*weights[2256] + tmp133*weights[2257] + tmp134*weights[2263] + tmp135*weights[2268] + tmp136*weights[2270] + tmp137*weights[2271] + tmp138*weights[2272] + tmp139*weights[2274] + tmp140*weights[2276] + tmp141*weights[2279] + tmp142*weights[2284] + tmp143*weights[2285] + tmp144*weights[2287] + tmp145*weights[2288] + tmp146*weights[2290] + tmp147*weights[2291] + tmp148*weights[2292] + tmp149*weights[2293] + tmp150*weights[2295] + tmp151*weights[2296] + tmp152*weights[2304] + tmp153*weights[2305] + tmp154*weights[2307] + tmp155*weights[2309] + tmp156*weights[2311] + tmp157*weights[2312] + tmp158*weights[2318] + tmp159*weights[2319] + tmp160*weights[2321] + tmp161*weights[2326] + tmp162*weights[2328] + tmp163*weights[2331] + tmp164*weights[2332] + tmp165*weights[2333] + tmp166*weights[2340] + tmp167*weights[2342] + tmp168*weights[2343] + tmp169*weights[2345] + tmp170*weights[2346] + tmp171*weights[2347] + tmp172*weights[2350] + tmp173*weights[2352] + tmp174*weights[2353] + tmp175*weights[2356] + tmp176*weights[2357] + tmp177*weights[2359] + tmp178*weights[2360] + tmp179*weights[2372] + tmp180*weights[2374] + tmp181*weights[2375] + tmp182*weights[2377] + tmp183*weights[2379] + tmp184*weights[2081] + tmp185*weights[2083] + tmp186*weights[2085] + tmp187*weights[2087] + tmp188*weights[2088] + tmp189*weights[2089] + tmp190*weights[2090] + tmp191*weights[2091] + tmp192*weights[2095] + tmp193*weights[2096] + tmp194*weights[2099] + tmp195*weights[2103] + tmp196*weights[2106] + tmp197*weights[2107] + tmp198*weights[2108] + tmp199*weights[2110] + tmp20*weights[2154] + tmp200*weights[2111] + tmp201*weights[2116] + tmp202*weights[2119] + tmp203*weights[2122] + tmp204*weights[2124] + tmp205*weights[2126] + tmp206*weights[2128] + tmp207*weights[2132] + tmp208*weights[2134] + tmp209*weights[2135] + tmp21*weights[2157] + tmp210*weights[2136] + tmp211*weights[2138] + tmp212*weights[2142] + tmp213*weights[2145] + tmp214*weights[2163] + tmp215*weights[2166] + tmp216*weights[2203] + tmp217*weights[2204] + tmp218*weights[2207] + tmp219*weights[2208] + tmp22*weights[2187] + tmp220*weights[2222] + tmp221*weights[2224] + tmp222*weights[2235] + tmp223*weights[2237] + tmp224*weights[2240] + tmp225*weights[2248] + tmp226*weights[2259] + tmp227*weights[2275] + tmp228*weights[2306] + tmp229*weights[2330] + tmp23*weights[2266] + tmp230*weights[2367] + tmp231*weights[2376] + tmp232*weights[2077] + tmp233*weights[2092] + tmp234*weights[2094] + tmp235*weights[2125] + tmp236*weights[2141] + tmp237*weights[2143] + tmp238*weights[2151] + tmp239*weights[2153] + tmp24*weights[2267] + tmp240*weights[2155] + tmp241*weights[2156] + tmp242*weights[2158] + tmp243*weights[2159] + tmp244*weights[2161] + tmp245*weights[2162] + tmp246*weights[2165] + tmp247*weights[2167] + tmp248*weights[2169] + tmp249*weights[2175] + tmp25*weights[2297] + tmp250*weights[2176] + tmp251*weights[2179] + tmp252*weights[2185] + tmp253*weights[2191] + tmp254*weights[2192] + tmp255*weights[2199] + tmp256*weights[2202] + tmp257*weights[2205] + tmp258*weights[2206] + tmp259*weights[2209] + tmp26*weights[2298] + tmp260*weights[2210] + tmp261*weights[2212] + tmp262*weights[2213] + tmp263*weights[2216] + tmp264*weights[2217] + tmp265*weights[2218] + tmp266*weights[2228] + tmp267*weights[2230] + tmp268*weights[2232] + tmp269*weights[2236] + tmp27*weights[2308] + tmp270*weights[2238] + tmp271*weights[2242] + tmp272*weights[2243] + tmp273*weights[2246] + tmp274*weights[2247] + tmp275*weights[2252] + tmp276*weights[2255] + tmp277*weights[2258] + tmp278*weights[2262] + tmp279*weights[2264] + tmp28*weights[2316] + tmp280*weights[2273] + tmp281*weights[2278] + tmp282*weights[2281] + tmp283*weights[2286] + tmp284*weights[2289] + tmp285*weights[2299] + tmp286*weights[2300] + tmp287*weights[2302] + tmp288*weights[2303] + tmp289*weights[2314] + tmp29*weights[2337] + tmp290*weights[2317] + tmp291*weights[2320] + tmp292*weights[2322] + tmp293*weights[2324] + tmp294*weights[2325] + tmp295*weights[2335] + tmp296*weights[2336] + tmp297*weights[2338] + tmp298*weights[2341] + tmp299*weights[2344] + tmp30*weights[2339] + tmp300*weights[2349] + tmp301*weights[2351] + tmp302*weights[2358] + tmp303*weights[2361] + tmp304*weights[2362] + tmp305*weights[2364] + tmp306*weights[2365] + tmp307*weights[2366] + tmp308*weights[2368] + tmp309*weights[2369] + tmp31*weights[2097] + tmp310*weights[2370] + tmp311*weights[2371] + tmp312*weights[2075] + tmp313*weights[2076] + tmp314*weights[2078] + tmp315*weights[2079] + tmp316*weights[2082] + tmp317*weights[2084] + tmp318*weights[2086] + tmp319*weights[2098] + tmp32*weights[2101] + tmp320*weights[2100] + tmp321*weights[2104] + tmp322*weights[2105] + tmp323*weights[2109] + tmp324*weights[2112] + tmp325*weights[2113] + tmp326*weights[2115] + tmp327*weights[2117] + tmp328*weights[2120] + tmp329*weights[2127] + tmp33*weights[2102] + tmp330*weights[2129] + tmp331*weights[2139] + tmp34*weights[2123] + tmp35*weights[2133] + tmp38*weights[2146] + tmp41*weights[2150] + tmp44*weights[2178] + tmp46*weights[2190] + tmp48*weights[2193] + tmp51*weights[2277] + tmp52*weights[2294] + tmp54*weights[2313] + tmp55*weights[2315] + tmp56*weights[2327] + tmp57*weights[2334] + tmp59*weights[2354] + tmp60*weights[2363] + tmp61*weights[2093] + tmp62*weights[2118] + tmp63*weights[2137] + tmp64*weights[2148] + tmp65*weights[2164] + tmp66*weights[2171] + tmp67*weights[2180] + tmp68*weights[2186] + tmp69*weights[2188] + tmp70*weights[2196] + tmp71*weights[2245] + tmp72*weights[2254] + tmp73*weights[2260] + tmp74*weights[2261] + tmp75*weights[2269] + tmp76*weights[2282] + tmp77*weights[2283] + tmp78*weights[2301] + tmp79*weights[2310] + tmp80*weights[2323] + tmp81*weights[2329] + tmp82*weights[2348] + tmp83*weights[2355] + tmp84*weights[2373] + tmp85*weights[2114] + tmp86*weights[2130] + tmp87*weights[2131] + tmp88*weights[2140] + tmp89*weights[2144] + tmp90*weights[2147] + tmp91*weights[2149] + tmp92*weights[2160] + tmp93*weights[2168] + tmp94*weights[2170] + tmp95*weights[2172] + tmp97*weights[2173] + tmp99*weights[2174] + weights[2080]*z[13] + weights[2121]*z[11] + weights[2152]*z[1] + weights[2184]*z[5] + weights[2195]*z[4] + weights[2233]*z[0] + weights[2265]*z[3] + weights[2280]*z[12] + weights[2378]*z[6] + z[20];
    out[84] =  weights[2390];
    out[85] =  weights[2383];
    out[86] =  weights[2385];
    out[87] =  tmp11*weights[2412] + tmp13*weights[2414] + tmp2*weights[2396] + tmp5*weights[2399] + tmp7*weights[2400] + tmp9*weights[2405] + weights[2382];
    out[88] =  tmp11*weights[2397] + tmp14*weights[2402] + tmp15*weights[2403] + tmp16*weights[2410] + tmp17*weights[2411] + tmp5*weights[2392] + weights[2389];
    out[89] =  tmp15*weights[2404] + tmp17*weights[2409] + tmp18*weights[2391] + tmp19*weights[2398] + tmp2*weights[2406] + tmp7*weights[2408] + weights[2388];
    out[90] =  tmp13*weights[2395] + tmp14*weights[2394] + tmp16*weights[2407] + tmp18*weights[2401] + tmp19*weights[2393] + tmp9*weights[2413] + weights[2384];
    out[91] =  weights[2387];
    out[92] =  weights[2380];
    out[93] =  weights[2381];
    out[94] =  weights[2386];
    out[95] =  tmp100*weights[2517] + tmp101*weights[2521] + tmp103*weights[2522] + tmp104*weights[2523] + tmp105*weights[2529] + tmp106*weights[2534] + tmp107*weights[2537] + tmp108*weights[2538] + tmp109*weights[2540] + tmp110*weights[2541] + tmp111*weights[2551] + tmp113*weights[2554] + tmp114*weights[2555] + tmp115*weights[2559] + tmp116*weights[2560] + tmp117*weights[2561] + tmp118*weights[2563] + tmp119*weights[2565] + tmp120*weights[2566] + tmp121*weights[2567] + tmp122*weights[2569] + tmp123*weights[2571] + tmp124*weights[2574] + tmp125*weights[2579] + tmp126*weights[2581] + tmp127*weights[2584] + tmp128*weights[2589] + tmp129*weights[2590] + tmp130*weights[2591] + tmp131*weights[2593] + tmp132*weights[2596] + tmp133*weights[2597] + tmp134*weights[2603] + tmp135*weights[2608] + tmp136*weights[2610] + tmp137*weights[2611] + tmp138*weights[2612] + tmp139*weights[2614] + tmp140*weights[2616] + tmp141*weights[2619] + tmp142*weights[2624] + tmp143*weights[2625] + tmp144*weights[2627] + tmp145*weights[2628] + tmp146*weights[2630] + tmp147*weights[2631] + tmp148*weights[2632] + tmp149*weights[2633] + tmp150*weights[2635] + tmp151*weights[2636] + tmp152*weights[2644] + tmp153*weights[2645] + tmp154*weights[2647] + tmp155*weights[2649] + tmp156*weights[2651] + tmp157*weights[2652] + tmp158*weights[2658] + tmp159*weights[2659] + tmp160*weights[2661] + tmp161*weights[2666] + tmp162*weights[2668] + tmp163*weights[2671] + tmp164*weights[2672] + tmp165*weights[2673] + tmp166*weights[2680] + tmp167*weights[2682] + tmp168*weights[2683] + tmp169*weights[2685] + tmp170*weights[2686] + tmp171*weights[2687] + tmp172*weights[2690] + tmp173*weights[2692] + tmp174*weights[2693] + tmp175*weights[2696] + tmp176*weights[2697] + tmp177*weights[2699] + tmp178*weights[2700] + tmp179*weights[2712] + tmp180*weights[2714] + tmp181*weights[2715] + tmp182*weights[2717] + tmp183*weights[2719] + tmp184*weights[2421] + tmp185*weights[2423] + tmp186*weights[2425] + tmp187*weights[2427] + tmp188*weights[2428] + tmp189*weights[2429] + tmp190*weights[2430] + tmp191*weights[2431] + tmp192*weights[2435] + tmp193*weights[2436] + tmp194*weights[2439] + tmp195*weights[2443] + tmp196*weights[2446] + tmp197*weights[2447] + tmp198*weights[2448] + tmp199*weights[2450] + tmp20*weights[2494] + tmp200*weights[2451] + tmp201*weights[2456] + tmp202*weights[2459] + tmp203*weights[2462] + tmp204*weights[2464] + tmp205*weights[2466] + tmp206*weights[2468] + tmp207*weights[2472] + tmp208*weights[2474] + tmp209*weights[2475] + tmp21*weights[2497] + tmp210*weights[2476] + tmp211*weights[2478] + tmp212*weights[2482] + tmp213*weights[2485] + tmp214*weights[2503] + tmp215*weights[2506] + tmp216*weights[2543] + tmp217*weights[2544] + tmp218*weights[2547] + tmp219*weights[2548] + tmp22*weights[2527] + tmp220*weights[2562] + tmp221*weights[2564] + tmp222*weights[2575] + tmp223*weights[2577] + tmp224*weights[2580] + tmp225*weights[2588] + tmp226*weights[2599] + tmp227*weights[2615] + tmp228*weights[2646] + tmp229*weights[2670] + tmp23*weights[2606] + tmp230*weights[2707] + tmp231*weights[2716] + tmp232*weights[2417] + tmp233*weights[2432] + tmp234*weights[2434] + tmp235*weights[2465] + tmp236*weights[2481] + tmp237*weights[2483] + tmp238*weights[2491] + tmp239*weights[2493] + tmp24*weights[2607] + tmp240*weights[2495] + tmp241*weights[2496] + tmp242*weights[2498] + tmp243*weights[2499] + tmp244*weights[2501] + tmp245*weights[2502] + tmp246*weights[2505] + tmp247*weights[2507] + tmp248*weights[2509] + tmp249*weights[2515] + tmp25*weights[2637] + tmp250*weights[2516] + tmp251*weights[2519] + tmp252*weights[2525] + tmp253*weights[2531] + tmp254*weights[2532] + tmp255*weights[2539] + tmp256*weights[2542] + tmp257*weights[2545] + tmp258*weights[2546] + tmp259*weights[2549] + tmp26*weights[2638] + tmp260*weights[2550] + tmp261*weights[2552] + tmp262*weights[2553] + tmp263*weights[2556] + tmp264*weights[2557] + tmp265*weights[2558] + tmp266*weights[2568] + tmp267*weights[2570] + tmp268*weights[2572] + tmp269*weights[2576] + tmp27*weights[2648] + tmp270*weights[2578] + tmp271*weights[2582] + tmp272*weights[2583] + tmp273*weights[2586] + tmp274*weights[2587] + tmp275*weights[2592] + tmp276*weights[2595] + tmp277*weights[2598] + tmp278*weights[2602] + tmp279*weights[2604] + tmp28*weights[2656] + tmp280*weights[2613] + tmp281*weights[2618] + tmp282*weights[2621] + tmp283*weights[2626] + tmp284*weights[2629] + tmp285*weights[2639] + tmp286*weights[2640] + tmp287*weights[2642] + tmp288*weights[2643] + tmp289*weights[2654] + tmp29*weights[2677] + tmp290*weights[2657] + tmp291*weights[2660] + tmp292*weights[2662] + tmp293*weights[2664] + tmp294*weights[2665] + tmp295*weights[2675] + tmp296*weights[2676] + tmp297*weights[2678] + tmp298*weights[2681] + tmp299*weights[2684] + tmp30*weights[2679] + tmp300*weights[2689] + tmp301*weights[2691] + tmp302*weights[2698] + tmp303*weights[2701] + tmp304*weights[2702] + tmp305*weights[2704] + tmp306*weights[2705] + tmp307*weights[2706] + tmp308*weights[2708] + tmp309*weights[2709] + tmp31*weights[2437] + tmp310*weights[2710] + tmp311*weights[2711] + tmp312*weights[2415] + tmp313*weights[2416] + tmp314*weights[2418] + tmp315*weights[2419] + tmp316*weights[2422] + tmp317*weights[2424] + tmp318*weights[2426] + tmp319*weights[2438] + tmp32*weights[2441] + tmp320*weights[2440] + tmp321*weights[2444] + tmp322*weights[2445] + tmp323*weights[2449] + tmp324*weights[2452] + tmp325*weights[2453] + tmp326*weights[2455] + tmp327*weights[2457] + tmp328*weights[2460] + tmp329*weights[2467] + tmp33*weights[2442] + tmp330*weights[2469] + tmp331*weights[2479] + tmp34*weights[2463] + tmp35*weights[2473] + tmp38*weights[2486] + tmp41*weights[2490] + tmp44*weights[2518] + tmp46*weights[2530] + tmp48*weights[2533] + tmp51*weights[2617] + tmp52*weights[2634] + tmp54*weights[2653] + tmp55*weights[2655] + tmp56*weights[2667] + tmp57*weights[2674] + tmp59*weights[2694] + tmp60*weights[2703] + tmp61*weights[2433] + tmp62*weights[2458] + tmp63*weights[2477] + tmp64*weights[2488] + tmp65*weights[2504] + tmp66*weights[2511] + tmp67*weights[2520] + tmp68*weights[2526] + tmp69*weights[2528] + tmp70*weights[2536] + tmp71*weights[2585] + tmp72*weights[2594] + tmp73*weights[2600] + tmp74*weights[2601] + tmp75*weights[2609] + tmp76*weights[2622] + tmp77*weights[2623] + tmp78*weights[2641] + tmp79*weights[2650] + tmp80*weights[2663] + tmp81*weights[2669] + tmp82*weights[2688] + tmp83*weights[2695] + tmp84*weights[2713] + tmp85*weights[2454] + tmp86*weights[2470] + tmp87*weights[2471] + tmp88*weights[2480] + tmp89*weights[2484] + tmp90*weights[2487] + tmp91*weights[2489] + tmp92*weights[2500] + tmp93*weights[2508] + tmp94*weights[2510] + tmp95*weights[2512] + tmp97*weights[2513] + tmp99*weights[2514] + weights[2420]*z[13] + weights[2461]*z[11] + weights[2492]*z[1] + weights[2524]*z[5] + weights[2535]*z[4] + weights[2573]*z[0] + weights[2605]*z[3] + weights[2620]*z[12] + weights[2718]*z[6] + z[21];
    out[96] =  weights[2730];
    out[97] =  weights[2723];
    out[98] =  weights[2725];
    out[99] =  tmp11*weights[2752] + tmp13*weights[2754] + tmp2*weights[2736] + tmp5*weights[2739] + tmp7*weights[2740] + tmp9*weights[2745] + weights[2722];
    out[100] =  tmp11*weights[2737] + tmp14*weights[2742] + tmp15*weights[2743] + tmp16*weights[2750] + tmp17*weights[2751] + tmp5*weights[2732] + weights[2729];
    out[101] =  tmp15*weights[2744] + tmp17*weights[2749] + tmp18*weights[2731] + tmp19*weights[2738] + tmp2*weights[2746] + tmp7*weights[2748] + weights[2728];
    out[102] =  tmp13*weights[2735] + tmp14*weights[2734] + tmp16*weights[2747] + tmp18*weights[2741] + tmp19*weights[2733] + tmp9*weights[2753] + weights[2724];
    out[103] =  weights[2727];
    out[104] =  weights[2720];
    out[105] =  weights[2721];
    out[106] =  weights[2726];
    out[107] =  tmp100*weights[2857] + tmp101*weights[2861] + tmp103*weights[2862] + tmp104*weights[2863] + tmp105*weights[2869] + tmp106*weights[2874] + tmp107*weights[2877] + tmp108*weights[2878] + tmp109*weights[2880] + tmp110*weights[2881] + tmp111*weights[2891] + tmp113*weights[2894] + tmp114*weights[2895] + tmp115*weights[2899] + tmp116*weights[2900] + tmp117*weights[2901] + tmp118*weights[2903] + tmp119*weights[2905] + tmp120*weights[2906] + tmp121*weights[2907] + tmp122*weights[2909] + tmp123*weights[2911] + tmp124*weights[2914] + tmp125*weights[2919] + tmp126*weights[2921] + tmp127*weights[2924] + tmp128*weights[2929] + tmp129*weights[2930] + tmp130*weights[2931] + tmp131*weights[2933] + tmp132*weights[2936] + tmp133*weights[2937] + tmp134*weights[2943] + tmp135*weights[2948] + tmp136*weights[2950] + tmp137*weights[2951] + tmp138*weights[2952] + tmp139*weights[2954] + tmp140*weights[2956] + tmp141*weights[2959] + tmp142*weights[2964] + tmp143*weights[2965] + tmp144*weights[2967] + tmp145*weights[2968] + tmp146*weights[2970] + tmp147*weights[2971] + tmp148*weights[2972] + tmp149*weights[2973] + tmp150*weights[2975] + tmp151*weights[2976] + tmp152*weights[2984] + tmp153*weights[2985] + tmp154*weights[2987] + tmp155*weights[2989] + tmp156*weights[2991] + tmp157*weights[2992] + tmp158*weights[2998] + tmp159*weights[2999] + tmp160*weights[3001] + tmp161*weights[3006] + tmp162*weights[3008] + tmp163*weights[3011] + tmp164*weights[3012] + tmp165*weights[3013] + tmp166*weights[3020] + tmp167*weights[3022] + tmp168*weights[3023] + tmp169*weights[3025] + tmp170*weights[3026] + tmp171*weights[3027] + tmp172*weights[3030] + tmp173*weights[3032] + tmp174*weights[3033] + tmp175*weights[3036] + tmp176*weights[3037] + tmp177*weights[3039] + tmp178*weights[3040] + tmp179*weights[3052] + tmp180*weights[3054] + tmp181*weights[3055] + tmp182*weights[3057] + tmp183*weights[3059] + tmp184*weights[2761] + tmp185*weights[2763] + tmp186*weights[2765] + tmp187*weights[2767] + tmp188*weights[2768] + tmp189*weights[2769] + tmp190*weights[2770] + tmp191*weights[2771] + tmp192*weights[2775] + tmp193*weights[2776] + tmp194*weights[2779] + tmp195*weights[2783] + tmp196*weights[2786] + tmp197*weights[2787] + tmp198*weights[2788] + tmp199*weights[2790] + tmp20*weights[2834] + tmp200*weights[2791] + tmp201*weights[2796] + tmp202*weights[2799] + tmp203*weights[2802] + tmp204*weights[2804] + tmp205*weights[2806] + tmp206*weights[2808] + tmp207*weights[2812] + tmp208*weights[2814] + tmp209*weights[2815] + tmp21*weights[2837] + tmp210*weights[2816] + tmp211*weights[2818] + tmp212*weights[2822] + tmp213*weights[2825] + tmp214*weights[2843] + tmp215*weights[2846] + tmp216*weights[2883] + tmp217*weights[2884] + tmp218*weights[2887] + tmp219*weights[2888] + tmp22*weights[2867] + tmp220*weights[2902] + tmp221*weights[2904] + tmp222*weights[2915] + tmp223*weights[2917] + tmp224*weights[2920] + tmp225*weights[2928] + tmp226*weights[2939] + tmp227*weights[2955] + tmp228*weights[2986] + tmp229*weights[3010] + tmp23*weights[2946] + tmp230*weights[3047] + tmp231*weights[3056] + tmp232*weights[2757] + tmp233*weights[2772] + tmp234*weights[2774] + tmp235*weights[2805] + tmp236*weights[2821] + tmp237*weights[2823] + tmp238*weights[2831] + tmp239*weights[2833] + tmp24*weights[2947] + tmp240*weights[2835] + tmp241*weights[2836] + tmp242*weights[2838] + tmp243*weights[2839] + tmp244*weights[2841] + tmp245*weights[2842] + tmp246*weights[2845] + tmp247*weights[2847] + tmp248*weights[2849] + tmp249*weights[2855] + tmp25*weights[2977] + tmp250*weights[2856] + tmp251*weights[2859] + tmp252*weights[2865] + tmp253*weights[2871] + tmp254*weights[2872] + tmp255*weights[2879] + tmp256*weights[2882] + tmp257*weights[2885] + tmp258*weights[2886] + tmp259*weights[2889] + tmp26*weights[2978] + tmp260*weights[2890] + tmp261*weights[2892] + tmp262*weights[2893] + tmp263*weights[2896] + tmp264*weights[2897] + tmp265*weights[2898] + tmp266*weights[2908] + tmp267*weights[2910] + tmp268*weights[2912] + tmp269*weights[2916] + tmp27*weights[2988] + tmp270*weights[2918] + tmp271*weights[2922] + tmp272*weights[2923] + tmp273*weights[2926] + tmp274*weights[2927] + tmp275*weights[2932] + tmp276*weights[2935] + tmp277*weights[2938] + tmp278*weights[2942] + tmp279*weights[2944] + tmp28*weights[2996] + tmp280*weights[2953] + tmp281*weights[2958] + tmp282*weights[2961] + tmp283*weights[2966] + tmp284*weights[2969] + tmp285*weights[2979] + tmp286*weights[2980] + tmp287*weights[2982] + tmp288*weights[2983] + tmp289*weights[2994] + tmp29*weights[3017] + tmp290*weights[2997] + tmp291*weights[3000] + tmp292*weights[3002] + tmp293*weights[3004] + tmp294*weights[3005] + tmp295*weights[3015] + tmp296*weights[3016] + tmp297*weights[3018] + tmp298*weights[3021] + tmp299*weights[3024] + tmp30*weights[3019] + tmp300*weights[3029] + tmp301*weights[3031] + tmp302*weights[3038] + tmp303*weights[3041] + tmp304*weights[3042] + tmp305*weights[3044] + tmp306*weights[3045] + tmp307*weights[3046] + tmp308*weights[3048] + tmp309*weights[3049] + tmp31*weights[2777] + tmp310*weights[3050] + tmp311*weights[3051] + tmp312*weights[2755] + tmp313*weights[2756] + tmp314*weights[2758] + tmp315*weights[2759] + tmp316*weights[2762] + tmp317*weights[2764] + tmp318*weights[2766] + tmp319*weights[2778] + tmp32*weights[2781] + tmp320*weights[2780] + tmp321*weights[2784] + tmp322*weights[2785] + tmp323*weights[2789] + tmp324*weights[2792] + tmp325*weights[2793] + tmp326*weights[2795] + tmp327*weights[2797] + tmp328*weights[2800] + tmp329*weights[2807] + tmp33*weights[2782] + tmp330*weights[2809] + tmp331*weights[2819] + tmp34*weights[2803] + tmp35*weights[2813] + tmp38*weights[2826] + tmp41*weights[2830] + tmp44*weights[2858] + tmp46*weights[2870] + tmp48*weights[2873] + tmp51*weights[2957] + tmp52*weights[2974] + tmp54*weights[2993] + tmp55*weights[2995] + tmp56*weights[3007] + tmp57*weights[3014] + tmp59*weights[3034] + tmp60*weights[3043] + tmp61*weights[2773] + tmp62*weights[2798] + tmp63*weights[2817] + tmp64*weights[2828] + tmp65*weights[2844] + tmp66*weights[2851] + tmp67*weights[2860] + tmp68*weights[2866] + tmp69*weights[2868] + tmp70*weights[2876] + tmp71*weights[2925] + tmp72*weights[2934] + tmp73*weights[2940] + tmp74*weights[2941] + tmp75*weights[2949] + tmp76*weights[2962] + tmp77*weights[2963] + tmp78*weights[2981] + tmp79*weights[2990] + tmp80*weights[3003] + tmp81*weights[3009] + tmp82*weights[3028] + tmp83*weights[3035] + tmp84*weights[3053] + tmp85*weights[2794] + tmp86*weights[2810] + tmp87*weights[2811] + tmp88*weights[2820] + tmp89*weights[2824] + tmp90*weights[2827] + tmp91*weights[2829] + tmp92*weights[2840] + tmp93*weights[2848] + tmp94*weights[2850] + tmp95*weights[2852] + tmp97*weights[2853] + tmp99*weights[2854] + weights[2760]*z[13] + weights[2801]*z[11] + weights[2832]*z[1] + weights[2864]*z[5] + weights[2875]*z[4] + weights[2913]*z[0] + weights[2945]*z[3] + weights[2960]*z[12] + weights[3058]*z[6] + z[22];
    out[108] =  weights[3070];
    out[109] =  weights[3063];
    out[110] =  weights[3065];
    out[111] =  tmp11*weights[3092] + tmp13*weights[3094] + tmp2*weights[3076] + tmp5*weights[3079] + tmp7*weights[3080] + tmp9*weights[3085] + weights[3062];
    out[112] =  tmp11*weights[3077] + tmp14*weights[3082] + tmp15*weights[3083] + tmp16*weights[3090] + tmp17*weights[3091] + tmp5*weights[3072] + weights[3069];
    out[113] =  tmp15*weights[3084] + tmp17*weights[3089] + tmp18*weights[3071] + tmp19*weights[3078] + tmp2*weights[3086] + tmp7*weights[3088] + weights[3068];
    out[114] =  tmp13*weights[3075] + tmp14*weights[3074] + tmp16*weights[3087] + tmp18*weights[3081] + tmp19*weights[3073] + tmp9*weights[3093] + weights[3064];
    out[115] =  weights[3067];
    out[116] =  weights[3060];
    out[117] =  weights[3061];
    out[118] =  weights[3066];
    out[119] =  tmp100*weights[3197] + tmp101*weights[3201] + tmp103*weights[3202] + tmp104*weights[3203] + tmp105*weights[3209] + tmp106*weights[3214] + tmp107*weights[3217] + tmp108*weights[3218] + tmp109*weights[3220] + tmp110*weights[3221] + tmp111*weights[3231] + tmp113*weights[3234] + tmp114*weights[3235] + tmp115*weights[3239] + tmp116*weights[3240] + tmp117*weights[3241] + tmp118*weights[3243] + tmp119*weights[3245] + tmp120*weights[3246] + tmp121*weights[3247] + tmp122*weights[3249] + tmp123*weights[3251] + tmp124*weights[3254] + tmp125*weights[3259] + tmp126*weights[3261] + tmp127*weights[3264] + tmp128*weights[3269] + tmp129*weights[3270] + tmp130*weights[3271] + tmp131*weights[3273] + tmp132*weights[3276] + tmp133*weights[3277] + tmp134*weights[3283] + tmp135*weights[3288] + tmp136*weights[3290] + tmp137*weights[3291] + tmp138*weights[3292] + tmp139*weights[3294] + tmp140*weights[3296] + tmp141*weights[3299] + tmp142*weights[3304] + tmp143*weights[3305] + tmp144*weights[3307] + tmp145*weights[3308] + tmp146*weights[3310] + tmp147*weights[3311] + tmp148*weights[3312] + tmp149*weights[3313] + tmp150*weights[3315] + tmp151*weights[3316] + tmp152*weights[3324] + tmp153*weights[3325] + tmp154*weights[3327] + tmp155*weights[3329] + tmp156*weights[3331] + tmp157*weights[3332] + tmp158*weights[3338] + tmp159*weights[3339] + tmp160*weights[3341] + tmp161*weights[3346] + tmp162*weights[3348] + tmp163*weights[3351] + tmp164*weights[3352] + tmp165*weights[3353] + tmp166*weights[3360] + tmp167*weights[3362] + tmp168*weights[3363] + tmp169*weights[3365] + tmp170*weights[3366] + tmp171*weights[3367] + tmp172*weights[3370] + tmp173*weights[3372] + tmp174*weights[3373] + tmp175*weights[3376] + tmp176*weights[3377] + tmp177*weights[3379] + tmp178*weights[3380] + tmp179*weights[3392] + tmp180*weights[3394] + tmp181*weights[3395] + tmp182*weights[3397] + tmp183*weights[3399] + tmp184*weights[3101] + tmp185*weights[3103] + tmp186*weights[3105] + tmp187*weights[3107] + tmp188*weights[3108] + tmp189*weights[3109] + tmp190*weights[3110] + tmp191*weights[3111] + tmp192*weights[3115] + tmp193*weights[3116] + tmp194*weights[3119] + tmp195*weights[3123] + tmp196*weights[3126] + tmp197*weights[3127] + tmp198*weights[3128] + tmp199*weights[3130] + tmp20*weights[3174] + tmp200*weights[3131] + tmp201*weights[3136] + tmp202*weights[3139] + tmp203*weights[3142] + tmp204*weights[3144] + tmp205*weights[3146] + tmp206*weights[3148] + tmp207*weights[3152] + tmp208*weights[3154] + tmp209*weights[3155] + tmp21*weights[3177] + tmp210*weights[3156] + tmp211*weights[3158] + tmp212*weights[3162] + tmp213*weights[3165] + tmp214*weights[3183] + tmp215*weights[3186] + tmp216*weights[3223] + tmp217*weights[3224] + tmp218*weights[3227] + tmp219*weights[3228] + tmp22*weights[3207] + tmp220*weights[3242] + tmp221*weights[3244] + tmp222*weights[3255] + tmp223*weights[3257] + tmp224*weights[3260] + tmp225*weights[3268] + tmp226*weights[3279] + tmp227*weights[3295] + tmp228*weights[3326] + tmp229*weights[3350] + tmp23*weights[3286] + tmp230*weights[3387] + tmp231*weights[3396] + tmp232*weights[3097] + tmp233*weights[3112] + tmp234*weights[3114] + tmp235*weights[3145] + tmp236*weights[3161] + tmp237*weights[3163] + tmp238*weights[3171] + tmp239*weights[3173] + tmp24*weights[3287] + tmp240*weights[3175] + tmp241*weights[3176] + tmp242*weights[3178] + tmp243*weights[3179] + tmp244*weights[3181] + tmp245*weights[3182] + tmp246*weights[3185] + tmp247*weights[3187] + tmp248*weights[3189] + tmp249*weights[3195] + tmp25*weights[3317] + tmp250*weights[3196] + tmp251*weights[3199] + tmp252*weights[3205] + tmp253*weights[3211] + tmp254*weights[3212] + tmp255*weights[3219] + tmp256*weights[3222] + tmp257*weights[3225] + tmp258*weights[3226] + tmp259*weights[3229] + tmp26*weights[3318] + tmp260*weights[3230] + tmp261*weights[3232] + tmp262*weights[3233] + tmp263*weights[3236] + tmp264*weights[3237] + tmp265*weights[3238] + tmp266*weights[3248] + tmp267*weights[3250] + tmp268*weights[3252] + tmp269*weights[3256] + tmp27*weights[3328] + tmp270*weights[3258] + tmp271*weights[3262] + tmp272*weights[3263] + tmp273*weights[3266] + tmp274*weights[3267] + tmp275*weights[3272] + tmp276*weights[3275] + tmp277*weights[3278] + tmp278*weights[3282] + tmp279*weights[3284] + tmp28*weights[3336] + tmp280*weights[3293] + tmp281*weights[3298] + tmp282*weights[3301] + tmp283*weights[3306] + tmp284*weights[3309] + tmp285*weights[3319] + tmp286*weights[3320] + tmp287*weights[3322] + tmp288*weights[3323] + tmp289*weights[3334] + tmp29*weights[3357] + tmp290*weights[3337] + tmp291*weights[3340] + tmp292*weights[3342] + tmp293*weights[3344] + tmp294*weights[3345] + tmp295*weights[3355] + tmp296*weights[3356] + tmp297*weights[3358] + tmp298*weights[3361] + tmp299*weights[3364] + tmp30*weights[3359] + tmp300*weights[3369] + tmp301*weights[3371] + tmp302*weights[3378] + tmp303*weights[3381] + tmp304*weights[3382] + tmp305*weights[3384] + tmp306*weights[3385] + tmp307*weights[3386] + tmp308*weights[3388] + tmp309*weights[3389] + tmp31*weights[3117] + tmp310*weights[3390] + tmp311*weights[3391] + tmp312*weights[3095] + tmp313*weights[3096] + tmp314*weights[3098] + tmp315*weights[3099] + tmp316*weights[3102] + tmp317*weights[3104] + tmp318*weights[3106] + tmp319*weights[3118] + tmp32*weights[3121] + tmp320*weights[3120] + tmp321*weights[3124] + tmp322*weights[3125] + tmp323*weights[3129] + tmp324*weights[3132] + tmp325*weights[3133] + tmp326*weights[3135] + tmp327*weights[3137] + tmp328*weights[3140] + tmp329*weights[3147] + tmp33*weights[3122] + tmp330*weights[3149] + tmp331*weights[3159] + tmp34*weights[3143] + tmp35*weights[3153] + tmp38*weights[3166] + tmp41*weights[3170] + tmp44*weights[3198] + tmp46*weights[3210] + tmp48*weights[3213] + tmp51*weights[3297] + tmp52*weights[3314] + tmp54*weights[3333] + tmp55*weights[3335] + tmp56*weights[3347] + tmp57*weights[3354] + tmp59*weights[3374] + tmp60*weights[3383] + tmp61*weights[3113] + tmp62*weights[3138] + tmp63*weights[3157] + tmp64*weights[3168] + tmp65*weights[3184] + tmp66*weights[3191] + tmp67*weights[3200] + tmp68*weights[3206] + tmp69*weights[3208] + tmp70*weights[3216] + tmp71*weights[3265] + tmp72*weights[3274] + tmp73*weights[3280] + tmp74*weights[3281] + tmp75*weights[3289] + tmp76*weights[3302] + tmp77*weights[3303] + tmp78*weights[3321] + tmp79*weights[3330] + tmp80*weights[3343] + tmp81*weights[3349] + tmp82*weights[3368] + tmp83*weights[3375] + tmp84*weights[3393] + tmp85*weights[3134] + tmp86*weights[3150] + tmp87*weights[3151] + tmp88*weights[3160] + tmp89*weights[3164] + tmp90*weights[3167] + tmp91*weights[3169] + tmp92*weights[3180] + tmp93*weights[3188] + tmp94*weights[3190] + tmp95*weights[3192] + tmp97*weights[3193] + tmp99*weights[3194] + weights[3100]*z[13] + weights[3141]*z[11] + weights[3172]*z[1] + weights[3204]*z[5] + weights[3215]*z[4] + weights[3253]*z[0] + weights[3285]*z[3] + weights[3300]*z[12] + weights[3398]*z[6] + z[23];
    out[120] =  weights[3410];
    out[121] =  weights[3403];
    out[122] =  weights[3405];
    out[123] =  tmp11*weights[3432] + tmp13*weights[3434] + tmp2*weights[3416] + tmp5*weights[3419] + tmp7*weights[3420] + tmp9*weights[3425] + weights[3402];
    out[124] =  tmp11*weights[3417] + tmp14*weights[3422] + tmp15*weights[3423] + tmp16*weights[3430] + tmp17*weights[3431] + tmp5*weights[3412] + weights[3409];
    out[125] =  tmp15*weights[3424] + tmp17*weights[3429] + tmp18*weights[3411] + tmp19*weights[3418] + tmp2*weights[3426] + tmp7*weights[3428] + weights[3408];
    out[126] =  tmp13*weights[3415] + tmp14*weights[3414] + tmp16*weights[3427] + tmp18*weights[3421] + tmp19*weights[3413] + tmp9*weights[3433] + weights[3404];
    out[127] =  weights[3407];
    out[128] =  weights[3400];
    out[129] =  weights[3401];
    out[130] =  weights[3406];
    out[131] =  tmp100*weights[3537] + tmp101*weights[3541] + tmp103*weights[3542] + tmp104*weights[3543] + tmp105*weights[3549] + tmp106*weights[3554] + tmp107*weights[3557] + tmp108*weights[3558] + tmp109*weights[3560] + tmp110*weights[3561] + tmp111*weights[3571] + tmp113*weights[3574] + tmp114*weights[3575] + tmp115*weights[3579] + tmp116*weights[3580] + tmp117*weights[3581] + tmp118*weights[3583] + tmp119*weights[3585] + tmp120*weights[3586] + tmp121*weights[3587] + tmp122*weights[3589] + tmp123*weights[3591] + tmp124*weights[3594] + tmp125*weights[3599] + tmp126*weights[3601] + tmp127*weights[3604] + tmp128*weights[3609] + tmp129*weights[3610] + tmp130*weights[3611] + tmp131*weights[3613] + tmp132*weights[3616] + tmp133*weights[3617] + tmp134*weights[3623] + tmp135*weights[3628] + tmp136*weights[3630] + tmp137*weights[3631] + tmp138*weights[3632] + tmp139*weights[3634] + tmp140*weights[3636] + tmp141*weights[3639] + tmp142*weights[3644] + tmp143*weights[3645] + tmp144*weights[3647] + tmp145*weights[3648] + tmp146*weights[3650] + tmp147*weights[3651] + tmp148*weights[3652] + tmp149*weights[3653] + tmp150*weights[3655] + tmp151*weights[3656] + tmp152*weights[3664] + tmp153*weights[3665] + tmp154*weights[3667] + tmp155*weights[3669] + tmp156*weights[3671] + tmp157*weights[3672] + tmp158*weights[3678] + tmp159*weights[3679] + tmp160*weights[3681] + tmp161*weights[3686] + tmp162*weights[3688] + tmp163*weights[3691] + tmp164*weights[3692] + tmp165*weights[3693] + tmp166*weights[3700] + tmp167*weights[3702] + tmp168*weights[3703] + tmp169*weights[3705] + tmp170*weights[3706] + tmp171*weights[3707] + tmp172*weights[3710] + tmp173*weights[3712] + tmp174*weights[3713] + tmp175*weights[3716] + tmp176*weights[3717] + tmp177*weights[3719] + tmp178*weights[3720] + tmp179*weights[3732] + tmp180*weights[3734] + tmp181*weights[3735] + tmp182*weights[3737] + tmp183*weights[3739] + tmp184*weights[3441] + tmp185*weights[3443] + tmp186*weights[3445] + tmp187*weights[3447] + tmp188*weights[3448] + tmp189*weights[3449] + tmp190*weights[3450] + tmp191*weights[3451] + tmp192*weights[3455] + tmp193*weights[3456] + tmp194*weights[3459] + tmp195*weights[3463] + tmp196*weights[3466] + tmp197*weights[3467] + tmp198*weights[3468] + tmp199*weights[3470] + tmp20*weights[3514] + tmp200*weights[3471] + tmp201*weights[3476] + tmp202*weights[3479] + tmp203*weights[3482] + tmp204*weights[3484] + tmp205*weights[3486] + tmp206*weights[3488] + tmp207*weights[3492] + tmp208*weights[3494] + tmp209*weights[3495] + tmp21*weights[3517] + tmp210*weights[3496] + tmp211*weights[3498] + tmp212*weights[3502] + tmp213*weights[3505] + tmp214*weights[3523] + tmp215*weights[3526] + tmp216*weights[3563] + tmp217*weights[3564] + tmp218*weights[3567] + tmp219*weights[3568] + tmp22*weights[3547] + tmp220*weights[3582] + tmp221*weights[3584] + tmp222*weights[3595] + tmp223*weights[3597] + tmp224*weights[3600] + tmp225*weights[3608] + tmp226*weights[3619] + tmp227*weights[3635] + tmp228*weights[3666] + tmp229*weights[3690] + tmp23*weights[3626] + tmp230*weights[3727] + tmp231*weights[3736] + tmp232*weights[3437] + tmp233*weights[3452] + tmp234*weights[3454] + tmp235*weights[3485] + tmp236*weights[3501] + tmp237*weights[3503] + tmp238*weights[3511] + tmp239*weights[3513] + tmp24*weights[3627] + tmp240*weights[3515] + tmp241*weights[3516] + tmp242*weights[3518] + tmp243*weights[3519] + tmp244*weights[3521] + tmp245*weights[3522] + tmp246*weights[3525] + tmp247*weights[3527] + tmp248*weights[3529] + tmp249*weights[3535] + tmp25*weights[3657] + tmp250*weights[3536] + tmp251*weights[3539] + tmp252*weights[3545] + tmp253*weights[3551] + tmp254*weights[3552] + tmp255*weights[3559] + tmp256*weights[3562] + tmp257*weights[3565] + tmp258*weights[3566] + tmp259*weights[3569] + tmp26*weights[3658] + tmp260*weights[3570] + tmp261*weights[3572] + tmp262*weights[3573] + tmp263*weights[3576] + tmp264*weights[3577] + tmp265*weights[3578] + tmp266*weights[3588] + tmp267*weights[3590] + tmp268*weights[3592] + tmp269*weights[3596] + tmp27*weights[3668] + tmp270*weights[3598] + tmp271*weights[3602] + tmp272*weights[3603] + tmp273*weights[3606] + tmp274*weights[3607] + tmp275*weights[3612] + tmp276*weights[3615] + tmp277*weights[3618] + tmp278*weights[3622] + tmp279*weights[3624] + tmp28*weights[3676] + tmp280*weights[3633] + tmp281*weights[3638] + tmp282*weights[3641] + tmp283*weights[3646] + tmp284*weights[3649] + tmp285*weights[3659] + tmp286*weights[3660] + tmp287*weights[3662] + tmp288*weights[3663] + tmp289*weights[3674] + tmp29*weights[3697] + tmp290*weights[3677] + tmp291*weights[3680] + tmp292*weights[3682] + tmp293*weights[3684] + tmp294*weights[3685] + tmp295*weights[3695] + tmp296*weights[3696] + tmp297*weights[3698] + tmp298*weights[3701] + tmp299*weights[3704] + tmp30*weights[3699] + tmp300*weights[3709] + tmp301*weights[3711] + tmp302*weights[3718] + tmp303*weights[3721] + tmp304*weights[3722] + tmp305*weights[3724] + tmp306*weights[3725] + tmp307*weights[3726] + tmp308*weights[3728] + tmp309*weights[3729] + tmp31*weights[3457] + tmp310*weights[3730] + tmp311*weights[3731] + tmp312*weights[3435] + tmp313*weights[3436] + tmp314*weights[3438] + tmp315*weights[3439] + tmp316*weights[3442] + tmp317*weights[3444] + tmp318*weights[3446] + tmp319*weights[3458] + tmp32*weights[3461] + tmp320*weights[3460] + tmp321*weights[3464] + tmp322*weights[3465] + tmp323*weights[3469] + tmp324*weights[3472] + tmp325*weights[3473] + tmp326*weights[3475] + tmp327*weights[3477] + tmp328*weights[3480] + tmp329*weights[3487] + tmp33*weights[3462] + tmp330*weights[3489] + tmp331*weights[3499] + tmp34*weights[3483] + tmp35*weights[3493] + tmp38*weights[3506] + tmp41*weights[3510] + tmp44*weights[3538] + tmp46*weights[3550] + tmp48*weights[3553] + tmp51*weights[3637] + tmp52*weights[3654] + tmp54*weights[3673] + tmp55*weights[3675] + tmp56*weights[3687] + tmp57*weights[3694] + tmp59*weights[3714] + tmp60*weights[3723] + tmp61*weights[3453] + tmp62*weights[3478] + tmp63*weights[3497] + tmp64*weights[3508] + tmp65*weights[3524] + tmp66*weights[3531] + tmp67*weights[3540] + tmp68*weights[3546] + tmp69*weights[3548] + tmp70*weights[3556] + tmp71*weights[3605] + tmp72*weights[3614] + tmp73*weights[3620] + tmp74*weights[3621] + tmp75*weights[3629] + tmp76*weights[3642] + tmp77*weights[3643] + tmp78*weights[3661] + tmp79*weights[3670] + tmp80*weights[3683] + tmp81*weights[3689] + tmp82*weights[3708] + tmp83*weights[3715] + tmp84*weights[3733] + tmp85*weights[3474] + tmp86*weights[3490] + tmp87*weights[3491] + tmp88*weights[3500] + tmp89*weights[3504] + tmp90*weights[3507] + tmp91*weights[3509] + tmp92*weights[3520] + tmp93*weights[3528] + tmp94*weights[3530] + tmp95*weights[3532] + tmp97*weights[3533] + tmp99*weights[3534] + weights[3440]*z[13] + weights[3481]*z[11] + weights[3512]*z[1] + weights[3544]*z[5] + weights[3555]*z[4] + weights[3593]*z[0] + weights[3625]*z[3] + weights[3640]*z[12] + weights[3738]*z[6] + z[24];
    
    
    }

    VectorXd rk4(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    {
        VectorXd k1 = delta*f(x, u, weights);
        VectorXd k2 = delta*f(x + .5*k1, u, weights);
        VectorXd k3 = delta*f(x + .5*k2, u, weights);
        VectorXd k4 = delta*f(x + k3, u, weights);

        VectorXd x_new = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        return x_new;
    }

    VectorXd continuous_dynamics(VectorXd x, VectorXd u, double weights[])
    {
        double Mg[NX*(NX+1)];
        double z[NX+NU+NV];

        // state
        for(int i = 0; i < NX; ++i) {
            z[i] = x(i);
        }
        // controls
        for(int i = 0; i < NU; ++i) {
            z[i+NX] = u(i);
        }
        // virtual controls
        for(int i = 0; i < NV; ++i) {
            z[i+NX+NU] = u(i+NU);
        }

        //z[0] = x(0); z[1] = x(1); z[2] = x(2); z[3] = x(3); z[4] = u(0);

        feval(z, weights, Mg);

        MatrixXd M(NX,NX);
        VectorXd g(NX);

        int idx = 0;
        for(int i = 0; i < NX; ++i) {
            for(int j = 0; j < NX; ++j) {
                M(i,j) = Mg[idx++];
            }
            g(i) = Mg[idx++];
        }

        VectorXd xdot(NX);
        xdot = M.lu().solve(-g);

        return xdot;
    }

    MatrixXd numerical_jacobian(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[])
    {
        int nX = x.rows();
        int nU = u.rows();

        // Create matrix, set it to all zeros
        MatrixXd jac(nX, nX+nU+1);
        jac.setZero(nX, nX+nU+1);

        int index = 0;

        MatrixXd I;
        I.setIdentity(nX, nX);
        for(int i = 0; i < nX; ++i) {
            jac.col(index) = rk4(f, x + .5*EPS*I.col(i), u, delta, weights) - rk4(f, x - .5*EPS*I.col(i), u, delta, weights);
            index++;
        }

        I.setIdentity(nU, nU);
        for(int i = 0; i < nU; ++i) {
            jac.col(index) = rk4(f, x, u + .5*EPS*I.col(i), delta, weights) - rk4(f, x, u - .5*EPS*I.col(i), delta, weights);
            index++;
        }

        jac.col(index) = rk4(f, x, u, delta + .5*EPS, weights) - rk4(f, x, u, delta - .5*EPS, weights);

        // Must divide by eps for finite differences formula
        jac /= EPS;

        return jac;
    }

    VectorXd dynamics_difference(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd x_next, VectorXd u, double delta, double weights[])
    {
        VectorXd simulated_x_next = rk4(f, x, u, delta, weights);
        return x_next - simulated_x_next;
    }

};

#endif /* SWIMMER_DYNAMICS_H_ */

            