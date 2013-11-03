from planning import *
import pylab as plt


def rotation_utils():
    return """
        __device__ {{ dtype }}4 get_quat({{ dtype }}3 r){            

            {{ dtype }} rotation_angle = sqrt(r.x*r.x + r.y*r.y+ r.z*r.z);

            {{ dtype }} s = sin(.5*rotation_angle), c = cos(.5*rotation_angle);
            {{ dtype }} t1 = s / (rotation_angle==0 ? 1.0 : rotation_angle);
            {{ dtype }}4 o;
            o.x = r.x * t1;
            o.y = r.y * t1;
            o.z = r.z * t1;
            o.w = c;
            return o;
            }

        __device__ {{ dtype }}3 drot({{ dtype }}3 r, {{ dtype }}4 dq){          

            {{ dtype }} phi = sqrt(r.x*r.x + r.y*r.y+ r.z*r.z);
            {{ dtype }} s = sin(.5*phi), c=cos(.5*phi);
            {{ dtype }}3 dr, e, de;            
            e.x = r.x/ phi; e.y = r.y/ phi; e.z = r.z/ phi;
            
            {{ dtype }} dphi = - 2.0* dq.w/s; 
            {{ dtype }} t = .5*c * dphi;
            de.x = (dq.x - e.x * t)/s;
            de.y = (dq.y - e.y * t)/s;
            de.z = (dq.z - e.z * t)/s;
            
            dr.x = phi* de.x + dphi* e.x;
            dr.y = phi* de.y + dphi* e.y;
            dr.z = phi* de.z + dphi* e.z;

            return dr;
            }


        __device__ {{ dtype }}4 quad_mult({{ dtype }}4 a, {{ dtype }}4 b){
            {{ dtype }}4 c;

            c.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
            c.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
            c.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
            c.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;

            return c; 
            }
    """

    

class Heli(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self, 12, 4, 
            control_bounds = [[-1.0,-1.0,-1.0,-1.0],[1.0,1.0,1.0,1.0]])

        start = np.zeros(self.nx)
        
        Environment.__init__(self, start, .01, noise = noise)
        
        u_weights = (13.20, -9.21, 14.84, -27.5)
        friction = (-3.47, -3.06, -2.58, -.048, -.12, -.0005)
        g = 9.81
        
        self.k_f = self.generate_k_f(u_weights,friction,g)
        
    def generate_k_f(self, u_weights, friction,g):
        
        tpl = Template( rotation_utils() +
        """
            
        __device__ void f(
                {{ dtype }} s[],
                {{ dtype }} u[], 
                {{ dtype }} ds[]){

            // s = angular velocity (heli frame), velocity (lab frame), 
            // rotation (Euler axis times angle), position

            // recover quaternion
            {{ dtype }}4 q, qi;
            {{ dtype }}3 rot;
            rot.x = s[6], rot.y = s[7], rot.z=s[8];
            q = get_quat(rot);  
            // inverse quaternion
            qi.x = -q.x; qi.y = -q.y; qi.z = -q.z; qi.w = q.w; 

            // velocity from lab frame to heli frame
            {{ dtype }}4 vl, vh;
            vl.x = s[3]; vl.y = s[4]; vl.z = s[5]; vl.w=0;
            vh = quad_mult(quad_mult(qi,vl),q);

            ds[9] = vl.x; ds[10] = vl.y; ds[11] = vl.z;

            // accels in heli frame 
            {{ dtype }}4 dvh, dvl; 
            dvh.x = vh.x * {{ fw[3] }};
            dvh.y = vh.y * {{ fw[4] }};
            dvh.z = vh.z * {{ fw[5] }} + u[3] * {{ uw[3] }};
            dvh.w = 0;
            // accels in lab frame
            dvl = quad_mult(quad_mult(q,dvh),qi);
            dvl.z += {{ g }};
            
            ds[3] = dvl.x; ds[4] = dvl.y; ds[5] = dvl.z; 

            // angular velocity in heli frame
            {{ dtype }}4 w; 
            w.x = s[0]; w.y = s[1]; w.z = s[2]; w.w = 0;

            // angular accel
            {{ dtype }}3 dw; 
            dw.x = w.x * {{ fw[0] }} + u[0] * {{ uw[0] }};
            dw.y = w.y * {{ fw[1] }} + u[1] * {{ uw[1] }};
            dw.z = w.z * {{ fw[2] }} + u[2] * {{ uw[2] }};
        
            ds[0] = dw.x; ds[1] = dw.y; ds[2] = dw.z;

            // rotation 
            {{ dtype }}4 dq;
            {{ dtype }}3 dr; 
            
            if (q.w==1.0) {
                ds[6] = w.x; ds[7] = w.y; ds[8] = w.z;
                return;
            }

            dq = quad_mult(w,q);
            dq.x *= .5; dq.y *= .5; dq.z *= .5; dq.w *= .5;            
            dr = drot(rot, dq);
            
            ds[6] = dr.x; ds[7] = dr.y; ds[8] = dr.z;

        }
        """
        )


        fn = tpl.render(
                uw = u_weights,
                fw = friction,
                g = g,
                dtype = cuda_dtype)

        return rowwise(fn,'heli')


class OptimisticHeli(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        if isinstance(predictor,object):
            predictor =  predictor(6+4+6)

        OptimisticDynamicalSystem.__init__(self, 12, 4, 
            [[-1.0,-1.0,-1.0,-1.0],[1.0,1.0,1.0,1.0]],
            6, predictor, **kwargs)

        g = 9.81
        self.k_f       = self.generate_k_f(g)
        self.k_pred_in = self.generate_k_pred_in()
        self.k_update  = self.generate_k_update(g)

    def generate_k_update(self,g):

        # kernel input : dx,x,u
        # kernel output: o vector used to update predictor

        tpl = Template( rotation_utils() +
        """
        __device__ void f(
                {{ dtype }} ds[],
                {{ dtype }}  s[], 
                {{ dtype }}  u[],
                {{ dtype }}  o[]
                ){

            // ds = time derivative of s
            // s = angular velocity (heli frame), velocity (lab frame), 
            // rotation (Euler axis times angle), position
            // u = controls
            // o = angular acceleration (heli frame), acceleration (heli frame),
            // angular velocity (heli frame), velocity (heli frame), controls

            // recover quaternion
            {{ dtype }}4 q, qi;
            {{ dtype }}3 rot;
            rot.x = s[6], rot.y = s[7], rot.z=s[8];
            q = get_quat(rot);  
            // inverse quaternion
            qi.x = -q.x; qi.y = -q.y; qi.z = -q.z; qi.w = q.w; 

            // velocity from lab frame to heli frame
            {{ dtype }}4 vl, vh;
            vl.x = s[3]; vl.y = s[4]; vl.z = s[5]; vl.w=0;
            vh = quad_mult(quad_mult(qi,vl),q);

            // accels to heli frame 
            {{ dtype }}4 dvl, dvh;
            dvl.x = ds[3]; dvl.y = ds[4]; dvl.z = ds[5]; dvl.w=0;
            dvl.z -= {{ g }};
            dvh = quad_mult(quad_mult(qi,dvl),q);

            // angular velocity in heli frame
            {{ dtype }}3 w; 
            w.x = s[0]; w.y = s[1]; w.z = s[2];

            // angular accel in heli frame
            {{ dtype }}3 dw; 
            dw.x = ds[0]; dw.y = ds[1]; dw.z = ds[2];

            // write out results 
            o[0] = dw.x; o[1] = dw.y; o[2] = dw.z;
            o[3] = dvh.x; o[4] = dvh.y; o[5] = dvh.z;
            o[6] = w.x; o[7] = w.y; o[8] = w.z;
            o[9] = vh.x; o[10] = vh.y; o[11] = vh.z;
            o[12] = u[0]; o[13] = u[1]; o[14] = u[2]; o[15]= u[3];
        }
        """
        )


        fn = tpl.render(g=g, dtype = cuda_dtype)

        return rowwise(fn,'heli_update')


    def generate_k_pred_in(self):


        tpl = Template( rotation_utils() +
        """
        __device__ void f(
                {{ dtype }}  s[],
                {{ dtype }}  u[], 
                {{ dtype }}  o[],
                {{ dtype }} xi[]
                ){

            // s  = angular velocity (heli frame), velocity (lab frame), 
            // rotation (Euler axis times angle), position
            // u  = controls, pseudo-controls
            // o  = out: angular velocity (heli frame), velocity (heli frame), 
            // controls
            // xi = out: pseudo controls

            // recover quaternion
            {{ dtype }}4 q, qi;
            {{ dtype }}3 rot;
            rot.x = s[6], rot.y = s[7], rot.z=s[8];
            q = get_quat(rot);  
            // inverse quaternion
            qi.x = -q.x; qi.y = -q.y; qi.z = -q.z; qi.w = q.w; 

            // velocity from lab frame to heli frame
            {{ dtype }}4 vl, vh;
            vl.x = s[3]; vl.y = s[4]; vl.z = s[5]; vl.w=0;
            vh = quad_mult(quad_mult(qi,vl),q);

            // angular velocity in heli frame
            {{ dtype }}3 w; 
            w.x = s[0]; w.y = s[1]; w.z = s[2];

            // write out results 
            o[0] = w.x; o[1] = w.y; o[2] = w.z;
            o[3] = vh.x; o[4] = vh.y; o[5] = vh.z;
            o[6] = u[0]; o[7] = u[1]; o[8] = u[2]; o[9]= u[3];
            
            for (int i=0;i<6;i++) xi[i] = u[4+i]; 
        }
        """
        )


        fn = tpl.render(dtype = cuda_dtype)

        return rowwise(fn,'heli_pred_in', output_inds=(2,3))


    def generate_k_f(self,g):
        
        tpl = Template( rotation_utils() +
        """
            
        __device__ void f(
                {{ dtype }} s[],
                {{ dtype }} o[],
                {{ dtype }} u[], 
                {{ dtype }} ds[]){

            // s  = angular velocity (heli frame), velocity (lab frame), 
            // rotation (Euler axis times angle), position
            // o  = predicted angular acceleration, velocity (both heli frame)
            // u  = controls
            // ds = state derivative to fill in

            // recover quaternion
            {{ dtype }}4 q, qi;
            {{ dtype }}3 rot;
            rot.x = s[6], rot.y = s[7], rot.z=s[8];
            q = get_quat(rot);  
            // inverse quaternion
            qi.x = -q.x; qi.y = -q.y; qi.z = -q.z; qi.w = q.w; 

            // velocity in lab frame
            {{ dtype }}4 vl;
            vl.x = s[3]; vl.y = s[4]; vl.z = s[5]; vl.w=0;

            ds[9] = vl.x; ds[10] = vl.y; ds[11] = vl.z;

            // accels in heli frame 
            {{ dtype }}4 dvh, dvl; 
            dvh.x = o[3]; dvh.y = o[4]; dvh.z = o[5];
            dvh.w = 0;
            // accels in lab frame
            dvl = quad_mult(quad_mult(q,dvh),qi);
            dvl.z += {{ g }};
            
            ds[3] = dvl.x; ds[4] = dvl.y; ds[5] = dvl.z; 

            // angular velocity in heli frame
            {{ dtype }}4 w; 
            w.x = s[0]; w.y = s[1]; w.z = s[2]; w.w = 0;

            // angular accel
            {{ dtype }}3 dw; 
            dw.x = o[0]; dw.y = o[1]; dw.z = o[2];
        
            ds[0] = dw.x; ds[1] = dw.y; ds[2] = dw.z;

            // rotation 
            {{ dtype }}4 dq;
            {{ dtype }}3 dr; 
            
            if (q.w==1.0) {
                ds[6] = w.x; ds[7] = w.y; ds[8] = w.z;
                return;
            }

            dq = quad_mult(w,q);
            dq.x *= .5; dq.y *= .5; dq.z *= .5; dq.w *= .5;            
            dr = drot(rot, dq);
            
            ds[6] = dr.x; ds[7] = dr.y; ds[8] = dr.z;

        }
        """
        )


        fn = tpl.render(g=g, dtype = cuda_dtype)

        return rowwise(fn,'heli_model_f')


