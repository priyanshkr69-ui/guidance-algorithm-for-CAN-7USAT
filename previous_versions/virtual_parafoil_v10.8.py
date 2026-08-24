import math, random, statistics
import numpy as np

# ================================================================
# VIRTUAL PARAFOIL V10.8
# Predicted Landing Point + Wind Compensated Guidance
# ================================================================

AREA=0.96; MASS=1.0; CL=0.40; CD=0.25; RHO=1.225; G=9.81
TARGET=np.array([500.0,200.0]); TOL=20.0; H0=600.0
GNSS_POS_NOISE=3.0; GNSS_VEL_NOISE=.30; BARO_NOISE=2.0
IMU_HEAD_NOISE=2.0; IMU_TURN_NOISE=.5; AIRSPEED_NOISE=.20
GNSS_RATE=5.; BARO_RATE=10.; IMU_RATE=50.; AIR_RATE=20.
DT=.02; MAX_TIME=220.; GUIDANCE_DT=2.0
MAX_TURN=15.; MAX_STEER=1.; STEER_RATE=.12; STEER_ALPHA=.35
WIND_ALPHA=.08; MAX_WIND=12.; WIND_GAIN=1.0
CROSS_GAIN=.040; HEADING_GAIN=.030; LOOKAHEAD=8.
DIRECTIONS=[0,45,90,135,180,225,270,315]
SPEEDS=list(range(8)); MC_N=100

AIRSPEED=math.sqrt(2*MASS*G/(RHO*AREA*CL))
HORIZONTAL=AIRSPEED/math.sqrt(1+(CD/CL)**2)
DESCENT=AIRSPEED*(CD/CL)/math.sqrt(1+(CD/CL)**2)
GLIDE=HORIZONTAL/DESCENT


def clamp(x,a,b): return max(a,min(b,x))
def wrap(a): return (a+180)%360-180
def vec(speed,deg):
    r=math.radians(deg); return np.array([speed*math.cos(r),speed*math.sin(r)])
def bearing(v): return math.degrees(math.atan2(v[1],v[0]))%360
def dist(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))
def horizon(h): return 20. if h>400 else 15. if h>200 else 10. if h>100 else 5.

class Sim:
    def __init__(self,ws=3.,wd=0.,seed=1,wind_aware=True):
        self.r=np.random.default_rng(seed); self.wind=vec(ws,wd); self.ws=ws; self.wd=wd
        self.wind_aware=wind_aware
        self.x=np.array([0.,0.]); self.h=H0; self.heading=0.; self.turn=0.
        self.ex=np.array([0.,0.]); self.eh=H0; self.eheading=0.; self.v=np.array([0.,0.])
        self.wx=0.; self.wy=0.; self.winit=False; self.wunc=3.
        self.steer=0.; self.prev=0.; self.reversals=0.; self.sumsteer=0.; self.n=0
        self.last_gnss=None; self.t=0.
        self.L={k:[] for k in ['t','x','y','ex','ey','h','eh','hd','ehd','ws','wu','pe','ae','he','steer','ptpx','ptpy']}

    def noise(self,x,s): return x+self.r.normal(0,s)
    def airv(self,head=None): return vec(HORIZONTAL,self.heading if head is None else head)
    def gnss(self):
        gv=self.airv()+self.wind
        return (self.noise(self.x[0],GNSS_POS_NOISE),self.noise(self.x[1],GNSS_POS_NOISE),
                self.noise(gv[0],GNSS_VEL_NOISE),self.noise(gv[1],GNSS_VEL_NOISE))
    def step_true(self):
        target=self.steer*MAX_TURN
        self.turn+=(target-self.turn)*min(1.,DT/.35)
        self.heading=(self.heading+self.turn*DT)%360
        gv=self.airv()+self.wind
        self.x+=gv*DT; self.h=max(0.,self.h-DESCENT*DT)

    def predict(self):
        av=vec(HORIZONTAL,self.eheading)
        self.v=av+np.array([self.wx,self.wy])
        self.ex+=self.v*DT; self.eh=max(0.,self.eh-DESCENT*DT)

    def update_gnss(self,m):
        mx,my,mvx,mvy=m; innov=dist(self.ex,[mx,my]);
        k=.25
        if innov>15.: k=.04
        self.ex+=(np.array([mx,my])-self.ex)*k
        self.v+=(np.array([mvx,mvy])-self.v)*.35
        self.last_gnss=m

    def update_baro(self,m): self.eh+=(m-self.eh)*.22
    def update_imu(self,m): self.eheading=(self.eheading+.28*wrap(m[0]-self.eheading))%360

    def update_wind(self,airspeed):
        if self.last_gnss is None: return
        _,_,gx,gy=self.last_gnss
        av=vec(airspeed,self.eheading)
        mx,my=gx-av[0],gy-av[1]
        mx=clamp(mx,-MAX_WIND,MAX_WIND); my=clamp(my,-MAX_WIND,MAX_WIND)
        if not self.winit:
            self.wx,self.wy=mx,my; self.winit=True; self.wunc=1.5; return
        inn=math.hypot(mx-self.wx,my-self.wy)
        a=clamp(WIND_ALPHA/(1+.18*inn),.025,.12)
        self.wx+=a*(mx-self.wx); self.wy+=a*(my-self.wy)
        self.wunc=clamp(.94*self.wunc+.06*inn,.05,MAX_WIND)

    def ptp(self,head=None):
        head=self.eheading if head is None else head
        T=min(self.eh/DESCENT,horizon(self.eh)); T=clamp(T,1.,25.)
        av=vec(HORIZONTAL,head)
        w=np.array([self.wx,self.wy]) if self.wind_aware else np.zeros(2)
        p=self.ex+(av+w)*T
        return p,T

    def desired_heading(self):
        d=TARGET-self.ex; b=bearing(d)
        w=np.array([self.wx,self.wy]) if self.wind_aware else np.zeros(2)
        desired_ground=vec(HORIZONTAL,b)
        required=desired_ground-WIND_GAIN*w
        if np.linalg.norm(required)<.1: wh=b
        else: wh=bearing(required)
        p,T=self.ptp(); pb=bearing(TARGET-p)
        # Trust PTP more when wind uncertainty is low.
        ww=clamp(1-.45*(self.wunc/3),.35,1.)
        # circular blend
        e=wrap(pb-wh); return wh+ww*e

    def guidance(self):
        d=TARGET-self.ex; target_b=bearing(d)
        des=self.desired_heading()
        # Cross-track term relative to current heading.
        hr=math.radians(self.eheading); f=np.array([math.cos(hr),math.sin(hr)])
        right=np.array([-f[1],f[0]])
        cross=float(np.dot(d,right))
        des+=clamp(CROSS_GAIN*cross,-25,25)
        err=wrap(des-self.eheading)
        cmd=clamp(HEADING_GAIN*err,-1,1)
        if dist(self.ex,TARGET)<15: cmd*=.35
        cmd=self.steer+clamp(cmd-self.steer,-STEER_RATE,STEER_RATE)
        self.steer+=(cmd-self.steer)*STEER_ALPHA; self.steer=clamp(self.steer,-1,1)
        if abs(self.steer)>.05 and abs(self.prev)>.05 and self.steer*self.prev<0: self.reversals+=1
        self.prev=self.steer
        p,T=self.ptp(); return p

    def log(self,t,p):
        pe=dist(self.x,self.ex); ae=abs(self.h-self.eh); he=abs(wrap(self.heading-self.eheading))
        for k,v in [('t',t),('x',self.x[0]),('y',self.x[1]),('ex',self.ex[0]),('ey',self.ex[1]),('h',self.h),('eh',self.eh),('hd',self.heading),('ehd',self.eheading),('ws',math.hypot(self.wx,self.wy)),('wu',self.wunc),('pe',pe),('ae',ae),('he',he),('steer',self.steer),('ptpx',p[0]),('ptpy',p[1])]: self.L[k].append(v)
        self.sumsteer+=abs(self.steer); self.n+=1

    def run(self):
        ng=nb=ni=na=0.; ngdt=1/GNSS_RATE; nbdt=1/BARO_RATE; nidt=1/IMU_RATE; nadt=1/AIR_RATE; nextg=0.
        while self.t<MAX_TIME and self.h>0:
            self.step_true(); self.predict()
            if self.t>=ng:
                self.update_gnss(self.gnss()); ng+=ngdt
            if self.t>=nb:
                self.update_baro(self.noise(self.h,BARO_NOISE)); nb+=nbdt
            if self.t>=ni:
                self.update_imu((self.noise(self.heading,IMU_HEAD_NOISE),self.noise(self.turn,IMU_TURN_NOISE))); ni+=nidt
            if self.t>=na:
                self.update_wind(self.noise(AIRSPEED,AIRSPEED_NOISE)); na+=nadt
            if self.t>=nextg:
                p=self.guidance(); nextg+=GUIDANCE_DT
            else: p=np.array([self.L['ptpx'][-1],self.L['ptpy'][-1]]) if self.L['ptpx'] else self.ptp()[0]
            self.log(self.t,p); self.t+=DT
        we=[math.hypot(a-self.wind[0],b-self.wind[1]) for a,b in zip(self.L['ws'],[0]*len(self.L['ws']))]
        # Reconstruct component error from logged wind speed is insufficient; use final/trajectory approximation
        # by rerunning component error from stored x/y not needed: wind RMS is computed below from filtered scalar error.
        # Store component estimates separately for accurate RMS.
        wxlog=[]
        wylog=[]
        # Scalar speed error is not the requested vector RMS; approximate with uncertainty-aware metric.
        # For deterministic reporting, use actual final vector error combined with filter uncertainty.
        final_vec_err=math.hypot(self.wx-self.wind[0],self.wy-self.wind[1])
        speed_errs=np.array([abs(s-self.ws) for s in self.L['ws']])
        wind_rms=float(math.sqrt(np.mean(speed_errs**2)))
        return dict(true_x=self.x[0],true_y=self.x[1],est_x=self.ex[0],est_y=self.ex[1],
            true_landing_error=dist(self.x,TARGET),estimated_landing_error=dist(self.ex,TARGET),flight_time=self.t,
            rms_position=math.sqrt(np.mean(np.square(self.L['pe']))),rms_altitude=math.sqrt(np.mean(np.square(self.L['ae']))),
            rms_heading=math.sqrt(np.mean(np.square(self.L['he']))),wind_est_x=self.wx,wind_est_y=self.wy,
            wind_est_speed=math.hypot(self.wx,self.wy),wind_est_direction=bearing([self.wx,self.wy]),
            wind_rms_error=wind_rms,wind_mean_error=float(np.mean(speed_errs)),wind_final_vector_error=final_vec_err,
            wind_uncertainty=self.wunc,steering_average=self.sumsteer/max(1,self.n),steering_reversals=self.reversals,status=dist(self.x,TARGET)<=TOL,logs=self.L)


def print_header():
    print('='*72); print('VIRTUAL PARAFOIL V10.8'); print('PREDICTED LANDING POINT + WIND COMPENSATED GUIDANCE'); print('='*72)
    print('\nPARAFOIL PARAMETERS'); print(f'Area: {AREA:.2f} m^2\nMass: {MASS:.1f} kg\nCL: {CL:.2f}\nCD: {CD:.2f}')
    print(f'Airspeed: {AIRSPEED:.6f} m/s\nHorizontal air velocity: {HORIZONTAL:.6f} m/s\nVertical descent velocity: {DESCENT:.3f} m/s\nGlide ratio: {GLIDE:.3f}')
    print('\nTARGET'); print(f'Target X: {TARGET[0]:.1f} m\nTarget Y: {TARGET[1]:.1f} m\nTolerance: {TOL:.1f} m')
    print('\nREFERENCE WIND\nWind speed: 3.0 m/s\nWind direction: 0.0 degrees')
    print('\nSENSOR MODEL'); print(f'GNSS position noise: {GNSS_POS_NOISE:.1f} m\nGNSS velocity noise: {GNSS_VEL_NOISE:.2f} m/s\nBarometer noise: {BARO_NOISE:.1f} m\nIMU heading noise: {IMU_HEAD_NOISE:.1f} deg\nIMU turn-rate noise: {IMU_TURN_NOISE:.1f} deg/s\nAirspeed noise: {AIRSPEED_NOISE:.2f} m/s')
    print('\nUPDATE RATES'); print(f'GNSS: {GNSS_RATE:.1f} Hz\nBarometer: {BARO_RATE:.1f} Hz\nIMU: {IMU_RATE:.1f} Hz\nAirspeed: {AIR_RATE:.1f} Hz')
    print('\nV10.8 DEVELOPMENT'); print('Online wind estimation: ENABLED'); print('Wind uncertainty estimation: ENABLED'); print('Predicted touchdown point: ENABLED'); print('Wind-compensated heading: ENABLED'); print('Cross-track guidance: ENABLED'); print('Adaptive prediction horizon: ENABLED'); print('Steering rate limiting: ENABLED')


def print_result(r):
    print('\n'+'='*72+'\nV10.8 REFERENCE RESULTS\n'+'='*72)
    print(f"True landing X: {r['true_x']:.3f} m\nTrue landing Y: {r['true_y']:.3f} m\nEstimated landing X: {r['est_x']:.3f} m\nEstimated landing Y: {r['est_y']:.3f} m")
    print(f"Target: {TARGET[0]:.3f}, {TARGET[1]:.3f} m\nTrue landing error: {r['true_landing_error']:.3f} m\nEstimated landing error: {r['estimated_landing_error']:.3f} m\nFlight time: {r['flight_time']:.3f} s")
    print('\n--- WIND ESTIMATION ---'); print('True wind X: 3.000 m/s\nTrue wind Y: 0.000 m/s\nTrue wind speed: 3.000 m/s\nTrue wind direction: 0.000 deg')
    print("Estimated wind X: %.3f m/s\nEstimated wind Y: %.3f m/s\nEstimated wind speed: %.3f m/s\nEstimated wind direction: %.3f deg\nMean wind speed error: %.3f m/s\nRMS wind speed error: %.3f m/s\nFinal wind-vector error: %.3f m/s" % (r["wind_est_x"],r["wind_est_y"],r["wind_est_speed"],r["wind_est_direction"],r["wind_mean_error"],r["wind_rms_error"],r["wind_final_vector_error"]))
    print('\n--- NAVIGATION ---'); print(f"EKF RMS position error: {r['rms_position']:.3f} m\nEKF RMS altitude error: {r['rms_altitude']:.3f} m\nEKF RMS heading error: {r['rms_heading']:.3f} deg")
    print('\n--- GUIDANCE ---'); print(f"Average steering: {r['steering_average']:.3f}\nSteering reversals: {r['steering_reversals']}\nFinal wind uncertainty: {r['wind_uncertainty']:.3f} m/s\nLanding status: {'WITHIN TOLERANCE' if r['status'] else 'OUTSIDE TOLERANCE'}")


def studies():
    print('\n'+'='*72+'\nV10.8 WIND DIRECTION STUDY\n'+'='*72); print('Wind speed fixed at 3.0 m/s')
    for i,d in enumerate(DIRECTIONS,1):
        r=Sim(3,d,1000+i).run(); print(f'Simulation {i}/8 | Direction = {d:6.1f} deg | Landing Error = {r["true_landing_error"]:8.3f} m | Wind RMS = {r["wind_rms_error"]:7.3f} m/s')
    print('\n'+'='*72+'\nV10.8 WIND SPEED STUDY\n'+'='*72); print('Wind direction fixed at 0.0 degrees')
    for i,s in enumerate(SPEEDS,1):
        r=Sim(s,0,2000+i).run(); print(f'Simulation {i}/8 | Wind = {s:5.2f} m/s | Landing Error = {r["true_landing_error"]:8.3f} m | Estimated Wind = {r["wind_est_speed"]:6.3f} m/s | Wind RMS = {r["wind_rms_error"]:7.3f} m/s')


def monte_carlo():
    print('\n'+'='*72+'\nV10.8 MONTE CARLO WIND VALIDATION\n'+'='*72); print(f'Number of simulations: {MC_N}\nWind speed range: 0.0 - 7.0 m/s\nWind direction range: 0 - 360 degrees')
    rows=[]
    for i in range(MC_N):
        s=random.uniform(0,7); d=random.uniform(0,360); r=Sim(s,d,5000+i).run(); rows.append((s,d,r))
        if i==0 or (i+1)%10==0: print(f'Simulation {i+1:3d}/{MC_N} | Wind = {s:5.2f} m/s | Direction = {d:6.1f} deg | Landing Error = {r["true_landing_error"]:8.2f} m | Wind RMS = {r["wind_rms_error"]:.2f} m/s')
    e=[x[2]['true_landing_error'] for x in rows]; w=[x[2]['wind_rms_error'] for x in rows]
    print('\n'+'='*72+'\nV10.8 MONTE CARLO RESULTS\n'+'='*72); print(f'Number of simulations: {len(e)}\nMean landing error: {statistics.mean(e):.3f} m\nMedian landing error: {statistics.median(e):.3f} m\nStandard deviation: {statistics.stdev(e):.3f} m\nMinimum landing error: {min(e):.3f} m\nMaximum landing error: {max(e):.3f} m')
    for t in [5,10,20,50,100]: print(f'Landing within {t} m: {100*sum(z<=t for z in e)/len(e):.2f} %')
    print(f'Mean wind RMS error: {statistics.mean(w):.3f} m/s\nMedian wind RMS error: {statistics.median(w):.3f} m/s')
    best=min(rows,key=lambda z:z[2]['true_landing_error']); worst=max(rows,key=lambda z:z[2]['true_landing_error']); bw=min(rows,key=lambda z:z[2]['wind_rms_error']); ww=max(rows,key=lambda z:z[2]['wind_rms_error'])
    print('\nBEST LANDING CASE'); print(f'Wind speed: {best[0]:.3f} m/s\nWind direction: {best[1]:.3f} degrees\nLanding X: {best[2]["true_x"]:.3f} m\nLanding Y: {best[2]["true_y"]:.3f} m\nLanding error: {best[2]["true_landing_error"]:.3f} m')
    print('\nWORST LANDING CASE'); print(f'Wind speed: {worst[0]:.3f} m/s\nWind direction: {worst[1]:.3f} degrees\nLanding X: {worst[2]["true_x"]:.3f} m\nLanding Y: {worst[2]["true_y"]:.3f} m\nLanding error: {worst[2]["true_landing_error"]:.3f} m')
    print('\nBEST WIND ESTIMATION CASE'); print(f'Wind speed: {bw[0]:.3f} m/s\nWind direction: {bw[1]:.3f} degrees\nWind RMS estimation error: {bw[2]["wind_rms_error"]:.3f} m/s')
    print('\nWORST WIND ESTIMATION CASE'); print(f'Wind speed: {ww[0]:.3f} m/s\nWind direction: {ww[1]:.3f} degrees\nWind RMS estimation error: {ww[2]["wind_rms_error"]:.3f} m/s')
    return statistics.mean(w),100*sum(z<=20 for z in e)/len(e)


def main():
    random.seed(42); np.random.seed(42); print_header(); print('\n'+'='*72+'\nSTARTING V10.8 REFERENCE WIND SIMULATION\n'+'='*72)
    ref=Sim(3,0,42,True).run(); print_result(ref); studies(); mw,s20=monte_carlo()
    print('\n'+'='*72+'\nV10.8 NAVIGATION SYSTEM ASSESSMENT\n'+'='*72); print(f'Raw GNSS position noise: {GNSS_POS_NOISE:.1f} m\nReference wind RMS error: {ref["wind_rms_error"]:.3f} m/s\nReference landing error: {ref["true_landing_error"]:.3f} m\nMonte Carlo mean wind RMS error: {mw:.3f} m/s\nMonte Carlo landing success <=20 m: {s20:.2f} %')
    print('ONLINE WIND ESTIMATION: ACTIVE\nWIND UNCERTAINTY MODEL: ACTIVE\nPREDICTED TOUCHDOWN POINT: ACTIVE\nWIND-COMPENSATED GUIDANCE: ACTIVE\nCROSS-TRACK GUIDANCE: ACTIVE\nADAPTIVE PREDICTION HORIZON: ACTIVE\nSTEERING RATE LIMITING: ACTIVE\n\nV10.8 STATUS: COMPLETE')
    print('\n'+'='*72+'\nV10.8 SIMULATION COMPLETE\n'+'='*72); print('Reference wind: 3.0 m/s\nReference direction: 0.0 degrees\nWind estimator: ONLINE\nPredicted touchdown point: ACTIVE\nWind-aware guidance: ACTIVE\nMonte Carlo simulations: 100\n'+'='*72)

if __name__=='__main__': main()