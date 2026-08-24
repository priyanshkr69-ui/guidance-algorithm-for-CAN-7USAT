import math, random, statistics
from dataclasses import dataclass

# ================================================================
# VIRTUAL PARAFOIL V10.6
# ONLINE WIND ESTIMATION + ADAPTIVE WIND VECTOR
# Corrected: result-return bug, consistent wind-vector convention,
# bounded estimator, sensor-health weighting, deterministic studies.
# ================================================================

@dataclass
class Params:
    AREA: float = 0.96
    MASS: float = 1.0
    CL: float = 0.40
    CD: float = 0.25
    RHO: float = 1.225
    G: float = 9.81
    TARGET_X: float = 500.0
    TARGET_Y: float = 200.0
    TOL: float = 20.0
    START_ALT: float = 600.0
    WIND_SPEED: float = 3.0
    WIND_DIR_DEG: float = 0.0
    GNSS_POS_NOISE: float = 3.0
    GNSS_VEL_NOISE: float = 0.30
    BARO_NOISE: float = 2.0
    IMU_HEADING_NOISE_DEG: float = 2.0
    IMU_TURN_NOISE_DEG_S: float = 0.5
    AIRSPEED_NOISE: float = 0.20
    GNSS_HZ: float = 5.0
    BARO_HZ: float = 10.0
    IMU_HZ: float = 50.0
    AIRSPEED_HZ: float = 20.0
    WIND_ALPHA: float = 0.08
    MAX_WIND_EST: float = 12.0
    GUIDANCE_DT: float = 2.0
    CANDIDATES: int = 21
    MAX_TURN_DEG_S: float = 15.0
    DT: float = 0.02
    MAX_TIME: float = 240.0
    SEED: int = 106

P = Params()


def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_deg(a): return (a + 180.0) % 360.0 - 180.0
def norm2(x, y): return math.hypot(x, y)
def angle_deg(x, y): return math.degrees(math.atan2(y, x)) % 360.0

def wind_xy(speed, direction_deg):
    r = math.radians(direction_deg)
    return speed * math.cos(r), speed * math.sin(r)

def airspeed_from_params():
    # Horizontal airspeed from lift/drag ratio approximation used by prior versions.
    # Keep the model internally consistent: glide ratio = CL/CD and horizontal
    # speed is selected to produce the configured descent rate.
    vz = 3.717
    glide = P.CL / P.CD
    vh = vz * glide
    return math.hypot(vh, vz), vh, vz, glide

AIRSPEED, V_AIR_H, V_DESC, GLIDE = airspeed_from_params()

# ---------------------------------------------------------------
# Sensor-health model
# ---------------------------------------------------------------
class Health:
    def __init__(self):
        self.gnss = self.imu = self.baro = self.airspeed = 1.0
    def update(self, name, good):
        h = getattr(self, name)
        # Slow degradation, quicker recovery.
        h += (0.025 if good else -0.012)
        setattr(self, name, clamp(h, 0.0, 1.0))

# ---------------------------------------------------------------
# Simple EKF-like navigation filter
# State: x,y,vx,vy,z,heading,turn_rate
# This is deliberately self-contained (no numpy/scipy dependency).
# ---------------------------------------------------------------
class NavigationFilter:
    def __init__(self):
        self.x = [0.0, 0.0, 0.0, 0.0, P.START_ALT, 0.0, 0.0]
        self.var = [25.0,25.0,4.0,4.0,16.0,math.radians(10)**2,math.radians(3)**2]
        self.health = Health()
        self.last_gnss = None
        self.last_baro = None
        self.last_imu = None
        self.pos_errors=[]; self.alt_errors=[]; self.head_errors=[]
        self.max_pos_error=0.0

    def predict(self, dt, air_heading=None):
        x,y,vx,vy,z,h,tr = self.x
        if air_heading is None: air_heading = h
        # Mild process model. Velocity is allowed to evolve from heading and
        # an air-relative speed estimate; wind is handled by guidance.
        self.x[0] += vx*dt
        self.x[1] += vy*dt
        self.x[4] = max(0.0, z - V_DESC*dt)
        self.x[5] = wrap_rad(h + tr*dt)
        # Uncertainty growth.
        self.var[0] += 0.12*dt + 0.02*dt*dt
        self.var[1] += 0.12*dt + 0.02*dt*dt
        self.var[2] += 0.04*dt
        self.var[3] += 0.04*dt
        self.var[4] += 0.20*dt
        self.var[5] += math.radians(0.08)**2 * dt
        self.var[6] += math.radians(0.15)**2 * dt

    def _scalar_update(self, idx, measurement, R, health, threshold_sigma=5.0):
        if health < 0.05: return False
        R_eff = R / max(health, 0.08)
        innovation = measurement - self.x[idx]
        sigma = math.sqrt(max(self.var[idx] + R_eff, 1e-12))
        if abs(innovation) > threshold_sigma*sigma:
            return False
        K = self.var[idx]/(self.var[idx]+R_eff)
        self.x[idx] += K*innovation
        self.var[idx] *= (1-K)
        return True

    def update_gnss(self, mx,my,mvx,mvy):
        # Position innovation gate in metres; velocity gate in m/s.
        hp=self.health.gnss
        accepted=True
        for idx,m in ((0,mx),(1,my),(2,mvx),(3,mvy)):
            R = P.GNSS_POS_NOISE**2 if idx<2 else P.GNSS_VEL_NOISE**2
            gate = 5.0 if idx<2 else 6.0
            ok=self._scalar_update(idx,m,R,hp,gate)
            accepted = accepted and ok
        self.health.update('gnss', accepted)
        return accepted

    def update_baro(self,mz):
        ok=self._scalar_update(4,mz,P.BARO_NOISE**2,self.health.baro,5.0)
        self.health.update('baro',ok)
        return ok

    def update_imu(self,mh,mtr):
        # Circular heading innovation.
        hp=self.health.imu
        innov=math.radians(wrap_deg(math.degrees(mh-self.x[5])))
        R=math.radians(P.IMU_HEADING_NOISE_DEG)**2/max(hp,0.08)
        sig=math.sqrt(self.var[5]+R)
        ok=abs(innov) <= math.radians(8.0) + 5*sig
        if ok:
            K=self.var[5]/(self.var[5]+R)
            self.x[5]=wrap_rad(self.x[5]+K*innov); self.var[5]*=(1-K)
            self.x[6]=self.x[6] + 0.15*(mtr-self.x[6])
        self.health.update('imu',ok)
        return ok

    def update_airspeed(self, measured):
        # Airspeed is scalar and mainly informs wind estimation; store health.
        good=abs(measured-AIRSPEED) < max(3*P.AIRSPEED_NOISE,1.0)
        self.health.update('airspeed',good)
        return good

    def log_error(self,true):
        pe=norm2(self.x[0]-true[0],self.x[1]-true[1])
        ae=abs(self.x[4]-true[2])
        he=abs(wrap_deg(math.degrees(self.x[5]-true[3])))
        self.pos_errors.append(pe); self.alt_errors.append(ae); self.head_errors.append(he)
        self.max_pos_error=max(self.max_pos_error,pe)


def wrap_rad(a): return (a+math.pi)%(2*math.pi)-math.pi

# ---------------------------------------------------------------
# Wind estimator
# ---------------------------------------------------------------
class WindEstimator:
    def __init__(self):
        self.wx=0.0; self.wy=0.0
        self.samples=[]; self.errors=[]
    def estimate(self, ground_vx, ground_vy, heading_rad, airspeed, weight=1.0):
        # Ground velocity = air-relative velocity + wind.
        ax=airspeed*math.cos(heading_rad); ay=airspeed*math.sin(heading_rad)
        rawx=ground_vx-ax; rawy=ground_vy-ay
        mag=norm2(rawx,rawy)
        if mag>P.MAX_WIND_EST:
            scale=P.MAX_WIND_EST/mag; rawx*=scale; rawy*=scale
        alpha=clamp(P.WIND_ALPHA*clamp(weight,0.1,1.0),0.01,0.20)
        self.wx=(1-alpha)*self.wx+alpha*rawx
        self.wy=(1-alpha)*self.wy+alpha*rawy
        return self.wx,self.wy
    def log_error(self,tx,ty): self.errors.append(norm2(self.wx-tx,self.wy-ty))
    @property
    def speed(self): return norm2(self.wx,self.wy)
    @property
    def direction(self): return angle_deg(self.wx,self.wy)

# ---------------------------------------------------------------
# Parafoil dynamics + guidance
# ---------------------------------------------------------------
class Vehicle:
    def __init__(self, windx, windy):
        self.x=0.0; self.y=0.0; self.z=P.START_ALT
        self.heading=0.0; self.turn_rate=0.0
        self.windx=windx; self.windy=windy
        self.steer=0.0
        self.steer_hist=[]
    def step(self,dt,command):
        maxr=math.radians(P.MAX_TURN_DEG_S)
        desired=command*maxr
        tau=0.55
        self.turn_rate += (desired-self.turn_rate)*clamp(dt/tau,0,1)
        self.turn_rate=clamp(self.turn_rate,-maxr,maxr)
        self.heading=wrap_rad(self.heading+self.turn_rate*dt)
        # Small steering-dependent airspeed loss, capped for realism.
        vh=V_AIR_H*(1.0-0.08*abs(command))
        avx=vh*math.cos(self.heading); avy=vh*math.sin(self.heading)
        self.x += (avx+self.windx)*dt
        self.y += (avy+self.windy)*dt
        self.z=max(0.0,self.z-V_DESC*dt)
        self.steer_hist.append(command)


def horizon(z):
    if z>400:return 20.0
    if z>200:return 15.0
    if z>100:return 10.0
    return 5.0


def predict_endpoint(x,y,z,heading,command,windx,windy):
    t=min(horizon(z), max(z/V_DESC,1.0))
    n=max(1,int(t/0.5)); dt=t/n
    h=heading; px=x; py=y
    tr=math.radians(P.MAX_TURN_DEG_S)*command
    vh=V_AIR_H*(1-0.08*abs(command))
    for _ in range(n):
        h=wrap_rad(h+tr*dt)
        px+=(vh*math.cos(h)+windx)*dt
        py+=(vh*math.sin(h)+windy)*dt
    return px,py


def choose_command(state,wind):
    x,y,_,_,z,h,_=state
    tx,ty=P.TARGET_X,P.TARGET_Y
    # Candidate steering values span full authority.
    vals=[-1.0+2.0*i/(P.CANDIDATES-1) for i in range(P.CANDIDATES)]
    best=None
    for c in vals:
        px,py=predict_endpoint(x,y,z,h,c,wind[0],wind[1])
        err=norm2(px-tx,py-ty)
        # Penalize excessive steering and target overshoot near ground.
        score=err+2.0*abs(c)+0.015*abs(c-state[6]/math.radians(P.MAX_TURN_DEG_S))
        if best is None or score<best[0]: best=(score,c)
    return best[1]

# ---------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------
def simulate(wind_speed,wind_dir_deg,seed=1,return_history=False):
    rng=random.Random(seed)
    wx,wy=wind_xy(wind_speed,wind_dir_deg)
    v=Vehicle(wx,wy); ekf=NavigationFilter(); we=WindEstimator()
    t=0.0; next_guid=0.0; gnss_count=baro_count=imu_count=air_count=0
    gnss_rej=0; gnss_drop=0; steering=[]; history=[]
    while v.z>0.0 and t<P.MAX_TIME:
        # Truth sensor values.
        gvx=V_AIR_H*math.cos(v.heading)+wx
        gvy=V_AIR_H*math.sin(v.heading)+wy
        mh=wrap_rad(v.heading+math.radians(rng.gauss(0,P.IMU_HEADING_NOISE_DEG)))
        mtr=v.turn_rate+math.radians(rng.gauss(0,P.IMU_TURN_NOISE_DEG_S))
        mx=v.x+rng.gauss(0,P.GNSS_POS_NOISE); my=v.y+rng.gauss(0,P.GNSS_POS_NOISE)
        mvx=gvx+rng.gauss(0,P.GNSS_VEL_NOISE); mvy=gvy+rng.gauss(0,P.GNSS_VEL_NOISE)
        mz=v.z+rng.gauss(0,P.BARO_NOISE)
        mas=AIRSPEED+rng.gauss(0,P.AIRSPEED_NOISE)

        ekf.predict(P.DT)
        if t+1e-9>=next_guid:
            c=choose_command(ekf.x,(we.wx,we.wy)); next_guid += P.GUIDANCE_DT
        else: c=steering[-1] if steering else 0.0
        v.step(P.DT,c); steering.append(c)

        if t+1e-9 >= gnss_count/P.GNSS_HZ:
            ok=ekf.update_gnss(mx,my,mvx,mvy)
            if not ok: gnss_rej+=1
            gnss_count+=1
            # Wind estimate only uses accepted GNSS velocity and healthy heading.
            if ok:
                w=we.estimate(mvx,mvy,ekf.x[5],mas,ekf.health.gnss*ekf.health.imu*ekf.health.airspeed)
            else:
                w=(we.wx,we.wy)
            gnss_drop += 0
        if t+1e-9 >= baro_count/P.BARO_HZ:
            ekf.update_baro(mz); baro_count+=1
        if t+1e-9 >= imu_count/P.IMU_HZ:
            ekf.update_imu(mh,mtr); imu_count+=1
        if t+1e-9 >= air_count/P.AIRSPEED_HZ:
            ekf.update_airspeed(mas); air_count+=1
        true=(v.x,v.y,v.z,v.heading)
        ekf.log_error(true); we.log_error(wx,wy)
        if return_history and len(history)%10==0:
            history.append((t,v.x,v.y,v.z,ekf.x[0],ekf.x[1],we.wx,we.wy,c))
        t+=P.DT

    true_err=norm2(v.x-P.TARGET_X,v.y-P.TARGET_Y)
    est_err=norm2(ekf.x[0]-P.TARGET_X,ekf.x[1]-P.TARGET_Y)
    rms=lambda a: math.sqrt(sum(q*q for q in a)/len(a)) if a else 0.0
    return {
        'wind_speed':wind_speed,'wind_dir':wind_dir_deg,'wind_x':wx,'wind_y':wy,
        'landing_x':v.x,'landing_y':v.y,'true_landing_error':true_err,
        'estimated_landing_x':ekf.x[0],'estimated_landing_y':ekf.x[1],
        'estimated_landing_error':est_err,'flight_time':t,
        'wind_est_x':we.wx,'wind_est_y':we.wy,'wind_est_speed':we.speed,
        'wind_est_dir':we.direction,'wind_mean_error':statistics.mean(we.errors) if we.errors else 0,
        'wind_rms_error':rms(we.errors),'ekf_rms_pos':rms(ekf.pos_errors),
        'ekf_rms_alt':rms(ekf.alt_errors),'ekf_rms_heading':rms(ekf.head_errors),
        'final_gnss_health':ekf.health.gnss,'final_imu_health':ekf.health.imu,
        'final_baro_health':ekf.health.baro,'final_airspeed_health':ekf.health.airspeed,
        'gnss_rejected':gnss_rej,'gnss_measurements':gnss_count,
        'average_steering':statistics.mean(abs(s) for s in steering) if steering else 0,
        'steering_reversals':sum(1 for a,b in zip(steering,steering[1:]) if a*b<0),
        'history':history,
    }


def print_header():
    print('='*72); print('VIRTUAL PARAFOIL V10.6'); print('ONLINE WIND ESTIMATION + ADAPTIVE WIND VECTOR'); print('='*72)
    print('\nPARAFOIL PARAMETERS'); print(f'Area: {P.AREA} m^2'); print(f'Mass: {P.MASS} kg'); print(f'CL: {P.CL}'); print(f'CD: {P.CD}')
    print(f'Airspeed: {AIRSPEED} m/s'); print(f'Horizontal air velocity: {V_AIR_H} m/s'); print(f'Vertical descent velocity: {V_DESC} m/s'); print(f'Glide ratio: {GLIDE}')
    print('\nTARGET'); print(f'Target X: {P.TARGET_X} m'); print(f'Target Y: {P.TARGET_Y} m'); print(f'Tolerance: {P.TOL} m')
    wx,wy=wind_xy(P.WIND_SPEED,P.WIND_DIR_DEG); print('\nREFERENCE WIND'); print(f'Wind speed: {P.WIND_SPEED} m/s'); print(f'Wind direction: {P.WIND_DIR_DEG} degrees'); print(f'Wind X: {wx} m/s'); print(f'Wind Y: {wy} m/s')
    print('\nSENSOR MODEL');
    for s in ['GNSS_POS_NOISE','GNSS_VEL_NOISE','BARO_NOISE','IMU_HEADING_NOISE_DEG','IMU_TURN_NOISE_DEG_S','AIRSPEED_NOISE']:
        print(f'{s}: {getattr(P,s)}')
    print('\nUPDATE RATES'); print(f'GNSS: {P.GNSS_HZ} Hz'); print(f'Barometer: {P.BARO_HZ} Hz'); print(f'IMU: {P.IMU_HZ} Hz'); print(f'Airspeed: {P.AIRSPEED_HZ} Hz')
    print('\nWIND ESTIMATOR'); print('Method: Ground velocity - air-relative velocity'); print(f'Wind filter alpha: {P.WIND_ALPHA}'); print(f'Maximum wind estimate: {P.MAX_WIND_EST} m/s'); print('Adaptive sensor-health weighting: ENABLED')
    print('\nGUIDANCE'); print(f'Guidance interval: {P.GUIDANCE_DT} s'); print(f'Candidate commands: {P.CANDIDATES}'); print(f'Maximum turn rate: {P.MAX_TURN_DEG_S} deg/s')


def print_reference(r):
    print('\n'+'='*72); print('V10.6 REFERENCE RESULTS'); print('='*72)
    print(f"True landing X: {r['landing_x']:.3f} m"); print(f"True landing Y: {r['landing_y']:.3f} m")
    print(f"Estimated landing X: {r['estimated_landing_x']:.3f} m"); print(f"Estimated landing Y: {r['estimated_landing_y']:.3f} m")
    print(f"Target: {P.TARGET_X:.3f}, {P.TARGET_Y:.3f} m"); print(f"True landing error: {r['true_landing_error']:.3f} m"); print(f"Estimated landing error: {r['estimated_landing_error']:.3f} m"); print(f"Flight time: {r['flight_time']:.3f} s")
    print('\n--- WIND ESTIMATION ---'); print(f"True wind X: {r['wind_x']:.3f} m/s"); print(f"True wind Y: {r['wind_y']:.3f} m/s"); print(f"True wind speed: {r['wind_speed']:.3f} m/s"); print(f"True wind direction: {r['wind_dir']:.3f} deg")
    print(f"Estimated wind X: {r['wind_est_x']:.3f} m/s"); print(f"Estimated wind Y: {r['wind_est_y']:.3f} m/s"); print(f"Estimated wind speed: {r['wind_est_speed']:.3f} m/s"); print(f"Estimated wind direction: {r['wind_est_dir']:.3f} deg"); print(f"Mean wind estimation error: {r['wind_mean_error']:.3f} m/s"); print(f"RMS wind estimation error: {r['wind_rms_error']:.3f} m/s")
    print('\n--- NAVIGATION ---'); print(f"EKF RMS position error: {r['ekf_rms_pos']:.3f} m"); print(f"EKF RMS altitude error: {r['ekf_rms_alt']:.3f} m"); print(f"EKF RMS heading error: {r['ekf_rms_heading']:.3f} deg")
    print('\n--- SENSOR HEALTH ---'); print(f"Final GNSS health: {r['final_gnss_health']:.3f}"); print(f"Final IMU health: {r['final_imu_health']:.3f}"); print(f"Final barometer health: {r['final_baro_health']:.3f}"); print(f"Final airspeed health: {r['final_airspeed_health']:.3f}")
    print('\n--- GUIDANCE ---'); print(f"Average steering: {r['average_steering']:.3f}"); print(f"Steering reversals: {r['steering_reversals']}"); print('Landing status:', 'WITHIN TOLERANCE' if r['true_landing_error']<=P.TOL else 'OUTSIDE TOLERANCE')


def study_directions():
    print('\n'+'='*72); print('V10.6 WIND DIRECTION STUDY'); print('='*72); print('Wind speed fixed at 3.0 m/s\n')
    out=[]
    for i,d in enumerate(range(0,360,45),1):
        r=simulate(3.0,float(d),P.SEED+100+i); out.append(r)
        print(f"Simulation {i}/8 | Direction = {d:6.1f} deg | Landing Error = {r['true_landing_error']:8.3f} m | Wind RMS Error = {r['wind_rms_error']:7.3f} m/s")
    return out


def study_speeds():
    print('\n'+'='*72); print('V10.6 WIND SPEED STUDY'); print('='*72); print('Wind direction fixed at 0.0 degrees\n')
    out=[]
    for i,s in enumerate(range(8),1):
        r=simulate(float(s),0.0,P.SEED+200+i); out.append(r)
        print(f"Simulation {i}/8 | Wind = {s:5.2f} m/s | Landing Error = {r['true_landing_error']:8.3f} m | Estimated Wind = {r['wind_est_speed']:6.3f} m/s | Wind RMS = {r['wind_rms_error']:7.3f} m/s")
    return out


def monte_carlo(n=100):
    print('\n'+'='*72); print('V10.6 MONTE CARLO WIND VALIDATION'); print('='*72); print(f'Number of simulations: {n}'); print('Wind speed range: 0.0 - 7.0 m/s'); print('Wind direction range: 0 - 360 degrees\n')
    rng=random.Random(P.SEED+500); results=[]
    for i in range(1,n+1):
        s=rng.uniform(0,7); d=rng.uniform(0,360)
        r=simulate(s,d,P.SEED+5000+i); results.append(r)
        if i==1 or i%10==0 or i==n:
            print(f"Simulation {i:3d}/{n} | Wind = {s:5.2f} m/s | Direction = {d:6.1f} deg | Landing Error = {r['true_landing_error']:8.2f} m | Wind RMS = {r['wind_rms_error']:6.2f} m/s")
    return results


def print_mc(results):
    errs=[r['true_landing_error'] for r in results]; wr=[r['wind_rms_error'] for r in results]
    print('\n'+'='*72); print('V10.6 MONTE CARLO RESULTS'); print('='*72); print(f'Number of simulations: {len(results)}')
    print(f'Mean landing error: {statistics.mean(errs):.3f} m'); print(f'Median landing error: {statistics.median(errs):.3f} m'); print(f'Standard deviation: {statistics.stdev(errs) if len(errs)>1 else 0:.3f} m'); print(f'Minimum landing error: {min(errs):.3f} m'); print(f'Maximum landing error: {max(errs):.3f} m')
    for t in (5,10,20): print(f'Landing within {t} m: {100*sum(e<=t for e in errs)/len(errs):.2f} %')
    print(f'Mean wind RMS error: {statistics.mean(wr):.3f} m/s'); print(f'Median wind RMS error: {statistics.median(wr):.3f} m/s')
    best=min(results,key=lambda r:r['true_landing_error']); worst=max(results,key=lambda r:r['true_landing_error']); bw=min(results,key=lambda r:r['wind_rms_error']); ww=max(results,key=lambda r:r['wind_rms_error'])
    print('\nBEST LANDING CASE'); print(f"Wind speed: {best['wind_speed']:.3f} m/s"); print(f"Wind direction: {best['wind_dir']:.3f} degrees"); print(f"Wind X: {best['wind_x']:.3f} m/s"); print(f"Wind Y: {best['wind_y']:.3f} m/s"); print(f"Landing X: {best['landing_x']:.3f} m"); print(f"Landing Y: {best['landing_y']:.3f} m"); print(f"Landing error: {best['true_landing_error']:.3f} m")
    print('\nWORST LANDING CASE'); print(f"Wind speed: {worst['wind_speed']:.3f} m/s"); print(f"Wind direction: {worst['wind_dir']:.3f} degrees"); print(f"Landing X: {worst['landing_x']:.3f} m"); print(f"Landing Y: {worst['landing_y']:.3f} m"); print(f"Landing error: {worst['true_landing_error']:.3f} m")
    print('\nBEST WIND ESTIMATION CASE'); print(f"Wind speed: {bw['wind_speed']:.3f} m/s"); print(f"Wind direction: {bw['wind_dir']:.3f} degrees"); print(f"Wind RMS estimation error: {bw['wind_rms_error']:.3f} m/s")
    print('\nWORST WIND ESTIMATION CASE'); print(f"Wind speed: {ww['wind_speed']:.3f} m/s"); print(f"Wind direction: {ww['wind_dir']:.3f} degrees"); print(f"Wind RMS estimation error: {ww['wind_rms_error']:.3f} m/s")
    return {'mean_landing_error':statistics.mean(errs),'median_landing_error':statistics.median(errs),'std_landing_error':statistics.stdev(errs) if len(errs)>1 else 0.0,'min_landing_error':min(errs),'max_landing_error':max(errs),'within5':sum(e<=5 for e in errs)/len(errs),'within10':sum(e<=10 for e in errs)/len(errs),'within20':sum(e<=20 for e in errs)/len(errs),'mean_wind_rms':statistics.mean(wr),'median_wind_rms':statistics.median(wr),'best':best,'worst':worst,'best_wind':bw,'worst_wind':ww}


def print_final_assessment(reference_result, mc):
    # IMPORTANT: reference_result is explicitly checked before indexing.
    if reference_result is None:
        raise RuntimeError('Reference simulation returned None. This should not occur.')
    print('\n'+'='*72); print('V10.6 NAVIGATION SYSTEM ASSESSMENT'); print('='*72)
    print(f"Raw GNSS position noise: {P.GNSS_POS_NOISE:.1f} m")
    print(f"Reference wind RMS estimation error: {reference_result['wind_rms_error']:.3f} m/s")
    print(f"Reference EKF RMS position error: {reference_result['ekf_rms_pos']:.3f} m")
    print(f"Monte Carlo mean wind RMS error: {mc['mean_wind_rms']:.3f} m/s")
    print(f"Monte Carlo landing success <=20 m: {100*mc['within20']:.2f} %")
    print('WIND ESTIMATION: ACTIVE'); print('ADAPTIVE SENSOR-HEALTH WEIGHTING: ACTIVE'); print('ONLINE WIND VECTOR: ACTIVE')
    print('\nV10.6 STATUS: COMPLETE')


def main():
    random.seed(P.SEED)
    print_header()
    print('\n'+'='*72); print('STARTING V10.6 REFERENCE WIND SIMULATION'); print('='*72)
    reference_result=simulate(P.WIND_SPEED,P.WIND_DIR_DEG,P.SEED+1)
    print_reference(reference_result)
    study_directions()
    study_speeds()
    mc_results=monte_carlo(100)
    mc=print_mc(mc_results)
    print_final_assessment(reference_result,mc)
    print('\n'+'='*72); print('V10.6 SIMULATION COMPLETE'); print('='*72)
    print('Reference wind:',P.WIND_SPEED,'m/s'); print('Reference direction:',P.WIND_DIR_DEG,'degrees'); print('Wind estimator: ONLINE'); print('Adaptive wind-vector estimation: ACTIVE'); print('Monte Carlo simulations: 100')
    print('='*72)

if __name__=='__main__': main()