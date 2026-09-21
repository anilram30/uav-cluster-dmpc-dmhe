"""
fixedwing.py -- Beard & McLain (2012) 6-DOF fixed-wing model, Aerosonde
first-edition (Appendix E) parameters. Framed as the small-scale surrogate of a
MALE-class platform (e.g. Gray-Eagle-class) for the fixed-wing cluster study.

State (12): [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]   (NED, body vel)
Control (4): [delta_e, delta_a, delta_r, delta_t]

All force/moment equations match the report ("Fixed-wing 6-DOF model") exactly,
including the sigmoid stall-blended lift coefficient.
"""
import numpy as np

G = 9.81

AEROSONDE = dict(
    m=13.5, Jx=0.8244, Jy=1.135, Jz=1.759, Jxz=0.1204,
    S=0.55, b=2.8956, c=0.18994, S_prop=0.2027, rho=1.2682, e=0.9,
    k_motor=80.0, k_Tp=0.0, k_Omega=0.0, C_prop=1.0,
    M=50.0, alpha0=0.4712, epsilon=0.1592,
    # longitudinal
    C_L0=0.28, C_Lalpha=3.45, C_Lq=0.0, C_Ldelta_e=-0.36,
    C_D0=0.03, C_Dalpha=0.30, C_Dp=0.0437, C_Dq=0.0, C_Ddelta_e=0.0,
    C_m0=-0.02338, C_malpha=-0.38, C_mq=-3.6, C_mdelta_e=-0.5,
    # lateral
    C_Y0=0.0, C_Ybeta=-0.98, C_Yp=0.0, C_Yr=0.0, C_Ydelta_a=0.0, C_Ydelta_r=-0.17,
    C_ell0=0.0, C_ellbeta=-0.12, C_ellp=-0.26, C_ellr=0.14,
    C_elldelta_a=0.08, C_elldelta_r=0.105,
    C_n0=0.0, C_nbeta=0.25, C_np=0.022, C_nr=-0.35,
    C_ndelta_a=0.06, C_ndelta_r=-0.032,
)


class FixedWing:
    def __init__(self, params=None):
        p = dict(AEROSONDE)
        if params:
            p.update(params)
        self.p = p
        # inertia-tensor products (Beard-McLain eq. 3.13 Gamma coefficients)
        Jx, Jy, Jz, Jxz = p['Jx'], p['Jy'], p['Jz'], p['Jxz']
        G0 = Jx*Jz - Jxz**2
        self.Gamma = G0
        self.G1 = Jxz*(Jx - Jy + Jz)/G0
        self.G2 = (Jz*(Jz - Jy) + Jxz**2)/G0
        self.G3 = Jz/G0
        self.G4 = Jxz/G0
        self.G5 = (Jz - Jx)/Jy
        self.G6 = Jxz/Jy
        self.G7 = ((Jx - Jy)*Jx + Jxz**2)/G0
        self.G8 = Jx/G0

    # --------------------------------------------------------------------- #
    #  Wind triangle:  Va, alpha, beta  (no wind here; ur=u etc.)
    # --------------------------------------------------------------------- #
    @staticmethod
    def airdata(u, v, w):
        Va = np.sqrt(u*u + v*v + w*w)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / max(Va, 1e-6), -1, 1))
        return Va, alpha, beta

    # --------------------------------------------------------------------- #
    #  Sigmoid-blended lift coefficient  (Beard-McLain eq. 4.9-4.10)
    # --------------------------------------------------------------------- #
    def CL(self, alpha):
        p = self.p
        M, a0 = p['M'], p['alpha0']
        num = 1 + np.exp(-M*(alpha - a0)) + np.exp(M*(alpha + a0))
        den = (1 + np.exp(-M*(alpha - a0))) * (1 + np.exp(M*(alpha + a0)))
        sigma = num / den
        CL_lin = p['C_L0'] + p['C_Lalpha']*alpha
        CL_stall = 2*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha)
        return (1 - sigma)*CL_lin + sigma*CL_stall

    def CD(self, alpha):
        p = self.p
        AR = p['b']**2 / p['S']
        return p['C_Dp'] + (p['C_L0'] + p['C_Lalpha']*alpha)**2 / (np.pi*p['e']*AR)

    # --------------------------------------------------------------------- #
    #  Forces and moments in the body frame
    # --------------------------------------------------------------------- #
    def forces_moments(self, x, u):
        p = self.p
        phi, theta, psi = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]
        de, da, dr, dt = u
        Va, alpha, beta = self.airdata(x[3], x[4], x[5])
        qbar = 0.5*p['rho']*Va**2
        S, b, c = p['S'], p['b'], p['c']

        # gravity in body frame
        fg = p['m']*G*np.array([-np.sin(theta),
                                np.cos(theta)*np.sin(phi),
                                np.cos(theta)*np.cos(phi)])

        # longitudinal aero: lift & drag in stability frame -> body
        ca, sa = np.cos(alpha), np.sin(alpha)
        cq = c/(2*max(Va, 1e-3))
        CLtot = self.CL(alpha) + p['C_Lq']*cq*qr + p['C_Ldelta_e']*de
        CDtot = self.CD(alpha) + p['C_Dq']*cq*qr + p['C_Ddelta_e']*de
        fx_a = qbar*S*(-CDtot*ca + CLtot*sa)
        fz_a = qbar*S*(-CDtot*sa - CLtot*ca)
        m_a = qbar*S*c*(p['C_m0'] + p['C_malpha']*alpha
                        + p['C_mq']*cq*qr + p['C_mdelta_e']*de)

        # lateral aero
        bp = b/(2*max(Va, 1e-3))
        fy_a = qbar*S*(p['C_Y0'] + p['C_Ybeta']*beta + p['C_Yp']*bp*pr
                       + p['C_Yr']*bp*rr + p['C_Ydelta_a']*da + p['C_Ydelta_r']*dr)
        l_a = qbar*S*b*(p['C_ell0'] + p['C_ellbeta']*beta + p['C_ellp']*bp*pr
                        + p['C_ellr']*bp*rr + p['C_elldelta_a']*da + p['C_elldelta_r']*dr)
        n_a = qbar*S*b*(p['C_n0'] + p['C_nbeta']*beta + p['C_np']*bp*pr
                        + p['C_nr']*bp*rr + p['C_ndelta_a']*da + p['C_ndelta_r']*dr)

        # propulsion (first-edition model): thrust along body x, reaction torque -x
        fx_p = 0.5*p['rho']*p['S_prop']*p['C_prop']*((p['k_motor']*dt)**2 - Va**2)
        l_p = -p['k_Tp']*(p['k_Omega']*dt)**2

        fx = fg[0] + fx_a + fx_p
        fy = fg[1] + fy_a
        fz = fg[2] + fz_a
        l = l_a + l_p
        m = m_a
        n = n_a
        return np.array([fx, fy, fz]), np.array([l, m, n]), (Va, alpha, beta)

    # --------------------------------------------------------------------- #
    #  Full 6-DOF derivative  (Beard-McLain eq. 3.14-3.17)
    # --------------------------------------------------------------------- #
    def deriv(self, x, u):
        p = self.p
        pn, pe, pd = x[0], x[1], x[2]
        uu, vv, ww = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]
        f, mom, _ = self.forces_moments(x, u)
        fx, fy, fz = f
        l, m, n = mom
        m_ = p['m']

        # position kinematics (body vel -> NED via rotation)
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)
        Rbv = np.array([
            [cth*cps, sph*sth*cps - cph*sps, cph*sth*cps + sph*sps],
            [cth*sps, sph*sth*sps + cph*cps, cph*sth*sps - sph*cps],
            [-sth,    sph*cth,               cph*cth],
        ])
        pos_dot = Rbv @ np.array([uu, vv, ww])

        # translational dynamics
        udot = rr*vv - qr*ww + fx/m_
        vdot = pr*ww - rr*uu + fy/m_
        wdot = qr*uu - pr*vv + fz/m_

        # attitude kinematics (Euler 3-2-1)
        phidot = pr + sph*np.tan(theta)*qr + cph*np.tan(theta)*rr
        thetadot = cph*qr - sph*rr
        psidot = (sph/cth)*qr + (cph/cth)*rr

        # rotational dynamics with Gamma coefficients
        pdot = (self.G1*pr*qr - self.G2*qr*rr) + (self.G3*l + self.G4*n)
        qdot = (self.G5*pr*rr - self.G6*(pr*pr - rr*rr)) + m/p['Jy']
        rdot = (self.G7*pr*qr - self.G1*qr*rr) + (self.G4*l + self.G8*n)

        return np.array([pos_dot[0], pos_dot[1], pos_dot[2],
                         udot, vdot, wdot,
                         phidot, thetadot, psidot,
                         pdot, qdot, rdot])

    def step(self, x, u, dt, substeps=1):
        h = dt/substeps
        for _ in range(substeps):
            k1 = self.deriv(x, u)
            k2 = self.deriv(x + 0.5*h*k1, u)
            k3 = self.deriv(x + 0.5*h*k2, u)
            k4 = self.deriv(x + h*k3, u)
            x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return x

    # --------------------------------------------------------------------- #
    #  Trim: straight-and-level at airspeed Va*  (solve for alpha, de, dt)
    # --------------------------------------------------------------------- #
    def trim_level(self, Va_star, from_guess=None):
        from scipy.optimize import fsolve

        def resid(z):
            alpha, de, dt = z
            # level flight: theta = alpha (gamma=0), body vel from Va*, alpha
            x = np.zeros(12)
            x[3] = Va_star*np.cos(alpha)
            x[5] = Va_star*np.sin(alpha)
            x[7] = alpha
            u = np.array([de, 0.0, 0.0, dt])
            d = self.deriv(x, u)
            # want udot=wdot=qdot=0
            return [d[3], d[5], d[10]]

        z0 = from_guess if from_guess is not None else [0.03, -0.1, 0.5]
        sol = fsolve(resid, z0, full_output=True)
        z, info, ier, msg = sol
        alpha, de, dt = z
        x = np.zeros(12)
        x[3] = Va_star*np.cos(alpha)
        x[5] = Va_star*np.sin(alpha)
        x[7] = alpha
        u = np.array([de, 0.0, 0.0, dt])
        return x, u, dict(alpha=alpha, delta_e=de, delta_t=dt,
                          residual=np.max(np.abs(resid(z))), ok=(ier == 1))
