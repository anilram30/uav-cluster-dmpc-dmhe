"""
fw_control.py -- cascaded fixed-wing autopilot + formation guidance
(Beard & McLain Ch. 6 style successive-loop closure), used for the winged-UAV
cluster demonstration.

Loops (all PI/PD, gains hand-tuned for the Aerosonde at Va~25 m/s):
  * airspeed  -> throttle         (PI on Va error, about trim throttle)
  * altitude  -> pitch command    (PI) -> elevator  (pitch PD)
  * course    -> roll command     (PI, wrapped) -> aileron (roll PD)
Guidance: track a moving formation slot -> commanded (course, altitude, airspeed).
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


class FWAutopilot:
    def __init__(self, trim_u, Va0=25.0, h0=100.0):
        self.de0, self.dt0 = trim_u[0], trim_u[3]
        self.Va0, self.h0 = Va0, h0
        # integrators
        self.i_Va = 0.0; self.i_h = 0.0; self.i_chi = 0.0
        # gains
        self.kp_V, self.ki_V = 0.10, 0.05           # airspeed -> throttle
        self.kp_h, self.ki_h = 0.020, 0.004         # altitude -> pitch (rad/m)
        self.kp_th, self.kd_th = -3.0, -0.4         # pitch -> elevator
        self.kp_chi, self.ki_chi = 1.1, 0.10        # course -> roll
        self.kp_phi, self.kd_phi = 0.9, 0.15        # roll -> aileron
        self.theta_max = np.deg2rad(25)
        self.phi_max = np.deg2rad(40)

    def command(self, x, chi_c, h_c, Va_c, dt):
        pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = x
        h = -pd
        Va = np.sqrt(u*u + v*v + w*w)
        chi = wrap(np.arctan2(pe*0 + np.sin(psi), np.cos(psi)))  # heading approx
        chi = psi                                    # small-sideslip: course~heading

        # airspeed -> throttle
        eV = Va_c - Va
        self.i_Va = np.clip(self.i_Va + eV*dt, -20, 20)
        dt_cmd = np.clip(self.dt0 + self.kp_V*eV + self.ki_V*self.i_Va, 0.0, 1.0)

        # altitude -> pitch command
        eh = h_c - h
        self.i_h = np.clip(self.i_h + eh*dt, -50, 50)
        theta_c = np.clip(self.kp_h*eh + self.ki_h*self.i_h, -self.theta_max, self.theta_max)
        # pitch -> elevator
        de_cmd = self.de0 + self.kp_th*(theta_c - theta) + self.kd_th*(-q)
        de_cmd = np.clip(de_cmd, np.deg2rad(-35), np.deg2rad(35))

        # course -> roll command
        echi = wrap(chi_c - chi)
        self.i_chi = np.clip(self.i_chi + echi*dt, -2, 2)
        phi_c = np.clip(self.kp_chi*echi + self.ki_chi*self.i_chi, -self.phi_max, self.phi_max)
        # roll -> aileron
        da_cmd = self.kp_phi*(phi_c - phi) + self.kd_phi*(-p)
        da_cmd = np.clip(da_cmd, np.deg2rad(-30), np.deg2rad(30))

        # coordinated turn: small rudder to damp yaw / sideslip
        beta = np.arcsin(np.clip(v/max(Va, 1e-3), -1, 1))
        dr_cmd = np.clip(-1.5*beta - 0.1*r, np.deg2rad(-25), np.deg2rad(25))
        return np.array([de_cmd, da_cmd, dr_cmd, dt_cmd])


def guidance_to_slot(x, slot, Va_nom=25.0, k_alongtrack=0.4, look=25.0):
    """
    Given current state and a desired 3-D slot position (NED), produce
    (course_cmd, altitude_cmd, airspeed_cmd). Course points a look-ahead
    'carrot' toward the slot; airspeed corrects along-track spacing error.
    """
    pn, pe, pd = x[0], x[1], x[2]
    sn, se, sd = slot
    dpn, dpe = sn - pn, se - pe
    dist = np.hypot(dpn, dpe)
    chi_c = np.arctan2(dpe, dpn)
    h_c = -sd
    # airspeed trims to close range but stays in a safe band
    Va_c = np.clip(Va_nom + k_alongtrack*(dist - 0.0), 16.0, 34.0)
    return chi_c, h_c, Va_c
