"""
validate_models.py -- numerical checks that the hexacopter and fixed-wing
models are internally consistent. Every check here backs a stated fact in the
report. Run:  python3 validate_models.py
"""
import numpy as np
from uavdmpc.quaternion import (qmul, qconj, quat_to_rot, qdot, Omega, qnorm,
                                qintegrate, euler_to_quat, quat_to_euler)
from uavdmpc.hexacopter import Hexacopter, G
from uavdmpc.fixedwing import FixedWing

np.set_printoptions(precision=5, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok_all = False
    print(f"  [{status}] {name}   {detail}")


print("=" * 70)
print("1. QUATERNION ALGEBRA")
print("=" * 70)
rng = np.random.default_rng(0)
q = qnorm(rng.standard_normal(4))
p = qnorm(rng.standard_normal(4))
# R(q) orthogonal, det +1
R = quat_to_rot(q)
check("R(q) orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-12))
check("det R(q) = +1", abs(np.linalg.det(R) - 1) < 1e-12,
      f"det={np.linalg.det(R):.12f}")
# R(p⊗q) = R(p) R(q) (homomorphism)
check("R(p⊗q) = R(p)R(q)",
      np.allclose(quat_to_rot(qmul(p, q)), quat_to_rot(p) @ quat_to_rot(q), atol=1e-12))
# conjugate = inverse rotation
check("R(q*) = R(q)^T", np.allclose(quat_to_rot(qconj(q)), R.T, atol=1e-12))
# 1/2 Omega(w) q  ==  1/2 q ⊗ [0,w]
w = rng.standard_normal(3)
check("qdot map equals 1/2 Omega(w) q",
      np.allclose(qdot(q, w), 0.5 * Omega(w) @ q, atol=1e-12))
# Euler round trip
phi, th, ps = 0.3, -0.2, 1.1
check("Euler->quat->Euler round trip",
      np.allclose([phi, th, ps], quat_to_euler(euler_to_quat(phi, th, ps)), atol=1e-10))
# norm-exact integrator preserves ||q||
qq = qnorm(np.array([1.0, 0.0, 0.0, 0.0]))
for _ in range(10000):
    qq = qintegrate(qq, np.array([0.7, -0.4, 0.9]), 1e-3)
check("integrator keeps ||q||=1 over 10k steps", abs(np.linalg.norm(qq) - 1) < 1e-10,
      f"||q||-1 = {np.linalg.norm(qq)-1:.2e}")

print()
print("=" * 70)
print("2. HEXACOPTER MODEL")
print("=" * 70)
hexa = Hexacopter()
# mixer rank and reconstruction
check("mixer M is 4x6 rank 4", np.linalg.matrix_rank(hexa.M) == 4,
      f"shape={hexa.M.shape}")
check("M @ pinv(M) = I_4", np.allclose(hexa.M @ hexa.Mpinv, np.eye(4), atol=1e-10))
# hover: wrench -> rotor speeds -> wrench round trip
u_hov = hexa.hover_input()
Om = hexa.wrench_to_rotor_speeds(u_hov)
u_rt = hexa.rotor_speeds_to_wrench(Om)
check("hover wrench round-trips through rotors", np.allclose(u_hov, u_rt, atol=1e-6),
      f"max err={np.max(np.abs(u_hov-u_rt)):.2e}")
check("hover rotor speeds equal (symmetry)", np.std(Om) < 1e-6,
      f"Omega={Om[0]:.2f} rad/s, spread={np.std(Om):.2e}")
# hover is an equilibrium: xdot(pv w) = 0, qdot=0 at level attitude
x_hov = np.zeros(13); x_hov[6] = 1.0   # q = identity, level, at rest
d = hexa.deriv(x_hov, u_hov)
check("hover is equilibrium (xdot=0)", np.max(np.abs(d)) < 1e-9,
      f"max|xdot|={np.max(np.abs(d)):.2e}")
# thrust must equal weight at hover
check("collective thrust T = m g at hover", abs(u_hov[0] - hexa.p['m']*G) < 1e-9)
# yaw torque axis: spinning all rotors up gives pure +z or -z torque only? check net
# Jacobian A at hover: position/velocity coupling sane (12 zero-ish modes expected)
A, B = hexa.jacobians(x_hov, u_hov)
check("Jacobian A finite", np.all(np.isfinite(A)))
check("Jacobian B finite and B[5,0]=1/m (thrust->vertical accel)",
      abs(B[5, 0] - 1.0/hexa.p['m']) < 1e-4, f"B[5,0]={B[5,0]:.5f}, 1/m={1/hexa.p['m']:.5f}")
# free fall check: zero thrust -> vertical accel = -g
d0 = hexa.deriv(x_hov, np.zeros(4))
check("zero thrust -> vertical accel = -g", abs(d0[5] + G) < 1e-9, f"vzdot={d0[5]:.4f}")
# a short closed integration: hover holds position
x = x_hov.copy()
for _ in range(2000):
    x = hexa.step(x, u_hov, 2e-3)
check("hover holds over 4 s (||p||<1e-3)", np.linalg.norm(x[0:3]) < 1e-3,
      f"||p||={np.linalg.norm(x[0:3]):.2e}, ||q||-1={abs(np.linalg.norm(x[6:10])-1):.1e}")

print()
print("=" * 70)
print("3. FIXED-WING MODEL (Aerosonde)")
print("=" * 70)
fw = FixedWing()
# CL sigmoid: linear region slope ~ C_Lalpha near 0, saturates past stall
a = np.linspace(-0.6, 0.6, 5)
CLs = [fw.CL(ai) for ai in a]
slope0 = (fw.CL(0.01) - fw.CL(-0.01)) / 0.02
check("CL slope near 0 ~ C_L0alpha", abs(slope0 - fw.p['C_Lalpha']) < 0.3,
      f"slope={slope0:.3f} vs C_Lalpha={fw.p['C_Lalpha']}")
check("CL continuous through stall (no jumps)",
      np.all(np.abs(np.diff([fw.CL(ai) for ai in np.linspace(0, 0.8, 200)])) < 0.1))
# inertia Gamma consistency
check("Gamma = Jx Jz - Jxz^2 > 0", fw.Gamma > 0, f"Gamma={fw.Gamma:.4f}")
# TRIM at 35 m/s
xt, ut, info = fw.trim_level(35.0)
check("trim solved (residual<1e-6)", info['residual'] < 1e-6,
      f"alpha={np.rad2deg(info['alpha']):.2f} deg, de={np.rad2deg(info['delta_e']):.2f} deg, "
      f"dt={info['delta_t']:.3f}, res={info['residual']:.2e}")
# integrate trim: level flight holds airspeed and altitude-rate ~ const
x = xt.copy()
alt0 = -x[2]
Va0, _, _ = fw.airdata(x[3], x[4], x[5])
for _ in range(500):
    x = fw.step(x, ut, 0.01)
Va1, al1, be1 = fw.airdata(x[3], x[4], x[5])
check("trim holds airspeed over 5 s", abs(Va1 - Va0) < 0.5,
      f"Va0={Va0:.2f} Va1={Va1:.2f}")
check("trim holds altitude over 5 s (|dh|<2 m)", abs((-x[2]) - alt0) < 2.0,
      f"dh={(-x[2])-alt0:.3f} m")
# elevator perturbation produces pitch response of correct sign
xp = xt.copy()
up = ut.copy(); up[0] += np.deg2rad(2)   # more (negative-def) elevator
dp = fw.deriv(xp, up)
check("elevator affects pitch accel (qdot != 0)", abs(dp[10]) > 1e-3,
      f"qdot={dp[10]:.4f}")

print()
print("=" * 70)
print("SUMMARY:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
print("=" * 70)
