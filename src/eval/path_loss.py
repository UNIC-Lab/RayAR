"""Pure-Python port of the C++ UTD ray-tracing path-loss computation.

This mirrors :file:`los2d_bfs/src/core/` so that energy errors can be
computed for *any* predicted chain, not only those pre-cached in PROP files.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants  (mirrors los2d_bfs/include/common/constants.hpp)
# ---------------------------------------------------------------------------

PI = float(np.float32(3.14159265358979))
C0 = 299792458.0  # speed of light  [m/s]
WEDGE_N = 1.5  # exterior wedge index  (270° interior → n=1.5)
EPSILON_R = 5.31  # relative permittivity (concrete)
SIGMA_C = 0.0326  # conductivity σ(f) = SIGMA_C · f^SIGMA_D
SIGMA_D = 0.8095

DEFAULT_FREQ_GHZ = 6.0

DEFAULT_TX_HEIGHT = 1.5
DEFAULT_RX_HEIGHT = 1.5
DEFAULT_VERTEX_HEIGHT = 20.0

# ---------------------------------------------------------------------------
# Free-Space Path Loss
# ---------------------------------------------------------------------------


def fspl(dist_m: float, freq_ghz: float = DEFAULT_FREQ_GHZ) -> float:
    """FSPL in dB.  ``dist_m`` is the 3-D slant distance."""
    d = max(float(dist_m), 1e-3)
    return 20.0 * math.log10(d) + 20.0 * math.log10(freq_ghz) + 32.45


# ---------------------------------------------------------------------------
# Fresnel integrals  (Cephes rational approximation — mirrors utd_transition.cpp)
# ---------------------------------------------------------------------------

# pylint: disable=bad-whitespace
_SN = np.array(
    [
        -2.99181919401019853726e3,
        7.08840045257738576863e5,
        -6.29741486205862506537e7,
        2.54890880573376359104e9,
        -4.42979518059697779103e10,
        3.18016297876567817986e11,
    ]
)
_SD = np.array(
    [  # monic
        2.81376268889994315696e2,
        4.55847810806532581675e4,
        5.17343888770096400730e6,
        4.19320245898111231129e8,
        2.24411795645340920940e10,
        6.07366389490084914091e11,
    ]
)
_CN = np.array(
    [
        -4.98843114573573548651e-8,
        9.50428062829859605134e-6,
        -6.45191435683965050962e-4,
        1.88843319396703850064e-2,
        -2.05525900955013891793e-1,
        9.99999999999999998822e-1,
    ]
)
_CD = np.array(
    [
        3.99982968972495980367e-12,
        9.15439215774657478799e-10,
        1.25001862479598821474e-7,
        1.22262789024179030997e-5,
        8.68029542941784300606e-4,
        4.12142090722199792936e-2,
        1.00000000000000000118e0,
    ]
)
_FN = np.array(
    [
        4.21543555043677546506e-1,
        1.43407919780758885261e-1,
        1.15220955073585758835e-2,
        3.45017939782574027900e-4,
        4.63613749287867322088e-6,
        3.05568983790257605827e-8,
        1.02304514164907233465e-10,
        1.72010743268161828879e-13,
        1.34283276233062758925e-16,
        3.76329711269987889006e-20,
    ]
)
_FD = np.array(
    [  # monic
        7.51586398353378947175e-1,
        1.16888925859191382142e-1,
        6.44051526508858611005e-3,
        1.55934409164153020873e-4,
        1.84627567348930545870e-6,
        1.12699224763999035261e-8,
        3.60140029589371370404e-11,
        5.88754533621578410014e-14,
        4.52001434074129701496e-17,
        1.25443237090011264384e-20,
    ]
)
_GN = np.array(
    [
        5.04442073643383265887e-1,
        1.97102833525523411709e-1,
        1.87648584092575249293e-2,
        6.84079380915393090172e-4,
        1.15138826111884280931e-5,
        9.82852443688422223854e-8,
        4.45344415861750144738e-10,
        1.08268041139020870318e-12,
        1.37555460633261799868e-15,
        8.36354435630677421531e-19,
        1.86958710162783235106e-22,
    ]
)
_GD = np.array(
    [  # monic
        1.47495759925128324529e0,
        3.37748989120019970451e-1,
        2.53603741420338795122e-2,
        8.14679107184306179049e-4,
        1.27545075667729153261e-5,
        1.04314589657571990585e-7,
        4.60680728515031256820e-10,
        1.10273215066240270757e-12,
        1.38796531259578871258e-15,
        8.39158816283118707363e-19,
        1.86958710162783236342e-22,
    ]
)


def _polevl(x: float, coef: np.ndarray) -> float:
    """Horner evaluation of polynomial (degree n, n+1 coefficients)."""
    y = float(coef[0])
    for i in range(1, len(coef)):
        y = y * x + float(coef[i])
    return y


def _p1evl(x: float, coef: np.ndarray) -> float:
    """Horner evaluation of *monic* polynomial (degree n, n coefficients)."""
    y = x + float(coef[0])
    for i in range(1, len(coef)):
        y = y * x + float(coef[i])
    return y


def _fresnel(z: float) -> Tuple[float, float]:
    """Return (C(z), S(z)) — Fresnel integrals via Cephes approximation."""
    x = abs(z)
    x2 = x * x

    if x2 < 2.5625:
        t = x2 * x2  # z^4
        c = x * _polevl(t, _CN) / _polevl(t, _CD)
        s = x * x2 * _polevl(t, _SN) / _p1evl(t, _SD)
    elif x < 192.0:
        u = 1.0 / (PI * x2)
        u2 = u * u
        f_raw = 1.0 - u2 * _polevl(u2, _FN) / _p1evl(u2, _FD)
        g_raw = u * _polevl(u2, _GN) / _p1evl(u2, _GD)
        half_pi_x2 = 0.5 * PI * x2
        cc = math.cos(half_pi_x2)
        ss = math.sin(half_pi_x2)
        pi_x = PI * x
        c = 0.5 + (f_raw * ss - g_raw * cc) / pi_x
        s = 0.5 - (f_raw * cc + g_raw * ss) / pi_x
    else:
        c = 0.5
        s = 0.5

    if z < 0.0:
        c = -c
        s = -s
    return c, s


# ---------------------------------------------------------------------------
# UTD transition function  F(X)
# ---------------------------------------------------------------------------


def _transition_F(X: float) -> complex:
    """UTD transition function — mirrors ``transition_F`` in utd_transition.cpp."""
    if X <= 0.0:
        return 0j
    if X < 1e-10:
        s = math.sqrt(PI * X)
        inv_sqrt2 = 0.70710678118654752440
        return complex(s * inv_sqrt2, s * inv_sqrt2)

    w = math.sqrt(2.0 * X / PI)
    c_val, s_val = _fresnel(w)

    dc = 0.5 - c_val
    ds = 0.5 - s_val
    sqrt_2piX = math.sqrt(2.0 * PI * X)
    cosX = math.cos(X)
    sinX = math.sin(X)

    real = sqrt_2piX * (ds * cosX - dc * sinX)
    imag = sqrt_2piX * (dc * cosX + ds * sinX)
    return complex(real, imag)


# ---------------------------------------------------------------------------
# UTD angles
# ---------------------------------------------------------------------------


def _utd_angles(
    in_az: float, out_az: float, nx: float, ny: float
) -> Tuple[float, float]:
    """Return (phi_prime, phi) — mirrors ``compute_utd_angles``."""
    TWO_PI = 2.0 * PI
    face0_az = math.atan2(ny, nx)
    phi_prime = (in_az + PI) - face0_az
    phi_prime = math.fmod(phi_prime, TWO_PI)
    if phi_prime < 0.0:
        phi_prime += TWO_PI
    phi = out_az - face0_az
    phi = math.fmod(phi, TWO_PI)
    if phi < 0.0:
        phi += TWO_PI
    return phi_prime, phi


# ---------------------------------------------------------------------------
# UTD diffraction coefficient  D(φ, φ')
# ---------------------------------------------------------------------------


def _safe_cot(x: float) -> float:
    s = math.sin(x)
    if abs(s) < 1e-10:
        return 1e10 if s >= 0.0 else -1e10
    return math.cos(x) / s


def _fresnel_coeffs(theta: float, eps_c: complex) -> Tuple[complex, complex]:
    """Return (R_TE, R_TM)."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    root = complex(math.sqrt((eps_c - sin_t * sin_t).real), 0.0)
    # Use cmath for complex sqrt
    import cmath

    root = cmath.sqrt(eps_c - sin_t * sin_t)
    r_te = (cos_t - root) / (cos_t + root)
    r_tm = (eps_c * cos_t - root) / (eps_c * cos_t + root)
    return r_te, r_tm


def _N_plus(beta: float, n: float) -> float:
    return round((beta + PI) / (2.0 * PI * n))


def _N_minus(beta: float, n: float) -> float:
    return round((beta - PI) / (2.0 * PI * n))


def _a_plus(beta: float, n: float) -> float:
    N = _N_plus(beta, n)
    arg = 2.0 * PI * n * N - beta
    half = math.cos(arg * 0.5)
    return 2.0 * half * half


def _a_minus(beta: float, n: float) -> float:
    N = _N_minus(beta, n)
    arg = 2.0 * PI * n * N - beta
    half = math.cos(arg * 0.5)
    return 2.0 * half * half


def _eval_utd_term(
    beta: float,
    n: float,
    k: float,
    L: float,
    use_plus: bool,
    reflection_coeff: complex,
) -> complex:
    """Evaluate one KP-UTD term with symmetric perturbation near singularities."""
    import cmath

    def calc_raw(beta_local: float) -> complex:
        a = _a_plus(beta_local, n) if use_plus else _a_minus(beta_local, n)
        arg = (PI + beta_local) if use_plus else (PI - beta_local)
        cot_val = _safe_cot(arg / (2.0 * n))
        return cot_val * _transition_F(k * L * a) * reflection_coeff

    a_center = _a_plus(beta, n) if use_plus else _a_minus(beta, n)
    if a_center < 1e-7:
        eps = 1e-5
        return (calc_raw(beta + eps) + calc_raw(beta - eps)) * 0.5
    return calc_raw(beta)


def _compute_utd_D(
    phi: float,
    phi_prime: float,
    d1: float,
    d2: float,
    freq_ghz: float,
) -> complex:
    """UTD diffraction coefficient — mirrors ``compute_utd_D``."""
    import cmath

    n = WEDGE_N
    j = 1j

    k = 2.0 * PI * freq_ghz * 1.0e9 / C0
    L = d1 * d2 / (d1 + d2)

    prefactor = -cmath.exp(-j * PI / 4.0) / (2.0 * n * math.sqrt(2.0 * PI * k))

    sigma = SIGMA_C * (freq_ghz**SIGMA_D)
    eps_c = complex(EPSILON_R, -(17.98 * sigma / freq_ghz))

    theta_0 = PI / 2.0 - phi_prime
    theta_n = PI / 2.0 - (n * PI - phi)

    r_te_0, r_tm_0 = _fresnel_coeffs(abs(theta_0), eps_c)
    r_te_n, r_tm_n = _fresnel_coeffs(abs(theta_n), eps_c)
    R_0 = (r_te_0 + r_tm_0) / 2.0
    R_n = (r_te_n + r_tm_n) / 2.0

    beta1 = phi - phi_prime
    beta2 = phi + phi_prime

    D1 = _eval_utd_term(beta1, n, k, L, True, complex(1.0, 0.0))
    D2 = _eval_utd_term(beta1, n, k, L, False, complex(1.0, 0.0))
    D3 = _eval_utd_term(beta2, n, k, L, True, R_0)
    D4 = _eval_utd_term(beta2, n, k, L, False, R_n)

    return prefactor * (D1 + D2 + D3 + D4)


# ---------------------------------------------------------------------------
# UTD diffraction loss for one reflection  (the public entry-point per vertex)
# ---------------------------------------------------------------------------


def utd_diff_loss(
    inc_az: float,
    out_az: float,
    nx: float,
    ny: float,
    d1: float,
    d2: float,
    freq_ghz: float = DEFAULT_FREQ_GHZ,
) -> float:
    """UTD diffraction loss [dB] at a single reflection vertex.

    Mirrors ``compute_utd_diff_loss`` in utd_diff_loss.cpp.
    """
    # Distance guard
    if d1 < 0.1 or d2 < 0.1:
        return 6.0
    # Normal validity
    norm_sq = nx * nx + ny * ny
    if norm_sq < 1e-6:
        return 6.0

    phi_prime, phi = _utd_angles(inc_az, out_az, nx, ny)

    # Validate UTD domain: try primary face, then alternate
    n_pi = WEDGE_N * PI
    if phi_prime > n_pi or phi > n_pi:
        pp2, p2 = _utd_angles(inc_az, out_az, -nx, -ny)
        if pp2 <= n_pi and p2 <= n_pi:
            phi_prime, phi = pp2, p2
        else:
            return 60.0

    import cmath

    D = _compute_utd_D(phi, phi_prime, d1, d2, freq_ghz)
    magnitude = abs(D)
    if magnitude < 1e-12:
        return 60.0

    # Physical envelope: diffraction field ≤ 0.5 × free-space field
    L_param = (d1 * d2) / (d1 + d2)
    max_physical_mag = math.sqrt(L_param) * 0.5
    if magnitude > max_physical_mag:
        magnitude = max_physical_mag

    geom_correction = 10.0 * math.log10(L_param)
    diff_loss = -20.0 * math.log10(magnitude)
    total = geom_correction + diff_loss

    if math.isnan(total) or math.isinf(total):
        total = 6.0

    return max(0.0, min(total, 80.0))


# ---------------------------------------------------------------------------
# Full-path loss
# ---------------------------------------------------------------------------


def _segment_3d(
    p1: np.ndarray,
    p2: np.ndarray,
    h1: float,
    h2: float,
) -> float:
    """3-D slant distance between two 2-D points at different heights."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    dz = h2 - h1
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_path_loss_db(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    chain: tuple[int, ...],
    vertex_xyz: np.ndarray,
    vertex_normal: np.ndarray,
    freq_ghz: float = DEFAULT_FREQ_GHZ,
    tx_height: float | None = None,
    rx_height: float | None = None,
    vertex_height: float | None = None,
) -> float:
    """Ray-trace a single chain and return the total path loss in dB.

    Only the chain's actual segments are evaluated.  Tx/Rx are at 1.5 m
    and every diffraction vertex is at 20 m unless explicitly overridden.
    """
    if tx_height is None:
        tx_height = DEFAULT_TX_HEIGHT
    if rx_height is None:
        rx_height = DEFAULT_RX_HEIGHT
    if vertex_height is None:
        vertex_height = DEFAULT_VERTEX_HEIGHT

    if not chain:
        dx = float(rx_xyz[0] - tx_xyz[0])
        dy = float(rx_xyz[1] - tx_xyz[1])
        return float(np.float32(fspl(math.hypot(dx, dy), freq_ghz)))

    first = int(chain[0])
    accumulated_dist = np.float32(
        _segment_3d(
            tx_xyz[:2],
            vertex_xyz[first, :2],
            tx_height,
            vertex_height,
        )
    )
    parent_dist = np.float32(0.0)
    accumulated_loss = np.float32(0.0)

    for index, raw_vi in enumerate(chain):
        vi = int(raw_vi)
        curr_pt = vertex_xyz[vi, :2]
        prev_pt = tx_xyz[:2] if index == 0 else vertex_xyz[int(chain[index - 1]), :2]
        d1 = (
            float(accumulated_dist)
            if index == 0
            else float(accumulated_dist - parent_dist)
        )

        final = index == len(chain) - 1
        if final:
            next_pt = rx_xyz[:2]
            d2 = _segment_3d(
                curr_pt,
                next_pt,
                vertex_height,
                rx_height,
            )
        else:
            next_vi = int(chain[index + 1])
            next_pt = vertex_xyz[next_vi, :2]
            d2 = _segment_3d(
                curr_pt,
                next_pt,
                vertex_height,
                vertex_height,
            )

        az_in = math.atan2(
            float(curr_pt[1] - prev_pt[1]),
            float(curr_pt[0] - prev_pt[0]),
        )
        az_out = math.atan2(
            float(next_pt[1] - curr_pt[1]),
            float(next_pt[0] - curr_pt[0]),
        )
        utd = utd_diff_loss(
            az_in,
            az_out,
            float(vertex_normal[vi, 0]),
            float(vertex_normal[vi, 1]),
            d1,
            d2,
            freq_ghz,
        )

        if final:
            total = (
                fspl(float(accumulated_dist) + d2, freq_ghz)
                + float(accumulated_loss)
                + utd
            )
            return float(np.float32(total))

        parent_dist = accumulated_dist
        accumulated_dist = np.float32(float(accumulated_dist) + d2)
        accumulated_loss = np.float32(float(accumulated_loss) + utd)

    raise AssertionError("unreachable")
