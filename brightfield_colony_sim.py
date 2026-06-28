import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
import chromatix.functional as cx

print("=" * 60)
print("Brightfield Colony Simulation — Halo & Shade-off Demo")
print("=" * 60)

# ── Physical parameters (same as single-cell sim) ─────────────────────
WAVELENGTH = 0.532   # µm — green LED
NA         = 0.80
N_MEDIUM   = 1.00
N_CELL     = 1.38
DN_CELL    = N_CELL - N_MEDIUM   # 0.38

# Grid — extended to fit 3×2 colony
SHAPE = (400, 400)
DX    = 0.05           # µm/pixel
FOV   = SHAPE[0] * DX  # 20 µm

# Cell geometry
CELL_RADIUS = 0.5
CELL_LENGTH = 4.0
CYL_HALF    = (CELL_LENGTH - 2 * CELL_RADIUS) / 2   # 1.5 µm
CELL_Z      = 2 * CELL_RADIUS                        # 1.0 µm
N_SLICES    = 20
DZ_SLICE    = CELL_Z / N_SLICES                      # 0.05 µm

# Partial coherence
NA_COND = 0.70 * NA
DN_IMAG = 0.002
N_ILL   = 61

# Camera noise
PHOTON_SCALE   = 5000
READ_NOISE_STD = 2.0
SEED = 42

# Focus depths to image
Z_DEPTHS = [0.0, -0.5, -1.0, -2.0]

# ── Colony layout: 6 cells in 2 rows × 3 columns ──────────────────────
# COL_SEP = centre-to-centre distance along the long axis (small end-cap gap).
# ROW_SEP = centre-to-centre distance across the short axis (cells nearly touching).
COL_SEP = CELL_LENGTH + 0.25   # 4.25 µm  →  0.25 µm gap between end caps
ROW_SEP = 2 * CELL_RADIUS + 0.1  # 1.10 µm  →  0.10 µm gap between sides

CELLS = [
    # (x_centre_µm, y_centre_µm, angle_deg)
    (-COL_SEP,  ROW_SEP / 2, 0),   # top-left
    (      0.0, ROW_SEP / 2, 0),   # top-centre
    ( COL_SEP,  ROW_SEP / 2, 0),   # top-right
    (-COL_SEP, -ROW_SEP / 2, 0),   # bottom-left
    (      0.0,-ROW_SEP / 2, 0),   # bottom-centre
    ( COL_SEP, -ROW_SEP / 2, 0),   # bottom-right
]

print(f"\n  Colony : {len(CELLS)} cells, {COL_SEP:.2f} µm column pitch, "
      f"{ROW_SEP:.2f} µm row pitch")
print(f"  Grid   : {SHAPE},  dx={DX} µm  →  FOV={FOV:.1f} µm")

# ── Pixel coordinate grids ─────────────────────────────────────────────
cy_pix = SHAPE[0] // 2
cx_pix = SHAPE[1] // 2
_ys = np.arange(SHAPE[0])
_xs = np.arange(SHAPE[1])
YY, XX = np.meshgrid(_ys, _xs, indexing="ij")   # (H, W)
YY_um = (YY - cy_pix) * DX   # physical y in µm
XX_um = (XX - cx_pix) * DX   # physical x in µm

# ── Per-cell phase and absorption projections ──────────────────────────
def cell_projections(cx_um, cy_um, angle_deg):
    """Projected phase and absorption maps for one spherocylinder."""
    theta  = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dX = XX_um - cx_um
    dY = YY_um - cy_um
    # Rotate to the cell's local coordinate frame
    d_along =  dX * cos_t + dY * sin_t   # along long axis
    d_perp  = -dX * sin_t + dY * cos_t   # perpendicular

    phase  = np.zeros(SHAPE, dtype=np.float64)
    absorb = np.zeros(SHAPE, dtype=np.float64)

    for k in range(N_SLICES):
        z_c   = -CELL_RADIUS + (k + 0.5) * DZ_SLICE
        r_eff = float(np.sqrt(max(0.0, CELL_RADIUS**2 - z_c**2)))

        body      = (np.abs(d_perp) <= r_eff) & (np.abs(d_along) <= CYL_HALF)
        cap_left  = (d_along + CYL_HALF)**2 + d_perp**2 <= r_eff**2
        cap_right = (d_along - CYL_HALF)**2 + d_perp**2 <= r_eff**2
        mask = (body | cap_left | cap_right).astype(np.float64)

        phase  += mask * DN_CELL * DZ_SLICE
        absorb += mask * DZ_SLICE   # occupancy × thickness

    phase  *= 2.0 * np.pi / WAVELENGTH
    absorb *= (2.0 * np.pi / WAVELENGTH) * DN_IMAG
    return phase, absorb


print("\n  Building colony phase map ...", flush=True)
phase_proj  = np.zeros(SHAPE, dtype=np.float64)
absorb_proj = np.zeros(SHAPE, dtype=np.float64)
for cx_um, cy_um, ang in CELLS:
    p, a = cell_projections(cx_um, cy_um, ang)
    phase_proj  += p
    absorb_proj += a

print(f"  Phase range : {phase_proj.min():.2f} … {phase_proj.max():.2f} rad")

# ── CTF (partial-coherence) imaging ───────────────────────────────────
_fy = np.fft.fftfreq(SHAPE[0], d=DX)
_fx = np.fft.fftfreq(SHAPE[1], d=DX)
_FX, _FY = np.meshgrid(_fx, _fy, indexing="ij")
_NU2    = _FX**2 + _FY**2
_PHI_FT = np.fft.fft2(phase_proj)
_ABS_FT = np.fft.fft2(absorb_proj)

# Fibonacci spiral condenser sampling
_phi_g  = 2.399963229
_k_arr  = np.arange(N_ILL)
_r_ill  = (NA_COND / WAVELENGTH) * np.sqrt((_k_arr + 0.5) / N_ILL)
_t_ill  = _phi_g * _k_arr
_ill_fx = np.concatenate([[0.0], _r_ill * np.cos(_t_ill)])
_ill_fy = np.concatenate([[0.0], _r_ill * np.sin(_t_ill)])
_N_ILL  = len(_ill_fx)


def partial_coherent_brightfield(z_f: float) -> np.ndarray:
    I_sum = np.zeros(SHAPE, dtype=np.float64)
    lam_n = WAVELENGTH / N_MEDIUM
    for fxi, fyi in zip(_ill_fx, _ill_fy):
        W       = np.pi * lam_n * z_f * (_NU2 + 2.0 * (_FX * fxi + _FY * fyi))
        P_shift = (np.sqrt((_FX + fxi)**2 + (_FY + fyi)**2) <= NA / WAVELENGTH).astype(np.float64)
        I_FT    = -2.0 * P_shift * (np.sin(W) * _PHI_FT + np.cos(W) * _ABS_FT)
        I_FT[0, 0] += float(SHAPE[0] * SHAPE[1])   # DC = flat background
        I_sum   += np.real(np.fft.ifft2(I_FT))
    return np.clip(I_sum / _N_ILL, 0.0, None)


def apply_sensor_noise(img: np.ndarray, key: jax.Array) -> np.ndarray:
    key_shot, key_read = random.split(key)
    i_max   = float(img.max()) + 1e-30
    photons = jnp.array(img) / i_max * PHOTON_SCALE
    shot    = cx.basic_sensor(photons, shot_noise_mode="poisson",
                               noise_key=key_shot, input_spacing=DX)
    return np.array(shot + READ_NOISE_STD * random.normal(key_read, shot.shape))


print("  Imaging at each depth ...", flush=True)
master_key   = random.PRNGKey(SEED)
images_clean = []
images_noisy = []

for k, z_f in enumerate(Z_DEPTHS):
    img = partial_coherent_brightfield(z_f)
    images_clean.append(img)
    images_noisy.append(apply_sensor_noise(img, random.fold_in(master_key, k)))
    print(f"    z = {z_f:+.1f} µm  σ={np.std(img):.4f}")

# ── Global normalisation (shared scale across all depths) ─────────────
def norm_global(imgs):
    lo = min(i.min() for i in imgs)
    hi = max(i.max() for i in imgs)
    return [(i - lo) / (hi - lo + 1e-30) for i in imgs]

imgs_clean_n = norm_global(images_clean)
imgs_noisy_n = norm_global(images_noisy)

# Display crop: colony spans ≈ ±6.25 µm in x and ±1.05 µm in y
XWIN = (-7.5, 7.5)
YWIN = (-3.0, 3.0)
EXT  = [-FOV/2, FOV/2, -FOV/2, FOV/2]

# ── Figure 1: Clean focus sweep ────────────────────────────────────────
n   = len(Z_DEPTHS)
fig1, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
fig1.suptitle(
    f"Brightfield colony — {len(CELLS)} spherocylinders "
    f"({2*CELL_RADIUS:.0f}×{CELL_LENGTH:.0f} µm each) | clean CTF model\n"
    f"λ={WAVELENGTH} µm  NA={NA}  n_cell={N_CELL}  σ_cond={NA_COND/NA:.2f}",
    fontsize=12, fontweight="bold",
)
for k, (z_f, img) in enumerate(zip(Z_DEPTHS, imgs_clean_n)):
    ax = axes[k]
    im = ax.imshow(img, cmap="gray", origin="lower", extent=EXT, vmin=0, vmax=1)
    ax.set_title(f"z = {z_f:+.1f} µm", fontsize=11, fontweight="bold")
    ax.set_xlabel("x (µm)", fontsize=9)
    if k == 0:
        ax.set_ylabel("y (µm)", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(*XWIN)
    ax.set_ylim(*YWIN)
    ax.tick_params(labelsize=8)
plt.colorbar(im, ax=axes[-1], label="Norm. intensity", shrink=0.85)
plt.tight_layout()
plt.savefig("colony_clean_focus_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Figure 1 saved → colony_clean_focus_sweep.png")

# ── Figure 2: Noisy focus sweep ────────────────────────────────────────
fig2, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
fig2.suptitle(
    f"Brightfield colony — {len(CELLS)} cells | CTF + camera noise\n"
    f"Poisson ({PHOTON_SCALE} max ph/px) + Gaussian read noise (σ={READ_NOISE_STD} e⁻)",
    fontsize=12, fontweight="bold",
)
for k, (z_f, img) in enumerate(zip(Z_DEPTHS, imgs_noisy_n)):
    ax = axes[k]
    im = ax.imshow(img, cmap="gray", origin="lower", extent=EXT, vmin=0, vmax=1)
    ax.set_title(f"z = {z_f:+.1f} µm", fontsize=11, fontweight="bold")
    ax.set_xlabel("x (µm)", fontsize=9)
    if k == 0:
        ax.set_ylabel("y (µm)", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(*XWIN)
    ax.set_ylim(*YWIN)
    ax.tick_params(labelsize=8)
plt.colorbar(im, ax=axes[-1], label="Norm. intensity", shrink=0.85)
plt.tight_layout()
plt.savefig("colony_noisy_focus_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 2 saved → colony_noisy_focus_sweep.png")

# ── Figure 3: Annotated image at z = -1 µm ────────────────────────────
# This is the key demonstration figure for halo and shade-off.
z_ann     = -1.0
img_ann   = imgs_clean_n[Z_DEPTHS.index(z_ann)]

fig3, ax3 = plt.subplots(figsize=(12, 6.5))
im3 = ax3.imshow(img_ann, cmap="gray", origin="lower", extent=EXT, vmin=0, vmax=1)
ax3.set_title(
    f"Colony brightfield at z = {z_ann:+.1f} µm — halo and shade-off effects\n"
    f"clean CTF model  |  λ={WAVELENGTH} µm  NA={NA}  n_cell={N_CELL}",
    fontsize=12, fontweight="bold",
)
ax3.set_xlabel("x (µm)", fontsize=11)
ax3.set_ylabel("y (µm)", fontsize=11)
ax3.set_xlim(*XWIN)
ax3.set_ylim(*YWIN)
ax3.tick_params(labelsize=9)

# Annotation 1 — HALO: bright ring at top edge of top-left cell
# Cell centre: (-COL_SEP, +ROW_SEP/2) = (-4.25, +0.55) µm
# Top boundary ≈ y = 0.55 + 0.5 = +1.05 µm; halo appears just outside
ax3.annotate(
    "Halo effect\n(bright ring at\ncell boundary)",
    xy=(-COL_SEP, ROW_SEP / 2 + CELL_RADIUS + 0.15),
    xytext=(-6.2, 2.45),
    fontsize=10.5, color="yellow", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="yellow", lw=2.0),
    ha="center",
)

# Annotation 2 — SHADE-OFF: gradual dimming across centre-bottom cell
# Cell centre: (0, -ROW_SEP/2) = (0, -0.55) µm
ax3.annotate(
    "Shade-off\n(intensity fades\ntoward interior)",
    xy=(0.0, -ROW_SEP / 2),
    xytext=(4.0, -2.3),
    fontsize=10.5, color="cyan", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="cyan", lw=2.0),
    ha="center",
)

# Annotation 3 — MERGED HALOS: where end caps of adjacent top-row cells face each other
# End cap of top-left at x = -COL_SEP + 2.0 = -2.25 µm
# End cap of top-centre at x = -2.0 µm
# Midpoint ≈ x = -2.125 µm, y = +0.55 µm
ax3.annotate(
    "Merged / suppressed\nhalo between\nadjacent cells",
    xy=(-COL_SEP / 2 + 0.05, ROW_SEP / 2),
    xytext=(1.5, 2.45),
    fontsize=10.5, color="orange", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="orange", lw=2.0),
    ha="center",
)

plt.colorbar(im3, ax=ax3, label="Norm. intensity", shrink=0.85)
plt.tight_layout()
plt.savefig("colony_annotated_halo_shadeoff.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 3 saved → colony_annotated_halo_shadeoff.png")

# ── Figure 4: Intensity line-profile through bottom row at z = -1 µm ──
# A horizontal cross-section makes the halo peaks and shade-off dip
# clearly visible as quantitative intensity values.
z_prof     = -1.0
img_raw    = images_clean[Z_DEPTHS.index(z_prof)]   # unnormalised — background ≈ 1.0
y_row_pix  = cy_pix + int(round(-ROW_SEP / 2 / DX))  # pixel row for y = -0.55 µm
profile    = img_raw[y_row_pix, :]
x_coords   = (np.arange(SHAPE[1]) - cx_pix) * DX
mask_x     = (x_coords >= XWIN[0]) & (x_coords <= XWIN[1])

fig4, ax4 = plt.subplots(figsize=(12, 4.5))
ax4.plot(x_coords[mask_x], profile[mask_x], color="steelblue", lw=1.8, label="intensity profile")
ax4.axhline(1.0, color="dimgray", lw=1.4, ls="--", label="background level (= 1)")

# Shade vertical bands for each bottom-row cell extent
labelled = False
for cx_um, cy_um, _ in CELLS:
    if abs(cy_um + ROW_SEP / 2) < 0.15:   # select bottom-row cells
        x_lo = cx_um - (CYL_HALF + CELL_RADIUS)
        x_hi = cx_um + (CYL_HALF + CELL_RADIUS)
        lbl  = "cell extent" if not labelled else ""
        ax4.axvspan(x_lo, x_hi, alpha=0.12, color="royalblue", label=lbl)
        ax4.axvline(cx_um, color="royalblue", lw=0.8, ls=":")
        labelled = True

# Annotate halo peaks and shade-off dip on the profile
# Left halo: just outside the left end cap of the leftmost bottom cell
x_halo_left = -COL_SEP - (CYL_HALF + CELL_RADIUS) - 0.15
col_halo    = int(np.clip(cx_pix + round(x_halo_left / DX), 0, SHAPE[1] - 1))
y_halo_val  = profile[col_halo]
ax4.annotate("halo\npeak", xy=(x_halo_left, y_halo_val),
    xytext=(-7.0, float(profile[mask_x].max()) * 0.97),
    fontsize=9.5, color="steelblue",
    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.3), ha="center",
)
# Shade-off dip: centre of the bottom-centre cell (x=0)
y_dip_val = float(img_raw[y_row_pix, cx_pix])
ax4.annotate("shade-off\n(dip inside cell)", xy=(0.0, y_dip_val),
    xytext=(2.2, y_dip_val - 0.07),
    fontsize=9.5, color="steelblue",
    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.3), ha="center",
)

ax4.set_xlabel("x (µm)", fontsize=11)
ax4.set_ylabel("Intensity (a.u.)", fontsize=11)
ax4.set_title(
    f"Intensity profile at y = {-ROW_SEP/2:.2f} µm (through bottom-row cells),  z = {z_prof:+.1f} µm\n"
    "Peaks above background line = halo effect  |  Dip inside cell = shade-off effect",
    fontsize=11, fontweight="bold",
)
ax4.set_xlim(*XWIN)
ax4.legend(fontsize=9, loc="lower right")
ax4.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("colony_intensity_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 4 saved → colony_intensity_profile.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Colony    : {len(CELLS)} spherocylinders, "
      f"{COL_SEP:.2f} µm column pitch, {ROW_SEP:.2f} µm row pitch")
print(f"  Cell      : d={2*CELL_RADIUS} µm, L={CELL_LENGTH} µm,  "
      f"n_cell={N_CELL}  (dn={DN_CELL})")
print(f"  Imaging   : partial coherence (σ={NA_COND/NA:.2f}), "
      f"dn_imag={DN_IMAG}, {_N_ILL} angles")
print(f"  Noise     : Poisson({PHOTON_SCALE} ph/px) + Gaussian(σ={READ_NOISE_STD} e⁻)")
print("  Outputs   : colony_clean_focus_sweep.png")
print("              colony_noisy_focus_sweep.png")
print("              colony_annotated_halo_shadeoff.png")
print("              colony_intensity_profile.png")
print("Done.")
