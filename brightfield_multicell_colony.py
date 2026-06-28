import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("Brightfield Multi-Cell Colony – Halo & Shade-off Demo")
print("=" * 60)

# ── Physical parameters (same optical setup as single-cell sim) ──────────────
WAVELENGTH  = 0.532    # µm
NA          = 0.80
N_MEDIUM    = 1.00
N_CELL      = 1.38
DN_CELL     = N_CELL - N_MEDIUM   # 0.38

SHAPE       = (512, 512)
DX          = 0.05     # µm/pixel
FOV         = SHAPE[0] * DX  # 25.6 µm

CELL_RADIUS = 0.5      # µm
CELL_LENGTH = 4.0      # µm
CYL_HALF    = (CELL_LENGTH - 2 * CELL_RADIUS) / 2   # 1.5 µm
CELL_Z      = 2 * CELL_RADIUS
N_SLICES    = 20
DZ_SLICE    = CELL_Z / N_SLICES

NA_COND     = 0.70 * NA
DN_IMAG     = 0.002
N_ILL       = 61

print(f"  Grid: {SHAPE}, dx={DX} µm, FOV={FOV:.1f} µm")
print(f"  Cell: d={2*CELL_RADIUS} µm, L={CELL_LENGTH} µm, n_cell={N_CELL}")
print(f"  Optics: λ={WAVELENGTH} µm, NA={NA}, σ={NA_COND/NA:.2f}")

# ── Pixel grids ───────────────────────────────────────────────────────────────
cy_pix       = SHAPE[0] // 2
cx_pix       = SHAPE[1] // 2
YY, XX       = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
half_cyl_pix = CYL_HALF / DX

# ── Frequency grids and condenser illumination angles ─────────────────────────
_FY, _FX = np.meshgrid(
    np.fft.fftfreq(SHAPE[0], d=DX),
    np.fft.fftfreq(SHAPE[1], d=DX),
    indexing="ij"
)
_NU2 = _FX**2 + _FY**2

phi_g       = 2.399963229   # golden angle in radians
kk          = np.arange(N_ILL)
r_ill       = (NA_COND / WAVELENGTH) * np.sqrt((kk + 0.5) / N_ILL)
t_ill       = phi_g * kk
ill_fx      = np.concatenate([[0.0], r_ill * np.cos(t_ill)])
ill_fy      = np.concatenate([[0.0], r_ill * np.sin(t_ill)])
N_ILL_TOTAL = len(ill_fx)

# Pre-compute effective radius per z-slice (same for every cell)
_z_c      = -CELL_RADIUS + (np.arange(N_SLICES) + 0.5) * DZ_SLICE
_r_eff_px = np.sqrt(np.maximum(0.0, CELL_RADIUS**2 - _z_c**2)) / DX


# ── Multi-cell phase map builder (vectorised over z-slices) ───────────────────
def build_colony_maps(cell_positions_um, cell_angles_rad=None):
    """
    Build projected phase and absorption maps for a list of cells.

    cell_positions_um : list of (x, y) µm relative to image centre
    cell_angles_rad   : long-axis orientation for each cell (0 = along X)
    """
    if cell_angles_rad is None:
        cell_angles_rad = [0.0] * len(cell_positions_um)

    phase_proj  = np.zeros(SHAPE, dtype=np.float64)
    absorb_proj = np.zeros(SHAPE, dtype=np.float64)

    for (cx_um, cy_um), theta in zip(cell_positions_um, cell_angles_rad):
        cx_off = cx_um / DX
        cy_off = cy_um / DX
        dX_raw = (XX - (cx_pix + cx_off)).astype(np.float64)
        dY_raw = (YY - (cy_pix + cy_off)).astype(np.float64)

        # Rotate pixel offsets into the cell-local frame
        cos_t  = np.cos(theta)
        sin_t  = np.sin(theta)
        dX_loc =  dX_raw * cos_t + dY_raw * sin_t   # (H, W)
        dY_loc = -dX_raw * sin_t + dY_raw * cos_t

        # Broadcast: compare (N_SLICES, 1, 1) radius against (1, H, W) offsets
        dX3 = dX_loc[np.newaxis]         # (1, H, W)
        dY3 = dY_loc[np.newaxis]
        r3  = _r_eff_px[:, None, None]   # (N_SLICES, 1, 1)

        body  = (np.abs(dY3) <= r3) & (np.abs(dX3) <= half_cyl_pix)
        cap_l = (dX3 + half_cyl_pix)**2 + dY3**2 <= r3**2
        cap_r = (dX3 - half_cyl_pix)**2 + dY3**2 <= r3**2
        mask3 = (body | cap_l | cap_r).astype(np.float32)   # (N_SLICES, H, W)

        # Sum over z-slices to get local thickness at each XY position
        thickness = mask3.sum(axis=0) * DZ_SLICE   # µm
        phase_proj  += thickness * DN_CELL
        absorb_proj += thickness

    phase_proj  *= 2.0 * np.pi / WAVELENGTH
    absorb_proj *= 2.0 * np.pi / WAVELENGTH * DN_IMAG
    return phase_proj, absorb_proj


# ── Partially coherent brightfield imaging (Hopkins CTF model) ────────────────
def image_colony(phase_proj, absorb_proj, z_f):
    """Compute the partially coherent brightfield intensity at defocus z_f µm."""
    PHI_FT = np.fft.fft2(phase_proj)
    ABS_FT = np.fft.fft2(absorb_proj)
    I_sum  = np.zeros(SHAPE, dtype=np.float64)
    lam_n  = WAVELENGTH / N_MEDIUM
    for fxi, fyi in zip(ill_fx, ill_fy):
        W       = np.pi * lam_n * z_f * (_NU2 + 2.0 * (_FX * fxi + _FY * fyi))
        P_shift = (np.sqrt((_FX + fxi)**2 + (_FY + fyi)**2) <= NA / WAVELENGTH).astype(np.float64)
        I_FT    = -2.0 * P_shift * (np.sin(W) * PHI_FT + np.cos(W) * ABS_FT)
        I_FT[0, 0] += float(SHAPE[0] * SHAPE[1])
        I_sum   += np.real(np.fft.ifft2(I_FT))
    return np.clip(I_sum / N_ILL_TOTAL, 0.0, None)


# ── Cell configurations ───────────────────────────────────────────────────────
# Cells are packed close together but not overlapping.
# gap_y < diffraction resolution (λ/2NA ≈ 0.33 µm) so cells appear touching in images.
gap_y = 0.05   # µm clearance between laterally adjacent cells
gap_x = 0.10   # µm clearance between end-to-end cells
cs    = CELL_LENGTH + gap_x   # 4.10 µm  (column spacing along X)
rs    = 2 * CELL_RADIUS + gap_y  # 1.05 µm  (row spacing along Y)

# 1. Single reference cell
pos_single = [(0.0, 0.0)]

# 2. Linear chain of 4 cells end-to-end along X
pos_chain = [((-1.5 + i) * cs, 0.0) for i in range(4)]

# 3. 2 × 4 cluster (8 cells: 2 rows, 4 columns)
n_r2, n_c2 = 2, 4
pos_cluster = [
    ((c - (n_c2 - 1) / 2.0) * cs, (r - (n_r2 - 1) / 2.0) * rs)
    for r in range(n_r2) for c in range(n_c2)
]

# 4. 3 × 4 microcolony (12 cells: 3 rows, 4 columns)
n_r4, n_c4 = 3, 4
pos_colony = [
    ((c - (n_c4 - 1) / 2.0) * cs, (r - (n_r4 - 1) / 2.0) * rs)
    for r in range(n_r4) for c in range(n_c4)
]

print(f"\n  Building phase maps …", flush=True)
maps_single  = build_colony_maps(pos_single)
maps_chain   = build_colony_maps(pos_chain)
maps_cluster = build_colony_maps(pos_cluster)
maps_colony  = build_colony_maps(pos_colony)
print("  Phase maps built.")

Z_DEMO  = -1.0   # defocus (µm) that most clearly shows halo
Z_SWEEP = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

print(f"\n  Computing images at z={Z_DEMO} µm …", flush=True)
img_single  = image_colony(*maps_single,  Z_DEMO)
img_chain   = image_colony(*maps_chain,   Z_DEMO)
img_cluster = image_colony(*maps_cluster, Z_DEMO)
img_colony  = image_colony(*maps_colony,  Z_DEMO)
print(f"  Done.  Focus sweep: {len(Z_SWEEP)} depths …", flush=True)

colony_sweep = []
for z in Z_SWEEP:
    print(f"    z = {z:+.1f} µm …", flush=True)
    colony_sweep.append(image_colony(*maps_colony, z))
print("  Focus sweep done.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-30)


def add_outlines(ax, positions):
    for cx_um, cy_um in positions:
        ax.add_patch(plt.Rectangle(
            (cx_um - CELL_LENGTH / 2, cy_um - CELL_RADIUS),
            CELL_LENGTH, 2 * CELL_RADIUS,
            lw=0.7, edgecolor="lime", facecolor="none", ls="--"
        ))


# ── Figure 1: Halo & shade-off — 4 cell densities at z = Z_DEMO ───────────────
print("\n  Saving Figure 1 …", flush=True)

fig1, axes = plt.subplots(1, 4, figsize=(22, 5.5))
fig1.suptitle(
    f"Brightfield images at z = {Z_DEMO:+.1f} µm defocus  (λ={WAVELENGTH} µm, NA={NA})\n"
    "Halo = bright ring at cell edges (defocus artifact of phase objects)  ·  "
    "Shade-off = interior brightening in dense colonies (overlapping halos raise local background)",
    fontsize=10, fontweight="bold"
)
panels_f1 = [
    (img_single,  "Single cell  (1 cell)\nReference",              [(-3.5, 3.5), (-2.5, 2.5)], pos_single),
    (img_chain,   "Linear chain  (4 cells)\nend-to-end",           [(-10.5, 10.5), (-3, 3)],   pos_chain),
    (img_cluster, "2 × 4 cluster  (8 cells)",                      [(-10.5, 10.5), (-3, 3)],   pos_cluster),
    (img_colony,  "3 × 4 microcolony  (12 cells)\n← shade-off →",  [(-10.5, 10.5), (-3, 3)],   pos_colony),
]
for ax, (img, title, (xlim, ylim), pos) in zip(axes, panels_f1):
    im = ax.imshow(norm01(img), cmap="gray", origin="lower",
                   extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    add_outlines(ax, pos)
    plt.colorbar(im, ax=ax, label="Norm. intensity", shrink=0.85)
fig1.tight_layout()
out1 = "brightfield_multicell_halo_shadeoff.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 1 saved → {out1}")


# ── Figure 2: Focus sweep of 3 × 4 colony ────────────────────────────────────
print("  Saving Figure 2 …", flush=True)

fig2, axes2 = plt.subplots(1, len(Z_SWEEP), figsize=(3.8 * len(Z_SWEEP), 5))
fig2.suptitle(
    f"3×4 Microcolony — focus depth sweep  "
    f"({n_r4} rows × {n_c4} cols, d={2*CELL_RADIUS:.0f} µm, L={CELL_LENGTH:.0f} µm)\n"
    "Halo rings narrow/expand with defocus; shade-off persists across all depths in the interior",
    fontsize=10, fontweight="bold"
)
g_min = min(i.min() for i in colony_sweep)
g_max = max(i.max() for i in colony_sweep)
for ax, z_f, img in zip(axes2, Z_SWEEP, colony_sweep):
    img_n = (img - g_min) / (g_max - g_min + 1e-30)
    im2 = ax.imshow(img_n, cmap="gray", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2], vmin=0, vmax=1)
    ax.set_title(f"z = {z_f:+.1f} µm", fontsize=9)
    ax.set_xlim(-11, 11)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x (µm)", fontsize=8)
    ax.tick_params(labelsize=7)
    if ax is axes2[0]:
        ax.set_ylabel("y (µm)", fontsize=8)
    else:
        ax.set_yticklabels([])
plt.colorbar(im2, ax=axes2[-1], label="Norm. intensity", shrink=0.8)
fig2.tight_layout()
out2 = "brightfield_multicell_focus_sweep.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 2 saved → {out2}")


# ── Figure 3: Intensity profiles showing shade-off quantitatively ─────────────
print("  Saving Figure 3 …", flush=True)

def bg_normalise(arr, margin=20):
    """Normalise to mean corner background so profiles have the same scale."""
    corners = [arr[:margin, :margin], arr[:margin, -margin:],
               arr[-margin:, :margin], arr[-margin:, -margin:]]
    bg = np.mean([c.mean() for c in corners])
    return arr / (bg + 1e-30)

prof_x       = (np.arange(SHAPE[1]) - cx_pix) * DX
row_mid      = cy_pix  # row index for y = 0

prof_single  = bg_normalise(img_single)[row_mid, :]
prof_chain   = bg_normalise(img_chain)[row_mid, :]
prof_cluster = bg_normalise(img_cluster)[row_mid, :]
prof_colony  = bg_normalise(img_colony)[row_mid, :]

fig3, ax3 = plt.subplots(figsize=(13, 4.5))
ax3.plot(prof_x, prof_single,  color="royalblue",   lw=2.0, alpha=0.9, label="Single cell (1)")
ax3.plot(prof_x, prof_chain,   color="seagreen",    lw=1.8, alpha=0.8, label="Chain (4 cells)")
ax3.plot(prof_x, prof_cluster, color="darkorange",  lw=1.8, alpha=0.8, label="2×4 cluster (8 cells)")
ax3.plot(prof_x, prof_colony,  color="crimson",     lw=2.0, alpha=0.9, label="3×4 colony (12 cells)")
ax3.axhline(1.0, color="gray", ls=":", lw=1.0, label="Background (1.0)")

# Shade the colony X extent
cx_min = min(p[0] for p in pos_colony) - CELL_LENGTH / 2
cx_max = max(p[0] for p in pos_colony) + CELL_LENGTH / 2
ax3.axvspan(cx_min, cx_max, alpha=0.07, color="red", label="Colony footprint")

ax3.set_xlabel("x (µm)", fontsize=11)
ax3.set_ylabel("Intensity / background", fontsize=11)
ax3.set_title(
    f"Horizontal intensity profiles at y = 0, z = {Z_DEMO:+.1f} µm\n"
    "Shade-off: interior intensity rises toward background as colony density increases "
    "(halos of neighbouring cells fill in dark dips)",
    fontsize=10
)
ax3.set_xlim(-13, 13)
ax3.set_ylim(0.70, 1.35)
ax3.legend(fontsize=9.5)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
out3 = "brightfield_multicell_profiles.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 3 saved → {out3}")


# ── Figure 4: Phase maps (shows the structure of packed cells) ────────────────
print("  Saving Figure 4 …", flush=True)

fig4, axes4 = plt.subplots(1, 4, figsize=(22, 5))
fig4.suptitle(
    "Projected phase maps φ(x,y) — structural input to the imaging model\n"
    "Colour = cumulative phase shift (rad) accumulated through cell thickness at each XY position",
    fontsize=10, fontweight="bold"
)
phase_panels = [
    (maps_single[0],  "Single cell",          [(-3.5, 3.5), (-2.5, 2.5)], pos_single),
    (maps_chain[0],   "Chain of 4",            [(-10.5, 10.5), (-3, 3)],  pos_chain),
    (maps_cluster[0], "2×4 cluster (8 cells)", [(-10.5, 10.5), (-3, 3)],  pos_cluster),
    (maps_colony[0],  "3×4 colony (12 cells)", [(-10.5, 10.5), (-3, 3)],  pos_colony),
]
for ax, (phase, title, (xlim, ylim), pos) in zip(axes4, phase_panels):
    im4 = ax.imshow(phase, cmap="hot", origin="lower",
                    extent=[-FOV/2, FOV/2, -FOV/2, FOV/2])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    plt.colorbar(im4, ax=ax, label="Phase shift (rad)", shrink=0.85)
fig4.tight_layout()
out4 = "brightfield_multicell_phase_maps.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 4 saved → {out4}")


print("\n" + "=" * 60)
print("OUTPUTS")
print("=" * 60)
print(f"  {out1}  — halo & shade-off comparison across 4 densities")
print(f"  {out2}  — 3×4 colony focus sweep (7 depths)")
print(f"  {out3}  — horizontal intensity profiles")
print(f"  {out4}  — projected phase maps")
print("\nKey observations:")
print("  • Halo  : bright ring at each cell edge, strongest at z ≈ ±1 µm defocus")
print("  • Shade-off: colony interior cells show reduced dark–light contrast")
print("    because halos from neighbouring cells raise the local background,")
print("    making interior cells harder to segment than isolated cells.")
print("Done.")
