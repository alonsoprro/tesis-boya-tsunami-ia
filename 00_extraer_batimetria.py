"""
00_extraer_batimetria.py - Transecto 1D de GEBCO -> perfil_arica.csv (columnas: x_km desde la costa, h_m).
Uso: python 00_extraer_batimetria.py gebco_arica.nc [--lat -19.5] [--xmax 300]
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("nc", nargs="?", default="gebco_arica.nc"); ap.add_argument("--lat", type=float, default=-19.5)
ap.add_argument("--xmax", type=float, default=300.0, help="largo del perfil mar adentro [km]")
a = ap.parse_args()

ds = xr.open_dataset(a.nc)
print("variables:", list(ds.data_vars), " coords:", list(ds.coords))
var = "elevation" if "elevation" in ds else list(ds.data_vars)[0]
tr = ds[var].sel(lat=a.lat, method="nearest").sortby("lon")
lon, z = tr["lon"].values, tr.values.astype(float)       # z>0 tierra, z<0 mar (GEBCO)

# 1) la costa = el punto más al OESTE donde el perfil pasa a tierra viniendo del mar (cruce z>=0)
land = np.flatnonzero(z >= 0)
i_coast = land[0] - 1 if land.size and land[0] > 0 else len(z) - 1   # último punto de mar antes de tierra
# (si hay islas/cerros costeros al oeste, revisa la figura)
lon, z = lon[:i_coast + 1], z[:i_coast + 1]

# 2) lon -> km a lo largo del paralelo, medidos DESDE la costa hacia mar adentro
x_km = (lon[-1] - lon) * 111.32 * np.cos(np.deg2rad(a.lat))
h = np.maximum(-z, 1.0)                                     # profundidad positiva; mínimo 1 m
m = x_km <= a.xmax
x_km, h = x_km[m][::-1], h[m][::-1]                         # x creciente (0 = costa)

# 3) remuestreo a malla regular de 0.5 km (= dx del modelo) y suavizado ligero (~2 km)
xg = np.arange(0, x_km.max(), 0.5)
hg = np.interp(xg, x_km, h)
k = 5; hg = np.convolve(np.pad(hg, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
np.savetxt("perfil_arica.csv", np.c_[xg, hg], delimiter=",", header="x_km,h_m", comments="", fmt="%.2f")
print(f"OK: {len(xg)} puntos, 0-{xg[-1]:.0f} km, h_max={hg.max():.0f} m, h(150)={np.interp(150,xg,hg):.0f} m, "
      f"h(175)={np.interp(175,xg,hg):.0f} m, h(200)={np.interp(200,xg,hg):.0f} m")
if xg[-1] < a.xmax - 1: print(f"AVISO: el .nc solo cubre {xg[-1]:.0f} km mar adentro; baja un recuadro más ancho al oeste.")

# 4) tiempo de viaje costa-boya con c=sqrt(g h):  T = integral dx / sqrt(g h)   (verifica la 'holgura')
c = np.sqrt(9.81 * hg)
for d in (150, 175, 200):
    j = xg <= d
    print(f"T_viaje boya {d} km -> costa = {np.trapezoid(1 / c[j], xg[j] * 1e3) / 60:.1f} min")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.fill_between(xg, -hg, -hg.max() * 1.05, color="0.8"); ax.plot(xg, -hg, "k")
ax.set(xlabel="distancia a la costa [km]", ylabel="cota [m]", title=f"Transecto GEBCO lat {a.lat}°")
plt.tight_layout(); plt.savefig("fig_00_batimetria.png", dpi=130)
