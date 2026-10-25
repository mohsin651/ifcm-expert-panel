# LIDC-IDRI prep: (1) every reader's attribute + malignancy ratings per nodule (pylidc DB, no images needed)
# (2) nodule crops (central axial slice +-2, 64x64 px at ~1 mm) for the vision side.
import os, glob, time, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc"], check=True, timeout=900)
import numpy as np
for a, t in [("int", int), ("bool", bool), ("float", float), ("object", object)]:   # pylidc uses removed numpy aliases
    if not hasattr(np, a): setattr(np, a, t)
import pandas as pd, configparser
if not hasattr(configparser, "SafeConfigParser"): configparser.SafeConfigParser = configparser.ConfigParser   # removed in py3.12

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_CROP_SCANS = 15 if SMOKE else 10**6
CROP = 64          # px
OUT = "/kaggle/working"

# pylidc expects <root>/<patient_id>/...; the Kaggle copy is sharded, so symlink every patient dir into one root
root = "/kaggle/working/lidc_root"; os.makedirs(root, exist_ok=True)
pdirs = glob.glob("/kaggle/input/**/LIDC-IDRI-[0-9][0-9][0-9][0-9]", recursive=True)
for d in pdirs:
    dst = os.path.join(root, os.path.basename(d))
    if not os.path.exists(dst): os.symlink(d, dst)
print("patient dirs found:", len(pdirs), flush=True)
open(os.path.expanduser("~/.pylidcrc"), "w").write(f"[dicom]\npath = {root}\nwarn = False\n")
import pylidc as pl

ATTRS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture", "malignancy"]
scans = pl.query(pl.Scan).all()
PART, NPART = int(os.environ.get("PART", "0")), 4
print("scans in pylidc DB:", len(scans), flush=True)

rows, t0, n_crop_scans, crops = [], time.time(), 0, {}
import signal
class ScanTimeout(Exception): pass
def _alarm(*_): raise ScanTimeout()
signal.signal(signal.SIGALRM, _alarm)
SCAN_LIMIT = 90     # s per scan; a stuck DICOM read must not hang the whole job
for si, scan in enumerate(scans):
    have_imgs = os.path.exists(os.path.join(root, scan.patient_id)) and n_crop_scans < N_CROP_SCANS and si % NPART == PART
    try:
        nods = scan.cluster_annotations()
    except Exception as e:
        print("cluster fail", scan.patient_id, repr(e)); continue
    vol = None
    if have_imgs: signal.alarm(SCAN_LIMIT)
    for ni, anns in enumerate(nods):
        nid = f"{scan.patient_id}_{scan.series_instance_uid[-8:]}_{ni}"
        for ri, a in enumerate(anns):
            r = {"nodule": nid, "patient": scan.patient_id, "reader_idx": ri, "n_readers": len(anns),
                 "diameter": a.diameter, "slice_thickness": scan.slice_thickness, "pixel_spacing": scan.pixel_spacing}
            r.update({k: getattr(a, k) for k in ATTRS})
            rows.append(r)
        if have_imgs:
            try:
                if vol is None:                                     # headers only: slice order by z, as pylidc does
                    import pydicom
                    fs = [os.path.join(dp, f) for dp, _, fn in os.walk(scan.get_path_to_dicom_files()) for f in fn if f.endswith(".dcm")]
                    hd = [(float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]), f) for f in fs]
                    vol = [f for _, f in sorted(hd)]
                c = np.mean([a.centroid for a in anns], 0).round().astype(int)      # (row, col, slice)
                half = int(round(CROP / 2 / scan.pixel_spacing))                    # ~CROP mm field of view
                r0, c0 = max(c[0] - half, 0), max(c[1] - half, 0)
                sl = []
                for d in (-2, -1, 0, 1, 2):
                    ds = pydicom.dcmread(vol[min(max(c[2] + d, 0), len(vol) - 1)])
                    img = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                    sl.append(img[r0:r0 + 2 * half, c0:c0 + 2 * half])
                patch = np.stack(sl, -1)
                crops[nid] = patch.astype(np.int16)
            except ScanTimeout:
                print("scan timeout", nid, flush=True); break
            except Exception as e:
                print("crop fail", nid, repr(e), flush=True)
    signal.alarm(0)
    if have_imgs and vol is not None:
        n_crop_scans += 1
        if n_crop_scans % 25 == 0: np.savez_compressed(f"{OUT}/lidc_crops_{PART}.npz", **crops)
    if si % 100 == 0:
        print(f"{si}/{len(scans)} scans, {len(rows)} ratings, {n_crop_scans} scans cropped, {time.time()-t0:.0f}s", flush=True)
    if SMOKE and n_crop_scans >= N_CROP_SCANS and si > 200:
        break

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/lidc_ratings.csv", index=False)
g = df.groupby("nodule")
nod = g.agg(n_readers=("n_readers", "first"), mal_mean=("malignancy", "mean"), mal_std=("malignancy", "std"),
            spic_std=("spiculation", "std"), margin_std=("margin", "std"), lob_std=("lobulation", "std"))
print("\nnodules:", len(nod), "| readers per nodule:", nod.n_readers.value_counts().sort_index().to_dict())
m = nod[nod.n_readers >= 3]
print("nodules with >=3 readers:", len(m))
print("malignancy rating std (>=3 readers) quantiles:", m.mal_std.quantile([.25, .5, .75, .9]).round(2).to_dict())
print("fraction with readers disagreeing on malignancy side (some <=2 and some >=4):",
      round(float(g.malignancy.apply(lambda v: (v.min() <= 2) and (v.max() >= 4)).loc[m.index].mean()), 3))
for k in ["spiculation", "margin", "lobulation", "texture", "sphericity", "calcification"]:
    print(f"   {k:14s} mean within-nodule std {g[k].std().loc[m.index].mean():.2f}")
np.savez_compressed(f"{OUT}/lidc_crops_{PART}.npz", **crops)
print("crops saved:", len(crops), "| total time", round(time.time() - t0), "s")
