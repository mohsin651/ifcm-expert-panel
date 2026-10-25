# CPU job: download the LIDC scans missing from the Kaggle copy straight from TCIA (public, no login),
# crop every rated nodule exactly as lidc-prep does (5 axial slices around the readers' mean centroid), delete the scan.
# Nodule ids match lidc-prep: <patient>_<last 8 chars of series UID>_<cluster index>.
import os, time, glob, io, zipfile, shutil, subprocess, sys, signal
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc"], check=True, timeout=900)
import numpy as np
for a, t in [("int", int), ("bool", bool), ("float", float), ("object", object)]:
    if not hasattr(np, a): setattr(np, a, t)
import configparser
if not hasattr(configparser, "SafeConfigParser"): configparser.SafeConfigParser = configparser.ConfigParser
import requests, pydicom
import pylidc as pl

SMOKE = os.environ.get("SMOKE", "0") == "1"
PART, NPART = int(os.environ.get("PART", "0")), 4
CROP = 64            # mm field of view, as in lidc-prep
SCAN_LIMIT = 900     # s per scan (download + crop)
OUT, TMP = "/kaggle/working", "/kaggle/tmp_scan"
URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="

scans = [s for s in pl.query(pl.Scan).all() if s.patient_id in set(MISSING)]
scans = sorted(scans, key=lambda s: s.patient_id)[PART::NPART]
if SMOKE: scans = scans[:3]
print(f"part {PART}: {len(scans)} scans to fetch", flush=True)

class ScanTimeout(Exception): pass
def _alarm(*_): raise ScanTimeout()
signal.signal(signal.SIGALRM, _alarm)

crops, t0, n_ok, n_fail = {}, time.time(), 0, 0
for si, scan in enumerate(scans):
    signal.alarm(SCAN_LIMIT)
    try:
        nods = scan.cluster_annotations()
        if not nods: continue
        shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
        r = requests.get(URL + scan.series_instance_uid, timeout=300)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(TMP)
        fs = [os.path.join(dp, f) for dp, _, fn in os.walk(TMP) for f in fn if f.endswith(".dcm")]
        hd = [(float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]), f) for f in fs]
        vol = [f for _, f in sorted(hd)]
        for ni, anns in enumerate(nods):
            nid = f"{scan.patient_id}_{scan.series_instance_uid[-8:]}_{ni}"
            c = np.mean([a.centroid for a in anns], 0).round().astype(int)      # (row, col, slice)
            half = int(round(CROP / 2 / scan.pixel_spacing))
            r0, c0 = max(c[0] - half, 0), max(c[1] - half, 0)
            sl = []
            for d in (-2, -1, 0, 1, 2):
                ds = pydicom.dcmread(vol[min(max(c[2] + d, 0), len(vol) - 1)])
                img = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                sl.append(img[r0:r0 + 2 * half, c0:c0 + 2 * half])
            crops[nid] = np.stack(sl, -1).astype(np.int16)
        n_ok += 1
        if len(fs) != len(set(f for _, f in hd)) or len(vol) < 10:
            print("warning: few slices", scan.patient_id, len(vol), flush=True)
    except ScanTimeout:
        n_fail += 1; print("timeout", scan.patient_id, flush=True)
    except Exception as e:
        n_fail += 1; print("fail", scan.patient_id, repr(e)[:200], flush=True)
    finally:
        signal.alarm(0); shutil.rmtree(TMP, ignore_errors=True)
    if si % 10 == 0:
        print(f"{si + 1}/{len(scans)} scans, ok {n_ok}, fail {n_fail}, crops {len(crops)}, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"{OUT}/lidc_crops_tcia_{PART}.npz", **crops)

np.savez_compressed(f"{OUT}/lidc_crops_tcia_{PART}.npz", **crops)
print(f"done: ok {n_ok}, fail {n_fail}, crops {len(crops)}, {time.time() - t0:.0f}s")
