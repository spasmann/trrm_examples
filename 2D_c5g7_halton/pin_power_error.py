import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import openmc

def reformat(arr, dev):
    arr = np.rot90(arr, 3)
    dev = np.rot90(dev, 3)
    val = arr.sum()
    factor = ((17 * 17 * 4) - (25 * 4)) / val
    arr = arr * factor
    dev = dev * factor
    return arr[0:34,0:34], dev[0:34,0:34]

def get_err(fname):
    # Load the statepoint file
    sp = openmc.StatePoint(fname)
    tally = sp.get_tally(scores=['fission'])
    fission = tally.get_slice(scores=['fission'])
    TIME = sp.runtime["total"]

    dim = 51
    fission.std_dev.shape = (dim, dim)
    fission.mean.shape = (dim, dim)

    trrm_power, trrm_std_dev = reformat(fission.mean, fission.std_dev)

    #np.savetxt("trrm.csv", trrm_power, delimiter=",")

    ref = np.loadtxt("reference.txt")
    ref = ref.reshape((34, 34))
    val = ref.sum()
    factor = ((17 * 17 * 4) - (25 * 4)) / val
    ref = ref * factor

    trrm_power[trrm_power == 0] = np.nan
    ref[ref == 0] = np.nan

    rel_diff = (trrm_power - ref) / ref

    rel_diff_squared = np.power(rel_diff, 2)

    RMS = math.sqrt(np.nanmean(rel_diff_squared))

    abs_percent_diff = np.absolute(rel_diff)

    RMS = 100.0 * RMS
    AAPE = 100.0 * np.nanmean(abs_percent_diff)
    MAX = 100.0 *np.nanmax(abs_percent_diff)

    return RMS, AAPE, MAX, TIME

N = 20
sim_num = np.arange(1,N)
RMS_halton = np.zeros(N)
AAPE_halton = np.zeros(N)
MAX_halton = np.zeros(N)
TIME_halton = np.zeros(N)

RMS_random = np.zeros(N)
AAPE_random = np.zeros(N)
MAX_random = np.zeros(N)
TIME_random = np.zeros(N)

for i in sim_num:
    fname_halton = f"results/halton_rays/sim_{i}.h5"
    fname_random = f"results/random_rays/sim_{i}.h5"
    RMS_halton[i], AAPE_halton[i], MAX_halton[i], TIME_halton[i] = get_err(fname_halton)
    RMS_random[i], AAPE_random[i], MAX_random[i], TIME_random[i] = get_err(fname_random)

RMS_FOM_halton = 1 / (RMS_halton.mean()**2 * TIME_halton.mean())
AAPE_FOM_halton = 1 / (AAPE_halton.mean()**2 * TIME_halton.mean())
MAX_FOM_halton = 1 / (MAX_halton.mean()**2 * TIME_halton.mean())

RMS_FOM_random = 1 / (RMS_random.mean()**2 * TIME_random.mean())
AAPE_FOM_random = 1 / (AAPE_random.mean()**2 * TIME_random.mean())
MAX_FOM_random = 1 / (MAX_random.mean()**2 * TIME_random.mean())

print("\n------> Halton Ray <------")
print("RMS Mean  = ", RMS_halton.mean())
print("RMS Sdev  = ", RMS_halton.std())
print("RMS FoM  = ", RMS_FOM_halton)

print("\nAAPE Mean = ", AAPE_halton.mean())
print("AAPE Sdev = ", AAPE_halton.std())
print("AAPE FoM  = ", AAPE_FOM_halton)

print("\nMAX Mean = ", MAX_halton.mean())
print("MAX Sdev = ", MAX_halton.std())
print("MAX FoM  = ", MAX_FOM_halton)

print("\n------> Random Ray <------")
print("RMS Mean  = ", RMS_random.mean())
print("RMS Sdev  = ", RMS_random.std())
print("RMS FoM  = ", RMS_FOM_random)

print("\nAAPE Mean = ", AAPE_random.mean())
print("AAPE Sdev = ", AAPE_random.std())
print("AAPE FoM  = ", AAPE_FOM_random)

print("\nMAX Mean = ", MAX_random.mean())
print("MAX Sdev = ", MAX_random.std())
print("MAX FoM  = ", MAX_FOM_random)

print("\n------> FoM Percent Increase <------")
print(f"RMS FoM Increase = {((RMS_FOM_halton - RMS_FOM_random)/RMS_FOM_random * 100 ).round(2)}%")
print(f"AAPE FoM Increase = {((AAPE_FOM_halton - AAPE_FOM_random)/AAPE_FOM_random * 100 ).round(2)}%")
print(f"MAX FoM Increase = {((MAX_FOM_halton - MAX_FOM_random)/MAX_FOM_random * 100 ).round(2)}%")

print()