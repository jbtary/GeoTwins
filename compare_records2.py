import numpy as np
import matplotlib.pyplot as plt
import nc_array_processing as dnc
import glob, os
from tqdm import tqdm
from obspy import Trace
from obspy.signal import PPSD
from obspy import UTCDateTime
from scipy import signal
from obspy import read, read_inventory

dnc.process.set_num_threads(80)
# Load DAS record
xsec_febus = dnc.read.vds('/mnt/geofib/1/2025-GRANGEGORMAN_NOISE/Virtual_Datasets/10-08_CH1_grangegorman.vds.h5',time=['2025-10-09T08:30:00.0000Z','2025-10-09T08:45:00.0000Z'],chan=[5,6],model='febus')

xsec_rmed,med = dnc.process.remove_median(dnc.process.detrend(xsec_febus))

#Load and remove BB response
BB_data = '/mnt/REPO/25-GRANGEGORMAN/MINISEED/2025/DB/TUD01/HHZ.D/DB.TUD01..HHZ.D.2025.282'
BB_response= '/mnt/REPO/25-GRANGEGORMAN/RESPONSES/DB.TUD01.xml'

st = read(BB_data)
st_processed = st.copy()
inv = read_inventory(BB_response)
inv.plot_response(min_freq=0.001, outfile="sensor_response.png")
pre_filt = (0.005, 0.008, 45, 50)
start = UTCDateTime("2025-10-09T08:30:00")
end = UTCDateTime("2025-10-09T08:45:00")

#Trim data
st_processed.trim(start, end, pad=True, fill_value=0)
print('trimmed')
st_processed.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt, water_level=60)
print('response removed')
data_clipped = st_processed[0].data
fs = st_processed[0].stats.sampling_rate
nperseg = int(150 * fs) 

#Broadband PSD
fBB, PxxBB = signal.welch(data_clipped, fs=fs, nperseg=nperseg)

Pxx_dbBB = 10 * np.log10(PxxBB)

print(xsec_rmed['data'].shape)
print(xsec_rmed['data'].min(),xsec_rmed['data'].max())
dt=.01
GL=20
import ipdb;ipdb.set_trace()
print(xsec_febus['data'].shape)
#displacement = np.cumsum(xsec_febus['data']/1e9) * (dt * GL)
#displacement_detrended = signal.detrend(displacement)

#DAS Psd
strain = np.cumsum(xsec_febus['data']/1e9) * dt
# Not always used - in progress
C=200
v_das = -C * strain

#f, Pxx = signal.welch(strain.flatten()/1e9, fs=100, nperseg=150*100)
f, Pxx = signal.welch(xsec_febus['data'].flatten()/1e9, fs=100, nperseg=150*100)

Pxx_db = 10 * np.log10(Pxx)

# Plot
plt.figure(figsize=(10, 5))
plt.semilogx(f, Pxx_db,label='DAS')
plt.semilogx(fBB,Pxx_dbBB,label='BB')# Log scale for frequency

plt.xlabel('Frequency (Hz)')
plt.ylabel('dB [rel to (units)^2/Hz]')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(.01,50)
plt.ylim(-200,-50)
plt.legend(loc='best')
plt.savefig("scipy_psd_both.png", dpi=300)
plt.close('all')

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xscale('log')
ax1.semilogx(f, Pxx_db, color='blue', label='DAS')
ax1.set_ylabel('dB [w.r.t to 1 strain²/Hz]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.semilogx(fBB, Pxx_dbBB, color='orange', label='BB')
ax2.set_ylabel('dB [rel to 1 (m/s)²/Hz]', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_xlim(0.01, 50)
#ax1.set_ylim(-170, -100)
#ax2.set_ylim(-170, -100)
ax1.grid(True, which="both", ls="-", alpha=0.3)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig("scipy_psd_twin_both2.png", dpi=300)
