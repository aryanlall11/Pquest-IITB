import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 0.01
T = 6*10**6
wo = 2.87*10**9;

n = 5000
l = 0.97
h = 1.03

w = np.linspace(l*wo, h*wo, n)
f = 1 - c*((T/2)**2)/((T/2)**2 + (w-wo)**2);
#x_volts = 10*np.sin(t/(2*np.pi))
plt.subplot(3,1,1)
plt.figure(figsize=(10,10))
plt.plot(w, f)
plt.title('Signal without noise')
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.savefig('NoNoise.png')
plt.show()

x_watts = f ** 2
target_snr_db = 67
# Calculate signal power and convert to dB 
sig_avg_watts = np.mean(x_watts)
sig_avg_db = 10 * np.log10(sig_avg_watts)
# Calculate noise according to [2] then convert to watts
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
# Noise up the original signal
f_n = f + noise_volts
# Plot signal with noise
plt.subplot(3,1,2)
plt.figure(figsize=(10,10))
plt.plot(w, f_n)
plt.title('Signal with noise')
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.savefig('Noise.png')
plt.show()

###########################################################################################
def center(wi):
    d_index = 1+round(10**6/(w[1]-w[0]))
    delta = 10*(w[np.int(d_index)]-w[0])
    p1 = 1-f_n[np.int(round((n-1)*(wi-l*wo)/((h-l)*wo)))]
    p2 = 1-f_n[np.int(round((n-1)*(wi+delta-l*wo)/((h-l)*wo)))]
    p3 = 1-f_n[np.int(round((n-1)*(wi-delta-l*wo)/((h-l)*wo)))]
    wc = wi - 0.5*delta*((1/p2)-(1/p3))/((1/p2)+(1/p3)-(2/p1))
    return wc;
###########################################################################################
    
fc = []
omega = []
j = w[w<2.858*10**9].shape[0]
for i in range(n-2*j):
    fc.append(center(w[i+j]))
    omega.append(w[i+j])
   
#print(fc)
plt.subplot(3,1,3)
plt.figure(figsize=(10,10))
x_coordinates = [w[j-n/25],w[n-j+n/25]]
y_coordinates = [2.87*10**(9), 2.87*10**(9)]
plt.plot(x_coordinates, y_coordinates)
plt.plot(omega, fc)
plt.title('Stabilization with f_c = 2.87GHz')
plt.xlabel('Starting Frequency')
plt.ylabel('Stabilized Frequency')
plt.savefig('Freq.png')
plt.show()
###########################################################################################
## Plot in dB
#y_watts = y_volts ** 2
#y_db = 10 * np.log10(y_watts)
#plt.subplot(2,1,2)
#plt.plot(t, 10* np.log10(y_volts**2))
#plt.title('Signal with noise (dB)')
#plt.ylabel('Power (dB)')
#plt.xlabel('Time (s)')
#plt.show()
