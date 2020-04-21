import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import butter,filtfilt
import matplotlib.animation as animation
from random import randint

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
ys_fil = []
w_local = []
i_local = []

f = f_ref = 2*2.2813*10**3             # In Hertz
no_samples_per_cycle = 20
fc = 300           # Cutoff frequency of LPF, Hz
fs = no_samples_per_cycle*f       # sample rate, Hz
no_of_cycles = 100       # no. of cycles in plot
nyq = 0.5 * fs  # Nyquist Frequency
#b, a = butter(2, fc/nyq, btype='low', analog=False)
alpha = np.exp(-2*np.pi*fc/fs)  # Used for defining filter

################################################### Lorentzian    

wo = 2.87*10**9
f_dev = 320*10**3

w_lo = 2.872*10**9
w_lo_init = w_lo
    
def lorent_val(t):
    c = 0.01
    T = 6*10**6
    w = w_lo + f_dev*np.cos(2*np.pi*f_ref*t)
    snr = randint(55, 62)       # 62-72
    noise_avg_watts = 10 ** (-snr / 10)
    n = np.random.normal(0, np.sqrt(noise_avg_watts), 1)[0]
    return (1 - c*((T/2)**2)/((T/2)**2 + (w-wo)**2)) + n;
    #return (1 - c*((T/2)**2)/((T/2)**2 + (w-wo)**2));


#################################################### Feed-Back

delta = 1.5*10**6      # delta (MHz) = 0.29*T(in MHz) - 0.24
kp=450
integral = 0
y_init=0

i_init=0
def feedback(i):  
    global xs,ys,ys_fil,w_local,i_local,w_lo,i_init, integral,y_init,w_lo_init,kp
    if(int(i*f)==i_init):
        print(i_init)
        i_init+=1
    # Signal generator
    #sig_v = np.cos(2*np.pi*f_ref*i)**2
    sig_v = (1/(1-lorent_val(i))) * np.cos(2*np.pi*f_ref*i);
    y = (1-alpha)*sig_v + alpha*y_init
    y_init = y
    xs.append(i)
    ys.append(y)
    integral += 100*y/fs
    i_local.append(integral)
    w_lo = w_lo_init - integral*delta
    w_local.append(w_lo)
    # Draw x and y lists
    ax.clear()
    #ax.plot(xs[-2*no_samples_per_cycle:], ys_fil[-2*no_samples_per_cycle:])
    ax.plot(xs, w_local)
    plt.title('Time vs LO Frequency')
    plt.ylabel('Local Oscillator Frequency')
    plt.xlabel('Time (s)')
    
# Set up plot to call animate() function perodically
ani = animation.FuncAnimation(fig, feedback,frames=np.linspace(0,no_of_cycles/f,(fs+1)*no_of_cycles/f))    # Refer no. of samples in one cycle
ani.save('Lock-in.mp4', writer = 'ffmpeg', fps = 30) 
plt.show()
###
plt.title('Lock-in signal')
plt.ylabel('Lock-in Voltage')
plt.xlabel('Time (s)')
plt.plot(xs,ys)
#plt.plot(xs[-len(w_local):],w_local)