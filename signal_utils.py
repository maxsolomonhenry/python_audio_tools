import numpy as np

def make_signal(f0=440, 
                num_harmonics=3,
                fm=5, 
                I=10, 
                dur=1, 
                fs=44100,
                rand_phase=False, 
                rand_vib_phase=False):
  """
      Simple generator for complex signals with modulation. 
  """
    
    t = np.linspace(0, dur, int(dur*fs), endpoint=False)
    x = np.zeros(t.shape)

    for h in range(1, num_harmonics + 1):
        mod_phase = rand_vib_phase * 2*np.pi*np.random.rand()
        modulation = I * h/fm * np.cos(2 * np.pi * fm * t + mod_phase)
        phase = rand_phase * 2 * np.pi * np.random.rand()
        x += np.cos(2 * np.pi * h*f * t + modulation + phase)

    return x
