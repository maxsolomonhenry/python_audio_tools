import numpy as np


class BlockBuffer:
    """Blocked buffer reading and processing of an input stream.

    Applies a given process to signal blocks of a given length, with a given hop size. To feed the stream, the class
    accepts a function get_input that takes a buffer length as its argument, and returns a signal of that length.

    The loop_through() method accepts an input stream in chunks of length "in_dur_s" (seconds), and builds up frames of
    length "out_dur_s" (seconds), hopping by "hop_size_s" (in seconds). Each frame is fed to "process," a function
    that is input as an argument.

    Args:
        SR (int)            :   Sample rate.
        in_dur_s (flt)      :   Input size, in seconds.
        hop_size_s (flt)    :   Analysis hop size, in seconds.
        out_dur_s (flt)     :   Frame size, in seconds.
        get_input (fcn)     :   A function that takes a sample size (int), and returns a signal of that length (np array).
        process (fcn)       :   A function that accepts a single frame of signal (as np array) and returns nothing.

    Example use:

    my_block = BlockBuffer(SR=16000,
                           in_dur_s=0.2,
                           hop_size_s=0.1,
                           out_dur_s=1.,
                           get_input=audio_helpers.get_istream,     # Request for input from, say, laptop microphone. 
                           process=net_helpers.predict_print)       # Make predictions with a neural net (that requires 1s of audio).

    my_block.loop_through()                                         # Instantiate the loop.


    """
    def __init__(self, get_input,
                 process,
                 SR=16000,
                 in_dur_s=.25,
                 hop_size_s=0.25,
                 out_dur_s=1.):
        self.get_input = get_input
        self.process = process
        self.SR = SR
        self.in_length = int(in_dur_s * self.SR)
        self.hop_size = int(hop_size_s * self.SR)
        self.buffer_size = int(out_dur_s * self.SR)
        self.num_out_hops = int(self.buffer_size // self.hop_size)
        self.num_in_hops = int(self.in_length // self.hop_size)
        self.num_bufs = self.num_out_hops
        self.sound_in = np.zeros(self.in_length)
        self.buffers = np.zeros((self.buffer_size, self.num_bufs))
        self.buf_ptrs = np.arange(self.num_bufs) * self.hop_size
        self.full_buf_idx = self.num_bufs - 1
        self.read_ptr = 0

    def loop_through(self):
        while True:
            self.sound_in = self.get_input(self.in_length)
            if not self.sound_in.any():
                return

            for h in range(self.num_in_hops):
                self.update_buffers()
                self.process(self.get_analysis_buf())
                self.update_pointers()

    def update_buffers(self):
        r_in = self.read_ptr
        r_out = r_in + self.hop_size

        read_chunk = self.sound_in[r_in:r_out]

        for b in range(self.num_bufs):
            w_in = self.buf_ptrs[b]
            w_out = w_in + self.hop_size
            self.buffers[w_in:w_out, b] = read_chunk

    def get_analysis_buf(self):
        return self.buffers[:, self.full_buf_idx]

    def update_pointers(self):
        self.full_buf_idx = np.mod(self.full_buf_idx - 1, self.num_bufs)
        self.buf_ptrs = np.mod(self.buf_ptrs + self.hop_size, self.buffer_size)
        self.read_ptr = np.mod(self.read_ptr + self.hop_size, self.in_length)
