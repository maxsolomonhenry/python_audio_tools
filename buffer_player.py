import numpy as np
import copy


class BufferPlayer:
    """Receive, and build up, regions of audio while sending out in fixed frames.

    Use the "tick" method, inspired by the Synthesis Toolkit (STK). Each tick outputs one hop-size of audio, while
    optionally receiving-and-passing a input. Ideal for processing that creates an audible tail, e.g. reverbs, ringing
    filters, etc. This will accept processed chunks, add them and their tails to a buffer, and output that buffer one
    hop at a time.

    ::Constructor::

    Args:
        hop_size (int)          :   In/out size in samples.
        num_future_hops(int)    :   Total buffer length as a multiple of hops.

    ::tick()::

    Args:
        sound_in (np array)     :   New sound to be added to the buffer.

    Returns:
        sound_out (np array)    :   Next output of length hop_size.

    Example use:

    my_buf = BufferPlayer(1024, 1000)
    audio_helpers.play_to_ostream( my_buf.tick(a_long_sound) )
    while(True):
        audio_helpers.play_to_ostream( my_buf.tick() )

    """

    def __init__(self, hop_size, num_future_hops):
        self.hop_size = hop_size
        self.buffer_size = int(hop_size * num_future_hops)
        self.buffer = np.zeros(self.buffer_size)
        self.ptr = 0

    def tick(self, sound_in=np.array([0])):
        self.place_in_buf(self.pad_sound(sound_in))

        # Grab from, then clear, buffer chunk.
        sound_out = copy.deepcopy(self.get_sound_out())
        self.clear_oldest_buf()

        self.ptr = self.advance_ptr(self.ptr)
        return sound_out

    def get_sound_out(self):
        return self.buffer[self.ptr:self.ptr+self.hop_size]

    def clear_oldest_buf(self):
        self.buffer[self.ptr:self.ptr + self.hop_size] = np.zeros(self.hop_size)

    def pad_sound(self, sound_in):
        pad_length = np.mod(len(sound_in), self.hop_size)
        return np.pad(sound_in, (0, pad_length))

    def place_in_buf(self, sound_in):
        num_hops = len(sound_in)//self.hop_size

        w_in = self.ptr
        w_out = w_in + self.hop_size
        r_in = 0
        r_out = r_in + self.hop_size

        for h in range(num_hops):
            self.buffer[w_in:w_out] += sound_in[r_in:r_out]

            w_in = self.advance_ptr(w_in)
            w_out = w_in + self.hop_size
            r_in += self.hop_size
            r_out = r_in + self.hop_size

    def advance_ptr(self, ptr):
        return np.mod(ptr + self.hop_size, self.buffer_size)


if __name__ == '__main__':
    my_buf = BufferPlayer(1024, 1000)

