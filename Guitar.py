import numpy as np



class The_String_of_Guitar:
    def __init__(self, pitch, starting_sample, the_freq_of_sampling, the_factor_of_stretch):
        self.the_pitch = pitch
        self.the_starting_sample = starting_sample
        self.the_sampling_freq = the_freq_of_sampling
        self.stretch_factor = the_factor_of_stretch
        self.The_initiation_of_wavetable()
        self.The_Present_Sample = 0
        self.The_Previous_Sample = 0


    def Getting_the_Sample(self):
        if self.The_Present_Sample >= self.the_starting_sample:
            The_Present_Sample_mod = self.The_Present_Sample % self.the_wavetable_of_guitar.size
            drawn_samples = np.random.binomial(1, 1 - 1 / self.stretch_factor)
            if drawn_samples == 0:
                self.the_wavetable_of_guitar[The_Present_Sample_mod] = 0.5 * (
                        self.the_wavetable_of_guitar[The_Present_Sample_mod] + self.The_Previous_Sample)
            The_Needed_sample = self.the_wavetable_of_guitar[The_Present_Sample_mod]
            self.The_Previous_Sample = The_Needed_sample
            self.The_Present_Sample += 1
        else:
            self.The_Present_Sample += 1
            The_Needed_sample = 0
        return The_Needed_sample

    def The_initiation_of_wavetable(self):
        the_size_of_piano_wavetable = self.the_sampling_freq // int(self.the_pitch)
        self.the_wavetable_of_guitar = (2 * np.random.randint(0, 2, the_size_of_piano_wavetable) - 1).astype(float)