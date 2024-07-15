from preprocessing.interpolation import InterpolationSignal


class Resample:
    def __init__(self, imu_signal, ia_signal, labels, input_freq, output_freq):
        self.imu_signal = imu_signal
        self.ia_signal = ia_signal
        self.labels = labels
        self.interpolated_factor = output_freq/input_freq

    def run_ia_resample(self):
        ia_signals_resampled = []
        for signal in self.ia_signal:
            interpolate_handler = InterpolationSignal(int(len(signal)*self.interpolated_factor))
            x = interpolate_handler.interpolate_signal(signal)
            ia_signals_resampled.append(x)
        return ia_signals_resampled

    def run_imu_resample(self):
        imu_signals_resampled = []
        for signal in self.imu_signal:
            interpolate_handler = InterpolationSignal(int(len(signal)*self.interpolated_factor))
            x = interpolate_handler.interpolate_signal(signal)
            imu_signals_resampled.append(x)
        return imu_signals_resampled

    def run_labels_resample(self):
        labels_resampled = []
        for label in self.labels:
            interpolate_handler = InterpolationSignal(int(len(label)*self.interpolated_factor))
            x = interpolate_handler.interpolate_df(label)
            labels_resampled.append(x)
        return labels_resampled

    def _run_resample(self):
        return self.run_imu_resample(), self.run_ia_resample(), self.run_labels_resample()

