use mel_spec::mel::{mel, log_mel_spectrogram, norm_mel};
use mel_spec::stft;
use ndarray::Array2;
use ndarray_npy::NpzReader;

//const MEL_FILTERS_FILENAME: &str = "mel_filters.npz";

pub(crate) struct LogMelSpectrogram {
    n_fft: usize,
    hop_size: usize,
    filters: Array2<f64>,
}

impl LogMelSpectrogram {
    pub fn new(n_fft: usize, sampling_rate: f64, n_mels: usize, hop_size: usize) -> Self {
        let mut npz = NpzReader::new(File::open(model_path.as_ref().join(MEL_FILTERS_FILENAME))?)?;
        let mel_filters = npz.by_name(MEL_FILTERS_NAME)?;
        let filters = mel(sampling_rate, n_fft, n_mels, None, None, false, true);
        Self { n_fft, hop_size, filters }
    }

    pub fn extract_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features: Vec<f32> = vec![];
        let mut stft = stft::Spectrogram::new(self.n_fft, self.hop_size);
        for hop in audio.chunks(self.hop_size) {
            if let Some(fft_frame) = stft.add(hop) {
                let mel = norm_mel(&log_mel_spectrogram(&fft_frame, &self.filters)).mapv(|v| v as f32);
                features.extend(mel);
            }
        }
        if audio.len() % self.hop_size != 0 {
            let n = self.hop_size - (audio.len() % self.hop_size);
            let zeros: Vec<f32> = vec![0.0; n];
            let fft_frame = stft.add(&zeros).unwrap();
            let mel = norm_mel(&log_mel_spectrogram(&fft_frame, &self.filters)).mapv(|v| v as f32);
            features.extend(mel);
        }
        features
    }
}

const SAMPLING_RATE: f64 = 16000.0;
const N_FFT: usize = 400;
const N_MELS: usize = 80;
const HOP_SIZE: usize = 160;
// feature_size = 128
// nb_max_frames = 3000

impl Default for LogMelSpectrogram {
    fn default() -> Self {
        Self::new(N_FFT, SAMPLING_RATE, N_MELS, HOP_SIZE)
    }
}