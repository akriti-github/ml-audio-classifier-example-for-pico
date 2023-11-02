#include "mfcc.h"
def preemphasis(xn, alpha = 0.97):
    # Using pre-empaphases with a certain alpha
    pre_e_xn = np.zeros((xn.shape))
    
    pre_e_xn[0] = xn[0]
    
    pre_e_xn[1:] = xn[1:] - alpha * xn[:-1]
    
    return pre_e_xn

def get_mel_from_hertz(hertz):
    return 2595 * np.log10(1 + (hertz/ 700))

def get_hertz_from_mel(mel):
    return 700 * (10**(mel / 2595) - 1)

def get_power_spectrum(xn_mag, fft_size=2048):
    return (1/fft_size) * np.power(xn_mag, 2)

def get_triangle_function(prev_freq, cur_freq, nex_freq, filter_banks, bin_fb):
    
    # Ascending Triangle
    
    for freq in range(int(prev_freq), int(cur_freq)):
        
        filter_banks[bin_fb-1,freq] = (freq - prev_freq)/(cur_freq-prev_freq)
        
    # Descending Triangle
    
    for freq in range(int(cur_freq+1), int(nex_freq)):
        
        filter_banks[bin_fb-1, freq] = (nex_freq-freq)/(nex_freq-cur_freq)
        
    # Triangle Tip
    
    filter_banks[bin_fb-1, int(cur_freq)] = 1
    
    return filter_banks

def mel_filter_banks(xn_pow, sr, number_filters, fft_size=2048):
    min_mel = 0
    max_mel = get_mel_from_hertz(sr/2)
    
    mel_freq_points = np.linspace(min_mel, max_mel, num=number_filters+2)
    hertz_freq_points = get_hertz_from_mel(mel_freq_points)
    
    corresponding_bins_hertz_points = np.floor((fft_size + 1) * hertz_freq_points / sr)
    
    # Filter banks have to be of shape number_filters * (fft_size/2) + 1
    filter_banks = np.zeros((number_filters, int(fft_size/2)+1))
    
    for bin_fb in range(1, number_filters+1):
        
        prev_bin = corresponding_bins_hertz_points[bin_fb-1]
        current_bin = corresponding_bins_hertz_points[bin_fb]
        next_bin = corresponding_bins_hertz_points[bin_fb+1]
        
        # Use the triangle function to get the values of the banks
        
        filter_banks = get_triangle_function(prev_bin, current_bin, next_bin, filter_banks, bin_fb)
        
    return filter_banks

def get_delta_values(x):
    delta_x = np.zeros(shape=x.shape)
    for i in range(1,x.shape[1]-1):
        prev_val = x[:,i-1]
        next_val = x[:,i+1]
        
        delta_x[:,i]  = (next_val - prev_val)/2
    
    return delta_x

def mfcc(xn, sr, number_filters, window_size = 500, hopsize=int(500/4), fft_size=512):
    
    # Pre-emphasis
    
    xn = preemphasis(xn)
    
    # Getting the STFT
        
    xn_stft = stft(xn, window_size= window_size, hopsize=hopsize, fft_size=fft_size)
    
    # Getting the Magnitude of the STFT
    
    xn_mag = np.abs(xn_stft)
    
    # Evaluating the Power spectrum for the magnitude
    
    xn_pow = get_power_spectrum(xn_mag, fft_size=fft_size)
    
    # To get the mel filter banks
    
    filter_banks = mel_filter_banks(xn_pow, sr, number_filters, fft_size=fft_size)
    
    machine_epsilon =  2.22044604925e-16
    
    filter_banks[filter_banks==0] = machine_epsilon
    
    
    # Multiply the filter_banks with the power spectrum
    
    filter_banks_res = np.dot(filter_banks, xn_pow.T)
    
    # Taking the log and the inverse DFT
    
    filter_banks_res = filter_banks_res + machine_epsilon
    
    log_filter_bank = np.log(filter_banks_res)
    
    idft = sp.fftpack.dct(log_filter_bank)
    
    # First 12 MFCC Values
    
    first_12 = idft[:12,:]
    
    # delta and delta-delta coefficients
    
    delta = get_delta_values(idft)
    
    delta_delta = get_delta_values(delta)
    
    # Getting Energy values of delta and delta-delta coefficients
    
    first_12_delta = delta[:12,:]
    
    first_12_delta_delta = delta_delta[:12,:]
    
    
    # Energy of the Cepstrum frame. Read from - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.8754&rep=rep1&type=pdf
    
    energy = np.sqrt(np.sum(np.power(first_12,2),axis=0)).reshape(1,-1)
    
    energy_delta = np.sqrt(np.sum(np.power(first_12_delta,2),axis=0)).reshape(1,-1)
    
    energy_delta_delta = np.sqrt(np.sum(np.power(first_12_delta_delta,2),axis=0)).reshape(1,-1)
    
    return np.vstack((energy, energy_delta, energy_delta_delta, first_12, first_12_delta, first_12_delta_delta)), filter_banks

mfcc_xn, filter_banks = mfcc(xn, sr, 40)


MFCC::MFCC(int num_mfcc_coeffs, int frame_size, int num_fft_points) :
    _num_mfcc_coeffs(num_mfcc_coeffs),
    _frame_size(frame_size),
    _num_fft_points(num_fft_points),
    _hanning_window(NULL),
    _mel_filterbank(NULL)
{
}

MFCC::~MFCC()
{
    if (_hanning_window != NULL) {
        delete [] _hanning_window;
        _hanning_window = NULL;
    }
    if (_mel_filterbank != NULL) {
        delete [] _mel_filterbank;
        _mel_filterbank = NULL;
    }
}

int MFCC::init()
{
    // Initialize the hanning window and Mel filterbank, similar to DSPPipeline::init

    return 1;
}

void MFCC::extract_mfcc(const int16_t* input, float32_t* output)
{
    int16_t windowed_input[_frame_size];
    int16_t fft_q15[_frame_size * 2];

    // Apply the MFCC pipeline: Hanning Window + FFT
    arm_mult_q15(_hanning_window, input, windowed_input, _frame_size);
    arm_rfft_q15(&_S_q15, windowed_input, fft_q15);

    // Calculate the magnitude spectrum (similar to DSPPipeline::calculate_spectrum)
    // Compute the power spectrum
    arm_cmplx_mag_q15(fft_q15, fft_mag_q15, _frame_size / 2 + 1);
    
    // Apply Mel filterbank to the power spectrum (specific to MFCC)
    float32_t mel_energies[_num_mfcc_coeffs];
    apply_mel_filterbank(fft_mag_q15, mel_energies);

    // Compute the logarithm of the mel energies
    for (int i = 0; i < _num_mfcc_coeffs; i++) {
        mel_energies[i] = logf(mel_energies[i]);
    }

    // Apply DCT (Discrete Cosine Transform) to obtain MFCC coefficients
    compute_mfcc(output, mel_energies);
}

void MFCC::apply_mel_filterbank(const int16_t* spectrum, float32_t* mel_energies)
{
    // Define the filter bank parameters
    int num_filter_banks = 13;
    int filter_bank_size = _frame_size / 2 + 1; // Half of the FFT size
    
    // Initialize the mel filterbank
    if (_mel_filterbank == NULL) {
        _mel_filterbank = new float32_t[num_filter_banks * filter_bank_size];
        
        // Initialize the mel filterbank with appropriate filter shapes
        // You can use equations like Triangular, Hanning, or other shapes for filters
        // Fill _mel_filterbank with filter coefficients based on filter bank parameters
        // Ensure that the coefficients sum to 1 for each filter
        // This step is essential and depends on your specific filterbank design.
    }
    
    // Apply the mel filter bank to the spectrum
    for (int i = 0; i < num_filter_banks; i++) {
        mel_energies[i] = 0.0;
        for (int j = 0; j < filter_bank_size; j++) {
            mel_energies[i] += _mel_filterbank[i * filter_bank_size + j] * spectrum[j];
        }
    }
}

void MFCC::compute_mfcc(float32_t* mfcc_output, const float32_t* mel_energies)
{
    // Define the number of MFCC coefficients
    int num_mfcc_coeffs = 9;
    
    // Initialize the DCT matrix (you can precompute it)
    // It's a matrix of size num_mfcc_coeffs x 13 (number of filter banks)
    // You can find precomputed DCT matrices in DSP libraries or compute it manually
    
    // Compute the DCT of the mel energies to obtain MFCC coefficients
    for (int i = 0; i < num_mfcc_coeffs; i++) {
        mfcc_output[i] = 0.0;
        for (int j = 0; j < 13; j++) {
            mfcc_output[i] += mel_energies[j] * dct_matrix[i * 13 + j];
        }
    }
}

