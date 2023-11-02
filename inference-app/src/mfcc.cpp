#include "mfcc.h"

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

