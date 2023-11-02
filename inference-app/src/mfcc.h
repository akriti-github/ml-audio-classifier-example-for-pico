#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>
#include <stddef.h>

class MFCC {
public:
    MFCC(int num_mfcc_coeffs, int frame_size, int num_fft_points);
    ~MFCC();

    int init();
    void extract_mfcc(const int16_t* input, float32_t* output);

private:
    int _num_mfcc_coeffs;
    int _frame_size;
    int _num_fft_points;
    int16_t* _hanning_window;
    int16_t* _mel_filterbank;

    // Other private member variables and functions specific to the implementation

    // Define any private functions used internally by the class
    void apply_mel_filterbank(const int16_t* spectrum, float32_t* mel_energies);
    void compute_mfcc(float32_t* mfcc_output, const float32_t* mel_energies);
};

#endif  // MFCC_H
