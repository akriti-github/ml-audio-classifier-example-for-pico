
#ifndef __KWS_MFCC_H__
#define __KWS_MFCC_H__

#include "arm_math.h"
#include "string.h"

#define SAMP_FREQ 16000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000

#define M_2PI 6.283185307179586476925286766559005

class MFCC{
  private:
    int num_mfcc_features;
    int frame_len;
    int frame_len_padded;
    int mfcc_dec_bits;
    float * frame;
    float * buffer;
    float * mel_energies;
    float * window_func;
    int32_t * fbank_filter_first;
    int32_t * fbank_filter_last;
    float ** mel_fbank;
    float * dct_matrix;
    arm_rfft_fast_instance_f32 * rfft;
    float * create_dct_matrix(int32_t input_length, int32_t coefficient_count); 
    float ** create_mel_fbank();
 
    static inline float InverseMelScale(float mel_freq) {
      return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
    }

    static inline float MelScale(float freq) {
      return 1127.0f * logf (1.0f + freq / 700.0f);
    }

  public:
    MFCC(int num_mfcc_features, int frame_len, int mfcc_dec_bits);
    ~MFCC();
    void mfcc_compute(const int16_t* data, q7_t* mfcc_out);
};

#endif
