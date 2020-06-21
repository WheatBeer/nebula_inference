#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

namespace nebula {

// Activation function
void elu_activation(float *m_output, unsigned m_size);
void hardtan_activation(float *m_output, unsigned m_size);
void leaky_activation(float *m_output, unsigned m_size);
void lhtan_activation(float *m_output, unsigned m_size);
void linear_activation(float *m_output, unsigned m_size);
void loggy_activation(float *m_output, unsigned m_size);
void logistic_activation(float *m_output, unsigned m_size);
void plse_activation(float *m_output, unsigned m_size);
void ramp_activation(float *m_output, unsigned m_size);
void relie_activation(float *m_output, unsigned m_size);
void relu_activation(float *m_output, unsigned m_size);
void stair_activation(float *m_output, unsigned m_size);
void tanh_activation(float *m_output, unsigned m_size);

} // namespace nebula 
#endif
