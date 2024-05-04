#ifndef TRAINER_INCLUDE_GUARD
#define TRAINER_INCLUDE_GUARD

#include "Training.h"

namespace Training {

/**
 * @brief A template class representing a trainer for neural networks.
 * 
 * @tparam Network_t The type of neural network used by the trainer.
 * @tparam Stream_t The type of data stream used by the trainer.
 * @tparam Tolerance_t The type representing the tolerance for convergence during training.
 * @tparam Step_t The type representing the step size for training.
 */
template <typename Network_t = network_t, typename Stream_t = stream_t, typename Tolerance_t = tolerance_t, typename Step_t = step_t> class Trainer {
public:
    
    using network_t = Network_t; /**< Type definition for the neural network type. */
    using stream_t = Stream_t; /**< Type definition for the data stream type. */
     using tolerance_t = Tolerance_t; /**< Type definition for the tolerance type. */
    using step_t = Step_t; /**< Type definition for the step size type. */

    /**
     * @brief Trains the neural network with the provided data stream.
     * 
     * @param net The neural network to train.
     * @param stream The data stream used for training.
     * @param tolerance The tolerance for convergence during training.
     * @param epochs The maximum number of epochs for training.
     * @param step The step size for training.
     */
    inline virtual void train(network_t& net, stream_t& stream, const tolerance_t tolerance, const size_t epochs, const step_t step) const = 0;
    
    
    /**
     * @brief Tests the neural network with the provided data stream.
     * 
     * @param net The neural network to test.
     * @param stream The data stream used for testing.
     */
    inline virtual void test(network_t& net, stream_t& stream) const = 0;
};
} // namespace Training


#endif

