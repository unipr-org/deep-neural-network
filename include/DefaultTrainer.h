#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "Network.h"
#include "DefaultNetwork.h"
#include "Trainer.h"
#include "Training.h"
#include <fstream>

namespace Training {

class DefaultTrainer : public Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t> {
public:
    using network_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::network_t;
    using stream_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::stream_t;
    using tolerance_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::tolerance_t;
    using step_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::step_t;

    inline void train(network_t& net, stream_t& stream, const tolerance_t tolerance, const size_t epochs, const step_t step) const override {
    }
    inline void test(network_t& net, stream_t& stream) const override {

    }
};
} // namespace Training


#endif

