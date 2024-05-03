#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "Network.h"
#include "DefaultNetwork.h"
#include "Trainer.h"
#include "Training.h"
#include <fstream>

namespace Training {

class DefaultTrainer : public Trainer<ANN::DefaultNetwork, stream_t> {
public:
    using network_t = typename Trainer<ANN::DefaultNetwork, stream_t>::network_t;
    using stream_t = typename Trainer<ANN::DefaultNetwork, stream_t>::stream_t;

    inline void train(network_t& net, stream_t& stream) const override {
    }
    inline void test(network_t& net, stream_t& stream) const override {

    }
};
} // namespace Training


#endif

