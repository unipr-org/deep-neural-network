#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "ANN.h"
#include "DefaultNetwork.h"
#include "Network.h"
#include "Trainer.h"
#include "Training.h"
#include <fstream>
#include <sstream>

namespace Training {

class DefaultTrainer : public Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t> {
  public:
	using network_t =
		typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::network_t;
	using stream_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::stream_t;
	using tolerance_t =
		typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::tolerance_t;
	using step_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::step_t;

	inline void train(network_t &net, stream_t &stream, const tolerance_t tolerance,
					  const size_t epochs, const step_t step) const override {
		std::vector<std::vector<ANN::data_t>> preactivation;
		std::vector<std::vector<ANN::data_t>> output;

		for (int l = 0; l < net.getSize(); ++l) {
			preactivation[l].reserve(net[l].getSize());
			output[l].reserve(net[l].getSize());
		}

		std::string line; // linea in cui viene letto il vettore del training set
		// Epoche
		for (size_t r = 1; r <= epochs; ++r) {

			std::vector<ANN::data_t> vectorTrainingSet; // vettore del training set
			getline(stream, line);						// leggo la linea del training set
			std::istringstream iss(line);
			ANN::data_t value;

			while (iss >> value) {
				vectorTrainingSet.push_back(value);
			}

			// Ora che ho il vettore del training set devo dividerlo nel vettore x e nel vettore
			// f(x)
			std::vector<ANN::data_t> x;
			std::vector<ANN::data_t> y;

			// Questa cosa va bene nel caso in cui abbiamo solo un neurone nell'ultimo layer.
			// TODO aggiungere un parametro che mi indichi quanti output devo avere
			y.push_back(vectorTrainingSet.back());
			x = std::move(vectorTrainingSet);

			/* net.evaluate(x, output, preactivation, output); // TODO rimuovere ridondanza vettore
			 * output */

			// ANN::data_t current_error = 0;
			// current_error = y.at(1) - output[k][j];
		}
	}
	inline void test(network_t &net, stream_t &stream) const override {}
};
} // namespace Training

#endif
