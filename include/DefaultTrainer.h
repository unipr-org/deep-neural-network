#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultLoader.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "Trainer.h"
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <ostream>
#include <spdlog/spdlog.h>
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
	using data_t = typename ANN::data_t;
	using data_v_t = typename ANN::data_v_t;
	using data_vv_t = typename ANN::data_vv_t;

	inline void train(network_t &net, stream_t &stream, const tolerance_t tolerance,
					  const size_t epochs, const step_t step) const override {

		spdlog::info("[train(network_t &net, stream_t &stream, const tolerance_t "
					  "tolerance, const "
					  "size_t epochs, const step_t step)] Starting train()");

		data_vv_t preactivation = net.getEmptyPreActivationVector();
		data_vv_t output = net.getEmptyOutputVector();

		std::string line; // linea in cui viene letto il vettore del training
						  // vectorTrainingSet
		data_t &netOutput = output[net.getSize() - 1][0];
		size_t netSize = net.getSize();

		// Epoche
		data_t error = 0.0;
		for (size_t r = 1; r <= epochs; ++r) {
			error = 0.0;
			size_t training_set_size = 0;

			while (getline(stream, line)) {
				std::vector<ANN::data_t> vectorTrainingSet; // vettore del training set
				++training_set_size;

				std::istringstream iss(line);
				data_t value;

				while (iss >> value) {
					vectorTrainingSet.push_back(value);
				}

				// Ora che ho il vettore del training set devo dividerlo nel vettore x e
				// nel vettore f(x)
				data_v_t x;
				data_t y;

				// Questa cosa va bene nel caso in cui abbiamo solo un neurone
				// nell'ultimo layer.
				// TODO aggiungere un parametro che mi indichi quanti output devo avere
				y = std::move(vectorTrainingSet.back());
				vectorTrainingSet.pop_back();
				x = std::move(vectorTrainingSet);

				/* std::cout << "preEvaluate netOutput: "; */
				/* std::cout << netOutput << std::endl; */
				net.evaluate(x, output, preactivation);

				spdlog::debug("[train(network_t &net, stream_t &stream, const "
							  "tolerance_t tolerance, const "
							  "size_t epochs, const step_t step)] Evaluated");

				/* std::cout << "netOutput: "; */
				/* std::cout << netOutput << std::endl; */
				data_t current_error = y - netOutput;
				/* std::cout << "current_error: "; */
				/* std::cout << current_error << std::endl; */
				spdlog::debug("[train(network_t &net, stream_t &stream, const "
							  "tolerance_t tolerance, const "
							  "size_t epochs, const step_t step)] Current error: {}",
							  current_error);

				backPropagate(net, x, output, preactivation, current_error, step);
				error += std::abs(current_error);

				spdlog::debug("[train(network_t &net, stream_t &stream, const "
							  "tolerance_t tolerance, const "
							  "size_t epochs, const step_t step)] line: {}",
							  training_set_size);
			}
			// Resetta eventuali errori di lettura
			stream.clear();

			// Sposta il cursore di lettura all'inizio del file
			stream.seekg(0, std::ios::beg);

			// spdlog::info("[train(network_t &net, stream_t &stream, const "
			// 			 "tolerance_t tolerance, "
			// 			 "const size_t epochs, const step_t step)] epoch: {}",
			// 			 r);

			error /= training_set_size;
			// spdlog::info("[train(network_t &net, stream_t &stream, const "
			// 			 "tolerance_t tolerance, "
			// 			 "const size_t epochs, const step_t step)] avg_error: {}",
			// 			 error);

			spdlog::info("epoch: {}\t error: {}", r, error); 

			if (error < tolerance) {
				Utils::DefaultLoader l;
				l.saveStatus(net);

				spdlog::info("[train(network_t &net, stream_t &stream, const "
						"tolerance_t tolerance, "
						"const size_t epochs, const step_t step)] Training ended after {} epochs with an avg error: {}",
						r, error);

				return;
			}
		}

		Utils::DefaultLoader l;
		l.saveStatus(net);

		spdlog::info("[train(network_t &net, stream_t &stream, const "
				"tolerance_t tolerance, "
				"const size_t epochs, const step_t step)] Training ended after {} epochs with an avg error: {}",
				epochs, error);

		return;
	}

	inline void backPropagate(network_t &net, const data_v_t &input, data_vv_t &output,
							  data_vv_t &preActivation, const data_t &current_error,
							  const step_t &step) const {

		size_t netSize = net.getSize();
		data_vv_t delta = net.getEmptyPreActivationVector();

		ANN::activationFunction_t g_d = ANN::sigmoid_d;

		delta.back()[0] = g_d(preActivation.back()[0]) * current_error;
		spdlog::debug("Evaluated delta for the last layer: {}", delta.back()[0]);

		for (size_t k = 0; k < output[netSize - 2].size(); ++k) {
			net[netSize - 1][0][k] += (step * delta[netSize - 1][0] * output[netSize - 2][k]);
		}
		spdlog::debug("Evaluated weight for the last layer");

		for (int i = netSize - 2; i >= 0; --i) {

			ANN::DefaultLayer &current_layer = net[i];
			ANN::DefaultLayer &following_layer = net[i + 1];

			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				data_t &current_elem = delta[i][j];

				// delta[i][j] evaluation
				current_elem = g_d(preActivation[i][j]);
				spdlog::debug("preActivation[{}][{}]: {}", i, j, preActivation[i][j]);

				data_t sum = 0.0;
				size_t next_layer_size = net[i + 1].getSize();

				for (size_t s = 0; s < next_layer_size; ++s) {
					spdlog::debug("delta[{}][{}]: {}", i + 1, s, delta[i + 1][s]);
					spdlog::debug("[{}][{}]:", s, j);
					spdlog::debug("following_layer[{}][{}]: {}", s, j, following_layer[s][j]);
					sum += (following_layer[s][j] * delta[i + 1][s]);
					spdlog::debug("sum: {}", sum);
				}
				current_elem *= sum;

				spdlog::debug("Evaluated delta[{}][{}]: {}", i, j, current_elem);

				// weight update
				if (i == 0) {
					// case first layer (output[-1] is the input)
					for (size_t k = 0; k <= input.size(); ++k) {
						current_layer[j][k] += step * current_elem * input[k];
						spdlog::debug("Evaluated w[{}][{}][{}]: {}", i, j, k, current_layer[j][k]);
					}
				} else {
					// case inner layers
					for (size_t k = 0; k < output[i - 1].size(); ++k) {
						current_layer[j][k] += step * current_elem * output[i - 1][k];
						spdlog::debug("Evaluated w[{}][{}][{}]: {}", i, j, k, current_layer[j][k]);
					}
				}
			}
			spdlog::debug("Finished layer");
		}

		spdlog::debug("Exit backPropagate()");
	}

	inline void test(network_t &net, stream_t &stream) const override {}
};
} // namespace Training

#endif
