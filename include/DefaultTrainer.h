#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "Trainer.h"
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
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

		spdlog::debug("[train(network_t &net, stream_t &stream, const tolerance_t tolerance, const "
					  "size_t epochs, const step_t step)] Starting train()");

		data_vv_t preactivation = net.getEmptyOutputVector();
		data_vv_t output = net.getEmptyOutputVector();

		std::string line; // linea in cui viene letto il vettore del training vectorTrainingSet
		data_t &netOutput = output[net.getSize() - 1][0];
		size_t netSize = net.getSize();
		size_t error = 0.0;

		// Epoche
		for (size_t r = 1; r <= epochs; ++r) {

			size_t training_set_size = 0;

			while (stream) {
				std::vector<ANN::data_t> vectorTrainingSet; // vettore del training set
				getline(stream, line); // leggo la linea del training training_set
				++training_set_size;

				std::istringstream iss(line);
				data_t value;

				while (iss >> value) {
					vectorTrainingSet.push_back(value);
				}

				// Ora che ho il vettore del training set devo dividerlo nel vettore x e nel vettore
				// f(x)
				data_v_t x;
				data_t y;

				// Questa cosa va bene nel caso in cui abbiamo solo un neurone nell'ultimo layer.
				// TODO aggiungere un parametro che mi indichi quanti output devo avere
				y = std::move(vectorTrainingSet.back());
				vectorTrainingSet.pop_back();

				x = std::move(vectorTrainingSet);

				net.evaluate(x, output, preactivation);
				for (auto v : output)
					ANN::operator<<(std::cout, v) << std::endl;

				spdlog::debug(
					"[train(network_t &net, stream_t &stream, const tolerance_t tolerance, const "
					"size_t epochs, const step_t step)] Evaluated");

				data_t current_error = y - netOutput;
				spdlog::debug(
					"[train(network_t &net, stream_t &stream, const tolerance_t tolerance, const "
					"size_t epochs, const step_t step)] Current error: {}",
					current_error);

				backPropagate(net, x, output, preactivation, current_error, step);
				error += std::abs(current_error);
			}

			error /= training_set_size;
			if (error < tolerance) {
				std::string status = net.getStatus(); // !TODO: sistemare
				std::ofstream model("model.txt");
				model << status;

				model.close();
				return;
			}

			spdlog::debug("[train(network_t &net, stream_t &stream, const tolerance_t tolerance, "
						  "const size_t epochs, const step_t step)] epoch: {}",
						  r);
		}

		std::string status = net.getStatus(); // !TODO: sistemare
		std::ofstream model("model.txt");
		model << status;

		model.close();
		return;
	}

	inline void backPropagate(network_t &net, const data_v_t &input, data_vv_t &output,
							  data_vv_t &preActivation, const data_t &current_error,
							  const step_t &step) const {

		size_t netSize = net.getSize();
		data_vv_t delta = net.getEmptyOutputVector();

		ANN::activationFunction_t g_d = ANN::sigmoid_d;

		delta[netSize - 1][0] = g_d(preActivation[netSize - 1][0]) * current_error;
		spdlog::debug("Evaluated delta for the last layer: {}", delta[netSize - 1][0]);

		for (size_t k = 0; k < output[netSize - 2].size(); ++k) {
			net[netSize - 1][0][k] += step * delta[netSize - 1][k] * output[netSize - 2][k];
		}
		spdlog::debug("Evaluated weight for the last layer");

		for (auto v : output)
			ANN::operator<<(std::cout, v) << std::endl;
		for (int i = netSize - 2; i >= 0; --i) {

			ANN::DefaultLayer &current_layer = net[i];
			ANN::DefaultLayer &following_layer = net[i + 1];

			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				data_t &current_elem = delta[i][j];

				// delta[i][j] evaluation
				current_elem = g_d(preActivation[i][j]);
				spdlog::debug("preActivation[{}][{}]: {}", i, j, preActivation[i][j]);

				data_t sum = 0.0;
				for (size_t s = 0; s < output[i + 1].size(); ++s) {
					spdlog::debug("delta[{}][{}]: {}", i + 1, s, delta[i + 1][s]);
					spdlog::debug("following_layer[{}][{}]: {}", s, j, following_layer[s][j]);
					sum += static_cast<long double>(following_layer[s][j] * delta[i + 1][s]);
					spdlog::debug("sum: {}", sum);
				}
				current_elem *= sum;

				spdlog::debug("Evaluated delta[{}][{}]: {}", i, j, current_elem);

				// weight update
				if (i == 0) {
					ANN::operator<<(std::cout, input) << std::endl;
					for (size_t k = 0; k <= input.size(); ++k) {
						current_layer[j][k] += step * current_elem * input[k];
						spdlog::debug("Evaluated w[{}][{}][{}]: {}", i, j, k, current_layer[j][k]);
					}
				} else {
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
