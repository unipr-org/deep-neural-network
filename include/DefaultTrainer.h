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

	inline void train(network_t &net, stream_t &trainingSetStream, stream_t &testSetStream, const tolerance_t tolerance,
					  const size_t epochs, const step_t step) const override {

		spdlog::info("Starting train()");

		data_vv_t preactivation = net.getEmptyPreActivationVector();
		data_vv_t output = net.getEmptyOutputVector();

		std::string line;
		data_t &netOutput = output[net.getSize() - 1][0];
		size_t netSize = net.getSize();
		size_t input_size = net[0][0].getWeights().size() - 1;
		
		std::vector<ANN::data_t> vectorTrainingSet;
		for(size_t i = 0; i < input_size; ++i) {
			vectorTrainingSet.push_back(0);
		}

		Utils::DefaultLoader l;
		data_t error = 0.0;
		data_t accuracy = 0.0;
		
		// Epochs
		for (size_t r = 1; r <= epochs; ++r) {
			error = 0.0;
			accuracy = 0.0;
			size_t training_set_size = 0;

			while (getline(trainingSetStream, line)) {

				++training_set_size;
				
				std::istringstream iss(line);
				data_t value;

				for(size_t i = 0; i < input_size; ++i) {
					iss >> value;
					vectorTrainingSet[i] = value;
				}

				data_t y;
				iss >> y;

				// This is fine in case we only have one neuron in the last layer
				// TODO add a parameter that tells me how many outputs I should have
				net.evaluate(vectorTrainingSet, output, preactivation);

				spdlog::debug("[train(...)] Evaluated");

				data_t current_error = y - netOutput;
				spdlog::debug("[train(...)] Current error: {}", current_error);

				// backPropagate(net, vectorTrainingSet, output, preactivation, current_error, step);
				backPropagateNormalized(net, vectorTrainingSet, output, preactivation, current_error, step);
				
				error += std::abs(current_error);

				spdlog::debug("[train(...)] line: {}", training_set_size);
			}
			// Back to file begin
			trainingSetStream.clear();
			trainingSetStream.seekg(0, std::ios::beg);

			l.saveStatus(net);

			error /= training_set_size;

			accuracy = test(net, testSetStream);

			spdlog::info("epoch: {}\t error: {}\t accuracy: {}", r, error, accuracy); 

			if (error <= tolerance) {
				spdlog::info("[train(...)] Training ended after {} epochs with an avg error of {} and an accuracy of {}", r, error, accuracy);
				return;
			}
		}

		spdlog::info("[train(...)] Training ended after {} epochs with an avg error: {} and an accuracy of {}", epochs, error, accuracy);
		return;
	}

	/**
	 * @brief Performs backpropagation to update the weights of the neural network.
	 *
	 * @param net The neural network to update.
	 * @param input The input data used for forward propagation.
	 * @param output The output data generated during forward propagation.
	 * @param preActivation The pre-activation values of the neurons in each layer.
	 * @param current_error The error value calculated during training.
	 * @param step The step size or learning rate for weight updates.
	 *
	 * This method performs backpropagation to update the weights of the neural network based on the calculated error
	 * during training. It iterates through each layer in the network, calculating the deltas and updating the weights
	 * accordingly.
	 */
	inline void backPropagate(network_t &net, const data_v_t &input, data_vv_t &output,
							  data_vv_t &preActivation, const data_t &current_error,
							  const step_t &step) const {

		size_t netSize = net.getSize();
		data_vv_t delta = net.getEmptyPreActivationVector();

		ANN::activationFunction_t g_d = ANN::getActivationDerivative(net[net.getSize() - 1][0].getActivationFunctionID());

		delta.back()[0] = g_d(preActivation.back()[0]) * current_error;

		spdlog::debug("Evaluated delta for the last layer: {}", delta.back()[0]);

		if(netSize == 1) { // SLP case
			for (size_t k = 0; k < input.size(); ++k) {
				net[netSize - 1][0][k] += (step * delta[netSize - 1][0] * input[k]);
			}
			spdlog::debug("Evaluated weight for the last layer (SLP)");
			return; // exit backPropagate
		}
		
		// MLP case below
		for (size_t k = 0; k < output[netSize - 2].size(); ++k) {
			net[netSize - 1][0][k] += (step * delta[netSize - 1][0] * output[netSize - 2][k]);
		}
		spdlog::debug("Evaluated weight for the last layer");

		for (int i = netSize - 2; i >= 0; --i) {

			ANN::DefaultLayer &current_layer = net[i];
			ANN::DefaultLayer &following_layer = net[i + 1];

			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				data_t &current_elem = delta[i][j];

				ANN::activationFunction_t g_d = ANN::getActivationDerivative(net[i][j].getActivationFunctionID());

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

	/**
	 * @brief Performs normalized backpropagation to update the weights of the neural network.
	 *
	 * @param net The neural network to update.
	 * @param input The input data used for forward propagation.
	 * @param output The output data generated during forward propagation.
	 * @param preActivation The pre-activation values of the neurons in each layer.
	 * @param current_error The error value calculated during training.
	 * @param step The step size or learning rate for weight updates.
	 *
	 * This method performs normalized backpropagation to update the weights of the neural network based on the
	 * calculated error during training. It iterates through each layer in the network, calculating the deltas and
	 * updating the weights accordingly. Additionally, it normalizes the step size based on the error magnitude.
	 */
	inline void backPropagateNormalized(network_t &net, const data_v_t &input, data_vv_t &output,
										data_vv_t &preActivation, const data_t &current_error,
										step_t step) const {

		size_t netSize = net.getSize();
		data_vv_t delta = net.getEmptyPreActivationVector();

		spdlog::debug("preActivation.back()[0]: {}", preActivation.back()[0]);
		spdlog::debug("current_error: {}", current_error);

		ANN::activationFunction_t g_d = ANN::getActivationDerivative(net[net.getSize() - 1][0].getActivationFunctionID());
		
		delta.back()[0] = g_d(preActivation.back()[0]) * current_error;

		spdlog::debug("Evaluated delta for the last layer: {}", delta.back()[0]);

		if(netSize == 1) { // SLP case
			for (size_t k = 0; k < input.size(); ++k) {
				net[netSize - 1][0][k] += (step * delta[netSize - 1][0] * input[k]);
			}
			spdlog::debug("Evaluated weight for the last layer (SLP)");
			return; // exit backPropagate
		}

		// MLP case below
		for (int i = netSize - 2; i >= 0; --i) {

			ANN::DefaultLayer &current_layer = net[i];
			ANN::DefaultLayer &following_layer = net[i + 1];

			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				data_t &current_elem = delta[i][j];

				ANN::activationFunction_t g_d = ANN::getActivationDerivative(net[i][j].getActivationFunctionID());

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
			}
			spdlog::debug("Finished layer");
		}
		data_t norm = 0;

		for (size_t i = 1; i < netSize - 1; ++i) {
			ANN::DefaultLayer &current_layer = net[i];
			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				auto &current_neuron = current_layer[j];
				for (size_t k = 0; k < current_neuron.getSize(); ++k) {
					norm += pow(delta[i][j] * output[i - 1][k], 2);
				}
			}
		}

		ANN::DefaultLayer &current_layer = net[0];
		for (size_t j = 0; j < current_layer.getSize(); ++j) {
			auto &current_neuron = current_layer[j];
			for (size_t k = 0; k < current_neuron.getSize(); ++k) {
				norm += pow(delta[0][j] * input[k], 2);
			}
		}

		step /= sqrt(norm);

		spdlog::debug("norm: {}, step: {}", norm, step);

		for (size_t k = 0; k < output[netSize - 2].size(); ++k) {
			net[netSize - 1][0][k] += (step * delta[netSize - 1][0] * output[netSize - 2][k]);
		}
		
		for (int i = netSize - 2; i >= 0; --i) {
			ANN::DefaultLayer &current_layer = net[i];
			ANN::DefaultLayer &following_layer = net[i + 1];

			for (size_t j = 0; j < current_layer.getSize(); ++j) {
				data_t &current_elem = delta[i][j];

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
		}
		spdlog::debug("Exit backPropagate()");
	}

	/**
	 * @brief Tests the neural network on a given data stream and calculates accuracy.
	 *
	 * @param net The neural network to test.
	 * @param stream The input data stream for testing.
	 * @return The accuracy of the neural network on the test data.
	 *
	 * This method tests the neural network on a given data stream and calculates the accuracy of the network's
	 * predictions. It iterates through the data stream, evaluates the network's output, and compares it with the
	 * expected output to determine accuracy.
	 */
	inline data_t test(network_t &net, stream_t &stream) const override {
		data_vv_t preactivation = net.getEmptyPreActivationVector();
		data_vv_t output = net.getEmptyOutputVector();

		std::string line;

		size_t input_size = net[0][0].getWeights().size() - 1;
		
		std::vector<ANN::data_t> vectorTrainingSet;
		for(size_t i = 0; i < input_size; ++i) {
			vectorTrainingSet.push_back(0);
		}

		data_t error = 0.0;
		
		error = 0.0;
		size_t test_set_size = 0;
		size_t correct = 0;

		while (getline(stream, line)) {

			++test_set_size;
			
			std::istringstream iss(line);
			data_t value;

			for(size_t i = 0; i < input_size; ++i) {
				iss >> value;
				vectorTrainingSet[i] = value;
			}

			data_t y;
			iss >> y;

			// This is fine in case we only have one neuron in the last layer
			// TODO add a parameter that tells me how many outputs I should have
			net.evaluate(vectorTrainingSet, output, preactivation);
			
			data_t result = abs(y - output.back()[0]);

			// spdlog::info("y: {}, output: {}, result: {}", y, output.back()[0], result);

			if(result < 0.01)
				++correct;

		}

		// Back to file begin
		stream.clear();
		stream.seekg(0, std::ios::beg);

		data_t accuracy = correct / test_set_size;

		return accuracy;
	}
};
} // namespace Training

#endif
