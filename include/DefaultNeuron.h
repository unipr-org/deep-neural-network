#ifndef DEFAULT_NEURON_INCLUDE_GUARD
#define DEFAULT_NEURON_INCLUDE_GUARD

#include "ANN.h"
#include "Neuron.h"
#include <cstddef>
#include <numeric>
#include <spdlog/spdlog.h>
#include <string>

namespace ANN {

class DefaultNeuron : public Neuron<> {
  public:
	using typename Neuron<>::weight_vector_t;
	using typename Neuron<>::data_vector_t;
	using typename Neuron<>::activationFunction_t;

  private:
	weight_vector_t _weights;
	activationFunction_t _activationFunction;

  public:
	/**
	 * @brief Default constructor for DefaultNeuron.
	 *
	 * This constructor creates a DefaultNeuron object without any parameters.
	 */
	inline DefaultNeuron() : _weights(), _activationFunction(tanh) {
		spdlog::debug("[DefaultNeuron::DefaultNeuron()] created DefaultNeuron");
	}

	/**
	 * @brief Constructor for DefaultNeuron with specified weights and activation function.
	 *
	 * @param weights The vector of weights to initialize the neuron.
	 * @param activationFunction The activation function to use. Default is sigmoid.
	 *
	 * This constructor creates a DefaultNeuron object with the specified weights and activation
	 * function.
	 */
	inline DefaultNeuron(const weight_vector_t &weights,
						 const activationFunction_t &activationFunction = tanh)
		: _weights(weights), _activationFunction(activationFunction) {

		spdlog::debug("[DefaultNeuron::DefaultNeuron(const weight_vector_t &, const "
					  "activationFunction_t &)] created DefaultNeuron");
	}

	inline ~DefaultNeuron() { spdlog::debug("[DefaultNeuron::~DefaultNeuron()]"); }

	inline void setWeights(const weight_vector_t &weights) override {
		spdlog::debug("[DefaultNeuron::setWeights(const weight_vector_t &)]");

		std::string msg = "old weights:\n";
		msg += _weights;
		msg += "\n";
		spdlog::debug(msg);

		_weights = weights;

		msg = "new weights:\n";
		msg += _weights;
		msg += "\n";
		spdlog::debug(msg);
	}

	inline void setActivationFunction(const activationFunction_t &activationFunction) override {
		_activationFunction = activationFunction;
		spdlog::debug("[DefaultNeuron::setActivationFunction(const activationFunction_t &)]");
	}

	inline weight_vector_t &getWeights() override { return _weights; }
	inline const weight_vector_t &getWeights() const override { return _weights; }
	inline size_t getSize() const override { return _weights.size(); }
	inline const activationFunction_t &getActivationFunction() const override {
		return _activationFunction;
	}

	inline data_t evaluate(const data_vector_t &input) const override {
		data_t innerProduct =
			std::inner_product(input.begin(), input.end(), _weights.begin(), data_t(0));
		data_t result = _activationFunction(innerProduct);

		std::string msg = "[DefaultNeuron::evaluate(const data_vector_t &)] input: ";
		msg += input;
		spdlog::debug(msg);

		msg = "[DefaultNeuron::evaluate(const data_vector_t &)] weights: ";
		msg += _weights;
		spdlog::debug(msg);

		spdlog::debug("[DefaultNeuron::evaluate(const data_vector_t &)] result: " +
					  std::to_string(result));
		return result;
	}

	inline void evaluate(const data_vector_t &input, data_t &output,
						 data_t &preActivation) const override {
		data_t innerProduct =
			std::inner_product(input.begin(), input.end(), _weights.begin(), data_t(0));

		preActivation = innerProduct;
		output = _activationFunction(innerProduct);
		spdlog::debug("innerProduct: {}, output: {}", innerProduct, output);

		std::string msg = "[DefaultNeuron::evaluate(const data_vector_t &input, data_t &output, "
						  "data_t &preActivation)] input: ";
		msg += input;
		spdlog::debug(msg);

		msg = "[DefaultNeuron::evaluate(const data_vector_t &input, data_t &output, data_t "
			  "&preActivation)] weights: ";
		msg += _weights;
		spdlog::debug(msg);

		spdlog::debug("[DefaultNeuron::evaluate(const data_vector_t &input, data_t &output, data_t "
					  "&preActivation)] output: " +
					  std::to_string(output));
	}

	inline weight_t &operator[](size_t index) override { return _weights[index]; }
	inline const weight_t &operator[](size_t index) const override { return _weights[index]; }

	inline std::ostream &operator<<(std::ostream &os) const override {
		size_t index = 0;

		std::string msg = "weights: ";
		msg += _weights;
		os << msg;

		return os;
	}

	// TODO scrivere documentazione
	inline void createWeights(size_t weights) {
		_weights = std::move(std::vector<weight_t>(weights));
	}
};

} // namespace ANN

#endif // NEURON_INCLUDE_GUARD
