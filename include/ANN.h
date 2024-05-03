#ifndef ANN_INCLUDE_GUARD
#define ANN_INCLUDE_GUARD

#include <cmath>
#include <functional>
#include <ostream>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>


namespace ANN {
/**
 * @brief Type definition for neural network weights.
*/
using weight_t = long double;

/**
 * @brief Type definition for data used in the neural network.
 */
using data_t = long double;

/**
 * @brief Type definition for activation functions used in the neural network.
 */
using activationFunction_t = std::function<data_t(data_t)>;

/**
 * @brief Output operator for vectors.
 * 
 * @tparam T The type of elements in the vector.
 * @param os The output stream to write to.
 * @param v The vector to output.
 * @return Reference to the output stream.
 * 
 * This function outputs a vector to the specified output stream.
 */
template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
	size_t index = 0;

	os << "{";
	for (auto it = v.begin(); it != v.end(); ++it, ++index) {
		os << *it;
		if (index != v.size() - 1)
			os << ", ";
	}
	os << "}";

	return os;
}

/**
 * @brief Concatenates a vector to a string.
 * 
 * @tparam T The type of elements in the vector.
 * @param string The string to concatenate to.
 * @param v The vector to concatenate.
 * @return Reference to the modified string.
 * 
 * This function concatenates a vector to a string.
 */
template <typename T> std::string &operator+=(std::string &string, const std::vector<T> &v) {
	size_t index = 0;

	string += "{";
	for (auto it = v.begin(); it != v.end(); ++it, ++index) {
		string += std::to_string(*it);
		if (index != v.size() - 1)
			string += ", ";
	}
	string += "}";

	return string;
}

/**
 * @brief Sigmoid activation function.
 * 
 * @param x The input value.
 * @return The output value after applying the sigmoid function.
 * 
 * The sigmoid function squashes its input into the range (0, 1).
 */
inline data_t sigmoid(data_t x) {
    data_t result = 1 / (1 + exp(-x));
    return result;
}

/**
 * @brief Derivative of the sigmoid activation function.
 * 
 * @param x The input value.
 * @return The derivative of the sigmoid function at the input value.
 */
inline data_t sigmoid_d(data_t x) {
    data_t result = sigmoid(x);
    return result * (1 - result);
}

/**
 * @brief Heaviside step function.
 * 
 * @param x The input value.
 * @return 1 if x >= 0, otherwise 0.
 */
inline data_t heaviside(data_t x) { return (x >= 0) ? 1 : 0; }

/**
 * @brief Identity activation function.
 * 
 * @param x The input value.
 * @return The input value unchanged.
 */
inline data_t identity(data_t x) { return x; }

/**
 * @brief Hyperbolic tangent (tanh) activation function.
 * 
 * @param x The input value.
 * @return The output value after applying the tanh function.
 * 
 * The tanh function squashes its input into the range (-1, 1).
 */
inline data_t tanh(data_t x) {
    data_t e1 = exp(x);
    data_t e2 = exp(-x);

    return (e1 - e2) / (e1 + e2);
}

/**
 * @brief Derivative of the hyperbolic tangent (tanh) activation function.
 * 
 * @param x The input value.
 * @return The derivative of the tanh function at the input value.
 */
inline data_t tanh_d(data_t x) {
    data_t e1 = exp(x);
    data_t e2 = exp(-x);

    return (1 - pow(tanh(x), 2));
}

} // namespace ANN

#endif // ANN_INCLUDE_GUARD
