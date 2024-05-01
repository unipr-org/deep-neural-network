#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD

#include <functional>
#include <ostream>
#include <vector>

namespace ANN {

using weight_t = long double;
using data_t = long double;
using activationFunction_t = std::function<data_t(data_t)>;

template <typename T> std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
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
} // namespace ANN

#endif // ANN_HH_INCLUSION_GUARD
