#ifndef DEFAULT_LOADER_INCLUDE_GUARD
#define DEFAULT_LOADER_INCLUDE_GUARD

#include "DefaultNetwork.h"
#include "Loader.h"
#include <cstddef>
#include <numeric>
#include <spdlog/spdlog.h>
#include <string>

namespace Utils {

class DefaultLoader : public Loader<ANN::DefaultNetwork> {
  public:
	using Loader<ANN::DefaultNetwork>::network_t;

    inline void saveStatus(const network_t& net) const override {
        std::string folderPath = "./model";
		std::string filename = folderPath + "/status.txt";

		if (mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			if (errno != EEXIST) {
				spdlog::error("Unable to create directory {}", folderPath);
				throw std::runtime_error("Unable to create directory " + folderPath);
			}
		}
		
		std::ofstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			throw std::runtime_error("Unable to open " + filename);
		}

		model << net.getStatus();

		model.close();

		spdlog::info("Status saved in {}", filename);
    }

    inline network_t* loadNetwork(const std::string& path = "./model/status.txt") const override {
		std::string filename = path;

		std::ifstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			throw std::runtime_error("Unable to open " + filename);
		}

		std::vector<int> topology;
		ANN::DefaultNetwork::layer_vector_t layers;
		std::string line;

		getline(model, line); // topology
		std::istringstream iss(line);
		int value;
		
		while (iss >> value) {
			topology.push_back(value);
		}

		for(auto item : topology){
			int c = 0;
			ANN::DefaultLayer::neuron_vector_t neurons;
			while(c < item){
				getline(model, line);
				std::istringstream iss(line);
				ANN::data_t value;
				ANN::DefaultNeuron::weight_vector_t weights;
				
				while (iss >> value) {
					weights.push_back(value);
				}
				
				ANN::DefaultNeuron n(weights, ANN::heaviside); // TODO remove activation function
				neurons.push_back(n);

				++c;
			}

			ANN::DefaultLayer l(neurons);
			layers.push_back(l);
		}

		model.close();

		return new ANN::DefaultNetwork(layers);
    }
};

} // namespace Utils

#endif // DEFAULT_LOADER_INCLUDE_GUARD
