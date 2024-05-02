#ifndef DEFAULT_LOADER_INCLUDE_GUARD
#define DEFAULT_LOADER_INCLUDE_GUARD

#include "DefaultNetwork.h"
#include <cstddef>
#include <numeric>
#include <spdlog/spdlog.h>
#include <string>

namespace ANN {

class DefaultLoader : public Loader<DefaultNetwork> {
  public:

  private:

  public:
    inline virtual void saveStatus(Network &net) const override {
        std::string folderPath = "./model";
		std::string filename = folderPath + "/status.txt";

		if (mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			if (errno != EEXIST) {
				spdlog::error("Unable to create directory {}", folderPath);
				return;
			}
		}
		
		std::ofstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			return;
		}

		spdlog::info("Saving topology");
		for (const auto &l : net.getLayers()) {
			model << l.getSize() << " ";
		}
		model << std::endl;

		spdlog::info("Saving weights");
		for (const auto &l : net.getLayers()) {
			auto neurons = l.getNeurons();

			for(const auto &n : neurons) {
				auto weights = n.getWeights();

				for(const auto &w : weights) {
					model << w << " ";
				}
				model << std::endl;
			}
		}

		spdlog::info("Status saved in {}", filename);

    }

    inline virtual Network& loadStatus() const override {
        std::string folderPath = "./model";
		std::string filename = folderPath + "/status.txt";

		std::ifstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			return -1;
		}

		std::vector<int> topology;
		DefaultNetwork::layer_vector_t layers;
		std::string line;

		getline(model, line); // topology
		std::istringstream iss(line);
		int value;
		
		while (iss >> value) {
			topology.push_back(value);
		}

		for(auto item : topology){
			int c = 0;
			DefaultLayer::neuron_vector_t neurons;
			while(c < item){
				getline(model, line);
				std::istringstream iss(line);
				data_t value;
				DefaultNeuron::weight_vector_t weights;
				
				while (iss >> value) {
					weights.push_back(value);
				}
				
				DefaultNeuron n(weights, heaviside);
				neurons.push_back(n);

				++c;
			}

			DefaultLayer l(neurons);
			layers.push_back(l);
		}

		model.close();

		DefaultNetwork net(layers);

		std::cout << net << std::endl;
        return net;
    }
};

} // namespace ANN

#endif // DEFAULT_LOADER_INCLUDE_GUARD
