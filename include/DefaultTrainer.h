#ifndef DEFAULT_TRAINER_INCLUDE_GUARD
#define DEFAULT_TRAINER_INCLUDE_GUARD

#include "ANN.h"
#include "Network.h"
#include "DefaultNetwork.h"
#include "Trainer.h"
#include "Training.h"
#include <fstream>
#include <sstream>

namespace Training {

class DefaultTrainer : public Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t> {
  public:
    using network_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::network_t;
    using stream_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::stream_t;
    using tolerance_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::tolerance_t;
    using step_t = typename Trainer<ANN::DefaultNetwork, stream_t, tolerance_t, step_t>::step_t;

  private:
    inline std::vector<std::vector<ANN::data_t>>& backward_propagate(network_t& net, std::vector<std::vector<ANN::data_t>>& preactivation, 
            const ANN::data_t& current_error, const step_t step) const override {
        
        ANN::activationFunction_t derivative = ANN::sigmoid_d;
        std::vector<std::vector<ANN::data_t>> delta;
        
        // Inizializzo lo spazio per delta
        for(int l = 0; l < net.getSize(); ++l) {
            delta[l].reserve(net[l].getSize());
        }

        delta[net.getSize() - 1][0] = derivative(preactivation[net.getSize() - 1][0] * current_error);

        for(int k = net.getSize() - 2; k >= 0; --k){
            ANN::data_t sum = 0;
           
            for(int j = 0; j < net[k].getSize(); ++j){

                for(int s = 0; s < net[k + 1].getSize(); ++s)
                    sum += net[k + 1][s][j] * delta[k + 1][s];
            
                delta[k][j] = derivative(preactivation[k][j]) * sum;
            }
        }
        return delta;
    }

    inline void updateWeights(network_t& net, const std::vector<std::vector<ANN::data_t>>& output, 
            const std::vector<std::vector<ANN::data_t>>& delta, const step_t step) {
        
        ANN::layer_vector_t layers = net.getLayers();
        for(int k = 0; k < layers.size(); ++k) {
            ANN::neuron_vector_t neurons = layers[k].getNeurons();
            for(int j = 0; j < neurons.size(); ++j) {
                ANN::weight_vector_t weights = neurons[j].getWeights();
                for(int i = 0; i < weights.size(); ++i) {
                    net[k][j][i] += step * delta[k][j] * output[k - 1][j];
                }
            }
        }
    }

  public:
    inline void train(network_t& net, stream_t& stream, const tolerance_t tolerance, const size_t epochs, const step_t step) const override {
        std::vector<std::vector<ANN::data_t>> preactivation;
        std::vector<std::vector<ANN::data_t>> output;

        for(int l = 0; l < net.getSize(); ++l) {
            preactivation[l].reserve(net[l].getSize());
            output[l].reserve(net[l].getSize());
        }

        std::string line; // linea in cui viene letto il vettore del training set 
        // Epoche
        for(size_t r = 1; r <= epochs; ++r) {
            // TODO In questo modo una epoca corrisponde ad una lettura di un vettore di un training set,
            // TODO bisogna scorrere tutto il file contenente i vettori del traning set per completare un'epoca.
            // TODO Conviene passare il path contenente il traning set e aprirlo direttamente questa funzione.
            std::vector<ANN::data_t> vectorTrainingSet; // vettore del training set
            getline(stream, line); // leggo la linea del training set
            std::istringstream iss(line); 
            ANN::data_t value;
            ANN::data_t current_error = 0;
            ANN::data_t error = 0;
            
            while (iss >> value) {
                vectorTrainingSet.push_back(value);
            }

            // Ora che ho il vettore del training set devo dividerlo nel vettore x e nel vettore f(x)
            std::vector<ANN::data_t> x;
            std::vector<ANN::data_t> y;
            
            // Questa cosa va bene nel caso in cui abbiamo solo un neurone nell'ultimo layer.
            // TODO aggiungere un parametro che mi indichi quanti output devo avere
            y.push_back(vectorTrainingSet.back());
            x = std::move(vectorTrainingSet);

            // TODO da rimuovere questo vettore fittizio
            std::vector<ANN::data_t> tmp; // vettore fittizio 
            net.evaluate(x, tmp, preactivation, output); // TODO rimuovere ridondanza vettore output

            current_error = y.at(1) - output[net.getSize() - 1][1]; // prendo l'output dell'uscita dell'ultimo neurone
            
            // Ritorno il delta
            std::vector<std::vector<ANN::data_t>> delta = backward_propagate(net, preactivation, current_error, step);

            // Aggiorno i pesi
            updateWeights(net, output, delta, step);
        }

    }

    inline void test(network_t& net, stream_t& stream) const override {

    }
};
} // namespace Training


#endif

