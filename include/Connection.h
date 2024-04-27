#ifndef CONNECTION_HH_INCLUSION_GUARD
#define CONNECTION_HH_INCLUSION_GUARD 1

class Connection {
private:
    double weight;
    int indexNeuronLinked;
public:
    Connection(int);
    double getWeight() { return weight; };
    int getIndexNeuronLinked() { return indexNeuronLinked; };
    void print();
};

#endif // CONNECTION_HH_INCLUSION_GUARD