#ifndef CONNECTION_HH_INCLUSION_GUARD
#define CONNECTION_HH_INCLUSION_GUARD 1

class Connection {
private:
    double weight;
    int indexNeuronLinked;
public:
    Connection(int);
    double getWeight();
};

#endif // CONNECTION_HH_INCLUSION_GUARD