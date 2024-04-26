#include "include/Connection.h"
#include <iostream>

using namespace std;

Connection::Connection(int index){
    weight = (double) rand() / RAND_MAX * 2.0 - 1.0;
    indexNeuronLinked = index;
}

double
Connection::getWeight(){
    return weight;
}

ostream& operator<<(ostream& os, Connection& conn) {
    os << conn.getWeight();
    return os;
}
