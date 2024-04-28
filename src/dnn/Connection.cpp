#include "Connection.h"
#include <iostream>

using namespace std;

/**
 * @brief Connection constructor.
 *
 * Constructs a Connection object with the given index.
 * It initializes the weight of the connection using a random value.
 *
 * @param index The index of the neuron to which the connection is linked.
 */
Connection::Connection(int index){
    weight = (double) rand() / RAND_MAX * 2.0 - 1.0;
    indexNeuronLinked = index;
}

/**
 * @brief Prints information about the connection.
 *
 * This method prints information about the connection, including its weight
 * and the index of the neuron to which it is linked.
 */
void Connection::print() {
    cout << "w: " << weight << ", to: " << indexNeuronLinked;
}

/**
 * @brief Overloaded << operator for printing Connection objects.
 *
 * This overloaded operator allows Connection objects to be printed directly to an output stream.
 * It prints the weight of the connection.
 *
 * @param os The output stream to which the Connection object is printed.
 * @param conn The Connection object to print.
 * @return The output stream after printing the Connection object.
 */
ostream& operator<<(ostream& os, Connection& conn) {
    os << conn.getWeight();
    return os;
}
