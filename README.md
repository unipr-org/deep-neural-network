# artificial-neural-network
How to compile:
```bash
g++ -o main.out main.cpp ./src/ANN.cpp ./src/Connection.cpp ./src/Layer.cpp ./src/Neuron.cpp -I./include -O1
```
How to execute:
```bash
./main.out
```

## Idee
Classe Artificial-Neural-Network
- Vettore contenente i layer
- Costruttore che inizializza la rete: viene passato un vettore che indica il numero di neuroni per ogni layer. Esempio: v = {4, 3, 5, 1}, avremo una ANN che ha 4 input, 2 strati nascosti e 1 solo neurone in uscita (non viene contato il bias qui, quindi andrebbe aggiunto successivamente).
- Algoritmo di back-propagation

Classe Layer
- Variabile che indica l'indice del layer nella ANN
- Vettore contenente i neuroni

Classe Neuron
- Variabile che indica l'indice del neurone nel Layer
- Vettore contenente le connessioni con i neuroni dello strato successivo
- Funzione di attivazione scelta
- Algoritmo di Feed-forward
- Funzione di preattivazione

Classe Connection
- Costruttore che inizializza il peso con valori random
- Variabile che indica il peso
- Variabile che tiene traccia del neurone di destinazione

```
for-each layer
	my-layer <- layer(i)
	for-each neuron
		my-neuron <- my-layer.neuron(j)
		for-each connection
			my-neuron.connection(k) <- update
		end for-each connection
	end for-each neuron
end for-each layer
```