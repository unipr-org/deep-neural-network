# artificial-neural-network

## How to compile

```bash
mkdir build
cd build
cmake ..
make
```

To turn on all types of logs use `cmake -DDEBUG=ON`.

## How to run tests

After the compilation phase, run

```bash
ctest # -V for verbose output
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

## TODO-List

- [ ] Implementare classi astratte (interfacce)
- [ ] Utilizzare il pattern _bridge_
- [ ] Rivedere struttura delle classi
- [ ] Rimuovere `bool isBias`
- [ ] Implementare algoritmo per la lettura dei pesi da file di testo (e quindi costruzione della rete)
