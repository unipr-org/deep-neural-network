# deep-neural-network

## How to compile

```bash
mkdir build
cd build
cmake ..
make
```

To turn on all types of logs use `cmake -DDEBUG=ON`.

## How to add and run tests

Filenames of tests **must** be compliant to the following patterns:

| Source | Input |
| ------------- | -------------- |
| `test_*.cpp` | `test_*.txt` |

and all tests must be in the `test` directory.

After the compilation phase, inside the `build` directory run, you can run all tests with:

```bash
ctest # -V for verbose output
```

To execute a single test: 

```bash
./test/test_<test_name>
```

or refer to the CMake [doc](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html#testing-using-ctest).

## Generate documentation

Simply run in the build directory

```bash
make documentation
```

and look for the `index.html` inside `doc/html/`.

---

## TODO-List

- [x] Utilizzo di C++ 11
- [x] Implementare classi astratte (interfacce)
- [x] Cambiare nomi variabili private (`_`)
- [x] Funzioni inline
- [x] Rimozione classe Connection
- [x] Rimuovere `bool isBias`
- [x] Utilizzo namespace e alias
- [x] Const correctess 
- [x] Operator []
- [x] Salvataggio stato rete
- [x] Funzione caricamento pesi
- [x] Caricamento struttura rete da file
- [ ] Classe Loader
- [ ] Adattare metodi per il salvataggio e caricamento della rete nella classe Loader
- [ ] Algoritmo di backpropagation
- [ ] Algoritmo di testing
- [ ] Update print `.dot`
- [ ] Generazione training e test set
- [ ] Lettura training set
- [ ] Lettura test set e statistiche
- [ ] Parallelizzazione (`thread`)
- [ ] Libreria matematica


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
