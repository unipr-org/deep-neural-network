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
- [x] Classe Loader
- [x] Adattare metodi per il salvataggio e caricamento della rete nella classe Loader
- [ ] Algoritmo di backpropagation
- [ ] Algoritmo di testing
- [ ] Update print `.dot`
- [ ] Generazione training e test set
- [ ] Lettura training set
- [ ] Lettura test set e statistiche
- [ ] Parallelizzazione (`thread`)
- [ ] Libreria matematica


## Feed-forward
```
function train(network, training-set, passo-appr):
	vector<vector<data_t>> pre-attivazione 
	vector<vector<data_t>> output
	
	for l in 1 to size(network)
		output[l].reserve(size(network[l]))
		pre-attivazione[l].reserve(size(network[l]))


	for p in 1..max-epochs:
		for-each (x, f(x)) in training-set:
			n.evaluate(x, pre-attivazione, output)
			
			current_error <- f(x) - output[size(network)][1]
			
			backward_propagate(network, pre-attivazione, current_error, passo-appr)
			
			error += |current_error|

		error /= size(traning-set)

		if error < tolleranza:
			save_status
			return

function backward_propagate(network, pre-attivazione, current_error, passo-appr):
	g_derivata <- getDerivative(network.getActivationFunction)
	vector<vector<data_t>> delta
	
	delta[size(network)][1] <- g_derivata(pre-att[size(network)][1] * current_error)

	for k in size(network) - 1 to 1: // scorro layers al contrario
		for j in 1 to size(network[k]):
			delta[k][j] <- g_derivata(pre-att[k][j]) * sum(network[k+1][s][j] * delta[k+1][s]) // con s da 1 a size(network[k+1])

```