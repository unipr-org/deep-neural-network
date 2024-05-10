# Deep Neural Network

## Dependencies

- [spdlog](https://github.com/gabime/spdlog)
- [doxygen](https://www.doxygen.nl/)

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

### Example test
You can run an example test with:
```bash
./test/test_DefaultTrainer
```

---

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
- [x] Algoritmo di backpropagation
- [x] Generazione training e test set
- [x] Lettura training set
- [x] Libreria matematica
- [x] Sistemare scelta del passo per l'algoritmo di back-prop
- [x] Ridirigere l'output della compilazione di doxygen da qualche altra parte
- [x] Algoritmo di testing
- [x] Implementare funzione test e statistiche
- [ ] Parallelizzazione (`thread`)
- [ ] Metodo getActivationFunction che restituisce g() e con una mappa otteniamo g’()
- [ ] Scrivere la documentazione mancante

