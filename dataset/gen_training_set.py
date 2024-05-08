import random


def generate_training_set(filename, num_samples):
    with open(filename, 'w') as file:
        for _ in range(num_samples):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 100)
            x3 = random.uniform(0, 100)
            y = x1  # La funzione da implementare: somma dei tre numeri di input
            file.write(f"{x1} {x2} {x3} {y}\n")


# Esempio di utilizzo: genera un training set con 1000 tuple e lo salva in 'training_set.txt'
generate_training_set('./training_set.txt', 100000)
generate_training_set('./test_set.txt', 20000)
