#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
persons = ['Farrah', 'Fred', 'Felicia']

plt.figure(figsize=(8, 6))

for i, person in enumerate(persons):
    plt.bar(person, fruit[:, i], bottom=np.sum(fruit[:, :i], axis=1), color=colors, label=fruits if i == 0 else None, width=0.5)

plt.xlabel('Person')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(fontsize='small')
plt.yticks(np.arange(0, 81, 10), fontsize='small')
plt.legend(title='Fruit', fontsize='small')
plt.ylim(0, 80)
plt.grid(axis='y')

plt.show()
