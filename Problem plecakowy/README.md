Podstawowe wiadomości przed wykonaniem zadania:

## Problem plecakowy

Problem plecakowy polega na tym, że każdy przedmiot ma wagę i wartość. Zadanie polega na załadowaniu plecaka tak, aby łączna wartość przedmiotów była jak największa przy nie przekroczeniu wagi jaką plecak może utrzymać;

## Metoda brute-force

Metoda brute-force polega na rozpatrzeniu wszytkich możliwych ułożeń przedmiotów w plecaku i znalezieniu tego, które najlepiej spełnia waruki zadania; czas wykonania powinien rosnąć wykładniczo wraz z liczbą przedmiotów;

## Metoda według podanej heurystyki

Do plecaka wkładamy stopniowo przedmioty o największym stosunku wartości do wagi. Zakładamy w tym przypadku niepodzielność wkładanych przedmiotów.

## Wskazówka na przyszłość

np.ndarray można mnożyć <br/>
Mogłeś użyć tego, zamiast iteracji po wziętych/niewzietych przedmiotach: <br/>

```
selected_items_weight = sum(list(np.array(list(selected_items)) * np.array(self.weights)))
selected_items_profit = sum(list(np.array(list(selected_items)) * np.array(self.profits)))
```

Zamiast: <br/>

```
for i in range(len(self.weights)):
    if selected_items[i] == 1:
        selected_items_weight += self.weights[i]
        selected_items_profit += self.profits[i]
    if selected_items_weight > self.capacity:
        break
```
