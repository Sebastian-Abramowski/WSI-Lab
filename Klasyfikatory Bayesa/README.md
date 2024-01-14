Podstawowe wiadomości przed wykonaniem zadania:

## Pomoce

- https://www.youtube.com/watch?v=O2L2Uv9pdDA
- https://www.youtube.com/watch?v=H3EjCKtlVog

## Sieć Bayesa

Acykliczny graf, w którym:

- wierzchołki odpowiadają dyskretnym zmienny losowym
- krawędzie reprezentują bezpośrednio występujące zależności między tymi zmiennymi
- dla każdej zmiennej dany jest jej rozkład prawdopodobieństwa – warunkowy jeżeli do zmiennej prowadzą krawędzie
- strzałki: przyczyna -> skutek

---

![](img/bayes_1.png)

## Naiwny klasyfikator Bayesa

- jego zadaniem, podobnie jak w przypadku drzewa decyzjnego, jest przypisanie danych do określonej klasy na podstawie ich cech
- wykorzystuje wnioskowanie bayesowskie do predyckji prawdopodobieństwa klas
- opiera się na prawdopodobieństwie warunkowym
- opiera też się na (priori) prawdopodobieństwie zajścia jakiegoś zdarzenia na podstawie zbioru trenującego
- mamy problem jeśli w jakimś prawdopodobieństwie warunkowym mamy 0

## Naiwność

- naiwność polega na tym, że zakładamy, że w ramach każdej klasy atrybuty są niezależne (w rzeczywistości jest to zazwyczaj niespełnione)

![](img/bayes_3.png)
