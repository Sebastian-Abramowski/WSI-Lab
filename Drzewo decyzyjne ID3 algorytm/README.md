Podstawowe wiadomości przed wykonaniem zadania:

## Drzewo decyzyjne

- Reprezentuje mechanizm decyzyjny
- Węzeł korzenia reprezentuje cały zestaw danych uczacych przed podziałem
- Węzły odpowiadają atrybutom, w węźle następuje podział zbioru danych na mniejsze podzbiory na podstawie określonych wartości danego atrybutu
- Krawędzie odpowiadają wyborom podejmowanym na podstawie atrybutów
- Liście reprezentują finalne decyzje, końcowe klasyfikacje, każdy liść jest przypisany do konkrentej klasy
- W uczeniu maszynowy konstruuje się drzewa na podstawie danych trenujących

## Indukcja dzrew decyzyjnych - ID3

- Metoda indukcji drzew klasyfikacyjnych
- Drzewa budowane są rekurencyjnie
- Liście zawierają klasy (konretne klasyfikacje)
- Dla danego x (konkrenty przypadek z zestawu danych, który zawiera wartości dla wszystkich atrybutów rozważanych w drzewie) ustalana jest ścieżka w drzewie i zwracana klasa z liścia kończącego tą ścieżkę
- Ogólne działanie: bierze konkretny przypadek, drzewo zwraca przypisaną klase do tego przypadku, na podstawie jego atrybutów

## Algorytm ID3

- Entropia - jest miarą nieporządku w danych - chodzi o to żeby ocenić jak dobrze atrybuty rozdzielają zestaw danych na podklasy podczas konstruowania drzewa (niska wartość entropi oznacza, że większość przypadków należy do jednej klasy):
  Entropia = -SUMA(PRAWDOPODOBIEŃSTWO_klasy \* log2PRAWDOPODOBIEŃSTWO_klasy)
- Minus w entropi jest po to, żeby entropia była na plusie
- Na początku entropia jest obliczana dla całego zestawu danych treningowych (Obrazuje to poczatkowy obraz chaosu w całym zbiorze danych)
- Entropia dla każdego atrybutu: ID3 oblicza entropię dla każdego możliwego podziału według róznych atrybutów
- Algorytm oblicza zysk informacyjny, zysk informacyjny jest mierzony jako:
  różnica między entropią początkową a ważoną średnia entropii po podziale (entropia każdego podzbioru jest ważona według proporcji, jaką ten podzbior stanowi w całym zbiorze danych)
- Jeśli zysk informacyjny jest wysoki, oznacza, to że dany atrybut dobrze rozdziela dane na bardziej jednorodne pozbiory. Wysoki zysk informacyjny oznacza, że ogólna entropia w danych została znacząco zredukowana
- Celem ID3 jest znalezienie takiego podziału, który najbardziej zmniejszy entropie w porównaniu z poczatkową entropią całego zbioru danych
- Proces jest iteracyjnie powatarzany. W każdym węźle drzewa, algorytm wybiera atrybut, który daje największy spadek entropii i dokonuje kolejnego podziału. W miarę schodzenia w dół drzewa, entropia w każym z jego podzbiorów maleje, co oznacza, że dane są coraz bardziej uporządkowane
- Proces ten kończy się, gdy algorytm osiągnie liście drzewa, w których każdy liść reprezentuje podzbiór danych o wystarczająco niskiej entropii (lub do osiągnięcia maksymalnej głębokości drzewa)

# Komentarz do początkowego kodu, część 1

- test_size = 0.1 - oznacza, że 10% danych jest użyte jako zestaw testowy, a pozostałe 90% danych jako zestaw treningowy
- random_state ten sam, gwarantuje otrzymanie za każdym razem dokładnie tego samego zestawu danych
- iris.data - reprezentuje dane wejściowe (cechy kwiatków), które są używane do budowania modelu (jest to lista list, które przetrzymują wartości czterech atrybutów np. [5.1, 3.5, 1.4, 0.2])
- iris.target - przechowuje informacje, do której klasy należy każdy przykład w zbiorze danych (jest to lista wartości, które reprezentują klase dla określonych danych atrybutów: mogą to być 0/1/2)
- np. x[0] reprezentuje wartości atrybutów, a y[0] klasę, do której kwiatek o takich atrybutach trafia
- iris.data i iris.target to tablice z Numpy

## Uczenie

- Drzewo decyzyjne jest reprezentowane jako zbior połączonych ze sobą Nodów
- Podczas budowania drzewa, każdy Node zapisuje informacje o podziale, te informacje są potem używane poczas procesu predykcji
- Dla normalnych danych, po prostu przechodzimy po drzewie korzystając już z informacji, które uzyskaliśmy z danych testowych

## Inne

- W algorytmie ID3, wartość graniczna dal podziału danej cechy jest wybierana na podstawie danych treningowych w procesie, który ma na celu znalezienie najlepszego punktu podzialu w celu maksymalizacji zysku informacyjnego
- W każdym węźle w build_tree wybierana jest cecha i wartość graniczna, która najlepiej dzieli dane na podstawie kryteriów takich jak entropia i zysk informacyjny
- Po zbudowaniu drzewa, można użyć metody predict do dokonywania predykcji na nowych danych, w trakcie predykcji dla każdego przykładu z nowego zbioru danych, algorytm przechodzi przez drzewo i zwraca decyzje klasyfikacyjną
- Średnia ważona entropii pomaga obliczyć jakość podziału danych, a wynik jest używany do obliczania zysku informacyjnego, który jest kluczowym kryterium wyboru cechy do podzialu w drzewie decyzyjnym
- Im niższa suma ważonej entropii, tym lepszy podział (oznacza to że lepiej segregujemy dane)
- Im większy zysk informacyjny, tym lepiej uporządkowane dane są PO TYM PODZIALE
- ID3 jest budowane rekurencyjne, a każdy węzeł może mieć różna liczbe potomków 0/2
