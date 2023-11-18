Podstawowe wiadomości przed wykonaniem zadania:

## Minimax

- dobrze nadaje się do gier o sumie zerowej, gdzie zysk jednego gracza oznacza stratę oponenta, najlepiej w grach dwuosobowych
- najważniejsza w nim jest funkcja ewaluująca stan obecnej planszy - jest to rodzaj funkcji heurystycznej
- wykorzystanie tej funkcji pozwala nam na nie reprezentowanie/ przeszukiwanie całego drzewa gry (wtedy byłoby za dużo możliwości, przeszukiwanie całego drzewa gry można stosować dla gier bardzo prostych)
- algorytm pozwala na analizowanie ścieżek o ograniczonej długości

## Złożoność

### Złożoność algorytmu jest zależna od głębokości drzewa i liczby ruchów na każdym poziomie

Złożoność najprostszego algorytmu to <strong>O(b^n)</strong>, gdzie b to średnia liczba możliwych ruchów na każdym poziomie drzewa (szerokość drzewa), a n to liczba poziomów drzewa. Prunning alha beta w najlepszym przypadku może zredukować złożoność do O(b^(n/2)).

## Prunning alpha-beta

### Może znacząco zmniejszyć liczbę potrzebnych operacji to osiągnięcia wyniku. Jego warunkiem jest <strong>alpha >= beta</strong>.

Opiera się na tym, że dana gałąź nie jest sprawdzana, jeśli na danym etapie gry, mamy świadomość już o lepszej możliwości, więc i tak ta gałąź nie zostanie wybrana, więc może być niesprawdzana.

- alpha - najlepszy wynik, który maksymalizujący gracz może zagwarantować na danym etapie sprawdzania (początkowo -oo)
- beta - najlepszy wynik, który minimalizujący gracz może zagwarantować na danym etapie sprawdzania (początkowo oo)
