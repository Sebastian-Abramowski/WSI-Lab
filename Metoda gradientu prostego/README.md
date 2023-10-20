Podstawowe wiadomości przed wykonaniem zadania:

## Gradient

Gradient wskazuje kierunek najszybszego wzrostu funkcji.

Gradient funkcji to wektor, którego składowe to pochodne cząstkowe funkcji po wszystkich jej argumentach.

## Gradient przykład

Gradient w tym przypadku to żółte wektory, dla każdego punktu (x, y) jest przyporzadkowany wektor.
<img src="img/gradient.png" alt="Example of gradient" width="75%">
Długość wektora gradientu pokazuje nam jak bardzo jest stromo (czyli jak szybko powierzchnia w danym punkcie się wznosi)

## Numpy

- np.arange(-2, 2, 0.05) - zwraca symetrycznie rozłożone wartości z przedziału [-2, 2) oddalone o 0.05

- np.exp - zwraca wartość funkcji eksponencjalnej dla wartości lub tablicy wartości f(x) = exp(x) = e^x

- np.meshgrid - bierze dwie tablice i tworzy z nich grid, zwraca listę numpy.ndarrays <br/>
  przykład:
  xvalues = np.array([0, 1, 2, 3, 4]) <br/>
  yvalues = np.array([0, 1, 2, 3, 4]) <br/>

xx, yy = np.meshgrid(xvalues, yvalues) <br/>
xx will be: <br/>
[[0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]] <br/>
<img src="img/meshgrid.png" alt="Example of gradient" width="75%">

## Matplotlib

plt.contour - tworzy wykres konturowy (przedstawienie wykresu 3d na wykresie 2d)
