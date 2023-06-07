from courbe import Courbe
import matplotlib.pyplot as plt

courbe_globale = Courbe.load_data()

courbes = courbe_globale.intervalles(41.5, 65)

for n, c in enumerate(courbes):
    r = c.find_expo([5, 7, 5][n])
    r.show(plt)
    print(f"equation {n+1}", r.equation)

plt.show()