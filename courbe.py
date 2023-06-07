import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class Courbe:
    def __init__(self, df) -> None:
        self._df = df

        self.poly_features = None
        self.X_poly = None
        self.lin_reg = None

    @classmethod
    def load_data(cls):
        print("reading data...")
        df = pd.read_csv("data.csv", delimiter=";").sort_values(by="col_x")
        print("data ready.")
        return cls(df)
    
    def intervalles(self, *intervalles):
        courbes = []
        #first
        df_mask = self._df['col_x'] <= intervalles[0]
        courbes.append(Courbe(self._df[df_mask].reset_index(drop=True)))

        for n_intervalle, intervalle in enumerate(intervalles[1:]):
            n_intervalle += 1
            df_mask = self._df['col_x'] > intervalles[n_intervalle-1]
            df_m = self._df[df_mask]
            df_mask = df_m['col_x'] <= intervalle
            courbes.append(Courbe(df_m[df_mask].reset_index(drop=True)))
        
        #last
        df_mask = self._df['col_x'] > intervalles[-1]
        courbes.append(Courbe(self._df[df_mask].reset_index(drop=True)))

        return tuple(courbes)
    
    @property
    def X(self):
        return self._df[["col_x"]]
    
    @property
    def Y(self):
        return self._df[["col_y"]]
    
    def show(self, pyplot):
        pyplot.scatter(self.X, self.Y)
    
    def find_expo(self, degree):
        return Regression(self, degree)
        


class Regression():
    def __init__(self, courbe, degree) -> None:
        print("searching the best curve...")
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(courbe.X)

        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, courbe.Y)
        print("best curve found.")

        self._courbe = courbe
        self._degree = degree
        self._poly_features = poly_features
        self._X_poly = X_poly
        self._lin_reg = lin_reg
    
    @property
    def equation(self):
        powers = self._poly_features.powers_
        coefficients = self._lin_reg.coef_.flatten()
        equation_parts = []

        for power, coefficient in zip(reversed(powers), reversed(coefficients)):
            assert len(power) == 1, f"power var is longer than 1 : {power}"
            power = power[0]
            if coefficient != 0 and power > 0:
                if power == 1:
                    power_str = "x"
                else:
                    power_str = f"x^{power}"
                equation_parts.append(f"{coefficient}{power_str}")

        intercept = self._lin_reg.intercept_[0]
        if intercept != 0:
            equation_parts.append(str(intercept))

        equation = ' + '.join(equation_parts)
        equation = equation.replace("e", "*10**")
        return equation
    
    def predict(self):
        return self._lin_reg.predict(self._X_poly)
        
    def show(self, pyplot):
        pyplot.plot(self._courbe.X, self.predict())
