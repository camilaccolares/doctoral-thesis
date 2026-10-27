import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pydea
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from datetime import datetime

begin = datetime.now()

df = pd.read_excel("3-dataset_completo.xlsx")

inputs = ["input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7", "input_8", "input_9", "input_10", "input_11", "input_12"]
outputs = ["output_1", "output_2", "output_3", "output_4", "output_5", "output_6", "output_7"]

def selecionar_variaveis_com_autoencoder(df, inputs, outputs, max_vars=7, plot=False):
    features = inputs + outputs
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Definir o autoencoder base
    autoencoder_base = MLPRegressor(random_state=42)

    # Definir a grade de hiperparâmetros para busca
    param_grid = {
        'hidden_layer_sizes': [(len(features)//2,), (len(features)//2, len(features)//4), (len(features),)],
        'activation': ['relu', 'tanh'],
        'max_iter': [500, 1000, 1500],
    }

    # Definir a métrica de score (MSE negativo para GridSearchCV)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Configurar GridSearchCV com validação cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(autoencoder_base, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_scaled, X_scaled)

    # Obter o melhor autoencoder
    autoencoder = grid_search.best_estimator_
    X_pred = autoencoder.predict(X_scaled)
    
    # Avaliação de performance do autoencoder otimizado
    mse = mean_squared_error(X_scaled, X_pred)
    r2 = r2_score(X_scaled, X_pred)
    print(f"\n--- Avaliação do Autoencoder Otimizado ---")
    print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")
    print(f"Melhor Score (MSE Negativo): {grid_search.best_score_:.4f}")
    print(f"Erro Quadrático Médio (MSE) no dataset completo: {mse:.4f}")
    print(f"Coeficiente de Determinação (R²) no dataset completo: {r2:.4f}")
    print(f"------------------------------------------")

    reconstruction_error = ((X_scaled - X_pred) ** 2).mean(axis=0)
    error_per_feature = pd.Series(reconstruction_error, index=features).sort_values(ascending=False)

    selected_vars = error_per_feature.head(max_vars).index.tolist()

    # Garante 1 input e 1 output
    selected_inputs = [v for v in selected_vars if v in inputs]
    selected_outputs = [v for v in selected_vars if v in outputs]

    if len(selected_inputs) == 0:
        best_input = error_per_feature[inputs].idxmax()
        if best_input not in selected_vars:
            selected_vars[-1] = best_input

    if len(selected_outputs) == 0:
        best_output = error_per_feature[outputs].idxmax()
        if best_output not in selected_vars:
            selected_vars[-1] = best_output

    selected_vars = list(set(selected_vars))
    selected_vars = error_per_feature[selected_vars].sort_values(ascending=False).index.tolist()

    if plot:
        plt.figure(figsize=(10, 6))
        plot_series = error_per_feature.sort_values(ascending=True)
        colors = ['#ff7f0e' if var in selected_vars else '#1f77b4' for var in plot_series.index]
        plot_series.plot(kind='barh', color=colors)
        plt.title("Erro de reconstrução por variável (Autoencoder)")
        plt.xlabel("Erro médio quadrático")
        plt.ylabel("Variável")
        legend_handles = [
            plt.Rectangle((0,0),1,1, color="#ff7f0e", label="Selecionada"),
            plt.Rectangle((0,0),1,1, color="#1f77b4", label="Não selecionada")
        ]
        plt.legend(handles=legend_handles)
        plt.tight_layout()
        plt.show()

    return selected_vars, error_per_feature

def explicar_variaveis_via_shap(df, selected_vars, all_vars, n_amostras=80):
    explicacoes = {}
    X = df[all_vars].dropna().copy()

    if len(X) > n_amostras:
        X = X.sample(n_amostras, random_state=42)

    for target_var in selected_vars:
        X_other = X.drop(columns=[target_var])
        y_target = X[target_var]

        model = RandomForestRegressor(random_state=42)
        model.fit(X_other, y_target)

        explainer = shap.Explainer(model, X_other)
        shap_values = explainer(X_other)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_series = pd.Series(mean_abs_shap, index=X_other.columns).sort_values(ascending=False)
        explicacoes[target_var] = shap_series

        # Plot SHAP
        print(f"\nReconstruindo: {target_var}")
        print(shap_series.head(5).to_frame("Importância média (SHAP)"))
        #shap.plots.bar(shap_values, max_display=10, show=True)

    return explicacoes

#main
selecionadas, erros = selecionar_variaveis_com_autoencoder(df, inputs, outputs)

print("Variáveis selecionadas:", selecionadas)

explicacoes = explicar_variaveis_via_shap(df, selecionadas, inputs + outputs)

lista_inputs = [i for i in selecionadas if "input" in i]
lista_outputs = [o for o in selecionadas if "output" in o]

df_inputs = df[lista_inputs]
df_outputs = df[lista_outputs]

problema = pydea.DEAProblem(df_inputs, df_outputs, returns="VRS")
resultado = problema.solve()
print(resultado["Efficiency"])
print("Variáveis selecionadas:", selecionadas)

end = datetime.now()

begin_time = begin.strftime("%H:%M:%S")
end_time = end.strftime("%H:%M:%S")
total_time = end - begin


k = resultado["Efficiency"].var()

arquivo_resultado = "82-resultados_xai_otimizado_2.txt"
with open(arquivo_resultado, "w") as f:
    f.write("Inicio: ")
    f.write(begin_time)
    f.write("\n")
    f.write("Termino: ")
    f.write(end_time)
    f.write("\n")
    f.write("Tempo total: ")
    f.write(str(total_time))
    f.write("\n")
    f.write("Variancia melhor resultado: ")
    f.write(str(k))
    f.write("\n")
    f.write("Eficiencias:")
    f.write("\n")
    f.write(str(resultado["Efficiency"]))
    f.write("\n")
    f.write("Fatores:")
    f.write("\n")
    f.write(str(selecionadas))



