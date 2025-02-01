import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#  ÉTAPE 1 : Implémentation de la méthode d'Euler pour le modèle SIRD


def sird_forecast(beta, gamma, mu, S0, I0, R0, D0, step, nb_jours):
    
    # Initialisation des listes qui vont stocker le temps et l'évolution de S, I, R, D
    time = [0]
    S = [S0]
    I = [I0]
    R = [R0]
    D = [D0]

    # Calcul du nombre total d'itérations à partir du nombre de jours et du pas "step"
    nb_points = int(nb_jours / step)

    # Boucle d'Euler : on avance pas à pas
    for _ in range(nb_points):
        # Récupérer les dernières valeurs connues de S, I, R, D
        s_current = S[-1]
        i_current = I[-1]
        r_current = R[-1]
        d_current = D[-1]

        # Calcul des dérivées (dS/dt, dI/dt, etc.) selon les équations du modèle
        dSdt = -beta * s_current * i_current
        dIdt = beta * s_current * i_current - gamma * i_current - mu * i_current
        dRdt = gamma * i_current
        dDdt = mu * i_current

        # Mise à jour des valeurs pour le prochain point de la simulation
        S.append(s_current + step * dSdt)
        I.append(i_current + step * dIdt)
        R.append(r_current + step * dRdt)
        D.append(d_current + step * dDdt)
        time.append(time[-1] + step)

    # Conversion des listes en tableaux NumPy pour faciliter l'utilisation ultérieure
    return np.array(time), np.array(S), np.array(I), np.array(R), np.array(D)


def plot_sird(time, S, I, R, D, title="Modèle SIRD", label_prefix=""):
    """
    Affiche les courbes S(t), I(t), R(t), D(t) sur un même graphique.
    label_prefix permet de distinguer plusieurs scénarios sur un même plot.
    """
    # On trace chaque courbe en fonction du temps
    plt.plot(time, S, label=f'{label_prefix}S(t)')
    plt.plot(time, I, label=f'{label_prefix}I(t)')
    plt.plot(time, R, label=f'{label_prefix}R(t)')
    plt.plot(time, D, label=f'{label_prefix}D(t)')
    plt.xlabel("Temps (jours)")
    plt.ylabel("% Population")
    plt.grid(True)
    plt.title(title)
    plt.legend()



                           
#  ÉTAPE 2 : Simulation et analyse pour un jeu de paramètres donné

def step2_simulation_example():

    # Définition des paramètres du modèle
    beta = 0.5
    gamma = 0.15
    mu = 0.015
    S0 = 0.99
    I0 = 0.01
    R0 = 0.0
    D0 = 0.0
    step = 0.01
    nb_jours = 200

    # Appel de la fonction sird_forecast pour obtenir les courbes S, I, R, D
    t, S, I, R, D = sird_forecast(beta, gamma, mu, S0, I0, R0, D0, step, nb_jours)

    # Affichage graphique
    plt.figure(figsize=(10,6))
    plot_sird(t, S, I, R, D,
              title="Étape 2 : Simulation SIRD (Beta=0.5, Gamma=0.15, Mu=0.015)")
    plt.show()



#ÉTAPE 3
# via une fonction de coût (MSE) + Grid Search

def mse(prediction, data):
    """ Calcule la Mean Squared Error (MSE) entre deux séries de même taille. """
    return np.mean((prediction - data)**2)

def cost_function_sird(beta, gamma, mu, S0, I0, R0, D0, step, nb_jours,
                       t_data, S_data, I_data, R_data, D_data):

    # 1) Simulation fine (pas=0.01)
    t_model, S_model, I_model, R_model, D_model = sird_forecast(
        beta, gamma, mu, S0, I0, R0, D0, step, nb_jours
    )

    # 2) Interpolation : on ramène la simulation aux instants t_data
    S_model_interp = np.interp(t_data, t_model, S_model)
    I_model_interp = np.interp(t_data, t_model, I_model)
    R_model_interp = np.interp(t_data, t_model, R_model)
    D_model_interp = np.interp(t_data, t_model, D_model)

    # 3) Calcul de la MSE entre ces "valeurs interpolées" et les données réelles
    mse_s = mse(S_model_interp, S_data)
    mse_i = mse(I_model_interp, I_data)
    mse_r = mse(R_model_interp, R_data)
    mse_d = mse(D_model_interp, D_data)

    # On renvoie la somme des MSE pour évaluer l'écart global du modèle par rapport aux données
    return mse_s + mse_i + mse_r + mse_d


def grid_search_sird(betas, gammas, mus,
                     S0, I0, R0, D0,
                     step, nb_jours,
                     t_data, S_data, I_data, R_data, D_data):
    
    # On initialise le "meilleur coût" avec l'infini pour trouver le minimum
    best_cost = float('inf')
    best_params = (None, None, None)

    # Double/triple boucles pour tester toutes les combinaisons de (beta, gamma, mu)
    for b in betas:
        for g in gammas:
            for m in mus:
                current_cost = cost_function_sird(
                    b, g, m,
                    S0, I0, R0, D0,
                    step, nb_jours,
                    t_data, S_data, I_data, R_data, D_data
                )
                # On met à jour les paramètres si on obtient un coût plus faible
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = (b, g, m)
    return best_params, best_cost


def step3_parameter_fitting():
    
    # 1) Lecture du fichier CSV contenant nos données réelles
    data = pd.read_csv("sird_dataset.csv")  
    t_data = data["Jour"].values
    S_data = data["Susceptibles"].values
    I_data = data["Infectés"].values
    R_data = data["Rétablis"].values
    D_data = data["Décès"].values

    # Conditions initiales : on récupère les premières valeurs de S, I, R, D
    S0 = S_data[0]
    I0 = I_data[0]
    R0 = R_data[0]
    D0 = D_data[0]

    # Paramètres de la simulation
    step = 0.01
    nb_jours = t_data[-1]  # on suppose que la dernière ligne = temps final

    # 2) On définit des plages de valeurs à tester pour beta, gamma, mu
    betas = np.linspace(0.25, 0.5, 6)
    gammas = np.linspace(0.08, 0.15, 8)
    mus = np.linspace(0.005, 0.015, 3)

    # 3) Grid Search pour trouver la combinaison (beta, gamma, mu) qui minimise l'erreur
    best_params, best_cost = grid_search_sird(
        betas, gammas, mus,
        S0, I0, R0, D0,
        step, nb_jours,
        t_data, S_data, I_data, R_data, D_data
    )
    best_beta, best_gamma, best_mu = best_params
    print("Meilleurs paramètres :", best_beta, best_gamma, best_mu)
    print("Coût minimal :", best_cost)

    # 4) On simule avec les meilleurs paramètres trouvés
    t_model, S_model, I_model, R_model, D_model = sird_forecast(
        best_beta, best_gamma, best_mu,
        S0, I0, R0, D0,
        step, nb_jours
    )

    # On interpole la solution du modèle pour l'afficher aux mêmes instants que les données
    S_model_interp = np.interp(t_data, t_model, S_model)
    I_model_interp = np.interp(t_data, t_model, I_model)
    R_model_interp = np.interp(t_data, t_model, R_model)
    D_model_interp = np.interp(t_data, t_model, D_model)

    # Affichage final : comparaison données vs. modèle
    plt.figure(figsize=(10,6))

    # -- Données : on les trace en points ("o")
    plt.plot(t_data, S_data, "bo", label="S_data")
    plt.plot(t_data, I_data, "ro", label="I_data")
    plt.plot(t_data, R_data, "go", label="R_data")
    plt.plot(t_data, D_data, "ko", label="D_data")

    # -- Modèle : on trace la courbe "continue"
    plt.plot(t_model, S_model, "b--", label="S_model (fine)")
    plt.plot(t_model, I_model, "r--", label="I_model (fine)")
    plt.plot(t_model, R_model, "g--", label="R_model (fine)")
    plt.plot(t_model, D_model, "k--", label="D_model (fine)")

    plt.title("Étape 3 : Ajustement des paramètres - Meilleur modèle vs Données (Interpolé)")
    plt.xlabel("Temps (jours)")
    plt.ylabel("% Population")
    plt.grid(True)
    plt.legend()
    plt.show()


#ÉTAPE 4
# Scénarios de contrôle : paramètre R0=beta/(gamma+mu).
# Expliquer R0<1 ou R0>1, puis appliquer une réduction de beta
# et comparer 2 scénarios sur un même plot.
###############################################################################
def step4_control_scenario():
   
    # On fixe gamma et mu
    gamma = 0.15
    mu = 0.015

    # Rappel théorique sur R0
    print("R0 = beta / (gamma+mu). Si R0>1 => l'épidémie se propage, sinon elle s'éteint")

    # Conditions initiales
    S0 = 0.99
    I0 = 0.01
    R0_ = 0.0
    D0 = 0.0
    step = 0.01
    nb_jours = 150

    # ===============================
    # Scénario A : pas d'intervention
    # ===============================
    beta_A = 0.5  # beta constant
    tA, SA, IA, RA, DA = sird_forecast(
        beta_A, gamma, mu,
        S0, I0, R0_, D0,
        step, nb_jours
    )
    print("Scénario A : R0_A =", beta_A / (gamma + mu))

    # ===========================================
    # Scénario B : intervention pour réduire beta
    # ===========================================
    beta_B_avant = 0.5
    beta_B_apres = 0.2
    T_interv = 10  # Jour où l'on applique la mesure de contrôle

    # Phase 1 : avant l'intervention (0 -> T_interv)
    tB1, SB1, IB1, RB1, DB1 = sird_forecast(
        beta_B_avant, gamma, mu,
        S0, I0, R0_, D0,
        step, T_interv
    )

    # On récupère les valeurs à la fin de la phase 1 pour initialiser la phase 2
    S0_ph2 = SB1[-1]
    I0_ph2 = IB1[-1]
    R0_ph2 = RB1[-1]
    D0_ph2 = DB1[-1]

    # Phase 2 : après l'intervention (T_interv -> fin)
    nb_jours_2 = nb_jours - T_interv
    tB2, SB2, IB2, RB2, DB2 = sird_forecast(
        beta_B_apres, gamma, mu,
        S0_ph2, I0_ph2, R0_ph2, D0_ph2,
        step, nb_jours_2
    )
    # Décalage de la phase 2 pour aligner l'axe du temps
    tB2 = tB2 + T_interv

    # Fusion des deux phases en un seul tableau (on enlève le premier point en double)
    tB = np.concatenate([tB1, tB2[1:]])
    SB = np.concatenate([SB1, SB2[1:]])
    IB = np.concatenate([IB1, IB2[1:]])
    RB = np.concatenate([RB1, RB2[1:]])
    DB = np.concatenate([DB1, DB2[1:]])

    # Calcul des R0 avant et après l'intervention
    R0_1 = beta_B_avant / (gamma + mu)
    R0_2 = beta_B_apres / (gamma + mu)
    print(f"Scénario B => Avant intervention : R0_1={R0_1:.2f}, Après={R0_2:.2f}")

    # Affichage comparatif des courbes d'infectés
    plt.figure(figsize=(10, 6))
    plt.plot(tA, IA, "r-",  label="Scénario A : I(t)")
    plt.plot(tB, IB, "r--", label="Scénario B : I(t)")

    plt.title("Étape 4 : Scénarios de contrôle (A vs. B)")
    plt.xlabel("Temps (jours)")
    plt.ylabel("% Infectés")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # 1) Étape 2 : Simulation simple
    print("=== Exécution de l'Étape 2 ===")
    step2_simulation_example()

    # 2) Étape 3 : Ajustement des paramètres (nécessite sird_dataset.csv
    #    avec colonnes Jour, Susceptibles, Infectés, Rétablis, Décès).
    print("\n=== Exécution de l'Étape 3 ===")
    step3_parameter_fitting()

    # 3) Étape 4 : Scénarios de contrôle
    print("\n=== Exécution de l'Étape 4 ===")
    step4_control_scenario()
