from nodes import Resistor, Capacitor, Inductor, SeriesNode, ParallelNode, ShuntNode, DriverNode
from mutator import TreeMutator
import numpy as np

# Mocking de la methode load_data du DriverNode pour éviter de nécessiter 
# de vrais fichiers ZMA et FRD pendant le test
def mock_load_data(self, filepath):
    # Retourne des arrays (freqs, mag, phase) factices
    return np.array([1000]), np.array([8]), np.array([0])
DriverNode.load_data = mock_load_data

def main():
    # 1. Création des composants de base
    tweeter = DriverNode("Tweeter", "dummy.frd", "dummy.zma")
    c1 = Capacitor(4.7e-6)
    c2 = Capacitor(1e-6)
    l1 = Inductor(0.5e-3)
    shunt_l1 = ShuntNode(l1)
    shunt_c2 = ShuntNode(c2)


    # 2. Assemblage d'un filtre passe-haut du 2ème ordre :
    # C1 en série avec (L1 en parallèle (shunt) avec le Tweeter)
    crossover_tree = SeriesNode(c1, ParallelNode(shunt_l1, ShuntNode(tweeter)))
    

    print("="*40)
    print("ARBRE ORIGINAL")
    print("="*40)
    crossover_tree.display()
    
    # On affiche graphiquement le premier arbre dans le navigateur
    print("\n[+] Ouverture de la visualisation graphique dans votre navigateur par défaut...")
    crossover_tree.draw()

    print("\n")

    # 3. Test des mutations
    mutator = TreeMutator()
    
    for i in range(3):
        print("="*40)
        print(f"MUTATION {i+1}")
        print("="*40)
        mutated_tree = mutator.mutate(crossover_tree)
        mutated_tree.display()
        print("\n")

if __name__ == "__main__":
    main()
