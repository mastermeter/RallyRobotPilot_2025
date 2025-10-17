import lzma
import pickle
import numpy as np
import sys
import os

# --- MODIFICATION CORRIGÉE ---
# Ajouter le répertoire du projet au chemin de Python pour permettre l'importation
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Importer la VRAIE classe depuis votre projet
from rallyrobopilot.sensing_message import SensingSnapshot
# --- FIN DE LA MODIFICATION ---


# --- À MODIFIER ---
# Remplacez ceci par le chemin vers votre fichier .npz
file_to_inspect = 'data/normal/record_alex_0n.npz'
# ------------------

# Redéfinir la méthode __str__ pour la classe importée
def custom_snapshot_str(self):
    """Retourne une représentation textuelle lisible de l'objet."""
    controls_str = (f"Avancer:{self.current_controls[0]}, "
                    f"Reculer:{self.current_controls[1]}, "
                    f"Gauche:{self.current_controls[2]}, "
                    f"Droite:{self.current_controls[3]}")
    
    return (
        f"  - Distances Raycast:\t{np.array(self.raycast_distances)}\n"
        f"  - Position Voiture:\t{self.car_position}\n"
        f"  - Vitesse Voiture:\t{self.car_speed:.2f}\n"
        f"  - Angle Voiture:\t{self.car_angle:.2f}\n"
        f"  - Contrôles Actuels:\t({controls_str})"
    )

# "Injecter" notre méthode d'affichage dans la classe originale
SensingSnapshot.__str__ = custom_snapshot_str


try:
    print(f"🔍 Inspection du fichier : {file_to_inspect}\n")

    with lzma.open(file_to_inspect, "rb") as f:
        all_snapshots = pickle.load(f)

    if all_snapshots:
        print(f"✅ Fichier chargé avec succès.")
        print(f"Nombre total d'enregistrements (snapshots) : {len(all_snapshots)}\n")

        print("--- Structure du premier enregistrement ---")
        first_snapshot = all_snapshots[0]
        print(first_snapshot)
        print("-----------------------------------------")

    else:
        print("Le fichier est vide ou n'a pas pu être lu.")

except FileNotFoundError:
    print(f"❌ Erreur : Le fichier '{file_to_inspect}' n'a pas été trouvé.")
except ModuleNotFoundError:
    print("❌ Erreur d'importation : Assurez-vous d'exécuter ce script depuis la racine de votre projet 'RallyRobotPilot_2025'.")
except Exception as e:
    print(f"❌ Une erreur inattendue est survenue : {e}")