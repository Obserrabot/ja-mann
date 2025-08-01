import socket
import json
import time
import math
from typing import List, Tuple, Optional

class RobotNavigator:
    """
    Navigation Controller für den Ball-Sammler Roboter
    Sendet stabile Kommandos an den Raspberry Pi
    """
    
    def __init__(self, raspberry_ip: str = "chainmasters.local", command_port: int = 8081):
        self.raspberry_ip = raspberry_ip
        self.command_port = command_port
        # Entscheidungs-Stabilität
        self.last_target = None
        self.last_decision_time = 0
        self.decision_cooldown = 3.0  # 3 Sekunden zwischen Entscheidungen (erhöht!)
        self.target_reached_distance = 50  # Pixel-Abstand für "erreicht"
        self.group_memory = {}  # Speichert Gruppen-Historie für Stabilität
        # Connection Management
        self.connection_failed = False
        self.last_connection_attempt = 0
        self.connection_retry_delay = 5.0  # 5 Sekunden zwischen Verbindungsversuchen
        self.is_connected = False
        self.sock = None
        self.sock_lock = None
        # Roboter Position (wird vom Raspberry gesendet oder geschätzt)
        self.robot_position = (960, 800)  # Unten-Mitte im 1920x1080 Bild (realistischer!)
        print(f"🤖 Robot Navigator initialisiert für {raspberry_ip}:{command_port}")
        self._init_socket()

    def _init_socket(self):
        import threading
        self.sock_lock = threading.Lock()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.raspberry_ip, self.command_port))
            self.is_connected = True
            self.connection_failed = False
            print(f"🔗 Verbindung zum Roboter aufgebaut!")
        except Exception as e:
            self.sock = None
            self.is_connected = False
            self.connection_failed = True
            self.last_connection_attempt = time.time()
            print(f"❌ Konnte Verbindung nicht aufbauen: {e}")

    def _close_socket(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.is_connected = False
    
    def send_command_to_robot(self, command: dict) -> bool:
        """
        Sendet Kommando per TCP an den Raspberry Pi über persistente Verbindung
        """
        import threading
        current_time = time.time()
        # Überprüfe ob Verbindung kürzlich fehlgeschlagen ist
        if self.connection_failed and (current_time - self.last_connection_attempt) < self.connection_retry_delay:
            return False  # Noch im Cooldown
        # Versuche ggf. Reconnect
        if not self.sock or not self.is_connected:
            self._init_socket()
            if not self.sock or not self.is_connected:
                return False
        try:
            with self.sock_lock:
                message = json.dumps(command).encode('utf-8')
                self.sock.sendall(message)
                response = self.sock.recv(1024).decode('utf-8')
                self.connection_failed = False
                self.is_connected = True
                print(f"🤖 Roboter Antwort: {response}")
                return True
        except Exception as e:
            self.connection_failed = True
            self.is_connected = False
            self.last_connection_attempt = current_time
            self._close_socket()
            print(f"❌ Roboter offline - retry in {self.connection_retry_delay}s: {e}")
            return False
    def close(self):
        """Verbindung zum Roboter sauber schließen."""
        self._close_socket()
    
    def calculate_mecanum_movement(self, target_x: int, target_y: int, robot_x: int, robot_y: int) -> dict:
        """
        Berechnet Mecanum-Rad Bewegung für Navigation zum Ziel
        """
        # Vektor zum Ziel
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < self.target_reached_distance:
            # Ziel erreicht - stoppen
            return {
                "action": "stop",
                "message": "Ziel erreicht"
            }
        
        # Normalisierte Richtung
        norm_x = dx / distance
        norm_y = dy / distance
        
        # Mecanum Rad Geschwindigkeiten berechnen
        # Für holonome Bewegung (seitlich + vorwärts möglich)
        speed = min(100, max(30, distance / 5))  # Geschwindigkeit basierend auf Entfernung
        
        # Mecanum Wheel Formula:
        # front_left = vx - vy - rotation
        # front_right = vx + vy + rotation  
        # back_left = vx + vy - rotation
        # back_right = vx - vy + rotation
        
        vx = norm_x * speed  # Vorwärts/Rückwärts
        vy = norm_y * speed  # Links/Rechts
        rotation = 0  # Keine Rotation für jetzt
        
        front_left = vx - vy - rotation
        front_right = vx + vy + rotation
        back_left = vx + vy - rotation
        back_right = vx - vy + rotation
        
        return {
            "action": "move",
            "wheels": {
                "front_left": int(front_left),
                "front_right": int(front_right),
                "back_left": int(back_left),
                "back_right": int(back_right)
            },
            "target": {"x": target_x, "y": target_y},
            "distance": int(distance),
            "speed": int(speed)
        }
    
    def select_best_target(self, ball_groups: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Wählt die beste Ball-Gruppe zum Anfahren aus
        Priorisiert: Größe der Gruppe, Entfernung, Stabilität (VERBESSERT)
        """
        if not ball_groups:
            return None
        
        # Bewertung jeder Gruppe mit verbesserter Stabilität
        best_group = None
        best_score = -1
        
        for group in ball_groups:
            if len(group) == 0:
                continue
                
            # Gruppen-Mittelpunkt
            center_x = sum(pt[0] for pt in group) / len(group)
            center_y = sum(pt[1] for pt in group) / len(group)
            
            # Entfernung zum Roboter
            distance = math.sqrt((center_x - self.robot_position[0])**2 + 
                               (center_y - self.robot_position[1])**2)
            
            # Score Berechnung (überarbeitet für mehr Stabilität):
            # - Anzahl Bälle (wichtigster Faktor)
            # - Nähe zum Roboter (sekundär)
            # - STARKE Stabilität (wenn es das letzte Ziel war)
            ball_count_score = len(group) * 150  # Erhöht: Gruppengroße wichtiger
            distance_score = max(0, 200 - distance * 0.5)  # Reduziert: Entfernung weniger wichtig
            
            # MASSIVE Stabilität für aktuelles Ziel (verhindert Springen)
            stability_score = 0
            if self.last_target:
                target_distance = math.sqrt((center_x - self.last_target[0])**2 + 
                                          (center_y - self.last_target[1])**2)
                if target_distance < 150:  # Größerer Toleranzbereich
                    stability_score = 300  # SEHR hoher Bonus für aktuelles Ziel
            
            total_score = ball_count_score + distance_score + stability_score
            
            if total_score > best_score:
                best_score = total_score
                best_group = (int(center_x), int(center_y), len(group))
        
        return best_group
    
    def navigate_to_balls(self, ball_groups: List[List[Tuple[int, int]]]) -> bool:
        """
        Hauptnavigations-Logik: Entscheidet und navigiert zu Ball-Gruppen
        """
        current_time = time.time()
        
        # Entscheidungs-Cooldown prüfen (Stabilität!)
        if current_time - self.last_decision_time < self.decision_cooldown:
            if self.last_target:
                # Weiter zum aktuellen Ziel fahren
                command = self.calculate_mecanum_movement(
                    self.last_target[0], self.last_target[1],
                    self.robot_position[0], self.robot_position[1]
                )
                return self.send_command_to_robot(command)
            return False
        
        # Neues Ziel auswählen
        target = self.select_best_target(ball_groups)
        
        if target is None:
            # Keine Bälle gefunden - stoppen
            stop_command = {"action": "stop", "message": "Keine Bälle gefunden"}
            return self.send_command_to_robot(stop_command)
        
        target_x, target_y, ball_count = target
        
        # Nur neue Entscheidung treffen wenn sich das Ziel SIGNIFIKANT geändert hat
        if (self.last_target is None or 
            abs(target_x - self.last_target[0]) > 120 or  # Erhöht: weniger empfindlich
            abs(target_y - self.last_target[1]) > 120):   # Verhindert Ziel-Wechsel bei kleinen Schwankungen
            
            self.last_target = (target_x, target_y)
            self.last_decision_time = current_time
            
            print(f"🎯 Neues Ziel: ({target_x}, {target_y}) mit {ball_count} Bällen")
        
        # Bewegungskommando berechnen und senden
        command = self.calculate_mecanum_movement(
            target_x, target_y,
            self.robot_position[0], self.robot_position[1]
        )
        
        return self.send_command_to_robot(command)
    
    def get_connection_status(self) -> dict:
        """
        Gibt aktuellen Verbindungsstatus zurück
        """
        current_time = time.time()
        
        if self.connection_failed:
            time_until_retry = max(0, self.connection_retry_delay - (current_time - self.last_connection_attempt))
            return {
                "connected": False,
                "status": "retry_cooldown",
                "retry_in": round(time_until_retry, 1)
            }
        elif self.is_connected:
            return {
                "connected": True,
                "status": "online"
            }
        else:
            return {
                "connected": False,
                "status": "never_connected"
            }
    
    def update_robot_position(self, x: int, y: int):
        """
        Update der Roboter-Position (kann vom Raspberry gesendet werden)
        """
        self.robot_position = (x, y)
    
    def emergency_stop(self):
        """
        Notfall-Stop für den Roboter
        """
        stop_command = {
            "action": "emergency_stop",
            "message": "Notfall-Stop ausgelöst"
        }
        return self.send_command_to_robot(stop_command)

# Globaler Navigator
robot_navigator = RobotNavigator()
