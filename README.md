# 🤖 Autonomer Ball-Sammler Roboter

Ein KI-gesteuerter Roboter mit Mecanum-Rädern, der automatisch Bälle auf einer 1x1m Fläche einsammelt.

Note:

venv aktivieren im root mit .\.venv\Scripts\Activate.ps1
## 🎯 System-Übersicht

```
Raspberry Pi (Roboter)          PC (KI-Verarbeitung)
├── Camera Stream      ────────> ├── YOLO Ball-Erkennung  
├── Mecanum-Räder      <──────── ├── Ball-Gruppierung
├── Motor Controller            ├── Navigation-KI
└── TCP Server (Port 8081)      └── Kommando-Sender
```

## 🚀 Setup

### Auf dem PC (Windows):
```bash
# PyTorch mit CUDA für RTX 3060
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# YOLO und OpenCV
pip install ultralytics opencv-python

# Navigation Module bereits in diesem Projekt enthalten
```

### Auf dem Raspberry Pi:
```bash
# System Update
sudo apt update && sudo apt upgrade -y

# Python Dependencies
sudo apt install python3-pip
pip3 install RPi.GPIO

# Kopiere raspberry_robot_server.py auf den Pi
scp raspberry_robot_server.py pi@chainmasters.local:~/

# Ausführbar machen und starten
chmod +x raspberry_robot_server.py
python3 raspberry_robot_server.py
```

## 🔧 Hardware-Anschlüsse (Raspberry Pi)

### Mecanum-Räder Motor Controller:
```
Motor Controller (z.B. L298N) → Raspberry Pi GPIO:

Front Left Motor:
├── Direction Pin 1 → GPIO 18
├── Direction Pin 2 → GPIO 19  
└── PWM Speed      → GPIO 12

Front Right Motor:
├── Direction Pin 1 → GPIO 20
├── Direction Pin 2 → GPIO 21
└── PWM Speed      → GPIO 13

Back Left Motor:
├── Direction Pin 1 → GPIO 22
├── Direction Pin 2 → GPIO 23
└── PWM Speed      → GPIO 16

Back Right Motor:
├── Direction Pin 1 → GPIO 24
├── Direction Pin 2 → GPIO 25
└── PWM Speed      → GPIO 26
```

### Stromversorgung:
- Raspberry Pi: 5V/3A USB-C
- Motoren: 12V/5A (separates Netzteil empfohlen)
- Motor Controller: Verbindung zu beiden Spannungen

## 🎮 Verwendung

### 1. Raspberry Pi starten:
```bash
# Auf dem Raspberry Pi
python3 raspberry_robot_server.py
```

### 2. PC KI-System starten:
```bash
# Auf dem PC (in diesem Ordner)
python main.py
```

### 3. Was passiert:
1. **Kamera-Stream**: Raspberry Pi sendet MJPEG Stream
2. **Ball-Erkennung**: PC erkennt Bälle mit YOLO KI
3. **Gruppierung**: Bälle werden nach Nähe gruppiert
4. **Entscheidung**: Größte Ball-Gruppe wird als Ziel gewählt
5. **Navigation**: PC berechnet Mecanum-Rad Bewegung
6. **Ausführung**: Raspberry Pi fährt zum Ziel

## 🧠 Entscheidungs-Logik

### Anti-Flacker System:
- **Cooldown**: 2 Sekunden zwischen Ziel-Änderungen
- **Stabilität**: Bevorzugt aktuelles Ziel bei kleinen Änderungen
- **Hysterese**: Nur große Ziel-Sprünge führen zu neuen Entscheidungen

### Ziel-Bewertung:
```python
Score = Ball_Anzahl × 100 + Nähe_Bonus + Stabilität_Bonus
```

- **Ball_Anzahl**: Hauptkriterium (mehr Bälle = höhere Priorität)
- **Nähe_Bonus**: Nähere Gruppen bevorzugt (max 300 Punkte)
- **Stabilität_Bonus**: Aktuelles Ziel bekommt +50 Punkte

### Mecanum-Rad Berechnung:
```python
# Holonome Bewegung (seitlich + vorwärts gleichzeitig)
front_left  = vx - vy - rotation
front_right = vx + vy + rotation  
back_left   = vx + vy - rotation
back_right  = vx - vy + rotation
```

## 📊 Performance Features

### PC (RTX 3060):
- **GPU Auslastung**: Optimiert für maximale RTX 3060 Nutzung
- **TensorFloat-32**: Beschleunigung für RTX 30xx Serie
- **Automatic Mixed Precision**: FP16 für höhere FPS
- **CUDNN Benchmark**: Optimiert für wiederkehrende Eingaben

### Stream Verarbeitung:
- **Buffer Minimierung**: Niedrige Latenz (1 Frame Buffer)
- **Auflösungs-Skalierung**: 3x Upscale für bessere Erkennung
- **Echtzeit-Visualisierung**: Live-Feedback mit Gruppen-Anzeige

## 🎯 Kalibrierung

### Ball-Gruppierung anpassen:
```python
# In main.py, Zeile ~140
ball_groups = group_balls(ball_centers, max_dist=100)  # Pixel-Abstand
```

### Navigation-Timing:
```python
# In robot_navigator.py
self.decision_cooldown = 2.0  # Sekunden zwischen Entscheidungen
self.target_reached_distance = 50  # Pixel für "Ziel erreicht"
```

### Motor-Geschwindigkeit:
```python
# In robot_navigator.py, calculate_mecanum_movement()
speed = min(100, max(30, distance / 5))  # Geschwindigkeit basierend auf Entfernung
```

## 🐛 Troubleshooting

### Kein Fenster sichtbar:
```bash
# Überprüfe OpenCV Installation
python -c "import cv2; print(cv2.__version__)"
```

### Roboter reagiert nicht:
```bash
# Teste Verbindung zum Raspberry Pi
telnet chainmasters.local 8081
```

### GPU nicht erkannt:
```bash
# Überprüfe CUDA Installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Niedrige Ball-Erkennung:
- Beleuchtung verbessern
- Ball-Kontrast zum Hintergrund erhöhen
- YOLO Confidence Threshold senken (conf=0.25 → 0.15)

## 🔄 Erweiterungsmöglichkeiten

### 1. Position-Tracking:
- Encoder an Rädern für präzise Position
- IMU/Gyroscope für Orientierung
- SLAM für Karten-Erstellung

### 2. Erweiterte KI:
- Reinforcement Learning für optimale Pfade
- Hindernis-Erkennung und -Umfahrung
- Multi-Roboter Koordination

### 3. Ball-Sammlung:
- Servo-gesteuerte Sammelmechanik
- Füllstand-Sensor im Sammelbehälter
- Automatische Entleerung an Basis-Station

### 4. Monitoring:
- Web-Interface für Live-Kontrolle
- Datenlogging von Routen und Performance
- Remote-Notfall-Stop über Web/App

## 📡 Netzwerk-Protokoll

### PC → Raspberry Pi Kommandos:
```json
{
  "action": "move",
  "wheels": {
    "front_left": 75,
    "front_right": 50,
    "back_left": 50,  
    "back_right": 75
  },
  "target": {"x": 320, "y": 240},
  "distance": 125,
  "speed": 65
}
```

### Raspberry Pi → PC Antworten:
```json
{
  "status": "ok",
  "action": "move",
  "position": {"x": 300, "y": 250},
  "battery": 87
}
```

## ⚡ Optimierung für 100 Bälle

### Ball-Dichte Herausforderungen:
- **Occlusion**: Überlappende Bälle schwer erkennbar
- **Lösung**: Multi-Angle Erkennung oder höhere Kamera-Position

### Effizienz-Strategien:
- **Spiralen-Muster**: Außen nach innen arbeiten
- **Cluster-First**: Größte Gruppen zuerst sammeln
- **Pfad-Optimierung**: Kürzeste Routen zwischen Gruppen

### Hardware-Empfehlungen:
- **Kamera**: 1080p/60fps für flüssige Erkennung
- **Motoren**: Hohe Drehmomente für schnelle Richtungsänderungen
- **Sammelmechanik**: Großer Trichter für mehrere Bälle gleichzeitig

---

**Viel Erfolg beim Ball-Sammeln! 🎾🤖**
