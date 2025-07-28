import cv2
from ultralytics import YOLO
import time
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from robot_navigator import robot_navigator

# MJPEG-Stream vom Pi/Roboter
stream_url = 'http://chainmasters.local:8080/?action=stream'

# RTX 3060 MAXIMUM POWER! 🚀 (mit CPU Fallback)
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.backends.cudnn.benchmark = True  # Optimiert für wiederkehrende Input-Größen
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 für RTX 30xx Series
    torch.backends.cudnn.allow_tf32 = True
    print(f"🔥 RTX 3060 BEAST MODE: {device}")
    print(f"💪 GPU Name: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    # CPU Maximum Performance Settings
    torch.set_num_threads(torch.get_num_threads())  # Alle CPU Kerne nutzen
    print(f"💻 CPU BEAST MODE: {device}")
    print(f"🔥 CPU Kerne: {torch.get_num_threads()}")
    print("⚠️ CUDA nicht verfügbar - installiere PyTorch mit CUDA für RTX 3060!")

print(f"🎯 CUDA verfügbar: {torch.cuda.is_available()}")
print(f"🔥 RTX 3060 BEAST MODE: {device}")
print(f"🎯 CUDA verfügbar: {torch.cuda.is_available()}")
print(f"� GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Keine GPU'}")

# Öffne den MJPEG-Stream wie eine Kamera
cap = cv2.VideoCapture(stream_url)
# Stream-Buffer minimieren für niedrigere Latenz
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# Threading für Stream-Reading
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# YOLOv8 BEAST MODE - optimiert für verfügbares Device
model = YOLO('yolov8s.pt')  # Können auch yolov8m.pt oder yolov8l.pt probieren
model.to(device)

# Warmup für GPU/CPU
if device == 'cuda:0':
    dummy_input = torch.randn(1, 3, 1280, 1280).to(device).float()
    print("🔥 GPU Warmup läuft...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.model(dummy_input)
    print("✅ GPU bereit für MAXIMUM PERFORMANCE!")
else:
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    print("🔥 CPU Warmup läuft...")
    with torch.no_grad():
        for _ in range(2):
            _ = model.model(dummy_input)
    print("✅ CPU bereit für MAXIMUM PERFORMANCE!")

if not cap.isOpened():
    print("❌ Stream konnte nicht geöffnet werden.")
    exit()

prev_time = time.time()
fps = 0
frame_count = 0
fps_display = 0

# Frame-Queue für Multi-Threading
frame_queue = []
max_queue_size = 3

# ThreadPoolExecutor für parallele Verarbeitung
executor = ThreadPoolExecutor(max_workers=4)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kein Frame empfangen.")
        break

    # FPS berechnen
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        fps_display = frame_count / (now - prev_time)
        prev_time = now
        frame_count = 0

    # Stream FPS Counter entfernt

    # --- BEAST MODE: KI-Ball-Erkennung (GPU/CPU optimiert) ---
    if device == 'cuda:0':
        # RTX 3060 Settings
        scale = 3.0  # Noch mehr Skalierung für maximale GPU-Auslastung
        imgsz = 1280  # Maximale YOLOv8 Auflösung für beste Qualität
        use_amp = True
    else:
        # CPU Optimized Settings
        scale = 2.0  # CPU bekommt auch mehr Qualität
        imgsz = 1280
        use_amp = False
    
    frame_up = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)  # Beste Qualität
    
    # Device-optimierte Inferenz
    if use_amp and device == 'cuda:0':
        with torch.cuda.amp.autocast():  # Nur bei GPU
            results = model(frame_up, 
                           imgsz=imgsz,
                           device=device,
                           half=True,
                           verbose=False,
                           conf=0.15,  # NIEDRIGER für mehr Ball-Detektionen
                           iou=0.3,    # NIEDRIGER für überlappende Bälle
                           max_det=200, # MEHR Detektionen
                           augment=True)
    else:
        # CPU oder GPU ohne AMP
        results = model(frame_up, 
                       imgsz=imgsz,
                       device=device,
                       half=False,  # CPU unterstützt kein FP16
                       verbose=False,
                       conf=0.2,    # NIEDRIGER für CPU
                       iou=0.3,     # NIEDRIGER für überlappende Bälle
                       max_det=150, # MEHR Detektionen
                       augment=True)
    # Alle Ball-Mittelpunkte sammeln
    ball_centers = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 32:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ball_centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Ball-Gruppen nach Abstand clustern (STABILISIERT)
    def group_balls(centers, max_dist=120):
        if not centers:
            return []
        
        groups = []
        used = set()
        
        # Sortiere Bälle nach Position für konsistente Gruppierung
        sorted_centers = sorted(centers, key=lambda p: (p[0], p[1]))
        
        for i, (x1, y1) in enumerate(sorted_centers):
            if i in used:
                continue
            group = [(x1, y1)]
            used.add(i)
            
            # Erweiterte Suche für stabile Gruppen
            for j, (x2, y2) in enumerate(sorted_centers):
                if j != i and j not in used:
                    dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
                    if dist < max_dist:
                        group.append((x2, y2))
                        used.add(j)
                        
                        # Erweitere Gruppe rekursiv (Ketten-Clustering)
                        for k, (x3, y3) in enumerate(sorted_centers):
                            if k not in used:
                                for gx, gy in group:
                                    chain_dist = ((gx-x3)**2 + (gy-y3)**2)**0.5
                                    if chain_dist < max_dist:
                                        group.append((x3, y3))
                                        used.add(k)
                                        break
            
            # Nur Gruppen mit min. Größe behalten (reduziert Flackern)
            if len(group) >= 1:  # Einzelbälle auch erlauben
                groups.append(group)
        
        return groups

    ball_groups = group_balls(ball_centers, max_dist=120)
    num_groups = len(ball_groups)
    cv2.putText(frame, f"Ball-Gruppen: {num_groups}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)
    
    # 🤖 ROBOTER NAVIGATION - Sende Kommandos an Raspberry Pi
    connection_status = robot_navigator.get_connection_status()
    
    if ball_groups:
        # Navigation-Logik läuft immer (auch offline für Ziel-Berechnung)
        navigation_success = robot_navigator.navigate_to_balls(ball_groups)
        
        if connection_status["connected"]:
            status_color = (0, 255, 0)
            status_text = "Roboter: Aktiv"
        elif connection_status["status"] == "retry_cooldown":
            status_color = (0, 165, 255)  # Orange
            retry_time = connection_status["retry_in"]
            status_text = f"Roboter: Retry in {retry_time}s"
        else:
            status_color = (0, 0, 255)  # Rot
            status_text = "Roboter: Offline"
    else:
        navigation_success = False
        status_color = (128, 128, 128)  # Grau
        status_text = "Roboter: Keine Baelle"
    
    cv2.putText(frame, status_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Aktuelles Ziel anzeigen (immer, auch wenn offline - zum Testen)
    if robot_navigator.last_target:
        target_x, target_y = robot_navigator.last_target
        cv2.circle(frame, (target_x, target_y), 20, (255, 0, 255), 3)
        cv2.putText(frame, "ZIEL", (target_x-20, target_y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # ROBOTER POSITION anzeigen (wo der Bot denkt dass er steht)
    robot_x, robot_y = robot_navigator.robot_position
    cv2.circle(frame, (robot_x, robot_y), 15, (0, 255, 255), 3)  # Gelber Kreis
    cv2.putText(frame, "ROBOT", (robot_x-25, robot_y-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    print(f"Ball-Gruppen erkannt: {num_groups}")

    # Jede Gruppe als dunkle Ellipse mit Ballanzahl markieren
    for group in ball_groups:
        if len(group) == 0:
            continue
        # Mittelpunkt und Bounding Box der Gruppe berechnen
        xs = [pt[0] for pt in group]
        ys = [pt[1] for pt in group]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = int(sum(xs) / len(xs))
        center_y = int(sum(ys) / len(ys))
        width = max(80, max_x - min_x + 40)
        height = max(80, max_y - min_y + 40)
        # Dunkle, halbtransparente Ellipse
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, center_y), (width//2, height//2), 0, 0, 360, (30,30,30), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        # Ballanzahl in die Ellipse schreiben (helles Gelb, dick)
        cv2.putText(frame, str(len(group)), (center_x-15, center_y+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 80), 4)

    # Performance-Anzeige für aktuelles Device
    if device == 'cuda:0':
        try:
            gpu_usage = torch.cuda.utilization() if torch.cuda.is_available() else 0
            gpu_temp = torch.cuda.temperature() if torch.cuda.is_available() else 0
            cv2.putText(frame, f"RTX 3060: {gpu_usage}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"GPU Temp: {gpu_temp}°C", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        except:
            cv2.putText(frame, "GPU Monitoring N/A", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"CPU BEAST MODE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Threads: {torch.get_num_threads()}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Live-Stream", frame)

    # FPS maximieren: Wartezeit auf 1ms reduzieren
    if cv2.waitKey(1) == 27:
        break

# Cleanup
executor.shutdown(wait=True)
cap.release()
cv2.destroyAllWindows()
if device == 'cuda:0':
    print("🔥 RTX 3060 Beast Mode beendet!")
else:
    print("💻 CPU Beast Mode beendet!")
print("\n🚀 Für RTX 3060 Support installiere:")
print("pip uninstall torch torchvision")
print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")