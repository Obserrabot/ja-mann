#!/usr/bin/env python3
"""
Raspberry Pi Roboter Server
Empfängt Navigation-Kommandos vom PC und steuert Mecanum-Räder

Installation auf Raspberry Pi:
sudo apt update
sudo apt install python3-pip
pip3 install RPi.GPIO

Hardware-Anschlüsse (Beispiel):
- Motor Controller: L298N oder ähnlich
- Front Left Motor: GPIO 18, 19
- Front Right Motor: GPIO 20, 21  
- Back Left Motor: GPIO 22, 23
- Back Right Motor: GPIO 24, 25
"""

import socket
import json
import threading
import time
try:
    import RPi.GPIO as GPIO
    RASPBERRY_PI = True
except ImportError:
    print("⚠️ RPi.GPIO nicht verfügbar - Simulation Mode")
    RASPBERRY_PI = False

class MecanumRobotController:
    def __init__(self):
        self.running = True
        
        # Motor GPIO Pins (anpassen je nach Hardware)
        self.motor_pins = {
            'front_left': {'pin1': 18, 'pin2': 19, 'pwm': 12},
            'front_right': {'pin1': 20, 'pin2': 21, 'pwm': 13}, 
            'back_left': {'pin1': 22, 'pin2': 23, 'pwm': 16},
            'back_right': {'pin1': 24, 'pin2': 25, 'pwm': 26}
        }
        
        if RASPBERRY_PI:
            self.setup_gpio()
        
        # Server für PC Kommunikation
        self.server_socket = None
        self.setup_server()
        
        print("🤖 Mecanum Robot Controller gestartet")
    
    def setup_gpio(self):
        """GPIO Setup für Motoren"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        self.pwm_objects = {}
        
        for motor, pins in self.motor_pins.items():
            # Direction Pins
            GPIO.setup(pins['pin1'], GPIO.OUT)
            GPIO.setup(pins['pin2'], GPIO.OUT)
            # PWM Pin für Geschwindigkeit
            GPIO.setup(pins['pwm'], GPIO.OUT)
            
            # PWM Object erstellen (1000Hz)
            pwm = GPIO.PWM(pins['pwm'], 1000)
            pwm.start(0)
            self.pwm_objects[motor] = pwm
            
        print("✅ GPIO für Mecanum-Räder konfiguriert")
    
    def setup_server(self):
        """TCP Server für PC Kommunikation"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 8081))
            self.server_socket.listen(5)
            print("🌐 Server lauscht auf Port 8081")
        except Exception as e:
            print(f"❌ Server Setup Fehler: {e}")
    
    def set_motor_speed(self, motor: str, speed: int):
        """
        Setzt Motorgeschwindigkeit (-100 bis +100)
        """
        if not RASPBERRY_PI:
            print(f"🔧 Simulation: {motor} = {speed}")
            return
            
        speed = max(-100, min(100, speed))  # Clamp auf ±100
        
        pins = self.motor_pins[motor]
        pwm = self.pwm_objects[motor]
        
        if speed > 0:
            # Vorwärts
            GPIO.output(pins['pin1'], GPIO.HIGH)
            GPIO.output(pins['pin2'], GPIO.LOW)
            pwm.ChangeDutyCycle(abs(speed))
        elif speed < 0:
            # Rückwärts  
            GPIO.output(pins['pin1'], GPIO.LOW)
            GPIO.output(pins['pin2'], GPIO.HIGH)
            pwm.ChangeDutyCycle(abs(speed))
        else:
            # Stop
            GPIO.output(pins['pin1'], GPIO.LOW)
            GPIO.output(pins['pin2'], GPIO.LOW)
            pwm.ChangeDutyCycle(0)
    
    def execute_movement(self, command: dict):
        """
        Führt Bewegungskommando aus
        """
        action = command.get('action', 'stop')
        
        if action == 'move':
            wheels = command.get('wheels', {})
            
            # Mecanum Räder setzen
            self.set_motor_speed('front_left', wheels.get('front_left', 0))
            self.set_motor_speed('front_right', wheels.get('front_right', 0))
            self.set_motor_speed('back_left', wheels.get('back_left', 0))
            self.set_motor_speed('back_right', wheels.get('back_right', 0))
            
            target = command.get('target', {})
            distance = command.get('distance', 0)
            
            print(f"🎯 Fahre zu ({target.get('x', 0)}, {target.get('y', 0)}) - {distance}px entfernt")
            
        elif action in ['stop', 'emergency_stop']:
            # Alle Motoren stoppen
            for motor in self.motor_pins.keys():
                self.set_motor_speed(motor, 0)
            print(f"🛑 Roboter gestoppt: {action}")
            
        return {"status": "ok", "action": action}
    
    def handle_client(self, client_socket, address):
        """
        Behandelt eingehende PC Verbindung
        """
        print(f"🔗 PC verbunden: {address}")
        
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                    
                # JSON Kommando parsen
                try:
                    command = json.loads(data.decode('utf-8'))
                    response = self.execute_movement(command)
                    
                    # Antwort senden
                    client_socket.send(json.dumps(response).encode('utf-8'))
                    
                except json.JSONDecodeError:
                    error_response = {"status": "error", "message": "Invalid JSON"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            print(f"❌ Client Fehler: {e}")
        finally:
            client_socket.close()
            print(f"🔌 PC getrennt: {address}")
    
    def run(self):
        """
        Haupt-Server Loop
        """
        if not self.server_socket:
            print("❌ Server nicht verfügbar")
            return
            
        print("🚀 Roboter bereit für Kommandos...")
        
        try:
            while self.running:
                client_socket, address = self.server_socket.accept()
                
                # Jeden Client in eigenem Thread behandeln
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\n🛑 Roboter wird heruntergefahren...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Cleanup beim Beenden
        """
        self.running = False
        
        if RASPBERRY_PI:
            # Alle Motoren stoppen
            for motor in self.motor_pins.keys():
                self.set_motor_speed(motor, 0)
            
            # PWM beenden
            for pwm in self.pwm_objects.values():
                pwm.stop()
                
            GPIO.cleanup()
            print("✅ GPIO bereinigt")
        
        if self.server_socket:
            self.server_socket.close()
        
        print("🤖 Roboter heruntergefahren")

if __name__ == "__main__":
    robot = MecanumRobotController()
    robot.run()
