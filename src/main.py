"""
🩺 MediSign: AI Bridge for Deaf Patient-Doctor Communication
Main application file with Gradio UI and real-time medical sign language recognition
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import mediapipe as mp
import gradio as gr
from gtts import gTTS
import pygame
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Medical vocabulary (30 terms)
MEDICAL_TERMS = [
    "pain", "fever", "cough", "headache", "vomiting",
    "doctor", "medicine", "hospital", "nurse", "injection",
    "tablet", "water", "food", "sleep", "tired",
    "hurt", "stomach", "chest", "back", "leg",
    "arm", "eye", "ear", "nose", "blood",
    "dizzy", "cold", "hot", "help", "stop"
]
NUM_CLASSES = len(MEDICAL_TERMS)

class CameraCapture:
    """Handles webcam recording for sign language capture"""
    
    @staticmethod
    def record_sign(duration=2.0):
        """
        Record sign from webcam for specified duration
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            np.ndarray: Normalized frame array or None if failed
        """
        cap = cv2.VideoCapture(0)
        frames = []
        print(f"🎥 RECORDING... Recording for {duration} seconds")
        
        start = cv2.getTickCount()
        while (cv2.getTickCount() - start) / cv2.getTickFrequency() < duration:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                cv2.imshow('MediSign Recording', frame)
                if cv2.waitKey(1) == ord(' '):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frames:
            frames = np.array(frames[:60])  # Max 60 frames
            if len(frames) < 60:
                frames = np.repeat(frames, 60//len(frames)+1, axis=0)[:60]
            return frames.astype(np.float32) / 255.0
        return None

class FeatureExtractor:
    """Extract CNN features from video frames"""
    
    def __init__(self):
        self.base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.base_model.trainable = False
    
    def extract(self, frames):
        """
        Extract features from frames using MobileNetV2
        
        Args:
            frames (np.ndarray): Video frames [N, H, W, C]
            
        Returns:
            np.ndarray: Feature vectors [N, 1280]
        """
        features = []
        for frame in frames:
            feat = self.base_model.predict(frame[None, ...], verbose=0)[0]
            features.append(feat)
        return np.array(features)

class MediSignModel:
    """BiLSTM + Attention model for sign language recognition"""
    
    @staticmethod
    def build():
        """Build MediSign model with BiLSTM and Attention"""
        cnn_input = layers.Input((60, 1280))
        
        # BiLSTM layer
        lstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(cnn_input)
        
        # Attention mechanism
        attention = layers.Dense(128, activation='tanh')(lstm)
        attention = layers.Dense(1, activation='softmax', name='attention_weights')(attention)
        context = layers.Multiply()([lstm, attention])
        context = layers.GlobalAveragePooling1D()(context)
        
        # Classification head
        x = layers.Dense(64, activation='relu')(context)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=cnn_input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

class AudioOutput:
    """Handle text-to-speech output"""
    
    @staticmethod
    def speak(text):
        """
        Convert text to speech and play audio
        
        Args:
            text (str): Text to speak
        """
        try:
            pygame.mixer.init()
            tts = gTTS(text, lang='en')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tts.save(tmp.name)
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
                os.remove(tmp.name)
        except Exception as e:
            print(f"Error in TTS: {e}")

class MediSignApp:
    """Main MediSign application"""
    
    def __init__(self):
        self.model = MediSignModel.build()
        self.feature_extractor = FeatureExtractor()
        self.camera = CameraCapture()
        self.audio = AudioOutput()
        self.history = []
    
    def predict(self):
        """
        Main prediction pipeline:
        Camera → Features → Prediction → Speech Output
        
        Returns:
            str: Formatted prediction result
        """
        try:
            # Step 1: Capture sign
            frames = self.camera.record_sign(duration=2.0)
            if frames is None:
                return "❌ No sign captured. Please try again."
            
            # Step 2: Extract features
            features = self.feature_extractor.extract(frames)
            
            # Step 3: Predict
            pred = self.model.predict(features[None, ...], verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = pred[class_id]
            
            term = MEDICAL_TERMS[class_id]
            
            # Step 4: Audio output
            self.audio.speak(term)
            
            # Log prediction
            self.history.append({
                'timestamp': datetime.now(),
                'term': term,
                'confidence': confidence,
                'class_id': class_id
            })
            
            return f"""
🩺 **MEDISIGN RECOGNITION**
━━━━━━━━━━━━━━━━━━━━━━━
✅ **Medical Term:** {term.upper()}
📊 **Confidence:** {confidence:.1%}
🔢 **Class ID:** {class_id}
📹 **Frames processed:** {len(frames)}
⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━
*Record another sign to continue*
            """
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}"
    
    def get_history(self):
        """Get prediction history as DataFrame"""
        if not self.history:
            return "No predictions yet"
        return pd.DataFrame(self.history)
    
    def launch_ui(self):
        """Launch Gradio UI"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🩺 **MediSign: Real-Time Medical Sign Translator**
            
            ### How to use:
            1. Click **PREDICT** button
            2. Allow camera access
            3. Perform your medical sign
            4. The system recognizes it and speaks the term aloud
            
            ### Medical Vocabulary:
            pain, fever, cough, headache, vomiting, doctor, medicine, hospital, nurse, injection, 
            tablet, water, food, sleep, tired, hurt, stomach, chest, back, leg, arm, eye, ear, 
            nose, blood, dizzy, cold, hot, help, stop
            """)
            
            with gr.Row():
                predict_btn = gr.Button("🎥 PREDICT (Record Sign)", size="lg")
            
            output = gr.Markdown()
            predict_btn.click(fn=self.predict, outputs=output)
        
        demo.launch(share=True, debug=False, server_port=7860)

def main():
    """Entry point"""
    print("🩺 MediSign Starting...")
    app = MediSignApp()
    app.launch_ui()

if __name__ == "__main__":
    main()