import streamlit as st
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import librosa
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

input_features = None  # Initialize

# Session state for dashboard
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Tabs
tab1, tab2 = st.tabs(["Voice Analysis", "Dashboard"])

with tab1:
    # Title
    st.title("Parkinson's Disease Detection using Vocal Data")

    # Description
    st.write("""
    This app uses a Support Vector Machine (SVM) model trained on vocal features to predict Parkinson's disease.
    Record your voice and click 'Predict' to get the result.
    """)

    st.header("Record Your Voice")
    webrtc_ctx = webrtc_streamer(
        key="voice", 
        mode=WebRtcMode.SENDONLY, 
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True}
    )
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            try:
                # Process audio (see Step 2)
                audio_data = np.concatenate([frame.to_ndarray() for frame in audio_frames])
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()  # Ensure 1D
                sr = 22050  # Sample rate
                # Extract vocal features for Parkinson's detection
                y = audio_data.astype(np.float32)
                
                # Fundamental frequency using PYIN
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
                f0_voiced = f0[voiced_flag]
                if len(f0_voiced) > 0:
                    mean_f0 = np.mean(f0_voiced)
                    std_f0 = np.std(f0_voiced)
                    jitter_percent = (std_f0 / mean_f0) * 100 if mean_f0 > 0 else 0
                    jitter_abs = std_f0 / sr
                else:
                    mean_f0 = 0
                    jitter_percent = 0
                    jitter_abs = 0
                
                # Jitter measures (approximations)
                rap = jitter_percent / 3  # Rough approximation
                ppq = jitter_percent / 2
                ddp = jitter_percent
                
                # Amplitude (RMS)
                rms = librosa.feature.rms(y=y)[0]
                mean_amp = np.mean(rms)
                std_amp = np.std(rms)
                shimmer = (std_amp / mean_amp) * 100 if mean_amp > 0 else 0
                shimmer_db = 20 * np.log10((mean_amp + std_amp) / mean_amp) if mean_amp > 0 else 0
                
                # Shimmer measures (approximations)
                apq3 = shimmer / 2
                apq5 = shimmer / 1.5
                apq = shimmer / 1.2
                dda = shimmer
                
                # NHR and HNR (Noise to Harmonics Ratio, Harmonics to Noise Ratio)
                # Approximation using spectral flatness or simple ratio
                nhr = np.mean(np.abs(y)) / (mean_amp + 1e-10)  # Rough
                hnr = 20 * np.log10(mean_amp / (nhr + 1e-10)) if nhr > 0 else 0
                
                # Nonlinear measures (placeholders with typical PD values)
                rpde = 0.5  # Typical for PD
                dfa = 0.7   # Typical for PD
                spread1 = -4.0  # Typical for PD
                spread2 = 0.3
                d2 = 2.5
                ppe = 0.3
                
                # Feature list matching the dataset (22 features)
                features = [
                    mean_f0,          # MDVP:Fo(Hz)
                    np.max(f0_voiced) if len(f0_voiced) > 0 else 0,  # MDVP:Fhi(Hz)
                    np.min(f0_voiced) if len(f0_voiced) > 0 else 0,  # MDVP:Flo(Hz)
                    jitter_percent / 100,  # MDVP:Jitter(%)
                    jitter_abs,       # MDVP:Jitter(Abs)
                    rap / 100,        # MDVP:RAP
                    ppq / 100,        # MDVP:PPQ
                    ddp / 100,        # Jitter:DDP
                    shimmer / 100,    # MDVP:Shimmer
                    shimmer_db,       # MDVP:Shimmer(dB)
                    apq3 / 100,       # Shimmer:APQ3
                    apq5 / 100,       # Shimmer:APQ5
                    apq / 100,        # MDVP:APQ
                    dda / 100,        # Shimmer:DDA
                    nhr,              # NHR
                    hnr,              # HNR
                    rpde,             # RPDE
                    dfa,              # DFA
                    spread1,          # spread1
                    spread2,          # spread2
                    d2,               # D2
                    ppe               # PPE
                ]
                input_features = np.array([features])
                st.success("Audio processed successfully! Click Predict.")
                st.write("Extracted features (first 5):", features[:5])  # Debug
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                input_features = np.array([[0] * 22])  # Fallback

    # Predict button
    if st.button("Predict"):
        if input_features is None:
            st.error("Please record audio first.")
        else:
            # Scale the input
            input_scaled = scaler.transform(input_features)
            st.write("Scaled features (first 5):", input_scaled[0][:5])  # Debug
            
            # Make prediction
            prediction = model.predict(input_scaled)
            
            # Display result
            if prediction[0] == 1:
                st.error("Parkinson's Disease Detected")
                result = "Detected"
            else:
                st.success("No Parkinson's Disease Detected")
                result = "Not Detected"
            
            # Store for dashboard
            st.session_state.predictions.append({
                'timestamp': pd.Timestamp.now(),
                'result': result,
                'features': features
            })

with tab2:
    st.title("Dashboard")
    st.write("Local tracking of predictions and features.")
    
    # Load from session state
    local_data = st.session_state.predictions
    
    if local_data:
        df = pd.DataFrame(local_data)
        st.subheader("Prediction History (Local)")
        st.dataframe(df[['timestamp', 'result']])
        
        # Simple chart
        results_count = df['result'].value_counts()
        st.bar_chart(results_count)
        
        # Feature visualization
        st.subheader("Feature Trends")
        feature_df = df['features'].apply(pd.Series)
        feature_df.columns = [f'Feature {i+1}' for i in range(22)]
        st.line_chart(feature_df.iloc[:, :5])
    else:
        st.write("No predictions yet.")