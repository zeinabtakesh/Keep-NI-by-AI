from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os, json
from datetime import datetime
import pandas as pd


app = Flask(__name__)
app.secret_key = "super_secret_key"
USERS_FILE = "users.json"

# ----------------- Utilities -----------------
def extract_frames(video_path, num_frames=5, as_pil=True):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        raise ValueError("Video has no frames")

    sample_indices = np.linspace(0, frame_count - 1, num=num_frames, endpoint=False, dtype=int)
    extracted_frames = []
    i = 0
    current_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i in sample_indices:
            if as_pil:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames.append(Image.fromarray(rgb))
            else:
                extracted_frames.append(frame)

            current_index += 1

        if current_index >= num_frames:
            break

        i += 1

    cap.release()
    return extracted_frames



def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ----------------- Sign Up -----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users = load_users()

        if username in users:
            flash('Username already exists.', 'warning')
            return redirect(url_for('signup'))

        users[username] = generate_password_hash(password)
        save_users(users)
        flash('Account created successfully. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('sign-up.html')

# ----------------- Login -----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users = load_users()

        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('sign-in.html')

# ----------------- Logout -----------------

@app.route('/logout')
def logout():
    session.clear()  # Completely clears all session variables
    return redirect(url_for('login'))  # Redirect directly to login

# ----------------- Main Dashboard -----------------
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    response = ""
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            from cctv_monitor import CCTVMonitor
            monitor = CCTVMonitor()
            response = monitor.query_events(query)

    return render_template('dashboard.html', username=session['user'], response=response)

# ----------------- No Caching -----------------
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/alerts', methods=['GET', 'POST'])
def alerts():
    if 'user' not in session:
        return redirect(url_for('login'))

    from cctv_monitor import CCTVMonitor
    monitor = CCTVMonitor()

    # Extract camera names from the DataFrame
    df = monitor.captions_df.sort_values(by="timestamp", ascending=False)
    available_cameras = df["camera_name"].unique().tolist()

    selected_camera = request.form.get("camera", "all")
    if selected_camera != "all":
        df = df[df["camera_name"] == selected_camera]

    alerts_data = df.to_dict(orient='records')
    return render_template(
        'alerts.html',
        username=session['user'],
        alerts=alerts_data,
        available_cameras=available_cameras,
        selected_camera=selected_camera
    )

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            from cctv_monitor import CCTVMonitor
            monitor = CCTVMonitor()
            response = monitor.query_events(query)

            # Save to chat history
            session['chat_history'].append((query, response))
            session.modified = True

            # ✅ Return JSON if it's an AJAX (JS) request
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {"reply": response}

    return render_template('chat.html', username=session['user'], chat_history=session['chat_history'])


@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    session['chat_history'] = []
    return redirect(url_for('chat'))

@app.route('/alerts/clear', methods=['POST'])
def clear_alerts():
    if 'user' not in session:
        return redirect(url_for('login'))

    from cctv_monitor import CCTVMonitor
    monitor = CCTVMonitor()

    # Clear DataFrame and save
    monitor.captions_df = pd.DataFrame(columns=[
        'timestamp', 'camera_name', 'caption',
        'is_suspicious', 'reason', 'confidence',
        'image_path'
    ])
    monitor.save_data()

    flash("All alerts cleared successfully.", "success")
    return redirect(url_for('alerts'))


@app.route('/report')
def report():
    if 'user' not in session:
        return redirect(url_for('login'))

    report_text = "No report found."
    from cctv_monitor import CCTVMonitor
    monitor = CCTVMonitor()
    report_text = monitor.generate_report()

    return render_template('report.html', username=session['user'], report=report_text)


@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        file = request.files.get('video')
        if not file:
            flash("No video file provided.", "danger")
            return redirect(url_for('dashboard'))

        # Save the uploaded video
        upload_folder = os.path.join("uploads")
        os.makedirs(upload_folder, exist_ok=True)
        video_path = os.path.join(upload_folder, file.filename)
        file.save(video_path)

        # Init helpers
        from cctv_monitor import CCTVMonitor
        from inference import ImageInferenceEngine
        monitor = CCTVMonitor()
        engine = ImageInferenceEngine()

        # Extract frames
        frames = extract_frames(video_path, num_frames=3, as_pil=True)
        if not frames:
            flash("No frames could be extracted from the uploaded video.", "danger")
            return redirect(url_for('dashboard'))

        for idx, frame in enumerate(frames):
            frame_path = os.path.join(monitor.footage_dir, f"video_frame_{idx}.jpg")
            frame.save(frame_path)

            # Generate caption
            caption = engine.caption_image(frame)
            print(f"Frame {idx} caption: {caption}")

            # Analyze with GPT
            analysis = monitor.analyze_suspicious_activity(caption)

            # Log in DataFrame
            new_row = pd.DataFrame([{
                'timestamp': datetime.now(),
                'camera_name': file.filename,
                'caption': caption,
                'is_suspicious': analysis.get('is_suspicious', False),
                'reason': analysis.get('reason', 'Normal activity'),
                'confidence': analysis.get('confidence', 0.0),
                'image_path': frame_path
            }])
            monitor.captions_df = pd.concat([monitor.captions_df, new_row], ignore_index=True)

        # Save alerts
        monitor.save_data()

        flash(f"Successfully extracted, captioned, and logged {len(frames)} frames.", "success")

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"An error occurred while processing the video: {str(e)}", "danger")

    return redirect(url_for('dashboard'))

@app.route('/clear-alerts-json', methods=['POST'])
def clear_alerts_json():
    try:
        from pathlib import Path
        alerts_json_path = Path("static/alerts.json")
        alerts_json_path.write_text("[]", encoding="utf-8")
        print("[BUZZ] alerts.json cleared from /clear-alerts-json route.")
        return '', 204
    except Exception as e:
        print(f"[ERROR] Failed to clear alerts.json from buzz button: {e}")
        return 'Error', 500

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        file = request.files.get('image')
        if not file:
            flash("No image file provided.", "danger")
            return redirect(url_for('dashboard'))

        # Save the uploaded image
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        
        # Ensure unique filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = os.path.join(upload_folder, filename)
        file.save(image_path)

        # Process the image with CCTVMonitor
        from cctv_monitor import CCTVMonitor
        monitor = CCTVMonitor()
        
        caption, analysis = monitor.process_uploaded_image(image_path, camera_name="Uploaded Image")
        
        if caption.startswith("Error"):
            flash(f"Error processing image: {caption}", "danger")
        else:
            suspicious_status = "Suspicious" if analysis.get('is_suspicious', False) else "Normal"
            flash(f"Image processed successfully. Caption: '{caption}'. Status: {suspicious_status}", "success")
        
        return redirect(url_for('alerts'))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"An error occurred while processing the image: {str(e)}", "danger")
        
    return redirect(url_for('dashboard'))

# ----------------- Start App -----------------
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
