from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os, json

app = Flask(__name__)
app.secret_key = "super_secret_key"
USERS_FILE = "users.json"

# ----------------- Utilities -----------------
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

#
# @app.route('/alerts', methods=['GET', 'POST'])
# def alerts():
#     if 'user' not in session:
#         return redirect(url_for('login'))
#
#     from cctv_monitor import CCTVMonitor
#     monitor = CCTVMonitor()
#
#     # Load and sort alerts
#     df = monitor.captions_df.sort_values(by="timestamp", ascending=False)
#     available_cameras = df["camera_name"].unique().tolist()
#
#     # Handle camera filter
#     selected_camera = request.form.get("camera", "all")
#     if selected_camera != "all":
#         df = df[df["camera_name"] == selected_camera]
#
#     # 🔊 Force first alert to be suspicious for testing
#     if df.shape[0] > 0:
#         df.iloc[0, df.columns.get_loc('is_suspicious')] = True
#         df.iloc[0, df.columns.get_loc('reason')] = "Forced test suspicious activity"
#         df.iloc[0, df.columns.get_loc('confidence')] = 0.99
#
#     alerts_data = df.to_dict(orient='records')
#     return render_template(
#         'alerts.html',
#         username=session['user'],
#         alerts=alerts_data,
#         available_cameras=available_cameras,
#         selected_camera=selected_camera
#     )

@app.route('/alerts', methods=['GET', 'POST'])
def alerts():
    if 'user' not in session:
        return redirect(url_for('login'))

    from cctv_monitor import CCTVMonitor
    monitor = CCTVMonitor()

    # Load and sort alerts
    df = monitor.captions_df.sort_values(by="timestamp", ascending=False)
    available_cameras = df["camera_name"].unique().tolist()

    # Handle camera filter
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


@app.route('/latest-alert')
def latest_alert():
    from cctv_monitor import CCTVMonitor
    monitor = CCTVMonitor()

    df = monitor.captions_df.sort_values(by="timestamp", ascending=False)

    if df.shape[0] > 0:
        latest = df.iloc[0]
        return {
            "is_suspicious": bool(latest["is_suspicious"]),
            "reason": latest["reason"],
            "timestamp": latest["timestamp"],
        }
    else:
        return {"is_suspicious": False}
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

            # Update chat history
            session['chat_history'].append((query, response))
            session.modified = True

            # 🔁 This handles the JS fetch call
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
    open("alerts.json", "w").write("[]")
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


@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    if 'user' not in session:
        return redirect(url_for('login'))

    def run_monitor():
        from cctv_monitor import CCTVMonitor
        monitor = CCTVMonitor()
        monitor.run()

    import threading
    thread = threading.Thread(target=run_monitor)
    thread.daemon = True
    thread.start()

    flash("Monitoring started. You can now check Alerts and Report.", "info")
    return redirect(url_for('dashboard'))

# ----------------- Start App -----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # ← Add host and port
