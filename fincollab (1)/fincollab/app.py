import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, date

def preprocess_user_features(users):
    """Convert user data into numerical features for clustering"""
    features = []
    
    for user in users:
        # Age from date of birth (convert to years)
        age = 0
        if user.dob:
            today = date.today()  # Use date instead of datetime
            # Convert datetime to date if necessary
            if isinstance(user.dob, datetime):
                user_dob = user.dob.date()
            else:
                user_dob = user.dob
            age = (today - user_dob).days / 365.25
        
        # Gender encoding
        gender_encoded = 0
        if user.gender:
            gender_lower = user.gender.lower()
            if gender_lower in ['male', 'm']:
                gender_encoded = 1
            elif gender_lower in ['female', 'f']:
                gender_encoded = 2
            else:
                gender_encoded = 3  # for other genders
        
        # Talent type encoding - improved version
        talent_encoded = 0
        if user.talent_type:
            # Simple encoding based on string hash, but bounded
            talent_encoded = hash(user.talent_type) % 100
        
        # Profile completeness score
        completeness = sum([
            1 if user.fullname else 0,
            1 if user.dob else 0,
            1 if user.gender else 0,
            1 if user.address else 0,
            1 if user.talent_type else 0,
            1 if user.bio else 0,
        ]) / 6.0
        
        # Account age in days - handle datetime comparison properly
        account_age = 0
        if user.created_at:
            if isinstance(user.created_at, datetime):
                account_age = (datetime.utcnow() - user.created_at).days
            else:
                # If it's already a date, convert to datetime for comparison
                created_dt = datetime.combine(user.created_at, datetime.min.time())
                account_age = (datetime.utcnow() - created_dt).days
        
        # Number of works
        works_count = len(user.works) if user.works else 0
        
        feature_vector = [
            age,
            gender_encoded,
            talent_encoded,
            completeness,
            account_age,
            works_count,
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def find_similar_users_kmeans(current_user, num_clusters=5, num_similar_users=10):
    """Find similar users using K-means clustering"""
    
    # Get all users except current user
    other_users = User.query.filter(User.id != current_user.id).all()
    
    if len(other_users) < 2:  # Need at least 2 users for clustering
        return other_users[:num_similar_users]
    
    # Prepare features for all users (including current user for clustering)
    all_users = [current_user] + other_users
    user_features = preprocess_user_features(all_users)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_features)
    
    # Determine optimal number of clusters (don't exceed number of users)
    n_clusters = min(num_clusters, len(all_users))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Find current user's cluster
    current_user_cluster = clusters[0]  # First element is current user
    
    # Find other users in the same cluster
    similar_users = []
    for i, user in enumerate(other_users):
        if clusters[i + 1] == current_user_cluster:  # +1 because first is current user
            similar_users.append(user)
    
    # If we have more similar users than needed, rank by similarity
    if len(similar_users) > num_similar_users:
        # Calculate similarity scores
        current_user_features = scaled_features[0:1]  # Current user's features
        similar_indices = [i + 1 for i, user in enumerate(other_users) 
                          if user in similar_users]
        similar_users_features = scaled_features[similar_indices]
        
        # Use cosine similarity
        similarities = cosine_similarity(current_user_features, similar_users_features)[0]
        
        # Sort by similarity score
        similar_users_with_scores = list(zip(similar_users, similarities))
        similar_users_with_scores.sort(key=lambda x: x[1], reverse=True)
        similar_users = [user for user, score in similar_users_with_scores[:num_similar_users]]
    
    return similar_users

# Simplified version that's more robust
def find_similar_users_simple(current_user, num_similar_users=10):
    """Simplified version that handles edge cases better"""
    
    other_users = User.query.filter(User.id != current_user.id).all()
    
    if not other_users:
        return []
    
    # For small number of users, just return all
    if len(other_users) <= num_similar_users:
        return other_users
    
    try:
        return find_similar_users_kmeans(current_user, num_similar_users=num_similar_users)
    except Exception as e:
        # Fallback: return random users or users with similar talent types
        print(f"K-means failed: {e}. Using fallback method.")
        return find_similar_users_fallback(current_user, other_users, num_similar_users)

def find_similar_users_fallback(current_user, other_users, num_similar_users=10):
    """Fallback method when K-means fails"""
    
    # Simple similarity based on talent type
    scored_users = []
    
    for user in other_users:
        score = 0
        
        # Talent type similarity
        if current_user.talent_type and user.talent_type:
            if current_user.talent_type.lower() == user.talent_type.lower():
                score += 3
            elif any(word in user.talent_type.lower() 
                    for word in current_user.talent_type.lower().split()):
                score += 1
        
        # Gender similarity (optional)
        if current_user.gender and user.gender:
            if current_user.gender.lower() == user.gender.lower():
                score += 1
        
        # Similar number of works
        current_works = len(current_user.works) if current_user.works else 0
        user_works = len(user.works) if user.works else 0
        works_diff = abs(current_works - user_works)
        if works_diff <= 5:  # Within 5 works
            score += 1
        
        scored_users.append((user, score))
    
    # Sort by score and return top users
    scored_users.sort(key=lambda x: x[1], reverse=True)
    return [user for user, score in scored_users[:num_similar_users]]

# Load the trained model and TF-IDF vectorizer
with open('model/model.pkl', 'rb') as f:
    model, tfidf_vectorizer = pickle.load(f)

def preprocess_text(text):
    return text

def predict_sentiment(text):
    text_processed = preprocess_text(text)
    text_vectorized = tfidf_vectorizer.transform([text_processed])
    prediction = model.predict(text_vectorized)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-to-something-random'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'

# -----------------------
# Database model
# -----------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    fullname = db.Column(db.String(120))
    dob = db.Column(db.Date)
    gender = db.Column(db.String(20))
    address = db.Column(db.String(250))
    talent_type = db.Column(db.String(120))
    profile_pic = db.Column(db.String(250))
    bio = db.Column(db.Text)
    password_hash = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    works = db.relationship("Work", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)
class Work(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    media_filename = db.Column(db.String(255), nullable=True)  # image/video
    media_type = db.Column(db.String(20), nullable=True)       # "image" or "video"

    likes = db.relationship("Like", backref="work", lazy=True)
    comments = db.relationship("Comment", backref="work", lazy=True)
    collab_requests = db.relationship("CollaborationRequest", backref="work", lazy=True)  # ✅ works now


class CollaborationRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    work_id = db.Column(db.Integer, db.ForeignKey('work.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="pending")  # pending, accepted, rejected

    sender = db.relationship("User", backref="sent_collabs")
    chat_messages = db.relationship("CollabMessage", backref="collab_request", lazy=True)


class CollabMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('collaboration_request.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship("User")

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    work_id = db.Column(db.Integer, db.ForeignKey('work.id'), nullable=False)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    work_id = db.Column(db.Integer, db.ForeignKey('work.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="comments")



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------
# Routes
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username', '').strip()
    if not username or not request.form.get('password'):
        flash('Username and password are required', 'danger')
        return redirect(url_for('index'))

    if User.query.filter_by(username=username).first():
        flash('Username already taken', 'danger')
        return redirect(url_for('index'))

    user = User(
        username=username,
        fullname=request.form.get('fullname'),
        gender=request.form.get('gender'),
        address=request.form.get('address'),
        talent_type=request.form.get('talent_type'),
        bio=request.form.get('bio')
    )

    dob = request.form.get('dob')
    if dob:
        try:
            user.dob = datetime.strptime(dob, "%Y-%m-%d").date()
        except ValueError:
            user.dob = None

    # handle profile pic
    file = request.files.get('profile_pic')
    if file and file.filename and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(f"{username}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        user.profile_pic = filename

    user.set_password(request.form.get('password'))
    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash('Account created — welcome!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('login_username')
    password = request.form.get('login_password')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        flash('Logged in successfully', 'success')
        return redirect(url_for('dashboard'))   # 👈 go to dashboard
    flash('Invalid credentials', 'danger')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    works = Work.query.filter_by(user_id=current_user.id).all()
    collab_requests = CollaborationRequest.query.join(Work).filter(
        Work.user_id == current_user.id
    ).order_by(CollaborationRequest.timestamp.desc()).all()

    received_requests = CollaborationRequest.query.join(Work).filter(
        Work.user_id == current_user.id
    ).order_by(CollaborationRequest.timestamp.desc()).all()

    sent_requests = CollaborationRequest.query.filter_by(sender_id=current_user.id).order_by(
        CollaborationRequest.timestamp.desc()
    ).all()

    # All accepted collabs I am part of (either sender or receiver)
    accepted_collabs = CollaborationRequest.query.filter(
        (CollaborationRequest.sender_id == current_user.id) |
        (Work.user_id == current_user.id)
    ).join(Work).filter(CollaborationRequest.status == "accepted").all()

    #session['uid']=current_user.id

    req = CollaborationRequest.query.get(current_user.id)
    other_users = User.query.filter(User.id != current_user.id).all()
    total_works = Work.query.count()
    total_collabs = CollaborationRequest.query.filter_by(status="accepted").count()
    total_talents = db.session.query(User.talent_type).distinct().count()
    total_categories = db.session.query(Work.media_type).distinct().count() if hasattr(Work, "category") else 0
    try:
        similar_users = find_similar_users_kmeans(current_user, num_similar_users=10)
    except Exception as e:
        print(f"Error in K-means: {e}")
        similar_users = find_similar_users_simple(current_user, num_similar_users=10)


    return render_template('dashboard.html',other_users=similar_users,req=req, received_requests=received_requests,works=works,collab_requests=collab_requests,sent_requests=sent_requests,accepted_collabs=accepted_collabs,total_works=total_works,total_collabs=total_collabs,total_talents=total_talents,total_categories=total_categories)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile/<username>')
@login_required
def profile(username):
    user = User.query.filter_by(username=username).first()
    return render_template('profile.html', user=user)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        current_user.fullname = request.form.get('fullname')
        dob = request.form.get('dob')
        if dob:
            try:
                current_user.dob = datetime.strptime(dob, "%Y-%m-%d").date()
            except ValueError:
                current_user.dob = None
        current_user.gender = request.form.get('gender')
        current_user.address = request.form.get('address')
        current_user.talent_type = request.form.get('talent_type')
        current_user.bio = request.form.get('bio')

        file = request.files.get('profile_pic')
        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(f"{current_user.username}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            current_user.profile_pic = filename

        new_password = request.form.get('password')
        if new_password:
            current_user.set_password(new_password)

        db.session.commit()
        flash('Profile updated', 'success')
        return redirect(url_for('profile', username=current_user.username))

    return render_template('edit_profile.html', user=current_user)
@app.route('/add_work', methods=['POST'])
@login_required
def add_work():
    title = request.form['title']
    description = request.form.get('description')
    media = request.files.get('media')

    filename, media_type = None, None
    if media and media.filename != '':
        filename = media.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        media.save(path)
        if filename.lower().endswith(('.png','.jpg','.jpeg','.gif')):
            media_type = "image"
        elif filename.lower().endswith(('.mp4','.mov','.avi')):
            media_type = "video"

    new_work = Work(user_id=current_user.id, title=title,
                    description=description, media_filename=filename,
                    media_type=media_type)
    db.session.add(new_work)
    db.session.commit()
    flash("Work uploaded!", "success")
    return redirect(url_for('dashboard'))

@app.route('/edit_work/<int:work_id>', methods=['POST'])
@login_required
def edit_work(work_id):
    work = Work.query.get_or_404(work_id)
    if work.user_id != current_user.id:
        flash("Not authorized", "danger")
        return redirect(url_for('dashboard'))

    work.title = request.form['title']
    work.description = request.form.get('description')

    media = request.files.get('media')
    if media and media.filename != '':
        filename = media.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        media.save(path)
        work.media_filename = filename
        if filename.lower().endswith(('.png','.jpg','.jpeg','.gif')):
            work.media_type = "image"
        elif filename.lower().endswith(('.mp4','.mov','.avi')):
            work.media_type = "video"

    db.session.commit()
    flash("Work updated!", "success")
    return redirect(url_for('dashboard'))

@app.route('/like/<int:work_id>')
@login_required
def like_work(work_id):
    work = Work.query.get_or_404(work_id)
    existing = Like.query.filter_by(user_id=current_user.id, work_id=work_id).first()
    if not existing:
        like = Like(user_id=current_user.id, work_id=work_id)
        db.session.add(like)
        db.session.commit()
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/collab_request/<int:work_id>')
@login_required
def collab_request(work_id):
    # TODO: handle collab request logic
    flash("Collaboration request sent!", "info")
    return redirect(url_for('dashboard'))

@app.route('/comment/<int:work_id>', methods=['POST'])
@login_required
def add_comment(work_id):
    work = Work.query.get_or_404(work_id)
    text = request.form['text']

    if text.strip():
        comment = Comment(user_id=current_user.id, work_id=work.id, text=text)
        db.session.add(comment)
        db.session.commit()
        flash("Comment added!", "success")
    else:
        flash("Comment cannot be empty.", "danger")

    return redirect(request.referrer or url_for('dashboard'))

@app.route('/collab/<int:work_id>', methods=['POST'])
@login_required
def send_collab_request(work_id):
    work = Work.query.get_or_404(work_id)
    message = request.form['message']

    if work.user_id == current_user.id:
        flash("You cannot request collaboration on your own work!", "danger")
        return redirect(url_for('dashboard'))

    req = CollaborationRequest(
        sender_id=current_user.id,
        work_id=work.id,
        message=message
    )
    db.session.add(req)
    db.session.commit()

    flash("Collaboration request sent!", "success")
    return redirect(request.referrer or url_for('dashboard'))


@app.route('/feed')
@login_required
def feed():
    search_query = request.args.get('q', '').strip()  # get query from URL parameter

    if search_query:
        # Filter works by title, description, or user fullname
        works = Work.query.join(User).filter(
            or_(
                Work.title.ilike(f"%{search_query}%"),
                Work.description.ilike(f"%{search_query}%"),
                User.fullname.ilike(f"%{search_query}%")
            )
        ).order_by(Work.id.desc()).all()
    else:
        works = Work.query.order_by(Work.id.desc()).all()  # show newest first

    req = CollaborationRequest.query.get(current_user.id)
    return render_template('feed.html', works=works, req=req, search_query=search_query)

@app.route('/collab/accept/<int:req_id>')
@login_required
def accept_collab(req_id):
    req = CollaborationRequest.query.get_or_404(req_id)
    if req.work.user_id != current_user.id:
        flash("Not authorized", "danger")
        return redirect(url_for('dashboard'))
    req.status = "accepted"
    db.session.commit()
    flash("Collaboration request accepted!", "success")
    return redirect(url_for('dashboard'))

@app.route('/collab/reject/<int:req_id>')
@login_required
def reject_collab(req_id):
    req = CollaborationRequest.query.get_or_404(req_id)
    if req.work.user_id != current_user.id:
        flash("Not authorized", "danger")
        return redirect(url_for('dashboard'))
    req.status = "rejected"
    db.session.commit()
    flash("Collaboration request rejected!", "danger")
    return redirect(url_for('dashboard'))

@app.route('/collab/message/<int:req_id>', methods=['POST'])
@login_required
def send_collab_message(req_id):
    req = CollaborationRequest.query.get_or_404(req_id)

    # Ensure current user is either the sender or the work owner
    if not (req.sender_id == current_user.id or req.work.user_id == current_user.id):
        flash("Not authorized to chat on this request", "danger")
        return redirect(url_for('dashboard'))

    if req.status != "accepted":
        flash("You can only chat on accepted requests!", "danger")
        return redirect(url_for('dashboard'))

    text = request.form['text']
    if text.strip():
        msg = CollabMessage(request_id=req.id, sender_id=current_user.id, text=text)
        db.session.add(msg)
        db.session.commit()
        flash("Message sent!", "success")

    return redirect(url_for('dashboard'))


@app.route('/collab/chat/<int:req_id>')
@login_required
def collab_chat(req_id):
    req = CollaborationRequest.query.get(req_id)

    # Ensure the logged-in user is either the sender or the work owner
    if not (req.sender_id == current_user.id or req.work.user_id == current_user.id):
        flash("Not authorized to view this chat", "danger")
        return redirect(url_for('dashboard'))

    if req.status != "accepted":
        flash("Chat only available for accepted collaborations", "danger")
        return redirect(url_for('dashboard'))

    messages = CollabMessage.query.filter_by(request_id=req.id).order_by(CollabMessage.timestamp.asc()).all()

    return render_template("collab_chat.html", req=req, messages=messages)

@app.route("/collaborations")
@login_required
def collaborations():
    # Accepted collaborations where user is either sender or work owner
    collabs = (
        CollaborationRequest.query
        .join(Work)
        .filter(
            (CollaborationRequest.status == "accepted") &
            ((CollaborationRequest.sender_id == current_user.id) | (Work.user_id == current_user.id))
        )
        .order_by(CollaborationRequest.timestamp.desc())
        .all()
    )
    #session['uid']=current_user.id

    req = CollaborationRequest.query.get(current_user.id)

    return render_template("collaborations.html", collabs=collabs,req=req)

@app.route("/user/<int:user_id>")
@login_required
def user_dashboard(user_id):
    user = User.query.get(user_id)

    # That user's works
    works = Work.query.filter_by(user_id=user.id).all()

    return render_template("user_dashboard.html", profile_user=user, works=works)



# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True,host='0.0.0.0',port="5050")
