from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import sqlite3
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename

app = Flask(__name__)
IMAGE_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpeg', 'jpg'}
app.secret_key = 'your_secret_key'

# Yardımcı fonksiyonlar
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        result TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')
    conn.commit()
    conn.close()

init_db()  # Veritabanı yapılandırmasını başlat

def create_admin(email, password):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (ad, soyad, email, password, role) VALUES (?, ?, ?, ?, ?)', 
                     ('Admin', 'User', email, password, 'admin'))
        conn.commit()
        print(f"{email} isimli kullanıcı başarıyla yönetici olarak tanımlandı.")
    except sqlite3.IntegrityError:
        print("Bu e-posta adresi zaten bir kullanıcıya aittir.")
    finally:
        conn.close()

# Admin kullanıcısını oluştur
create_admin('ydnceren00@gmail.com', '123')

# Flask-Login yapılandırması
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'giris'  # Eğer giriş yapılmadıysa, yönlendirilecek sayfa

# Kullanıcı sınıfı
class User(UserMixin):
    def __init__(self, user_id, role):
        self.id = user_id
        self.role = role

    @property
    def is_admin(self):
        return self.role == 'admin'

# Kullanıcı yükleme fonksiyonu
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user['id'], user['role'])
    return None

# Ana sayfa
@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Dosya seçilmedi!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Dosya seçilmedi!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Dosya formatına göre sonuç belirleme
                file_format = filename.rsplit('.', 1)[1].lower()
                if file_format == 'png':
                    result = 'Orijinal Resim'
                elif file_format in ['jpeg', 'jpg']:
                    result = 'Manipüle Edilmiş Resim'

                conn = get_db_connection()
                conn.execute('INSERT INTO uploads (user_id, filename, result) VALUES (?, ?, ?)', (current_user.id, filename, result))
                conn.commit()
                conn.close()
                
                flash('Dosya başarıyla yüklendi!', 'success')
                return redirect(url_for('rapor'))
            
            except Exception as e:
                print(f"Dosya kaydetme hatası: {e}")
                flash('Dosya kaydetme hatası!', 'danger')
                return redirect(request.url)
    
    return render_template('upload.html')

# Raporlama Sayfası
@app.route('/rapor')
@login_required
def rapor():
    conn = get_db_connection()
    uploads = conn.execute('SELECT * FROM uploads WHERE user_id = ?', (current_user.id,)).fetchall()
    conn.close()
    return render_template('rapor.html', uploads=uploads)

# Yardım sayfası
@app.route('/yardim')
def help():
    return render_template('yardim.html')

# Kayıt işlemi
@app.route('/kayit', methods=['GET', 'POST'])
def kayit():
    if request.method == 'POST':
        ad = request.form['ad']
        soyad = request.form['soyad']
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (ad, soyad, email, password, role) VALUES (?, ?, ?, ?, ?)', 
                         (ad, soyad, email, password, 'user'))
            conn.commit()
            flash('Başarıyla kayıt oldunuz!', 'success')
            return redirect(url_for('giris'))  # Kayıt başarılıysa giriş sayfasına yönlendir
        except sqlite3.IntegrityError:
            flash('Bu e-posta adresi zaten kayıtlı.', 'danger')
        finally:
            conn.close()
    return render_template('kayit.html')

# Giriş işlemi
@app.route('/giris', methods=['GET', 'POST'])
def giris():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()
        if user:
            user_obj = User(user['id'], user['role'])
            login_user(user_obj)
            session['ad'] = user['ad']
            session['soyad'] = user['soyad']
            return redirect(url_for('upload_file'))  # Giriş başarılıysa ana sayfaya yönlendir
        else:
            flash('Geçersiz e-posta veya şifre', 'danger')
    return render_template('giris.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        
        file_format = filename.rsplit('.', 1)[1].lower()
        if file_format == 'png':
            message = 'Orijinal Resim'
        elif file_format in ['jpeg', 'jpg']:
            message = 'Manipüle Edilmiş Resim'

        return render_template('result.html', message=message)

    return redirect(url_for('upload_file'))



# Çıkış işlemi
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('ad', None)
    session.pop('soyad', None)
    flash('Çıkış yapıldı', 'success')
    return redirect(url_for('giris'))  # Çıkış işlemi yapıldıktan sonra giriş sayfasına yönlendir

if __name__ == '__main__':
    app.run(debug=False)
