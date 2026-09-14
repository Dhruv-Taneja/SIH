const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const crypto = require('crypto');
const { PythonShell } = require('python-shell');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (req, file, cb) => {
    const allowedExt = /jpeg|jpg|png|pdf/;
    const ext = allowedExt.test(path.extname(file.originalname).toLowerCase());
    // allow octet-stream for camera captures renamed .png, plus pdf
    const mimeOk = /image|pdf|octet-stream/.test(file.mimetype);
    if (ext && mimeOk) cb(null, true);
    else cb(new Error('Only images (jpg, jpeg, png) and PDFs are allowed'));
  }
});

const ALLOWED_TYPES = ['auto', 'aadhaar', 'pan', 'passport', 'driving_licence', 'marksheet', 'certificate', 'generic'];

// ---------------------------------------------------------------------------
// Organization auth (local demo-grade: scrypt hashes + opaque session tokens,
// JSON file store, HttpOnly cookies. No external identity service —
// documents and credentials never leave this machine.)
// ---------------------------------------------------------------------------
const DATA_DIR = path.join(__dirname, 'data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');
const SESSIONS_FILE = path.join(DATA_DIR, 'sessions.json');
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

function readJson(file, fallback) {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(file)) return fallback;
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch { return fallback; }
}
function writeJson(file, obj) {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(file, JSON.stringify(obj, null, 2));
}
function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return `scrypt$${salt}$${hash}`;
}
function checkPassword(password, stored) {
  try {
    const [, salt, hash] = String(stored).split('$');
    const test = crypto.scryptSync(password, salt, 64).toString('hex');
    return crypto.timingSafeEqual(Buffer.from(test, 'hex'), Buffer.from(hash, 'hex'));
  } catch { return false; }
}
function parseCookies(req) {
  const out = {};
  (req.headers.cookie || '').split(';').forEach(p => {
    const i = p.indexOf('=');
    if (i > 0) out[p.slice(0, i).trim()] = decodeURIComponent(p.slice(i + 1).trim());
  });
  return out;
}
function setSession(res, email) {
  const sessions = readJson(SESSIONS_FILE, {});
  const token = crypto.randomBytes(32).toString('hex');
  sessions[token] = { email, created: Date.now() };
  writeJson(SESSIONS_FILE, sessions);
  res.setHeader('Set-Cookie',
    `dv_session=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${SESSION_TTL_MS / 1000}`);
}
function clearSession(req, res) {
  const token = parseCookies(req).dv_session;
  if (token) {
    const sessions = readJson(SESSIONS_FILE, {});
    delete sessions[token];
    writeJson(SESSIONS_FILE, sessions);
  }
  res.setHeader('Set-Cookie', 'dv_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0');
}
function getSessionOrg(req) {
  const token = parseCookies(req).dv_session;
  if (!token) return null;
  const sessions = readJson(SESSIONS_FILE, {});
  const s = sessions[token];
  if (!s || Date.now() - s.created > SESSION_TTL_MS) return null;
  const users = readJson(USERS_FILE, {});
  const u = users[String(s.email).toLowerCase()];
  return u ? { orgName: u.orgName, email: u.email, created: u.created } : null;
}
function requireAuth(req, res, next) {
  const org = getSessionOrg(req);
  if (!org) return res.status(401).json({ error: 'Sign in required', auth: false });
  req.org = org;
  next();
}
// Verification is open to guests too — organizations may sign in, but it is
// optional. Attaches the org when present (for logging) without rejecting.
function optionalAuth(req, res, next) {
  req.org = getSessionOrg(req);
  next();
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// POST /api/auth/signup { orgName, email, password }
app.post('/api/auth/signup', (req, res) => {
  const orgName = String(req.body.orgName || '').trim();
  const email = String(req.body.email || '').trim().toLowerCase();
  const password = String(req.body.password || '');
  if (orgName.length < 2) return res.status(400).json({ error: 'Organization name required' });
  if (!EMAIL_RE.test(email)) return res.status(400).json({ error: 'Valid email required' });
  if (password.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters' });
  const users = readJson(USERS_FILE, {});
  if (users[email]) return res.status(409).json({ error: 'An account with this email already exists' });
  users[email] = { orgName, email, pass: hashPassword(password), created: new Date().toISOString() };
  writeJson(USERS_FILE, users);
  setSession(res, email);
  res.json({ ok: true, org: { orgName, email } });
});

// POST /api/auth/login { email, password }
app.post('/api/auth/login', (req, res) => {
  const email = String(req.body.email || '').trim().toLowerCase();
  const password = String(req.body.password || '');
  const users = readJson(USERS_FILE, {});
  const u = users[email];
  if (!u || !checkPassword(password, u.pass)) {
    return res.status(401).json({ error: 'Incorrect email or password' });
  }
  setSession(res, email);
  res.json({ ok: true, org: { orgName: u.orgName, email: u.email } });
});

// POST /api/auth/logout
app.post('/api/auth/logout', (req, res) => {
  clearSession(req, res);
  res.json({ ok: true });
});

// GET /api/auth/me → current org or 401
app.get('/api/auth/me', (req, res) => {
  const org = getSessionOrg(req);
  if (!org) return res.status(401).json({ auth: false });
  res.json({ auth: true, org });
});

// POST /api/upload — accepts `document` file + `documentType`, returns verification JSON
app.post('/api/upload', optionalAuth, upload.single('document'), async (req, res) => {
  let filePath = null;
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    filePath = req.file.path;
    let documentType = (req.body.documentType || 'generic').toLowerCase();
    if (!ALLOWED_TYPES.includes(documentType)) documentType = 'generic';

    console.log(`[upload] ${req.org ? req.org.orgName : 'guest'} → ${req.file.originalname} → ${documentType}`);
    const result = await processDocument(filePath, documentType);
    res.json(result);
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message || 'Verification failed' });
  } finally {
    if (filePath) fs.unlink(filePath, (err) => {
      if (err) console.error('Failed to delete temp file:', err);
    });
  }
});

// JS → Python bridge (python-shell v5 = Promise API)
async function processDocument(filePath, documentType) {
  const options = {
    mode: 'text',
    pythonPath: 'python3',
    pythonOptions: ['-u'],
    scriptPath: path.join(__dirname, 'ml'),
    args: [filePath, documentType]
  };
  try {
    const results = await PythonShell.run('verify_document.py', options);
    if (!results || !results.length) throw new Error('Empty response from ML service');
    const lastLine = results[results.length - 1];
    return JSON.parse(lastLine);
  } catch (err) {
    console.error('Python ML error:', err.message);
    throw new Error(
      'ML service failed. Did you run `pip install -r ml/requirements.txt`? Details: ' + err.message
    );
  }
}

app.get('/api/health', async (req, res) => {
  // Report which OCR backends are actually usable, so the UI can warn
  // before the user uploads a photo (digital PDFs always work via pypdf).
  let engines = null;
  try {
    const out = await PythonShell.run('check_engines.py', {
      mode: 'text',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, 'ml')
    });
    engines = JSON.parse(out[out.length - 1]);
  } catch (e) {
    engines = { error: 'engine check failed: ' + e.message };
  }
  res.json({ status: 'ok', service: 'document-verification', engines });
});

app.get('/api/doc-types', (req, res) => {
  res.json(ALLOWED_TYPES.map(t => ({ value: t })));
});

// Multer / validation errors → clean JSON
app.use((err, req, res, next) => {
  if (err) return res.status(400).json({ error: err.message });
  next();
});

app.listen(PORT, () => {
  console.log(`DocuVerify running on http://localhost:${PORT}`);
  console.log(`Make sure Python deps are installed: pip install -r ml/requirements.txt`);
});
