// DocuVerify organization auth — sign in / sign up / session gate.
// Owns ONLY the #auth panel + verify gating; the upload flow in app.js
// is untouched except for the 401 handoff it already dispatches.
(function () {
  async function me() {
    try {
      const r = await fetch('/api/auth/me');
      if (!r.ok) return null;
      return (await r.json()).org;
    } catch (e) { return null; }
  }

  function showSignedIn(org) {
    document.getElementById('authSignedOut').classList.add('hidden');
    document.getElementById('authSignedIn').classList.remove('hidden');
    document.getElementById('welcomeMsg').textContent =
      `Welcome, ${org.orgName} 👋`;
    const bar = document.getElementById('accountBar');
    bar.textContent = `🏢 ${org.orgName}`;
    bar.classList.remove('hidden');
    const v = document.getElementById('verify');
    if (v) v.classList.remove('locked');
  }

  function showSignedOut() {
    document.getElementById('authSignedOut').classList.remove('hidden');
    document.getElementById('authSignedIn').classList.add('hidden');
    document.getElementById('accountBar').classList.add('hidden');
    // Verification stays open for guests — no locking.
    const v = document.getElementById('verify');
    if (v) v.classList.remove('locked');
  }

  function status(msg, cls) {
    const el = document.getElementById('authStatus');
    el.textContent = msg;
    el.className = 'status ' + (cls || '');
  }

  async function refresh() {
    const org = await me();
    if (org) showSignedIn(org); else showSignedOut();
  }

  document.addEventListener('DOMContentLoaded', () => {
    const tabIn = document.getElementById('tabSignin');
    const tabUp = document.getElementById('tabSignup');
    const formIn = document.getElementById('signinForm');
    const formUp = document.getElementById('signupForm');
    tabIn.addEventListener('click', () => {
      tabIn.classList.add('active'); tabUp.classList.remove('active');
      formIn.classList.remove('hidden'); formUp.classList.add('hidden');
      status('');
    });
    tabUp.addEventListener('click', () => {
      tabUp.classList.add('active'); tabIn.classList.remove('active');
      formUp.classList.remove('hidden'); formIn.classList.add('hidden');
      status('');
    });

    formIn.addEventListener('submit', async (e) => {
      e.preventDefault();
      status('Signing in…', 'loading');
      try {
        const r = await fetch('/api/auth/login', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: document.getElementById('signinEmail').value,
            password: document.getElementById('signinPass').value
          })
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Sign in failed');
        status('');
        showSignedIn(d.org);
        document.getElementById('verify').scrollIntoView({ behavior: 'smooth' });
      } catch (err) { status('Error: ' + err.message, 'error'); }
    });

    formUp.addEventListener('submit', async (e) => {
      e.preventDefault();
      status('Creating account…', 'loading');
      try {
        const r = await fetch('/api/auth/signup', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            orgName: document.getElementById('signupOrg').value,
            email: document.getElementById('signupEmail').value,
            password: document.getElementById('signupPass').value
          })
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Sign up failed');
        status('');
        showSignedIn(d.org);
        document.getElementById('verify').scrollIntoView({ behavior: 'smooth' });
      } catch (err) { status('Error: ' + err.message, 'error'); }
    });

    document.getElementById('signoutBtn').addEventListener('click', async () => {
      await fetch('/api/auth/logout', { method: 'POST' });
      showSignedOut();
      document.getElementById('auth').scrollIntoView({ behavior: 'smooth' });
    });

    // Fired by app.js when /api/upload returns 401 (expired session).
    document.addEventListener('docuverify:auth-required', () => {
      showSignedOut();
      status('Session expired — please sign in again.', 'error');
      document.getElementById('auth').scrollIntoView({ behavior: 'smooth' });
    });

    refresh();
  });
})();
