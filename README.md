# 👁️ Computer Vision Tutorial — Zero to Practitioner

A fully interactive, self-contained CV tutorial built with React + Vite.  
Hosted free on GitHub Pages. No backend. No database. No API keys.

**Live site:** `https://YOUR-USERNAME.github.io/cv-tutorial/`

---

## 📦 What's Inside

| Tab | Content |
|-----|---------|
| 💻 Challenges | 12 LeetCode-style CV coding problems with solutions |
| 🏗️ Projects | 6 real-world projects (PlantVillage, ChestX-ray14, Cityscapes…) |
| 🚀 Deploy & Host | 5 deployment strategies with copy-paste code |
| 🧪 Quiz | 20-question interactive quiz |

---

## 🚀 Deploy to GitHub Pages in 5 Steps

### Step 1 — Create a GitHub repo

Go to github.com → **New repository**  
Name it: `cv-tutorial` (or anything you like)  
Set to **Public** (required for free Pages hosting)

### Step 2 — Edit the base path

Open `vite.config.js` and change one line:

```js
base: '/cv-tutorial/',   // ← replace with your actual repo name
```

### Step 3 — Enable GitHub Pages

In your repo: **Settings → Pages → Source → GitHub Actions**

### Step 4 — Push your code

```bash
# First time
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/cv-tutorial.git
git push -u origin main
```

GitHub Actions will automatically build and deploy.  
Wait ~2 minutes, then visit:  
`https://YOUR-USERNAME.github.io/cv-tutorial/`

### Step 5 — Update anytime

```bash
# After any change to App.jsx:
git add .
git commit -m "Update tutorial"
git push
# Auto-deploys in ~90 seconds
```

---

## 🛠️ Local Development

```bash
npm install       # install dependencies (only once)
npm run dev       # start dev server at http://localhost:5173
npm run build     # build for production → dist/
npm run preview   # preview the production build locally
```

---

## 📁 Project Structure

```
cv-tutorial/
├── .github/
│   └── workflows/
│       └── deploy.yml       ← GitHub Actions auto-deploy
├── src/
│   ├── main.jsx             ← React entry point
│   └── App.jsx              ← The entire tutorial (2000 lines)
├── index.html               ← HTML shell
├── vite.config.js           ← base path for GitHub Pages
├── package.json
└── .gitignore
```

---

## ✏️ Customising

All content lives in `src/App.jsx`:

| What to change | Where in App.jsx |
|---------------|-----------------|
| Add a challenge | `CHALLENGES` array (top of file) |
| Add a project | `PROJECTS` array |
| Add quiz questions | `QUIZ_BANK` array |
| Change colours | `P` palette object |
| Add a hosting option | `HOSTING` array |

---

## 🌐 Alternative Free Hosts

| Platform | Command | URL format |
|----------|---------|------------|
| **Netlify** | drag & drop `dist/` folder | `random-name.netlify.app` |
| **Vercel** | `npx vercel` | `project.vercel.app` |
| **Cloudflare Pages** | connect GitHub repo | `project.pages.dev` |
| **GitHub Pages** | push to main | `username.github.io/repo` |

For Netlify/Vercel/Cloudflare, set `base: '/'` in `vite.config.js` (no subdirectory needed).

---

## 📄 License

MIT — use freely, modify, and share.
