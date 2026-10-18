# CONTRIBUTING.md — How to Contribute to VideoAnomalyDetection

Thank you for contributing! Follow these guidelines to keep our codebase clean and professional.

---

## Branch Strategy (Git Flow)

```
main          ← production-ready, protected
develop       ← integration branch (merge PRs here first)
feature/xxx   ← new features (branched from develop)
bugfix/xxx    ← bug fixes (branched from develop)
hotfix/xxx    ← urgent production fixes (branched from main)
experiment/xxx← ML experiments (not merged to main, archived)
```

## Step-by-Step Workflow

### 1. Clone the repo (first time)
```bash
git clone https://github.com/<your-org>/VideoAnomalyDetection.git
cd VideoAnomalyDetection
```

### 2. Set up your environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Create your feature branch
```bash
git checkout develop            # always branch from develop
git pull origin develop         # get latest
git checkout -b feature/your-feature-name
```

### 4. Work, commit often
```bash
git add <files>
git commit -m "feat: short description of what you did"
```

**Commit message prefixes:**
| Prefix | When to use |
|--------|------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring (no new feature) |
| `test:` | Adding or fixing tests |
| `chore:` | Build process, tooling |
| `exp:` | ML experiment |

### 5. Push and open a Pull Request
```bash
git push origin feature/your-feature-name
```
Then go to GitHub → **New Pull Request** → base: `develop` ← compare: `feature/your-feature-name`

### 6. Code Review
- At least **1 teammate must approve** before merging
- Resolve all comments
- Do NOT merge your own PR without review

---

## Rules
- ❌ Never push directly to `main` or `develop`
- ❌ Never commit secrets, API keys, or large data files
- ✅ Always run tests before pushing: `pytest tests/`
- ✅ Keep PRs small and focused (one feature/fix per PR)
