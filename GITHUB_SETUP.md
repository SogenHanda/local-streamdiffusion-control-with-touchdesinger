# GitHub publishing

The repository is prepared to publish the Python backend and TouchDesigner
project together. Large local files are excluded by `.gitignore`.

Private repository:

```text
SogenHanda/local-streamdiffusion-control-with-touchdesinger
```

After creating the empty private repository on GitHub, run from this directory:

```powershell
git remote add origin https://github.com/SogenHanda/local-streamdiffusion-control-with-touchdesinger.git
git commit -m "Generalize and organize Local StreamDiffusion projects"
git push -u origin main
```

The previous project remote is preserved as `legacy-origin`; there is currently
no `origin`. Inspect remotes before changing anything:

```powershell
git remote -v
```

If `origin` is added with the wrong URL, replace only that remote:

```powershell
git remote set-url origin https://github.com/SogenHanda/local-streamdiffusion-control-with-touchdesinger.git
```

Do not force-push over an unrelated repository.
