# Mario — Progress Log 

## Branch: Mario/Expanded-Models

---

## April 22, 2026

### Done
- - Migrated from Greene (Adarsh's cluster) to NYU Cloud Bursting (via OOD)
  - Slurm account: ds_ga_1003-2026sp
  - 300 GPU-hour budget (course allocation)
  - Spot instance architecture; all Slurm scripts require --requeue
- Conda env working: /scratch/mty236/conda_envs/musicvae (Python 3.9)
- Installed requirements.txt + tqdm: librosa 0.11.0, torch 2.8.0+cu128, numpy 1.23.5
- NSynth train set downloaded and extracted: 289,205 audio files total, 65,474 bass files
- MoisesDB downloaded (88.82 GB) and extracted: 240 tracks, 275 bass files
- SLURM scripts written for dataset downloads + unzip + preprocessing
- Preprocessing script written (scripts/preprocess_pitch_shifts.py)
  - Uses real librosa.effects.pitch_shift in waveform space
  - Generates 5 discrete shifts per source file: {-6, -3, 0, +3, +6} semitones
  - For MoisesDB: 20 random crops per stem × 5 shifts
  - Parallelized via multiprocessing.Pool
- Preprocessing script tested on single NSynth file: 5 npy files produced, shape (128, 126), dtype float32
- Full preprocessing not yet submitted — pending code review before committing compute
- Mel Spectrograms seem unhelpful for Latent Representation Analysis
- Full Metrics Re-evaluation is required

- ### Bugs identified in original pipeline (pending fix)
- nsynth_preprocessed.py and moisesdb_preprocessed.py use np.roll on mel axis instead of real pitch-shift in waveform space. This means the "symmetry constraint" in the paper isn't trained on actual pitch-shift equivariance; it's trained on mel-axis rotations, which have no acoustic meaning.
- vae.py loss implements only style invariance (Lsym = MSE(zs(x), zs(Tg x))). Content equivariance (zc(Tg x) = ρ(g)zc(x)) is missing from both the loss and the forward pass — the model doesn't receive the shift amount g as input.
- dci.py Informativeness uses training-set accuracy (clf.fit(z,f); clf.score(z,f) on same data). This is memorization, not informativeness. Explains why DCI-I = 0.959 across all three models.
- srr.py computes ratio on log-mel values directly, not on linear power. Units are meaningless; didn't catch the April 20 decoder collapse until weeks later.
- MIG has no random_state set for mutual_info_classif, making runs nondeterministic.

### Issues encountered and resolved
- First MoisesDB unzip (job 171066) silently failed: only 115 of 2585 wav files extracted despite exit code 0:0. Cause: `unzip -q` flag suppressed warnings. Resolved: re-ran without -q and captured output, which revealed that unzip was aborting due to zip-bomb detection (MoisesDB's wav-heavy content trips the compression-ratio heuristic).
- Second unzip (job 171116) resolved by setting UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE before unzip. Successfully extracted all 2585 wav files.
- Initial pip install landed 6.9 GB of packages in ~/.local/ instead of the conda env because conda wasn't properly activated before pip. Cleaned up and re-installed after sourcing /share/apps/pyenv/py3.9/etc/profile.d/conda.sh and activating explicitly. Verified correct site-packages path via `python -c "import site; print(site.getsitepackages())"` before second attempt.

### Directory layout on HPC
- Repo: /scratch/mty236/ML_Project/Disentangled-Representations-Learning-for-Music-Processing
- Data: /scratch/mty236/ML_Project/data/{nsynth_raw, moisesdb_raw, nsynth_shifted, moisesdb_shifted}
- Logs: /scratch/mty236/ML_Project/logs/
- Scripts: /scratch/mty236/ML_Project/scripts/ (committed to git)
- Conda env: /scratch/mty236/conda_envs/musicvae

### Next steps
- Review preprocessing script design choices before committing compute (5 vs 13 shift values, 20 crops per MoisesDB stem, filename shift encoding)
- Draft fixes for dci.py (cross-validated informativeness), srr.py (linear power units), mig.py (random_state)



