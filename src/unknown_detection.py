"""Module 9: unknown / unsupported-case detection.

Three transparent mechanisms, each calibrated on *validation* data only:
  1. FeatureOOD     - distance of a feature vector to its k nearest training vectors. The threshold is a
                      high percentile of distances computed for held-out (validation) samples.
  2. uncertain band - a probability interval around the decision threshold inside which validation
                      predictions were unreliable (see scripts/train_models.py).
  3. scope checks   - rule-based checks that the input is of a kind the method supports (duration, face
                      present, text length, vocabulary coverage ...).

A low-confidence or out-of-scope input becomes "Unknown / Review". It is NOT treated as fraud.
"""
import numpy as np


class FeatureOOD:
    """k-NN distance novelty detector on standardised features."""

    def __init__(self, k=3):
        self.k = k
        self.mu = self.sd = self.ref = None
        self.threshold = None
        self.percentile = None

    def fit(self, X_train):
        X = np.asarray(X_train, float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        self.ref = (X - self.mu) / self.sd
        return self

    def distance(self, x, exclude_self=False):
        z = (np.asarray(x, float).reshape(1, -1) - self.mu) / self.sd
        d = np.sort(np.linalg.norm(self.ref - z, axis=1))
        if exclude_self:
            d = d[1:]
        return float(np.mean(d[:self.k]))

    def calibrate(self, X_val, percentile=95):
        """Set the threshold from distances of validation samples that were NOT used in fit()."""
        d = [self.distance(x) for x in np.asarray(X_val, float)]
        self.threshold = float(np.percentile(d, percentile))
        self.percentile = percentile
        return self

    def check(self, x):
        d = self.distance(x)
        return {"distance": round(d, 3), "threshold": round(self.threshold, 3),
                "out_of_distribution": bool(d > self.threshold)}


def overlap_band(y_true, proba, threshold, q=0.02):
    """Uncertain score interval = region where the two classes' held-out scores overlap.

    low  = q-quantile of the POSITIVE-class scores   (below it almost no positive example was ever seen)
    high = (1-q)-quantile of the NEGATIVE-class scores (above it almost no negative example was ever seen)
    A new score inside [low, high] looks like both classes -> "Unknown / Review". Scores outside are
    only "confident" in the sense that the opposite class was rare there on held-out data; this is NOT a
    calibrated probability. Returns the interval plus honest diagnostics (fraction inside, accuracy inside
    vs outside) so the effect of the band can be inspected.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    low = float(np.quantile(proba[y_true == 1], q))
    high = float(np.quantile(proba[y_true == 0], 1 - q))
    low, high = min(low, threshold), max(high, threshold)
    inside = (proba >= low) & (proba <= high)
    correct = (proba >= threshold).astype(int) == y_true
    return {"low": low, "high": high, "q": q, "n": int(len(y_true)), "fraction_inside": float(inside.mean()),
            "accuracy_outside": float(correct[~inside].mean()) if (~inside).any() else None,
            "accuracy_inside": float(correct[inside].mean()) if inside.any() else None,
            "errors_total": int((~correct).sum()), "errors_inside": int((~correct & inside).sum())}


def uncertain_band(y_true, proba, threshold, target_acc=0.95, max_half_width=0.3):
    """(Used by the video model.) Smallest symmetric band [t-w, t+w] such that predictions OUTSIDE it reach `target_acc` on validation.

    If even the widest allowed band cannot reach the target, the widest band is returned (flagged).
    NOTE: if overall accuracy already exceeds `target_acc` this returns a zero-width band (the criterion is met
    trivially), which is why the text model now uses `overlap_band` instead.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    for w in np.arange(0.0, max_half_width + 1e-9, 0.02):
        out = np.abs(proba - threshold) > w
        if out.sum() < max(10, 0.2 * len(y_true)):
            break
        acc = np.mean((proba[out] >= threshold).astype(int) == y_true[out])
        if acc >= target_acc:
            return float(max(threshold - w, 0.0)), float(min(threshold + w, 1.0)), float(w)
    return float(max(threshold - max_half_width, 0.0)), float(min(threshold + max_half_width, 1.0)), max_half_width


# ------------------------------------------------------------------- scope (rule) checks ----
def video_scope_issues(meta, faces_found, n_frames_used):
    issues = []
    if n_frames_used < 6:
        issues.append(f"Only {n_frames_used} frames could be analysed (too few for a stable estimate).")
    if faces_found == 0:
        issues.append("No frontal face was detected in any sampled frame; the detector was trained on "
                      "face-centred talking-head clips.")
    elif faces_found / max(n_frames_used, 1) < 0.5:
        issues.append(f"A face was found in only {faces_found}/{n_frames_used} sampled frames.")
    if meta.get("duration_s") and meta["duration_s"] < 1.0:
        issues.append("Clip is shorter than 1 second.")
    if min(meta.get("width", 0), meta.get("height", 0)) < 200:
        issues.append("Very low resolution (<200 px): forensic statistics are unreliable.")
    return issues


def audio_scope_issues(info):
    issues = []
    if info["analysed_duration_s"] < 1.0:
        issues.append("Audio is shorter than 1 second.")
    if info["silence_ratio"] > 0.85:
        issues.append(f"{info['silence_ratio']:.0%} of the audio is near-silent.")
    if info.get("original_sample_rate") and info["original_sample_rate"] < 8000:
        issues.append("Original sample rate is below 8 kHz.")
    return issues
