"""M4: layout choice, stable face-anchored crops, and the ffmpeg filter for each layout.

Single-speaker crops use one representative face position for the whole clip. A fixed crop avoids
camera shake caused by small face-detector changes between sampled frames.
"""
import math

import numpy as np

from .faces import motion_centroid, sample_frames, shot_starts

OUT_W, OUT_H = 1080, 1920
LAYOUTS = ("auto", "single", "two_speakers", "gameplay_facecam", "no_face", "fit_blur")
# Preset names: "gameplay_facecam" = gaming (facecam on top, game below, blurred background);
# "fit_blur" / "no_face" = normal video (16:9 centred, blurred above and below).


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def one_euro(xs, dt, min_cutoff=0.8, beta=1.5, resets=()):
    """Adaptive low-pass: smooth when still, responsive when moving. `resets` = shot-change indices."""
    def alpha(cutoff):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
    out, x_hat, dx_hat = [], xs[0], 0.0
    for i, x in enumerate(xs):
        if i == 0 or i in resets:
            x_hat, dx_hat = x, 0.0
        else:
            dx = (x - out[-1]) / dt
            dx_hat += alpha(1.0) * (dx - dx_hat)
            x_hat += alpha(min_cutoff + beta * abs(dx_hat)) * (x - x_hat)
        out.append(x_hat)
    return out


def limit_pan(xs, dt, max_step, resets=()):
    out = [xs[0]]
    for i in range(1, len(xs)):
        step = xs[i] - out[-1]
        out.append(xs[i] if i in resets else out[-1] + max(-max_step * dt, min(max_step * dt, step)))
    return out


def _interp(vals):
    """Fill missing (None) values: linear between detections, hold at the ends."""
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        return None
    return list(np.interp(range(len(vals)), idx, [vals[i] for i in idx]))


def choose_layout(faces) -> tuple[str, dict]:
    """faces: per sample, a list of (x0,y0,x1,y1) boxes in 0-1 coordinates."""
    n = len(faces)
    with_face = [f for f in faces if f]
    if n == 0 or len(with_face) / n < 0.15:
        return "no_face", {}
    primary = [max(f, key=_area) for f in with_face]
    med = [float(np.median([b[k] for b in primary])) for k in range(4)]
    cx, cy = (med[0] + med[2]) / 2, (med[1] + med[3]) / 2
    spread = max(np.std([(b[0] + b[2]) / 2 for b in primary]), np.std([(b[1] + b[3]) / 2 for b in primary]))
    if _area(med) < 0.03 and (cx < 0.35 or cx > 0.65) and (cy < 0.4 or cy > 0.6) and spread < 0.06:
        return "gameplay_facecam", {"facecam": med}
    two = [f for f in with_face if len([b for b in f if _area(b) >= 0.012]) >= 2]
    if len(two) / len(with_face) >= 0.35:
        pair = []
        for f in two:
            big = sorted([b for b in f if _area(b) >= 0.012], key=_area, reverse=True)[:2]
            pair.append(sorted(big, key=lambda b: b[0]))
        return "two_speakers", {"left": [float(np.median([p[0][k] for p in pair])) for k in range(4)],
                                "right": [float(np.median([p[1][k] for p in pair])) for k in range(4)]}
    return "single", {}


def find_facecam(faces):
    """Find a persistent small corner face, which is more likely to be a streamer's webcam."""
    clusters = []
    for frame_i, found in enumerate(faces):
        for box in found:
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            area = _area(box)
            if area > 0.10 or not (cx < 0.38 or cx > 0.62 or cy < 0.38 or cy > 0.62):
                continue
            cluster = next((c for c in clusters
                            if abs(c["cx"] - cx) < 0.12 and abs(c["cy"] - cy) < 0.12), None)
            if cluster is None:
                cluster = {"boxes": [], "frames": set(), "cx": cx, "cy": cy}
                clusters.append(cluster)
            cluster["boxes"].append(box)
            cluster["frames"].add(frame_i)
            cluster["cx"] = float(np.median([(b[0] + b[2]) / 2 for b in cluster["boxes"]]))
            cluster["cy"] = float(np.median([(b[1] + b[3]) / 2 for b in cluster["boxes"]]))

    if not clusters:
        return None
    best = max(clusters, key=lambda c: (len(c["frames"]), float(np.median([_area(b) for b in c["boxes"]]))))
    min_frames = max(1, int(np.ceil(len(faces) * 0.12)))
    if len(best["frames"]) < min_frames:
        return None
    return [float(np.median([b[k] for b in best["boxes"]])) for k in range(4)]


def _even(v):
    return max(2, int(round(v / 2)) * 2)


def _simplify(points, tol=2.0):
    """Drop path points that a straight line through their neighbours already explains."""
    if len(points) <= 2:
        return points
    keep = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:]):
        t = (b[0] - keep[-1][0]) / max(c[0] - keep[-1][0], 1e-9)
        if abs(keep[-1][1] + t * (c[1] - keep[-1][1]) - b[1]) > tol or abs(keep[-1][2] + t * (c[2] - keep[-1][2]) - b[2]) > tol:
            keep.append(b)
    keep.append(points[-1])
    return keep


def track_single(faces, times, resets, W, H, cfg_f, dt):
    """Build a fixed 9:16 crop around the median detected face position for the whole clip."""
    prim, last = [], None
    for f in faces:
        if not f:
            prim.append(None)
            continue
        if last is not None:  # keep following the same person: the nearest face unless a much bigger one appears
            near = min(f, key=lambda b: abs((b[0] + b[2]) / 2 - (last[0] + last[2]) / 2))
            big = max(f, key=_area)
            last = near if _area(near) >= 0.5 * _area(big) else big
        else:
            last = max(f, key=_area)
        prim.append(last)
    n = len(prim)
    fh_med = float(np.median([b[3] - b[1] for b in prim if b])) if any(prim) else 0.3
    fh_hi = float(np.percentile([b[3] - b[1] for b in prim if b], 90)) if any(prim) else 0.3
    ch = min(1.0, max(0.6, fh_med / 0.30, fh_hi / 0.55))
    ch_px = _even(ch * H)
    cw_px = min(_even(ch_px * 9 / 16), W // 2 * 2)
    detected = [b for b in prim if b]
    cx = float(np.median([(b[0] + b[2]) / 2 for b in detected])) if detected else 0.5
    cy = float(np.median([(b[1] + b[3]) / 2 for b in detected])) if detected else 0.4
    left = min(max(cx * W - cw_px / 2, 0), W - cw_px)
    top = min(max(cy * H - 0.38 * ch_px, 0), H - ch_px)
    m = cfg_f["edge_margin_pct"] / 100.0
    bad = 0
    face_i = [i for i, b in enumerate(prim) if b]
    for i in face_i:  # report how often a moving face falls outside this stable crop
        x0, y0, x1, y1 = prim[i][0] * W, prim[i][1] * H, prim[i][2] * W, prim[i][3] * H
        mx, my = m * cw_px, m * ch_px
        inside = (x0 >= left + mx - 1 and x1 <= left + cw_px - mx + 1 and y0 >= top + my - 1
                  and y1 <= top + ch_px - my + 1)
        bad += 0 if inside else 1
    path = [[round(times[0], 2), round(left, 1), round(top, 1)]] if times else [[0.0, round(left, 1), round(top, 1)]]
    return {"cw": cw_px, "ch": ch_px, "path": path, "violations": round(bad / max(len(face_i), 1), 3)}, prim


def _expr(path, k):
    """Piecewise-linear ffmpeg expression in t through path points [(t, x, y)] using column k (1=x, 2=y)."""
    if len(path) == 1:
        return f"{path[0][k]:.1f}"
    e = f"{path[-1][k]:.1f}"
    for a, b in reversed(list(zip(path, path[1:]))):
        dt = max(b[0] - a[0], 1e-3)
        e = f"if(lt(t,{b[0]:.2f}),{a[k]:.1f}+({b[k] - a[k]:.1f})*(t-{a[0]:.2f})/{dt:.2f},{e})"
    return e


def _static_crop(box, W, H, aspect, face_to_crop=3.2, headroom=0.38):
    """A fixed crop (x, y, w, h in px) of the given aspect around a face box, with headroom."""
    fh = (box[3] - box[1]) * H
    ch = min(H, _even(max(fh * face_to_crop, 0.45 * H)))
    cw = _even(ch * aspect)
    if cw > W:
        cw, ch = _even(W), _even(W / aspect)
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    x = min(max(cx - cw / 2, 0), W - cw)
    y = min(max(cy - headroom * ch, 0), H - ch)
    return [int(x), int(y), int(cw), int(ch)]


def frame_clip(cfg, ffmpeg, video, src_w, src_h, start, dur, detector, layout_override=None, positions=None) -> dict:
    cf = cfg["framing"]
    positions = {**cf, **(positions or {})}
    fps = cf["face_sample_fps"]
    frames, times = sample_frames(ffmpeg, video, start, dur, src_w, src_h, fps, cf["analysis_height"])
    dt = 1.0 / fps
    forced = layout_override or cf["layout"]
    if forced not in LAYOUTS:
        raise ValueError(f"Unknown layout {forced!r}")
    faces = [detector.detect(f) for f in frames] if (forced not in ("no_face", "fit_blur") and detector is not None) else [[] for _ in frames]
    resets = set(shot_starts(frames)[1:])
    chosen, info = choose_layout(faces)
    if forced == "gameplay_facecam":
        facecam = find_facecam(faces)
        if facecam is not None:
            info = {**info, "facecam": facecam}
        else:
            info.pop("facecam", None)
    layout = chosen if forced == "auto" else forced
    fr = {"layout": layout, "auto_layout": chosen, "forced": forced != "auto", "W": src_w, "H": src_h,
          "samples": [{"t": t, "boxes": [[round(v, 3) for v in b] for b in f]} for t, f in zip(times, faces)],
          "style": positions["facecam_style"],
          "face_ratio": round(sum(1 for f in faces if f) / max(len(faces), 1), 3), "violations": 0.0, "regions": []}
    if layout == "single" and any(faces):
        crop, _ = track_single(faces, times, resets, src_w, src_h, cf, dt)
        fr["crop"] = crop
        fr["regions"] = [{"dst": [0, 0, OUT_W, OUT_H], "cw": crop["cw"], "ch": crop["ch"], "path": crop["path"]}]
        fr["violations"] = crop["violations"]
    elif layout == "single":  # forced single but no face found: say so and fit instead of guessing a crop
        fr["layout"], fr["note"] = "no_face", "no face found; used fit-over-blur instead of a blind crop"
    elif layout == "two_speakers":
        if "left" not in info:  # forced, but we never saw two faces: split at 1/4 and 3/4 of the frame
            info = {"left": [0.15, 0.25, 0.35, 0.6], "right": [0.65, 0.25, 0.85, 0.6]}
            fr["note"] = "two faces not detected; split the frame in halves"
        fr["regions"] = [{"dst": [0, 0, OUT_W, 960], "static": _static_crop(info["left"], src_w, src_h, 1080 / 960)},
                         {"dst": [0, 960, OUT_W, 960], "static": _static_crop(info["right"], src_w, src_h, 1080 / 960)}]
    elif layout == "gameplay_facecam":
        if "facecam" not in info:
            info = {"facecam": [0.72, 0.68, 0.98, 0.98]}
        box = info["facecam"]
        pad = 0.9  # grow the face box into a facecam window
        w, h = box[2] - box[0], box[3] - box[1]
        fc = [max(box[0] - pad * w, 0), max(box[1] - pad * h, 0), min(box[2] + pad * w, 1), min(box[3] + pad * h, 1)]
        fr["facecam_box"] = fc
        mx = one_euro(motion_centroid(frames), dt, resets=resets)
        gh = src_h
        gw = min(_even(gh * 1080 / 1152), src_w)
        gpath = _simplify([[round(t, 2), round(min(max(x * src_w - gw / 2, 0), src_w - gw), 1), 0.0]
                           for t, x in zip(times, mx)])
        fr["game"] = {"cw": gw, "ch": gh, "path": gpath}
        fr["regions"] = [{"dst": [0, 0, OUT_W, 1152], "cw": gw, "ch": gh, "path": gpath}]
        if fr["style"] == "split":
            game_size, face_size = (OUT_W, 608), (OUT_W, 608)
            game_default, face_default = (0, 770), (0, 150)
        elif fr["style"] == "corner":
            game_size, face_size = (OUT_W, 608), (400, 300)
            game_default, face_default = (0, 656), (40, 160)
        else:
            game_size, face_size = (OUT_W, 1152), (OUT_W, 768)
            game_default, face_default = (0, 0), (0, 1152)

        def layer_position(name, default, size):
            x = min(max(round(float(positions.get(f"{name}_x", default[0]))), 0), OUT_W - size[0])
            y = min(max(round(float(positions.get(f"{name}_y", default[1]))), 0), OUT_H - size[1])
            return [x, y, *size]

        fr["composition"] = {
            "gameplay": layer_position("gameplay", game_default, game_size),
            "facecam": layer_position("facecam", face_default, face_size),
        }
    return fr


def build_filter(fr: dict, style: str = "stack") -> str:
    """Return a filter_complex that turns [0:v] into a 1080x1920 30 fps [v]. Subtitles are added after."""
    W, H, layout = fr["W"], fr["H"], fr["layout"]
    tail = "fps=30,format=yuv420p"
    if layout == "single":
        c = fr["crop"]
        return (f"[0:v]crop={c['cw']}:{c['ch']}:x='{_expr(c['path'], 1)}':y='{_expr(c['path'], 2)}',"
                f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,{tail}[v]")
    if layout == "two_speakers":
        parts = []
        for k, r in enumerate(fr["regions"]):
            x, y, w, h = r["static"]
            parts.append(f"[s{k}]crop={w}:{h}:{x}:{y},scale=1080:960,setsar=1[r{k}]")
        return f"[0:v]split=2[s0][s1];{';'.join(parts)};[r0][r1]vstack,{tail}[v]"
    if layout == "gameplay_facecam":
        fb, g = fr["facecam_box"], fr["game"]
        composition = fr.get("composition", {})
        gx, gy, gw_out, gh_out = composition.get("gameplay", [0, 770, OUT_W, 608])
        fx_out, fy_out, fw_out, fh_out = composition.get("facecam", [0, 150, OUT_W, 608])
        fx, fy = int(fb[0] * W), int(fb[1] * H)
        fw, fh = _even((fb[2] - fb[0]) * W), _even((fb[3] - fb[1]) * H)
        fw, fh = min(fw, W - fx), min(fh, H - fy)
        if style in ("split", "stack"):
            game_scale = (f"[g0]scale={gw_out}:{gh_out}[gs];" if style == "split" else
                          f"[g0]crop={g['cw']}:{g['ch']}:x='{_expr(g['path'], 1)}':y=0,scale={gw_out}:{gh_out}[gs];")
            return (f"[0:v]split=3[bg][g0][f0];"
                    f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=40:5[b];"
                    f"[f0]crop={fw}:{fh}:{fx}:{fy},scale={fw_out}:{fh_out}:force_original_aspect_ratio=increase,"
                    f"crop={fw_out}:{fh_out}[fc];"
                    f"{game_scale}"
                    f"[b][gs]overlay={gx}:{gy}[t1];[t1][fc]overlay={fx_out}:{fy_out},{tail}[v]")
        if style == "corner":
            r = 28
            return (f"[0:v]split=3[bg][g0][f0];"
                    f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=30:3[b];"
                    f"[g0]scale={gw_out}:-2[g];[b][g]overlay={gx}:{gy}[base];"
                    f"[f0]crop={fw}:{fh}:{fx}:{fy},scale=400:300,format=yuva420p,"
                    f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':"
                    f"a='if(lte(hypot(max(abs(X-W/2)-(W/2-{r}),0),max(abs(Y-H/2)-(H/2-{r}),0)),{r}),255,0)'[fcr];"
                    f"[base][fcr]overlay={fx_out}:{fy_out},{tail}[v]")
    return (f"[0:v]split=2[bg][fg];[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=40:5[b];[fg]scale={OUT_W}:-2[f];[b][f]overlay=(W-w)/2:(H-h)/2,{tail}[v]")


def _at(path, t, k):
    return float(np.interp(t, [p[0] for p in path], [p[k] for p in path]))


def faces_in_output(fr: dict, t: float) -> list[tuple[float, float, float, float]]:
    """Face boxes mapped into 1080x1920 output pixels at clip time t (used to keep captions off faces)."""
    if not fr["samples"]:
        return []
    k = min(range(len(fr["samples"])), key=lambda i: abs(fr["samples"][i]["t"] - t))
    boxes = fr["samples"][k]["boxes"]
    W, H, out = fr["W"], fr["H"], []
    layout = fr["layout"]
    if layout == "single" and fr["regions"]:
        r = fr["regions"][0]
        x, y = _at(r["path"], t, 1), _at(r["path"], t, 2)
        sx, sy = OUT_W / r["cw"], OUT_H / r["ch"]
        out = [((b[0] * W - x) * sx, (b[1] * H - y) * sy, (b[2] * W - x) * sx, (b[3] * H - y) * sy) for b in boxes]
    elif layout == "two_speakers":
        for r in fr["regions"]:
            cx, cy, cw, ch = r["static"]
            dx, dy, dw, dh = r["dst"]
            for b in boxes:
                if cx <= (b[0] + b[2]) / 2 * W <= cx + cw:
                    out.append(((b[0] * W - cx) * dw / cw + dx, (b[1] * H - cy) * dh / ch + dy,
                                (b[2] * W - cx) * dw / cw + dx, (b[3] * H - cy) * dh / ch + dy))
    elif layout == "gameplay_facecam" and "facecam_box" in fr:
        fb = fr["facecam_box"]
        composition = fr.get("composition", {})
        fx_out, fy_out, fw_out, fh_out = composition.get("facecam", [0, 150, OUT_W, 608])
        fx, fy = fb[0] * W, fb[1] * H
        fw, fh = (fb[2] - fb[0]) * W, (fb[3] - fb[1]) * H
        for b in boxes:
            x0, y0, x1, y1 = b[0] * W - fx, b[1] * H - fy, b[2] * W - fx, b[3] * H - fy
            if fr.get("style") == "split":
                sc = max(fw_out / fw, fh_out / fh)
                ox, oy = (fw * sc - fw_out) / 2, (fh * sc - fh_out) / 2
                out.append((x0 * sc - ox + fx_out, y0 * sc - oy + fy_out,
                            x1 * sc - ox + fx_out, y1 * sc - oy + fy_out))
            elif fr.get("style") == "corner":
                sx, sy = fw_out / fw, fh_out / fh
                out.append((fx_out + x0 * sx, fy_out + y0 * sy, fx_out + x1 * sx, fy_out + y1 * sy))
            else:
                sc = max(fw_out / fw, fh_out / fh)
                ox, oy = (fw * sc - fw_out) / 2, (fh * sc - fh_out) / 2
                out.append((x0 * sc - ox + fx_out, y0 * sc - oy + fy_out,
                            x1 * sc - ox + fx_out, y1 * sc - oy + fy_out))
    return [tuple(round(v, 1) for v in b) for b in out]
