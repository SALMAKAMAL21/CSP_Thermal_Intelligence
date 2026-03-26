"""
annotation_app.py - Version 4
Annotation CSP hybride :
  - Florence-2 pour la detection initiale (bounding boxes globales)
  - CV classique (couleur + geometrie + Hough) pour separer tube vs miroir
  - SAM2 pour la segmentation precise

Usage : streamlit run scripts/annotation_app.py
"""

import sys, types, importlib.util

def _make_fake_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    mod.__loader__ = None; mod.__path__ = []; mod.__package__ = name.split('.')[0]; mod.__file__ = None
    return mod

for _m in ['flash_attn','flash_attn.bert_padding','flash_attn.flash_attn_interface',
           'flash_attn.flash_attn_triton','flash_attn.layers','flash_attn.layers.rotary']:
    if _m not in sys.modules:
        sys.modules[_m] = _make_fake_module(_m)

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import time

st.set_page_config(page_title="CSP Auto-Annotation", page_icon="🌞", layout="wide")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASS_NAMES  = {0: "receiver_tube", 1: "parabolic_mirror"}
CLASS_COLORS = {0: (34, 197, 94), 1: (251, 146, 60)}


# ── Chargement modeles ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_florence2(model_size="large"):
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
    from pathlib import Path
    local_paths = [
        Path(f"models/checkpoints/Florence-2-{model_size}"),
        Path(f"models/Florence-2-{model_size}"),
        Path.home() / f".cache/huggingface/hub/models--microsoft--Florence-2-{model_size}/snapshots",
    ]
    model_id = f"microsoft/Florence-2-{model_size}"
    for p in local_paths:
        if p.exists():
            if p.name == "snapshots" and p.is_dir():
                snaps = sorted(p.iterdir())
                if snaps: model_id = str(snaps[-1]); break
            elif (p / "config.json").exists():
                model_id = str(p); break
    st.write(f"Modele : {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    def _patch(cfg):
        for a in ("forced_bos_token_id","forced_eos_token_id"):
            if not hasattr(cfg, a): setattr(cfg, a, None)
    _patch(config)
    for sub in ("text_config","vision_config","language_config"):
        if hasattr(config, sub): _patch(getattr(config, sub))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.float32,
        trust_remote_code=True, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


@st.cache_resource(show_spinner=False)
def load_sam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from pathlib import Path
    candidates = [
        ("configs/sam2.1/sam2.1_hiera_l.yaml","models/checkpoints/sam2.1_hiera_large.pt"),
        ("configs/sam2.1/sam2.1_hiera_s.yaml","models/checkpoints/sam2.1_hiera_small.pt"),
        ("configs/sam2.1/sam2.1_hiera_t.yaml","models/checkpoints/sam2.1_hiera_tiny.pt"),
    ]
    for cfg, ckpt in candidates:
        if Path(ckpt).exists():
            return SAM2ImagePredictor(build_sam2(cfg, ckpt, device=DEVICE))
    raise FileNotFoundError("Aucun checkpoint SAM2 trouve.")


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION PAR VISION CLASSIQUE (pas de modele IA)
# ══════════════════════════════════════════════════════════════════════════════

def detect_cv(image_np, debug=False):
    """
    Detection des tubes et miroirs par analyse d'image classique.
    
    Tube recepteur :
      - Ligne fine horizontale sombre (metal + verre)
      - Position : dans le tiers superieur de l'image (au foyer)
      - Detectable par transformee de Hough sur lignes
    
    Miroir parabolique :
      - Grandes regions claires/brillantes
      - Haute luminosite, surface lisse
      - Detectable par seuillage de luminosite + contours larges
    """
    h, w = image_np.shape[:2]
    gray   = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hsv    = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    boxes_tubes, boxes_mirrors = [], []
    debug_imgs = {}

    # ── 1. DETECTION DES MIROIRS ──────────────────────────────────────────────
    # Les miroirs sont les zones les plus brillantes (reflet du ciel/soleil)
    # Luminosite V du HSV > 180 = zones reflechissantes
    
    V = hsv[:,:,2]  # canal luminosite
    S = hsv[:,:,1]  # canal saturation (miroirs = faible saturation, gris/blanc)
    
    # Masque miroir : brillant ET peu sature (reflet metallique/verre)
    mirror_mask = ((V > 160) & (S < 80)).astype(np.uint8) * 255
    
    # Nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mirror_mask = cv2.morphologyEx(mirror_mask, cv2.MORPH_CLOSE, kernel)
    mirror_mask = cv2.morphologyEx(mirror_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))
    
    if debug: debug_imgs["mirror_mask"] = mirror_mask

    contours_m, _ = cv2.findContours(mirror_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_m:
        area = cv2.contourArea(cnt)
        # Filtrer : miroir = grande surface (> 1% image) et forme large
        if area < (h * w * 0.01):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / (bh + 1e-5)
        # Miroir : plus large que haut (ar > 1.2) et surface significative
        if ar > 1.2 and bw > w * 0.05:
            # Fusionner les boxes proches horizontalement
            boxes_mirrors.append([x, y, x+bw, y+bh])

    # Fusionner les miroirs en groupes horizontaux
    boxes_mirrors = _merge_horizontal_boxes(boxes_mirrors, gap=w//10)

    # ── 2. DETECTION DU TUBE RECEPTEUR ────────────────────────────────────────
    # Le tube est une ligne fine horizontale sombre en haut du champ de miroirs
    # Approche : chercher des contours fins et tres allonges horizontalement
    
    # Zone de recherche : tiers superieur-milieu de l'image (le tube est au foyer)
    roi_y1 = int(h * 0.05)
    roi_y2 = int(h * 0.70)
    roi = gray[roi_y1:roi_y2, :]

    # Detecter les bords
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    
    if debug: debug_imgs["edges"] = edges

    # Transformee de Hough pour lignes horizontales
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180,
        threshold=60,
        minLineLength=w // 5,   # ligne doit faire au moins 1/5 de la largeur
        maxLineGap=w // 8
    )

    tube_lines = []
    if lines is not None:
        for line in lines:
            x1, y1_l, x2, y2_l = line[0]
            angle = abs(np.degrees(np.arctan2(y2_l - y1_l, x2 - x1)))
            length = np.sqrt((x2-x1)**2 + (y2_l-y1_l)**2)
            # Garder seulement les lignes quasi-horizontales (< 15 degres)
            if angle < 15 or angle > 165:
                tube_lines.append((x1, y1_l + roi_y1, x2, y2_l + roi_y1, length))

    if debug: debug_imgs["lines_count"] = len(tube_lines) if tube_lines else 0

    # Grouper les lignes proches en un seul tube
    if tube_lines:
        # Trier par longueur decroissante, garder les plus longues
        tube_lines.sort(key=lambda l: l[4], reverse=True)
        
        # Clusterer par position Y (lignes dans une bande de 20px = meme tube)
        clusters = []
        for line in tube_lines:
            y_mid = (line[1] + line[3]) // 2
            placed = False
            for cluster in clusters:
                if abs(cluster["y_mid"] - y_mid) < 25:
                    cluster["lines"].append(line)
                    cluster["y_mid"] = (cluster["y_mid"] + y_mid) // 2
                    placed = True
                    break
            if not placed:
                clusters.append({"y_mid": y_mid, "lines": [line]})

        for cluster in clusters:
            lns = cluster["lines"]
            xs = [l[0] for l in lns] + [l[2] for l in lns]
            ys = [l[1] for l in lns] + [l[3] for l in lns]
            x1, x2 = min(xs), max(xs)
            y1_c, y2_c = min(ys), max(ys)
            bw = x2 - x1
            bh = max(y2_c - y1_c, 8)  # hauteur minimum 8px
            ar = bw / bh
            # Tube : tres allonge (ar > 5) et assez long (> 15% largeur image)
            if ar > 5 and bw > w * 0.15:
                # Padding vertical pour englober le tube entier
                pad = max(10, bh)
                boxes_tubes.append([
                    max(0, x1 - 5),
                    max(0, y1_c - pad),
                    min(w, x2 + 5),
                    min(h, y2_c + pad)
                ])

    # ── 3. NETTOYER : supprimer miroirs qui contiennent le tube ───────────────
    # Si une box miroir chevauche fortement une box tube, reajuster le miroir
    cleaned_mirrors = []
    for mb in boxes_mirrors:
        overlaps_tube = False
        for tb in boxes_tubes:
            iou = _iou(mb, tb)
            # Si le tube est contenu dans le miroir ET le miroir est grand
            if iou > 0.3:
                overlaps_tube = True
        if not overlaps_tube:
            cleaned_mirrors.append(mb)
        else:
            # Garder le miroir mais en excluant la zone du tube
            cleaned_mirrors.append(mb)  # SAM2 va affiner

    all_boxes, all_ids = [], []
    for b in boxes_tubes:
        all_boxes.append(b); all_ids.append(0)
    for b in cleaned_mirrors:
        all_boxes.append(b); all_ids.append(1)

    return all_boxes, all_ids, debug_imgs


def _merge_horizontal_boxes(boxes, gap=50):
    """Fusionne les boxes proches horizontalement."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:
        last = merged[-1]
        # Chevauche ou proche horizontalement ET meme bande verticale
        if b[0] <= last[2] + gap and abs(b[1] - last[1]) < 50:
            merged[-1] = [
                min(last[0], b[0]), min(last[1], b[1]),
                max(last[2], b[2]), max(last[3], b[3])
            ]
        else:
            merged.append(b)
    return merged


def _iou(b1, b2):
    xi1,yi1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    xi2,yi2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter == 0: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)


# ── Florence-2 comme alternative ─────────────────────────────────────────────

def detect_florence(image_pil, model, processor):
    """Detection Florence-2 OD generique comme fallback."""
    inputs = processor(text="<OD>", images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3,
        )
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        text, task="<OD>",
        image_size=(image_pil.width, image_pil.height)
    )
    raw = result.get("<OD>", {})
    return raw.get("bboxes", []), raw.get("labels", [])


# ── Segmentation SAM2 ─────────────────────────────────────────────────────────

def segment_elements(image_np, boxes, class_ids, predictor):
    predictor.set_image(image_np)
    masks_list, scores_list = [], []

    for box, cid in zip(boxes, class_ids):
        b = np.array(box, dtype=float)
        masks, scores, _ = predictor.predict(
            box=b[np.newaxis], multimask_output=True
        )
        if cid == 0:
            # Tube : masque le plus fin (aspect ratio le plus eleve)
            best = 0; best_ar = 0
            for i, m in enumerate(masks):
                if m.sum() == 0: continue
                rows = np.any(m, axis=1).sum()
                cols = np.any(m, axis=0).sum()
                ar = cols / (rows + 1e-5)
                score_combined = ar * 0.7 + float(scores[i]) * 0.3
                if score_combined > best_ar:
                    best_ar = score_combined; best = i
        else:
            # Miroir : masque le plus grand et le mieux score
            best = np.argmax(scores)
        masks_list.append(masks[best])
        scores_list.append(float(scores[best]))

    if DEVICE == "mps": torch.mps.empty_cache()
    return (np.array(masks_list) if masks_list else np.array([])), scores_list


# ── Dessin ────────────────────────────────────────────────────────────────────

def draw_annotations(image_np, boxes, masks, class_ids, scores,
                     show_masks, show_boxes, show_labels, opacity):
    result = image_np.copy()
    h, w = result.shape[:2]
    for i, (box, cid) in enumerate(zip(boxes, class_ids)):
        color = CLASS_COLORS[cid]
        if show_masks and i < len(masks):
            mask = masks[i].astype(bool)
            overlay = result.copy(); overlay[mask] = color
            result = cv2.addWeighted(overlay, opacity, result, 1-opacity, 0)
            cnts,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, cnts, -1, color, 2)
        if show_boxes:
            x1,y1,x2,y2 = [int(v) for v in box]
            cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
        if show_labels:
            x1,y1 = int(box[0]),int(box[1])
            s = f"{scores[i]:.2f}" if i < len(scores) else ""
            text = f"{CLASS_NAMES[cid]} {s}"
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1-6, th+4)
            cv2.rectangle(result,(x1,ty-th-4),(x1+tw+6,ty+2),color,-1)
            cv2.putText(result,text,(x1+3,ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return result


def mask_to_yolo(mask, class_id, h, w):
    cnts,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50: return None
    eps = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,eps,True)
    if len(approx) < 3: return None
    pts = np.clip(approx.reshape(-1,2).astype(float)/np.array([w,h]),0,1)
    return f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x,y in pts])


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🌞 CSP Auto-Annotation")
    st.caption(f"CV classique + SAM2  ·  Device : **{DEVICE.upper()}**")

    with st.sidebar:
        st.header("Parametres")
        
        detection_mode = st.radio(
            "Mode de detection",
            ["CV Classique (recommande)", "Florence-2 + CV"],
            index=0,
            help="CV Classique = rapide et fiable pour les CSP\nFlorence-2 + CV = charge le modele IA en plus"
        )
        
        st.divider()
        st.subheader("Parametres CV")
        mirror_brightness = st.slider("Seuil luminosite miroir", 100, 220, 160, 5,
            help="Plus bas = detecte plus de surfaces reflechissantes")
        mirror_saturation = st.slider("Seuil saturation miroir", 30, 120, 80, 5,
            help="Plus bas = regions plus grises/blanches uniquement")
        tube_min_length = st.slider("Longueur min tube (% largeur)", 5, 40, 15, 5,
            help="Longueur minimum d'une ligne pour etre un tube")
        
        st.divider()
        st.subheader("Affichage")
        show_masks  = st.toggle("Masques SAM2",    value=True)
        show_boxes  = st.toggle("Bounding boxes",  value=True)
        show_labels = st.toggle("Labels",          value=True)
        show_debug  = st.toggle("Mode debug (masques intermediaires)", value=False)
        opacity     = st.slider("Opacite", 0.1, 0.9, 0.4, 0.05)

        st.divider()
        st.caption("🟢 **receiver_tube** — Tube recepteur HCE")
        st.caption("🟠 **parabolic_mirror** — Miroir parabolique")

    uploaded = st.file_uploader(
        "Upload une image CSP", type=["jpg","jpeg","png","bmp","tiff"]
    )
    if uploaded is None:
        st.info("Upload une image CSP pour demarrer.")
        return

    image_pil = Image.open(uploaded).convert("RGB")
    image_np  = np.array(image_pil)
    h, w      = image_np.shape[:2]

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.subheader("Image originale")
        st.image(image_pil, use_container_width=True)
        st.caption(f"{w}x{h}px · {uploaded.size/1024:.1f}KB")

    if not st.button("Annoter automatiquement", type="primary", use_container_width=True):
        return

    with st.status("Annotation en cours...", expanded=True) as status:
        t0 = time.time()

        # ── Detection ─────────────────────────────────────────────────────────
        st.write("Detection par analyse d'image (luminosite + geometrie)...")

        # Utiliser les parametres de la sidebar
        # Patcher temporairement les seuils dans detect_cv
        import types as _types
        _orig_detect = detect_cv

        def detect_cv_custom(img, debug=False):
            h2, w2 = img.shape[:2]
            gray2  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hsv2   = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            V2 = hsv2[:,:,2]; S2 = hsv2[:,:,1]
            debug_imgs = {}

            # Miroirs avec seuils ajustables
            mirror_mask2 = ((V2 > mirror_brightness) & (S2 < mirror_saturation)).astype(np.uint8)*255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            mirror_mask2 = cv2.morphologyEx(mirror_mask2, cv2.MORPH_CLOSE, kernel)
            mirror_mask2 = cv2.morphologyEx(mirror_mask2, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))
            if debug: debug_imgs["Masque miroirs"] = mirror_mask2

            boxes_mirrors2 = []
            cnts2,_ = cv2.findContours(mirror_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts2:
                area = cv2.contourArea(cnt)
                if area < (h2*w2*0.01): continue
                x2b,y2b,bw2,bh2 = cv2.boundingRect(cnt)
                ar2 = bw2/(bh2+1e-5)
                if ar2 > 1.2 and bw2 > w2*0.05:
                    boxes_mirrors2.append([x2b,y2b,x2b+bw2,y2b+bh2])
            boxes_mirrors2 = _merge_horizontal_boxes(boxes_mirrors2, gap=w2//10)

            # Tubes
            roi_y1 = int(h2*0.05); roi_y2 = int(h2*0.70)
            roi2 = gray2[roi_y1:roi_y2,:]
            blurred2 = cv2.GaussianBlur(roi2,(5,5),0)
            edges2   = cv2.Canny(blurred2,30,100)
            if debug: debug_imgs["Bords (Canny)"] = edges2

            min_len = int(w2 * tube_min_length / 100)
            lines2 = cv2.HoughLinesP(edges2,1,np.pi/180,60,
                                     minLineLength=min_len, maxLineGap=w2//8)
            boxes_tubes2 = []
            tube_lines2 = []
            if lines2 is not None:
                for line2 in lines2:
                    x1l,y1l,x2l,y2l = line2[0]
                    angle2 = abs(np.degrees(np.arctan2(y2l-y1l,x2l-x1l)))
                    length2 = np.sqrt((x2l-x1l)**2+(y2l-y1l)**2)
                    if angle2 < 15 or angle2 > 165:
                        tube_lines2.append((x1l,y1l+roi_y1,x2l,y2l+roi_y1,length2))

            if tube_lines2:
                tube_lines2.sort(key=lambda l:l[4],reverse=True)
                clusters2 = []
                for line2 in tube_lines2:
                    y_mid2 = (line2[1]+line2[3])//2
                    placed2 = False
                    for cl2 in clusters2:
                        if abs(cl2["y_mid"]-y_mid2) < 25:
                            cl2["lines"].append(line2); cl2["y_mid"]=(cl2["y_mid"]+y_mid2)//2
                            placed2=True; break
                    if not placed2: clusters2.append({"y_mid":y_mid2,"lines":[line2]})

                for cl2 in clusters2:
                    lns2 = cl2["lines"]
                    xs2=[l[0] for l in lns2]+[l[2] for l in lns2]
                    ys2=[l[1] for l in lns2]+[l[3] for l in lns2]
                    x1c,x2c=min(xs2),max(xs2)
                    y1c,y2c=min(ys2),max(ys2)
                    bw2c=x2c-x1c; bh2c=max(y2c-y1c,8)
                    ar2c=bw2c/bh2c
                    if ar2c>5 and bw2c>w2*0.15:
                        pad2=max(10,bh2c)
                        boxes_tubes2.append([max(0,x1c-5),max(0,y1c-pad2),
                                             min(w2,x2c+5),min(h2,y2c+pad2)])

            all_boxes2,all_ids2=[],[]
            for b in boxes_tubes2: all_boxes2.append(b); all_ids2.append(0)
            for b in boxes_mirrors2: all_boxes2.append(b); all_ids2.append(1)
            return all_boxes2, all_ids2, debug_imgs

        boxes, class_ids, debug_imgs = detect_cv_custom(image_np, debug=show_debug)

        t_det = time.time()-t0
        n_tubes   = sum(1 for c in class_ids if c==0)
        n_mirrors = sum(1 for c in class_ids if c==1)
        st.write(f"Detecte : {n_tubes} tube(s), {n_mirrors} miroir(s) en {t_det:.2f}s")

        if show_debug and debug_imgs:
            st.write("**Masques intermediaires (debug) :**")
            dcols = st.columns(len(debug_imgs))
            for col, (name, img) in zip(dcols, debug_imgs.items()):
                col.image(img, caption=name, use_container_width=True)

        if not boxes:
            status.update(label="Aucun element detecte", state="error")
            st.warning(
                "Aucun element detecte avec les seuils actuels.\n\n"
                "**Ajuste dans la sidebar :**\n"
                "- Baisse le seuil de luminosite miroir (si miroirs sombres)\n"
                "- Baisse la longueur min tube (si tube court)"
            )
            return

        st.write("Chargement SAM2...")
        try:
            sam2 = load_sam2()
        except FileNotFoundError as e:
            status.update(label="Erreur SAM2", state="error")
            st.error(str(e)); return

        st.write("Segmentation precise avec SAM2...")
        t2 = time.time()
        masks, scores = segment_elements(image_np, boxes, class_ids, sam2)
        t_seg = time.time()-t2
        st.write(f"Segmentation terminee en {t_seg:.1f}s")
        status.update(label=f"Termine ✓ — {n_tubes} tube(s), {n_mirrors} miroir(s)", state="complete")

    annotated = draw_annotations(
        image_np, boxes, masks, class_ids, scores,
        show_masks, show_boxes, show_labels, opacity
    )

    with col_result:
        st.subheader("Resultat annote")
        st.image(annotated, use_container_width=True)
        st.caption(f"Detection CV: {t_det:.2f}s · SAM2: {t_seg:.1f}s · Total: {t_det+t_seg:.1f}s")

    st.subheader("Detections")
    import pandas as pd
    rows=[]
    for i,(box,cid) in enumerate(zip(boxes,class_ids)):
        x1,y1,x2,y2=[int(v) for v in box]
        rows.append({"ID":i,"Classe":CLASS_NAMES[cid],
                     "Score":f"{scores[i]:.3f}" if i<len(scores) else "-",
                     "X1":x1,"Y1":y1,"X2":x2,"Y2":y2,
                     "Largeur":x2-x1,"Hauteur":y2-y1,
                     "Ratio L/H":f"{(x2-x1)/(y2-y1+1e-5):.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    m1,m2=st.columns(2)
    m1.metric("Receiver tubes",    n_tubes)
    m2.metric("Parabolic mirrors", n_mirrors)

    st.subheader("Export YOLO11")
    if len(masks)>0:
        yolo_lines=[mask_to_yolo(masks[i],class_ids[i],h,w)
                    for i in range(len(masks)) if mask_to_yolo(masks[i],class_ids[i],h,w)]
        yolo_txt="\n".join(yolo_lines)
        c1,c2=st.columns(2)
        with c1:
            st.text_area("Annotation .txt YOLO",value=yolo_txt,height=160)
            st.download_button("Telecharger .txt",data=yolo_txt,
                file_name=f"{uploaded.name.rsplit('.',1)[0]}.txt",
                mime="text/plain",use_container_width=True)
        with c2:
            import io; buf=io.BytesIO()
            Image.fromarray(annotated).save(buf,format="JPEG",quality=95)
            st.download_button("Telecharger image annotee",data=buf.getvalue(),
                file_name=f"{uploaded.name.rsplit('.',1)[0]}_annotated.jpg",
                mime="image/jpeg",use_container_width=True)


if __name__ == "__main__":
    main()
