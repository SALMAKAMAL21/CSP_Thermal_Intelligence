"""
annotation_app.py
Annotation automatique des éléments CSP via Florence-2.
Utilise HuggingFace Spaces (Gradio Client) — aucun modèle local, gratuit.

Installation :
    pip install streamlit gradio-client opencv-python pillow numpy pandas

Usage :
    streamlit run scripts/annotation_app.py
"""

import streamlit as st
import numpy as np
import cv2
import json
import io
import time
import tempfile
import os
from PIL import Image

st.set_page_config(
    page_title="CSP Auto-Annotation",
    page_icon="🌞",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        --bg-dark: #05070d;
        --bg-card: #0f1624;
        --bg-card-hover: #141d2f;
        --border-muted: #1f2a3d;
        --text-primary: #f5f7fb;
        --accent: #22d3ee;
        --accent-strong: #38bdf8;
    }
    body { background-color: var(--bg-dark); }
    .stApp { background: var(--bg-dark); color: var(--text-primary); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    section[data-testid="stSidebar"] { background: #04050a; }
    section[data-testid="stSidebar"] .stSidebarContent { padding: 2rem 1.5rem; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stSlider > div { color: var(--text-primary); }
    div[data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: inset 0 0 0 1px rgba(34,211,238,0.05);
    }
    div[data-testid="stFileUploader"]:focus-within {
        box-shadow: inset 0 0 0 1px var(--accent), 0 -4px 0 0 rgba(0,0,0,0.7) inset;
    }
    .hero-card {
        background: linear-gradient(135deg,#0f172a,#101936);
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid var(--border-muted);
        box-shadow: 0 25px 65px rgba(15,23,42,0.55);
        margin-bottom: 1.5rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
        background: rgba(79,70,229,0.15);
        padding: .35rem .9rem;
        border-radius: 999px;
        border: 1px solid rgba(99,102,241,0.4);
    }
    .hero-card h1 { margin-top: 1rem; color: var(--text-primary); }
    .hero-card p { color: #cbd5f5; max-width: 60ch; }
    .quick-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
        gap: 0.9rem;
        margin-bottom: 1.25rem;
    }
    .quick-card {
        background: var(--bg-card);
        border-radius: 16px;
        border: 1px solid var(--border-muted);
        padding: 0.95rem 1rem;
        font-size: 0.95rem;
        color: #d1d8f0;
        position: relative;
        transition: all 0.2s ease;
    }
    .quick-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        border-top: 4px solid transparent;
        transition: border-color 0.2s ease;
    }
    .quick-card:hover, .quick-card:focus-within { background: var(--bg-card-hover); }
    .quick-card:hover::before, .quick-card:focus-within::before { border-top-color: var(--accent-strong); }
    .quick-card span {
        display: inline-flex;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(34,211,238,0.15);
        color: var(--accent);
        font-weight: 600;
        align-items: center;
        justify-content: center;
        margin-right: 0.6rem;
    }
    .quick-card strong { color: var(--text-primary); }
    .stButton>button {
        background: linear-gradient(90deg,#0ea5e9,#22d3ee);
        border: none;
        color: #03121e;
        font-weight: 600;
        border-radius: 14px;
        padding: 0.65rem 1.2rem;
    }
    .stButton>button:hover { filter: brightness(1.05); }
    .stStatus, [data-testid="stStatus"] {
        border-radius: 16px;
        border: 1px solid var(--border-muted);
        background: var(--bg-card);
        color: var(--text-primary);
    }
    .stDataFrame { border: 1px solid var(--border-muted); border-radius: 16px; }
    .metric-row div[data-testid="stMetricValue"] { color: var(--text-primary); }
    .compact-tip {
        background: rgba(34,211,238,0.08);
        border: 1px solid rgba(34,211,238,0.25);
        color: #9de0ff;
        padding: 0.75rem 1rem;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_MAP = {
    "receiver tube":     0, "receiver": 0, "tube": 0, "hce": 0,
    "heat collector":    0, "absorber": 0,
    "parabolic mirror":  1, "mirror": 1, "reflector": 1, "parabolic": 1,
    "support":           2, "support structure": 2, "structure": 2,
    "frame": 2, "pylon": 2, "stand": 2,
    "torque tube":       3, "torque": 3, "drive": 3,
}

CLASS_NAMES  = {0: "receiver_tube", 1: "parabolic_mirror",
                2: "support_structure", 3: "torque_tube"}
CLASS_COLORS = {0: (34,197,94), 1: (251,146,60), 2: (99,102,241), 3: (236,72,153)}
CLASS_FR     = {0: "Tube récepteur HCE", 1: "Miroir parabolique",
                2: "Structure de support", 3: "Tube de torsion"}

# ── HuggingFace Spaces disponibles avec Florence-2 ────────────────────────────
# Ces Spaces publics font tourner Florence-2 gratuitement
HF_SPACES = {
    "multimodalart/florence2-finetuning": {
        "fn_index": 0,
        "desc": "Florence-2 finetuning demo (Microsoft official)"
    },
    "gokaygokay/Florence-2": {
        "fn_index": 0,
        "desc": "Florence-2 large — détection objets"
    },
}

# Prompt grounding CSP
GROUNDING_PROMPT = "receiver tube, parabolic mirror, support structure, torque tube"


# ── Appel via Gradio Client ───────────────────────────────────────────────────

def call_florence2_space(image_pil: Image.Image, hf_token: str) -> dict:
    """
    Appelle Florence-2 via le Space HuggingFace gokaygokay/Florence-2.
    Utilise la task <CAPTION_TO_PHRASE_GROUNDING> avec prompt CSP.
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        return {"error": "gradio-client non installé. Lance : pip install gradio-client"}

    # Sauvegarder l'image temporairement
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_pil.save(tmp.name, format="JPEG", quality=90)
        tmp_path = tmp.name

    client = None
    try:
        client = Client("gokaygokay/Florence-2", hf_token=hf_token or None)

        # Tentative 1 : CAPTION_TO_PHRASE_GROUNDING avec prompt CSP
        try:
            result = client.predict(
                image=handle_file(tmp_path),
                task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
                text_input=GROUNDING_PROMPT,
                model_id="microsoft/Florence-2-large",
                api_name="/process_image"
            )
            return {"status": "ok", "result": result, "task": "<CAPTION_TO_PHRASE_GROUNDING>"}
        except Exception:
            pass

        # Tentative 2 : OD générique si grounding échoue
        result = client.predict(
            image=handle_file(tmp_path),
            task_prompt="<OD>",
            text_input="",
            model_id="microsoft/Florence-2-large",
            api_name="/process_image"
        )
        return {"status": "ok", "result": result, "task": "<OD>"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_florence_output(result) -> list:
    """
    Parse la sortie brute du Space Florence-2.
    Le Space retourne généralement un dict ou une string JSON.
    """
    detections = []

    # Cas 1 : result est un tuple (image_annotée, json_string)
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, str):
                try:
                    data = json.loads(item)
                    detections.extend(_extract_from_dict(data))
                except Exception:
                    pass
            elif isinstance(item, dict):
                detections.extend(_extract_from_dict(item))

    # Cas 2 : result est un dict direct
    elif isinstance(result, dict):
        detections.extend(_extract_from_dict(result))

    # Cas 3 : result est une string JSON
    elif isinstance(result, str):
        try:
            data = json.loads(result)
            detections.extend(_extract_from_dict(data))
        except Exception:
            pass

    return detections


def _extract_from_dict(data: dict) -> list:
    """Extrait les bboxes et labels d'un dict Florence-2."""
    detections = []

    # Format Florence-2 standard : {"bboxes": [...], "labels": [...]}
    bboxes = data.get("bboxes", data.get("boxes", data.get("bbox", [])))
    labels = data.get("labels", data.get("label", []))

    # Clé imbriquée possible : {"<OD>": {"bboxes": ..., "labels": ...}}
    for key in ["<OD>", "<CAPTION_TO_PHRASE_GROUNDING>",
                "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"]:
        if key in data and isinstance(data[key], dict):
            sub = data[key]
            bboxes = sub.get("bboxes", sub.get("boxes", []))
            labels = sub.get("labels", sub.get("label", []))
            break

    for bbox, label in zip(bboxes, labels):
        if len(bbox) != 4:
            continue
        cid = _label_to_class_id(str(label))
        detections.append({
            "class_id":   cid,
            "class_name": CLASS_NAMES.get(cid, str(label)),
            "label_raw":  str(label),
            "bbox":       [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "confidence": 0.85,
        })

    return detections


def _label_to_class_id(label: str) -> int:
    label_lower = label.lower()
    for key, cid in CLASS_MAP.items():
        if key in label_lower:
            return cid
    return 0


# ── Dessin annotations ────────────────────────────────────────────────────────

def draw_annotations(image_np, detections, show_boxes, show_labels, opacity):
    result = image_np.copy()
    h, w   = result.shape[:2]

    for det in detections:
        cid   = det.get("class_id", 0)
        color = CLASS_COLORS.get(cid, (255,255,255))
        bbox  = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1:
            continue

        overlay = result.copy()
        overlay[y1:y2, x1:x2] = (
            overlay[y1:y2, x1:x2] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)
        result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)

        if show_boxes:
            cv2.rectangle(result, (x1,y1), (x2,y2), color, 2)

        if show_labels:
            name  = CLASS_NAMES.get(cid, det.get("class_name",""))
            conf  = det.get("confidence", 0)
            text  = f"{name} {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1-6, th+4)
            cv2.rectangle(result,(x1,ty-th-4),(x1+tw+6,ty+2),color,-1)
            cv2.putText(result,text,(x1+3,ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return result


def bbox_to_yolo(bbox, class_id, h, w):
    x1,y1,x2,y2 = bbox
    return (f"{class_id} "
            f"{((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill">Green Energy Park · SaaS Lab</div>
            <h1>Annotation CSP nouvelle génération</h1>
            <p>Automatisez vos annotations de tubes récepteurs et miroirs via Florence-2 hébergé sur HuggingFace.
            Interface sombre, feedback instantané, export YOLO prêt pour l'entraînement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="quick-grid">
            <div class="quick-card"><span>1</span><strong>Charger</strong><br/>Déposez votre image CSP haute résolution.</div>
            <div class="quick-card"><span>2</span><strong>Annoter</strong><br/>Florence-2 génère les boîtes en quelques secondes.</div>
            <div class="quick-card"><span>3</span><strong>Exporter</strong><br/>Téléchargez l'image annotée et le label YOLO.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")

        hf_token = st.text_input(
            "HuggingFace Token (optionnel)",
            type="password",
            placeholder="hf_...",
            help="Optionnel — accélère l'accès et évite les rate limits"
        )

        st.markdown(
            """
            <div class="compact-tip">
                <strong>Astuce vitesse :</strong> crée un token HuggingFace (Read) pour éviter la file d'attente.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Affichage")
        show_boxes  = st.toggle("Bounding boxes", value=True)
        show_labels = st.toggle("Labels",         value=True)
        opacity     = st.slider("Opacité", 0.1, 0.8, 0.35, 0.05)

        st.divider()
        st.subheader("Légende")
        icons = ["🟢","🟠","🟣","🩷"]
        for cid, name in CLASS_NAMES.items():
            st.caption(f"{icons[cid]} **{name}** — {CLASS_FR[cid]}")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Téléverser une image CSP",
        type=["jpg","jpeg","png","bmp","tiff"]
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="quick-grid">
                <div class="quick-card" tabindex="0"><span>1</span>Glissez votre image dans la zone ci-dessus.</div>
                <div class="quick-card" tabindex="0"><span>2</span>Appuyez sur <strong>Annoter</strong> pour lancer Florence-2.</div>
                <div class="quick-card" tabindex="0"><span>3</span>Récupérez vos exports YOLO11 instantanément.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    # Redimensionner pour le Space (max 1024px)
    ratio = min(1024 / max(orig_w, orig_h), 1.0)
    img_resized = image_pil.resize(
        (int(orig_w*ratio), int(orig_h*ratio)), Image.LANCZOS
    ) if ratio < 1.0 else image_pil
    res_w, res_h = img_resized.size

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.subheader("Image originale")
        st.image(image_pil, use_container_width=True)
        st.caption(f"{orig_w}×{orig_h}px · {uploaded.size/1024:.1f}KB")

    if not st.button("Annoter avec Florence-2", type="primary", use_container_width=True):
        return

    with st.status("Connexion au Space Florence-2...", expanded=True) as status:
        t0 = time.time()
        st.write("Envoi au Space `gokaygokay/Florence-2`...")
        st.write(f"Prompt : `<CAPTION_TO_PHRASE_GROUNDING>` + *{GROUNDING_PROMPT}*")

        resp = call_florence2_space(img_resized, hf_token)

        if "error" in resp:
            status.update(label="Erreur", state="error")
            st.error(f"Erreur : {resp['error']}")
            st.markdown(
                "**Solutions :**\n"
                "1. Vérifie ta connexion internet\n"
                "2. Le Space est peut-être temporairement indisponible — "
                "réessaie dans 1-2 min\n"
                "3. Va directement sur "
                "[gokaygokay/Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2) "
                "pour vérifier qu'il est en ligne"
            )
            return

        t_total = time.time() - t0
        raw_result = resp.get("result")

        # Parser les détections
        detections = parse_florence_output(raw_result)

        # Remettre les bbox à l'échelle originale si redimensionné
        if ratio < 1.0:
            sx, sy = orig_w/res_w, orig_h/res_h
            for det in detections:
                b = det["bbox"]
                det["bbox"] = [int(b[0]*sx), int(b[1]*sy),
                               int(b[2]*sx), int(b[3]*sy)]

        if not detections:
            status.update(label="Aucun élément détecté", state="error")
            st.warning("Aucun élément CSP détecté.")
            with st.expander("Réponse brute Florence-2 (debug)"):
                st.write(raw_result)
            return

        status.update(
            label=f"Annotation terminée ✓ — {len(detections)} éléments en {t_total:.1f}s",
            state="complete"
        )

    # ── Résultat ──────────────────────────────────────────────────────────────
    annotated = draw_annotations(
        image_np, detections, show_boxes, show_labels, opacity
    )

    with col_result:
        st.subheader("Résultat annoté")
        st.image(annotated, use_container_width=True)
        st.caption(f"{len(detections)} élément(s) · {t_total:.1f}s")

    # ── Tableau ───────────────────────────────────────────────────────────────
    st.subheader("Détections")
    import pandas as pd
    rows = []
    for i, det in enumerate(detections):
        cid = det.get("class_id", 0)
        b   = det.get("bbox", [0,0,0,0])
        x1,y1,x2,y2 = [int(v) for v in b]
        rows.append({
            "ID": i,
            "Classe":    CLASS_NAMES.get(cid,"?"),
            "Label raw": det.get("label_raw",""),
            "Conf.":     f"{det.get('confidence',0):.2f}",
            "X1":x1,"Y1":y1,"X2":x2,"Y2":y2,
            "W":x2-x1,"H":y2-y1,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    counts = {n:0 for n in CLASS_NAMES.values()}
    for det in detections:
        counts[CLASS_NAMES.get(det.get("class_id",0),"receiver_tube")] += 1
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Receiver tubes",    counts["receiver_tube"])
    m2.metric("Parabolic mirrors", counts["parabolic_mirror"])
    m3.metric("Support structures",counts["support_structure"])
    m4.metric("Torque tubes",      counts["torque_tube"])

    # ── Export YOLO ───────────────────────────────────────────────────────────
    st.subheader("Export YOLO11")
    yolo_lines = [
        bbox_to_yolo(d["bbox"], d["class_id"], orig_h, orig_w)
        for d in detections if len(d.get("bbox",[])) == 4
    ]
    yolo_txt = "\n".join(yolo_lines)

    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Annotation .txt YOLO", value=yolo_txt, height=150)
        st.download_button(
            "Télécharger .txt",
            data=yolo_txt,
            file_name=f"{uploaded.name.rsplit('.',1)[0]}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with c2:
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG", quality=95)
        st.download_button(
            "Télécharger image annotée",
            data=buf.getvalue(),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_annotated.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    with st.expander("Réponse brute Florence-2 (debug)"):
        st.write(raw_result)


if __name__ == "__main__":
    main()
