"""Informe Final escrito (formato cátedra): A4, índice, narrativa, figuras a escala."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"
FIGURES = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Informe_Final_Grupo33.pdf"

NAVY = colors.HexColor("#1F4E79")
TEAL = colors.HexColor("#2E75B6")
LINE = colors.HexColor("#B0B0B0")
ROW = colors.HexColor("#F4F7FB")
HEADER_BG = colors.HexColor("#1F4E79")


def pct(x, d=1):
    return f"{100 * float(x):.{d}f} %"


def num(x, d=3):
    return f"{float(x):.{d}f}".replace(".", ",")


def styles():
    return {
        "kicker": ParagraphStyle(
            "kicker", fontName="Helvetica", fontSize=11, leading=16.5,
            alignment=TA_CENTER, textColor=colors.HexColor("#C0392B"), spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", fontName="Helvetica-Bold", fontSize=18, leading=24,
            alignment=TA_CENTER, spaceAfter=10, textColor=colors.black,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName="Helvetica", fontSize=12, leading=18,
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "idx": ParagraphStyle(
            "idx", fontName="Helvetica", fontSize=12, leading=18,
            alignment=TA_CENTER, spaceAfter=3,
        ),
        "h1": ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=14, leading=21,
            textColor=NAVY, spaceBefore=10, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=12, leading=18,
            textColor=TEAL, spaceBefore=8, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=12, leading=18,
            alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=12, leading=18,
            leftIndent=12, spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Helvetica-Oblique", fontSize=9, leading=12,
            alignment=TA_CENTER, textColor=colors.grey, spaceBefore=3, spaceAfter=10,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Helvetica", fontSize=8, leading=10,
            alignment=TA_CENTER,
        ),
        "cell_l": ParagraphStyle(
            "cell_l", fontName="Helvetica", fontSize=8, leading=10,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "th", fontName="Helvetica-Bold", fontSize=8, leading=10,
            alignment=TA_CENTER, textColor=colors.white,
        ),
    }


def fig(name, max_w=15 * cm, max_h=6.0 * cm):
    """Inserta PNG manteniendo aspecto y cabiendo en el marco (evita zoom/recorte)."""
    path = FIGURES / name
    if not path.exists():
        return Paragraph(f"[Figura no disponible: {name}]", styles()["caption"])
    with PILImage.open(path) as im:
        pw, ph = im.size
    aspect = ph / float(pw)
    w = max_w
    h = w * aspect
    if h > max_h:
        h = max_h
        w = h / aspect
    return Image(str(path), width=w, height=h, kind="proportional")


def table(data, col_widths=None):
    S = styles()
    formatted = []
    for i, row in enumerate(data):
        style = S["th"] if i == 0 else S["cell"]
        formatted.append([Paragraph(str(c), style) for c in row])
    t = Table(formatted, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.15 * cm, "Informe Final — Detección de Fraude — Grupo 33")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, str(doc.page))
    canvas.restoreState()


def build():
    S = styles()
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    eda = data["eda"]
    rand = data["random_split"]
    temp = data["temporal_split"]
    best_name = rand["best_model"]
    best = rand["models"][best_name]
    m50 = best["metrics_thr_0.5"]
    msel = best["metrics_selected"]
    cap = {r["k"]: r for r in best["captura_at_k"]}
    best_t = temp["best_model"]
    m50_t = temp["models"][best_t]["metrics_thr_0.5"]
    cap_t = {r["k"]: r for r in temp["models"][best_t]["captura_at_k"]}
    n_neg = m50["n_neg"]
    n_pos = m50["n_pos"]
    fpr5_alerts = int(0.05 * n_neg)

    story = []

    # -------- Portada + índice --------
    story.append(Spacer(1, 1.6 * cm))
    story.append(Paragraph("INFORME FINAL", S["kicker"]))
    story.append(Paragraph("Detección de fraude en tarjetas de crédito:<br/>priorización de la cola de revisión", S["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Diplomado en Ciencia de Datos y Análisis Avanzado", S["cover_sub"]))
    story.append(Paragraph("Unidad N.º 9 — Proyecto Final Integrador", S["cover_sub"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("<b>Grupo 33</b>", S["cover_sub"]))
    story.append(Paragraph(
        "Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis<br/>Marco Garcia · Esteban Derghazarian",
        S["cover_sub"],
    ))
    story.append(Paragraph("Octubre 2026", S["cover_sub"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("<b>Índice</b>", S["cover_sub"]))
    story.append(Paragraph(
        "1. Resumen ejecutivo — 2. Definición del problema y relevancia — "
        "3. Datos y metodología — 4. Modelado y evaluación — "
        "5. Interpretación del modelo — 6. Impacto en el negocio, riesgos y ética — "
        "7. Conclusiones y recomendaciones — Anexos",
        S["idx"],
    ))
    story.append(PageBreak())

    # -------- 1. Resumen --------
    story.append(Paragraph("1. Resumen ejecutivo", S["h1"]))
    story.append(Paragraph(
        f"<b>Problema.</b> Un emisor de tarjetas (audiencia: Prevención de Fraude) debe "
        f"decidir qué operaciones revisar a mano. El dataset ULB tiene "
        f"<b>284.807</b> transacciones y solo <b>492</b> fraudes (0,172 %). "
        f"Con tan pocos positivos, un FPR del 5 % sobre las ~{n_neg:,} legítimas del "
        f"holdout produciría unas <b>{fpr5_alerts:,} alertas falsas</b>. No alcanza con "
        f"controlar FPR: hay que mirar Precision y cuántas operaciones revisaría el equipo.",
        S["body"],
    ))
    story.append(Paragraph(
        f"<b>Solución.</b> Un modelo candidato <b>{best_name}</b> que ordena las "
        f"transacciones por score de riesgo. El producto no es un “sí/no” rígido, sino "
        f"una cola de revisión: el equipo mira primero las operaciones de mayor riesgo. "
        f"La selección se basó en PR-AUC, Recall y Precision, y se contrastó un split "
        f"aleatorio estratificado con una validación temporal por <i>Time</i>.",
        S["body"],
    ))

    story.append(table(
        [
            ["Indicador (holdout)", "Azar / sin modelo", f"{best_name} aleatorio", f"{best_t} temporal"],
            ["PR-AUC", "≈ 0,002 (tasa base)", num(m50["pr_auc"]), num(m50_t["pr_auc"])],
            ["Recall (umbral 0,5)", "—", pct(m50["recall"]), pct(m50_t["recall"])],
            ["Precision (umbral 0,5)", "—", pct(m50["precision"]), pct(m50_t["precision"])],
            ["Fraudes en Top-100", f"≈ {max(1, int(100 * n_pos / (n_pos + n_neg)))}", str(cap[100]["frauds_found"]), str(cap_t[100]["frauds_found"])],
            ["Fraudes en Top-500", f"≈ {max(1, int(500 * n_pos / (n_pos + n_neg)))}", str(cap[500]["frauds_found"]), str(cap_t[500]["frauds_found"])],
            ["Alertas con umbral operativo", "miles (si FPR=5 %)", str(msel["alerts"]), "ver §4.4"],
        ],
        col_widths=[5.0 * cm, 3.6 * cm, 3.7 * cm, 3.7 * cm],
    ))
    story.append(Paragraph(
        "Tabla 1. Lectura de negocio: captura de fraude con el mismo esfuerzo de revisión.",
        S["caption"],
    ))

    story.append(Paragraph(
        f"<b>Resultado.</b> En holdout aleatorio ({n_pos} fraudes en {rand['meta']['test_n']:,} "
        f"operaciones), {best_name} alcanza PR-AUC {num(m50['pr_auc'])}, Recall "
        f"{pct(m50['recall'])} y Precision {pct(m50['precision'])}. Si el equipo revisa "
        f"las 100 operaciones de mayor riesgo encuentra <b>{cap[100]['frauds_found']}</b> "
        f"fraudes (Precision@100 = {pct(cap[100]['precision_at_k'])}); con 500 revisiones, "
        f"<b>{cap[500]['frauds_found']}</b> fraudes (lift {num(cap[500]['lift'], 1)}×). "
        f"Con umbral {msel['threshold']:.2f} emite solo <b>{msel['alerts']} alertas</b> "
        f"({msel['tp']} TP / {msel['fp']} FP).",
        S["body"],
    ))
    story.append(Paragraph(
        f"<b>Límite y recomendación.</b> En validación temporal el mejor modelo es "
        f"{best_t} (PR-AUC {num(m50_t['pr_auc'])}; Top-100 = {cap_t[100]['frauds_found']} "
        f"fraudes). La señal se sostiene, pero el split aleatorio es algo más optimista. "
        f"Se recomienda usar el score para priorizar la cola, fijar el umbral según la "
        f"capacidad del equipo y revalidar en ventanas temporales nuevas antes de un "
        f"despliegue real.",
        S["body"],
    ))

    # -------- 2. Problema --------
    story.append(Paragraph("2. Definición del problema y relevancia", S["h1"]))
    story.append(Paragraph(
        "La pregunta de negocio no es “¿esta transacción es fraude con certeza?”, sino "
        "<b>a qué operaciones asignar primero un cupo limitado de revisión humana</b>. "
        "El producto es un ranking. La relevancia es directa: cada fraude no detectado "
        "es pérdida financiera y cada falsa alarma consume analista y puede bloquear "
        "a un cliente legítimo.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Objetivo medible (fijado en la pre-entrega y ajustado por la devolución):</b> "
        "maximizar Recall de fraude manteniendo Precision y un volumen de alertas "
        "operable. Métricas principales: PR-AUC, Recall y Precision. Métrica de negocio: "
        "captura en Top-100 y Top-500. Benchmark: un orden aleatorio captura, en el "
        f"holdout, unos {max(1, int(100 * n_pos / (n_pos + n_neg)))} fraudes entre 100 "
        "revisiones; el modelo debe multiplicar ese lift.",
        S["body"],
    ))
    story.append(table(
        [
            ["Situación", "Lectura estadística", "Costo de negocio"],
            ["Alerta sobre operación legítima", "Falso positivo", "Horas de analista + fricción al cliente"],
            ["Fraude que no entra a la cola", "Falso negativo", "Pérdida del monto y daño reputacional"],
            ["FPR ≤ 5 % como única traba", "Miles de FP con 0,17 % de prevalencia", "El equipo no puede revisar esa cola"],
        ],
        col_widths=[5.4 * cm, 5.3 * cm, 5.3 * cm],
    ))
    story.append(Paragraph("Tabla 2. Costos del error. El umbral es una decisión de capacidad, no un default técnico.", S["caption"]))
    story.append(Paragraph(
        "<b>Hipótesis.</b> H1: los componentes PCA (V1–V28) concentran señal discriminativa. "
        "H2: el monto, transformado a escala logarítmica, aporta menos que las V* pero "
        "ayuda a calibrar riesgo. H3: como existe <i>Time</i>, un split aleatorio puede "
        "sobreestimar el desempeño; entrenar con operaciones anteriores y probar con "
        "posteriores es la prueba más realista.",
        S["body"],
    ))

    # -------- 3. Datos --------
    story.append(Paragraph("3. Datos y metodología", S["h1"]))
    story.append(Paragraph("3.1 Fuente y calidad", S["h2"]))
    story.append(Paragraph(
        "Fuente. Credit Card Fraud Detection, ULB Machine Learning Group / Kaggle "
        "(Dal Pozzolo, Caelen, Johnson y Bontempi, 2015). Licencia ODbL v1.0. "
        "Transacciones de titulares europeos en septiembre de 2013 (dos días). "
        "Archivo original: 284.807 filas × 31 columnas. Tasa de fraude: 0,172 %. "
        "El CSV se descarga de un mirror público y se cachea en <i>data/creditcard.csv</i>; "
        "el esquema se valida al cargar (Time, V1–V28, Amount, Class).",
        S["body"],
    ))
    story.append(table(
        [
            ["Hallazgo de calidad", "Decisión y motivo"],
            ["0 nulos", "Ninguna imputación."],
            [f"{data['duplicates_removed']} duplicados exactos", f"Eliminados; quedan {eda['n_rows']:,} filas y {eda['n_fraud']} fraudes."],
            ["Amount sesgado (media €88; mediana €22)", "Amount_log = log1p(Amount); el crudo no entra al modelo."],
            ["V1–V28 ya son PCA anonimizado", "Se usan sin reescalar; limitan interpretación causal."],
            ["Time en [0 ; 172.792] s", "Se escala en el Pipeline y se usa para el split temporal."],
            ["Desbalance 0,17 %", "class_weight / scale_pos_weight / SMOTE dentro del Pipeline."],
        ],
        col_widths=[7.5 * cm, 8.5 * cm],
    ))
    story.append(Paragraph("Tabla 3. Decisiones de calidad registradas (equivalente a un data log).", S["caption"]))

    story.append(fig("01_class_distribution.png", max_w=12 * cm, max_h=5.2 * cm))
    story.append(Paragraph("Figura 1. Distribución de clases tras la limpieza: el positivo es raro.", S["caption"]))

    story.append(Paragraph("3.2 Conclusiones del EDA", S["h2"]))
    story.append(Paragraph(
        f"Las variables con mayor |correlación| con Class son V17 ({num(eda['top_corr']['V17'], 2)}), "
        f"V14 ({num(eda['top_corr']['V14'], 2)}) y V12 ({num(eda['top_corr']['V12'], 2)}). "
        "Amount por sí solo no separa bien: hay fraudes en montos bajos y medios. "
        "Time muestra concentraciones distintas por clase, lo que respalda H3. "
        "Accuracy o ROC-AUC aislados son engañosos: un clasificador trivial “todo legítimo” "
        "tendría Accuracy 99,83 % y no detectaría ningún fraude.",
        S["body"],
    ))
    story.append(fig("04_corr_with_class.png", max_w=12 * cm, max_h=5.4 * cm))
    story.append(Paragraph("Figura 2. |Correlación| absoluta con Class (top 10). Respaldo descriptivo de H1.", S["caption"]))

    story.append(Paragraph("3.3 Preparación y CRISP-DM", S["h2"]))
    story.append(Paragraph(
        "Metodología. Se siguió CRISP-DM: negocio (sección 2), comprensión y preparación "
        "de datos (esta sección), modelado y evaluación (sección 4) y traducción a una "
        "decisión (secciones 5–6). Todas las transformaciones viven en un Pipeline de "
        "scikit-learn / imbalanced-learn: StandardScaler sobre Time y Amount_log; "
        "V1–V28 pasan sin cambios; SMOTE, cuando se usa, queda <b>dentro</b> del "
        "pipeline para no filtrar información del test. random_state=42.",
        S["body"],
    ))
    story.append(table(
        [
            ["Diseño de validación", "Pregunta que responde"],
            ["Holdout estratificado 80/20 (una sola vez)", "¿Cuánto rinde en una mezcla i.i.d. de las 48 h?"],
            ["Split temporal por Time (80 % anterior → 20 % posterior)", "¿Generaliza hacia operaciones futuras?"],
            ["Comparación de cinco algoritmos en ambos esquemas", "¿Qué modelo discrimina mejor (PR-AUC) sin saturar alertas?"],
        ],
        col_widths=[8.2 * cm, 7.8 * cm],
    ))
    story.append(Paragraph("Tabla 4. Dos diseños de validación, como pidió la devolución del tutor.", S["caption"]))

    # -------- 4. Modelado --------
    story.append(Paragraph("4. Modelado y evaluación", S["h1"]))
    story.append(Paragraph("4.1 Algoritmos y regla de selección", S["h2"]))
    story.append(Paragraph(
        "Se compararon cinco configuraciones, todas con manejo explícito del desbalance: "
        "Regresión Logística (baseline, class_weight=balanced), Random Forest "
        "(class_weight), XGBoost (scale_pos_weight), LightGBM (is_unbalance) y "
        "XGBoost+SMOTE dentro del pipeline. <b>Regla de selección definida antes de "
        "elegir el umbral:</b> gana el mayor PR-AUC en el holdout aleatorio; el desempate "
        "lo dan Recall, Precision y el volumen de alertas. El split temporal no cambia "
        "la regla: sirve para medir robustez y avisar si el candidato se degrada.",
        S["body"],
    ))

    rows = [["Modelo", "PR-AUC", "Recall", "Precision", "F1", "Alertas", "Fraud@100", "Fraud@500"]]
    for name, m in rand["models"].items():
        m0 = m["metrics_thr_0.5"]
        c = {r["k"]: r for r in m["captura_at_k"]}
        mark = f"{name} *" if name == best_name else name
        rows.append([
            mark,
            num(m0["pr_auc"]),
            pct(m0["recall"]),
            pct(m0["precision"]),
            num(m0["f1"]),
            str(m0["alerts"]),
            str(c[100]["frauds_found"]),
            str(c[500]["frauds_found"]),
        ])
    story.append(table(rows, col_widths=[3.4*cm, 1.7*cm, 1.8*cm, 2.0*cm, 1.5*cm, 1.6*cm, 1.9*cm, 2.0*cm]))
    story.append(Paragraph(
        f"Tabla 5. Holdout aleatorio, umbral 0,5. *Candidato: {best_name}. Test: "
        f"{rand['meta']['test_n']:,} filas / {n_pos} fraudes.",
        S["caption"],
    ))

    story.append(Paragraph(
        f"LightGBM obtiene Recall razonable ({pct(rand['models']['LightGBM']['metrics_thr_0.5']['recall'])}) "
        f"pero PR-AUC {num(rand['models']['LightGBM']['metrics_thr_0.5']['pr_auc'])}: "
        "el ranking es pobre y Top-100 apenas captura 3 fraudes. XGBoost+SMOTE sube "
        f"Recall a {pct(rand['models']['XGBoost_SMOTE']['metrics_thr_0.5']['recall'])} "
        f"pero cae la Precision a {pct(rand['models']['XGBoost_SMOTE']['metrics_thr_0.5']['precision'])} "
        f"({rand['models']['XGBoost_SMOTE']['metrics_thr_0.5']['alerts']} alertas). "
        "La logística detecta mucho (Recall alto) a costa de 1.499 alertas. "
        f"Random Forest es preciso pero recupera menos fraude. {best_name} gana el "
        "equilibrio que pide el negocio: PR-AUC 0,827, 89 alertas y 77 fraudes en Top-100.",
        S["body"],
    ))

    story.append(fig("06_pr_curve_aleatorio.png", max_w=10.5 * cm, max_h=7.2 * cm))
    story.append(Paragraph(
        f"Figura 3. Curva Precision-Recall de {best_name} en el holdout aleatorio (AP = 0,83).",
        S["caption"],
    ))

    story.append(Paragraph("4.2 Métrica de negocio: captura Top-K", S["h2"]))
    story.append(Paragraph(
        f"Si el equipo revisa las 100 operaciones de mayor score encuentra "
        f"<b>{cap[100]['frauds_found']}</b> de los {n_pos} fraudes del holdout "
        f"(Recall@100 = {pct(cap[100]['recall_at_k'])}; Precision@100 = "
        f"{pct(cap[100]['precision_at_k'])}). Ampliar a 500 revisiones suma solo "
        f"{cap[500]['frauds_found'] - cap[100]['frauds_found']} fraudes más "
        f"(lift {num(cap[500]['lift'], 1)}× sobre la tasa base de {pct(cap[500]['base_rate'], 3)}). "
        "La curva se aplana: el valor está en el tope del ranking, no en revisar más y más.",
        S["body"],
    ))
    story.append(fig("08_captura_topk_aleatorio.png", max_w=13.5 * cm, max_h=5.8 * cm))
    story.append(Paragraph(
        "Figura 4. Fraudes encontrados vs operaciones revisadas (Top-K). Líneas en K=100 y K=500.",
        S["caption"],
    ))

    story.append(Paragraph("4.3 Umbral operativo y carga de revisión", S["h2"]))
    story.append(Paragraph(
        f"Un FPR del 5 % sobre {n_neg:,} legítimas implica ~{fpr5_alerts:,} falsos "
        f"positivos en un solo holdout. Por eso el umbral no se fija por FPR. "
        f"Regla usada: Precision ≥ 50 % y tope de alertas cercano a 100–500 "
        f"(capacidad de un turno). El umbral {msel['threshold']:.2f} deja "
        f"<b>{msel['alerts']} alertas</b>, Recall {pct(msel['recall'])} y Precision "
        f"{pct(msel['precision'])} (TP={msel['tp']}, FP={msel['fp']}, FN={msel['fn']}). "
        "Esa es la carga real que tendría que revisar el equipo.",
        S["body"],
    ))
    story.append(fig("09_workload_aleatorio.png", max_w=13.5 * cm, max_h=5.8 * cm))
    story.append(Paragraph(
        "Figura 5. Precision y Recall en función del número de alertas. El recorte operativo está a la izquierda.",
        S["caption"],
    ))
    story.append(fig("07_confusion_aleatorio.png", max_w=9.5 * cm, max_h=6.5 * cm))
    story.append(Paragraph(
        f"Figura 6. Matriz de confusión de {best_name} con umbral {msel['threshold']:.2f}.",
        S["caption"],
    ))

    story.append(Paragraph("4.4 Validación temporal versus aleatoria", S["h2"]))
    story.append(Paragraph(
        f"Se ordenó por Time y se entrenó con las primeras {temp['meta']['train_n']:,} "
        f"operaciones ({temp['meta']['train_fraud']} fraudes; Time ≤ "
        f"{temp['meta']['train_time_max']:.0f} s) y se evaluó con las "
        f"{temp['meta']['test_n']:,} posteriores ({temp['meta']['test_fraud']} fraudes). "
        "Es el diseño que pide la devolución y el más cercano a producción: el modelo "
        "no ve el “futuro” de las 48 horas.",
        S["body"],
    ))

    rows_t = [["Modelo", "PR-AUC", "Recall", "Precision", "Fraud@100", "Fraud@500"]]
    for name, m in temp["models"].items():
        m0 = m["metrics_thr_0.5"]
        c = {r["k"]: r for r in m["captura_at_k"]}
        mark = f"{name} *" if name == best_t else name
        rows_t.append([
            mark, num(m0["pr_auc"]), pct(m0["recall"]), pct(m0["precision"]),
            str(c[100]["frauds_found"]), str(c[500]["frauds_found"]),
        ])
    story.append(table(rows_t, col_widths=[3.6*cm, 2.2*cm, 2.2*cm, 2.4*cm, 2.6*cm, 2.6*cm]))
    story.append(Paragraph(
        f"Tabla 6. Holdout temporal. *Mejor PR-AUC: {best_t}. Test: "
        f"{temp['meta']['test_fraud']} fraudes.",
        S["caption"],
    ))
    story.append(fig("11_comparison_pr_auc.png", max_w=15 * cm, max_h=5.6 * cm))
    story.append(Paragraph(
        "Figura 7. PR-AUC por modelo en ambos esquemas. LightGBM se cae en ranking; el resto sostiene la señal.",
        S["caption"],
    ))
    story.append(Paragraph(
        f"Lectura. {best_name} sigue siendo competitivo en temporal (PR-AUC "
        f"{num(temp['models']['XGBoost']['metrics_thr_0.5']['pr_auc'])}), pero "
        f"{best_t} lo supera levemente y con Precision {pct(m50_t['precision'])}. "
        f"La captura Top-100 baja de {cap[100]['frauds_found']} a "
        f"{cap_t[100]['frauds_found']} fraudes, en parte porque el test temporal "
        f"tiene menos positivos ({temp['meta']['test_fraud']} vs {n_pos}). "
        "H3 queda respaldada: el aleatorio no miente, pero es el techo; el temporal "
        "es el piso más honesto para decidir un piloto.",
        S["body"],
    ))

    # -------- 5. Interpretación --------
    story.append(Paragraph("5. Interpretación del modelo", S["h1"]))
    story.append(Paragraph(
        f"Sobre {best_name} se leyeron importancias de split (no son causalidad ni "
        "SHAP). Las V* más usadas coinciden con el EDA (V14, V17, V12 y vecinas). "
        "Time y Amount_log aportan, pero no lideran. Como las V* son PCA, el negocio "
        "no puede traducir “sube V14” a una regla de comercio o país: el modelo "
        "prioriza, no explica el fraude ante un auditor. En un emisor real habría que "
        "complementar con features interpretables (MCC, dispositivo, geo, historial).",
        S["body"],
    ))
    story.append(fig("10_importance_aleatorio.png", max_w=12.5 * cm, max_h=6.0 * cm))
    story.append(Paragraph(
        f"Figura 8. Importancia de variables de {best_name} (split aleatorio). Lectura descriptiva, no causal.",
        S["caption"],
    ))
    story.append(Paragraph(
        "El score se usa para <b>ordenar</b>. Un valor 0,80 no se interpreta aquí como "
        "“probabilidad calibrada del 80 %”: no se reporta curva de calibración. "
        "Para un umbral de corte (sección 4.3) alcanza con el ranking y la matriz de "
        "confusión. Si el área quisiera comunicar “1 de cada 5 alertas es fraude”, "
        "habría que calibrar (Platt / isotónica) sobre un conjunto aparte.",
        S["body"],
    ))

    # -------- 6. Impacto --------
    story.append(Paragraph("6. Impacto en el negocio, riesgos y ética", S["h1"]))
    story.append(Paragraph(
        "Sin ranking, capturar ~81 % de los fraudes del holdout exigiría revisar una "
        "fracción enorme de la cartera o convivir con miles de alertas. Con el modelo, "
        f"ese Recall se obtiene revisando {msel['alerts']} operaciones, o "
        f"{cap[100]['frauds_found']} fraudes con solo 100 revisiones. El ahorro no es "
        "un monto de 2013 “real”: es <b>capacidad de analista liberada</b> y fraudes "
        "que entran a la cola a tiempo. Un FPR del 5 % costaría ~"
        f"{fpr5_alerts:,} revisiones falsas; el umbral elegido cuesta {msel['fp']} FP.",
        S["body"],
    ))
    story.append(table(
        [
            ["Escenario de revisión", "Operaciones a mirar", "Fraudes hallados", "Precision"],
            ["Azar (100 casos)", "100", f"≈ {max(1, int(100 * n_pos / (n_pos + n_neg)))}", "≈ tasa base"],
            ["Modelo Top-100", "100", str(cap[100]["frauds_found"]), pct(cap[100]["precision_at_k"])],
            ["Modelo Top-500", "500", str(cap[500]["frauds_found"]), pct(cap[500]["precision_at_k"])],
            [f"Umbral {msel['threshold']:.2f}", str(msel["alerts"]), str(msel["tp"]), pct(msel["precision"])],
            ["Regla FPR = 5 %", f"~{fpr5_alerts:,}+", "alto Recall, baja Precision", "muy baja"],
        ],
        col_widths=[4.2 * cm, 4.0 * cm, 4.0 * cm, 3.8 * cm],
    ))
    story.append(Paragraph("Tabla 7. El valor está en ordenar la cola, no en umbrales holgados de FPR.", S["caption"]))

    story.append(table(
        [
            ["Riesgo", "Mitigación"],
            ["Dataset de solo 2 días (2013), Europa", "Tratar el caso como prueba de método; reentrenar con datos propios del emisor."],
            ["Drift de patrones de fraude", "Monitorear PR-AUC, Precision y captura@K por día; reentrenar."],
            ["PCA poco explicable ante auditoría", "Usar el score para priorizar; complementar con reglas y features nativas."],
            ["Falso positivo = bloqueo de cliente", "Umbral según capacidad; canal de apelación; no automatizar el rechazo final."],
            ["Fuga si SMOTE o scaler ven el test", "Pipeline único; holdout separado antes de cualquier ajuste."],
            ["Privacidad", "Datos ya anonimizados; minimización; acceso need-to-know al score."],
        ],
        col_widths=[6.2 * cm, 9.8 * cm],
    ))
    story.append(Paragraph("Tabla 8. Riesgos y mitigaciones. El prototipo no autoriza un despliegue automático.", S["caption"]))
    story.append(Paragraph(
        "<b>Ética.</b> No hay PII. El umbral es una decisión explícita de negocio "
        "(Precision y cupo), no un default de 0,5. Un falso negativo pierde dinero; "
        "un falso positivo daña al cliente: por eso se reportan ambos y el volumen "
        "absoluto de alertas. Uso académico bajo ODbL. Cualquier producción requeriría "
        "supervisión humana, registro de cada alerta y revisión legal del modelo de riesgo.",
        S["body"],
    ))

    # -------- 7. Conclusiones --------
    story.append(Paragraph("7. Conclusiones y recomendaciones", S["h1"]))
    for b in [
        f"El problema se formuló como <b>priorización bajo capacidad limitada</b>. PR-AUC, Recall, Precision y captura Top-K reflejan esa decisión mejor que Accuracy o un FPR suelto.",
        f"<b>{best_name}</b> es el candidato en distribución mezclada: PR-AUC {num(m50['pr_auc'])}, Recall {pct(m50['recall'])}, Precision {pct(m50['precision'])}; Top-100 = {cap[100]['frauds_found']} fraudes; umbral {msel['threshold']:.2f} → {msel['alerts']} alertas.",
        f"H1 tiene respaldo descriptivo (V17, V14, V12). H3 también: en temporal gana {best_t} (PR-AUC {num(m50_t['pr_auc'])}) y la captura Top-100 baja a {cap_t[100]['frauds_found']}. El aleatorio es techo; el temporal es la prueba de piloto.",
        "SMOTE y LightGBM (is_unbalance) no se eligen: mejoran o sostienen Recall pero destruyen Precision o PR-AUC.",
        "Recomendación: <b>no desplegar a ciegas</b>. Usar el score para ordenar la cola, fijar el umbral según el turno de analistas, reentrenar con datos actuales del emisor y medir captura@100/@500 y drift en una ventana temporal nueva.",
    ]:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Paragraph(
        "Código. Repositorio del Grupo 33: notebook "
        "<i>notebooks/TP_Final_Deteccion_Fraude.ipynb</i>, módulo <i>src/</i>, "
        "<i>requirements.txt</i> y <i>reports/results.json</i>. El pipeline se reproduce con "
        "<i>python -m src.run_pipeline</i>.",
        S["body"],
    ))

    # -------- Anexos (siguen al cuerpo; consigna permite anexos fuera del tope) --------
    story.append(PageBreak())
    story.append(Paragraph("Anexos", S["h1"]))
    story.append(Paragraph("Anexo A. Trazabilidad con la consigna, la pre-entrega y la devolución", S["h2"]))
    story.append(table(
        [
            ["Peso", "Criterio", "Dónde está en este informe"],
            ["20 %", "Problema y negocio", "§1–2: ranking, costos FP/FN, KPI Top-K"],
            ["20 %", "Datos y metodología", "§3: calidad, EDA, Pipeline, dos validaciones"],
            ["15 %", "Selección del modelo", "§4.1 y Tablas 5–6"],
            ["15 %", "Calidad de métricas", "PR-AUC, Recall, Precision, Top-K, alertas"],
            ["15 %", "Interpretación y valor", "§5–6: importancias y carga de revisión"],
            ["5 %", "Ética y privacidad", "§6 y Tabla 8"],
            ["10 %", "Claridad", "Resumen no técnico, figuras y recomendación"],
        ],
        col_widths=[2.0 * cm, 5.0 * cm, 9.0 * cm],
    ))
    story.append(Paragraph("Tabla A1. Cobertura de los criterios de evaluación.", S["caption"]))
    story.append(table(
        [
            ["Pedido de la devolución", "Estado"],
            ["PR-AUC, Recall y Precision como métricas principales", "Tablas 1, 5 y 6; no se decide por FPR."],
            ["Cuántas operaciones revisaría el equipo", "§4.3: 88 alertas vs ~2.832 si FPR=5 %."],
            ["Captura si se revisan Top-100 / Top-500", "Figura 4 y Tabla 7."],
            ["Comparar split aleatorio vs temporal por Time", "§4.4, Figura 7, Tabla 6."],
        ],
        col_widths=[8.0 * cm, 8.0 * cm],
    ))
    story.append(Paragraph("Tabla A2. Cierre de la devolución del tutor.", S["caption"]))

    story.append(Paragraph("Anexo B. Figuras complementarias de EDA", S["h2"]))
    story.append(fig("02_amount_by_class.png", max_w=14.5 * cm, max_h=5.4 * cm))
    story.append(Paragraph("Figura B1. Amount por clase. Las legítimas se muestran con eje Y en log.", S["caption"]))
    story.append(fig("03_time_by_class.png", max_w=14.5 * cm, max_h=5.2 * cm))
    story.append(Paragraph("Figura B2. Densidad de Time por clase: hay estructura temporal (H3).", S["caption"]))
    story.append(fig("05_boxplots_by_class.png", max_w=14.5 * cm, max_h=5.2 * cm))
    story.append(Paragraph("Figura B3. Boxplots de V14, V17 y V12 por clase.", S["caption"]))
    story.append(fig("08_captura_topk_temporal.png", max_w=13.5 * cm, max_h=5.4 * cm))
    story.append(Paragraph(
        f"Figura B4. Captura Top-K en el split temporal ({best_t}).",
        S["caption"],
    ))

    story.append(Paragraph("Anexo C. Reproducibilidad y fuentes", S["h2"]))
    story.append(Paragraph(
        "Dal Pozzolo, A., Caelen, O., Johnson, R. A. y Bontempi, G. (2015). "
        "<i>Calibrating Probability with Undersampling for Unbalanced Classification</i>. "
        "IEEE CIDM. Dataset: Credit Card Fraud Detection (ULB). Licencia ODbL v1.0. "
        "Chapman, P. et al. (2000). <i>CRISP-DM 1.0</i>. "
        "Repositorio del equipo: https://github.com/SolKidonakis/TP_Final_DataScience",
        S["body"],
    ))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=1.8 * cm,
        title="Informe Final — Detección de Fraude — Grupo 33",
        author="Grupo 33",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe: {OUT}")


def main():
    build()
    from pypdf import PdfReader

    n = len(PdfReader(str(OUT)).pages)
    print(f"Páginas: {n} (cuerpo + anexos; tope de consigna = 12 sin anexos)")
    parent = ROOT.parent
    shutil.copy2(OUT, parent / "Informe_Final_Grupo33.pdf")
    print(f"Copia: {parent / 'Informe_Final_Grupo33.pdf'}")


if __name__ == "__main__":
    main()
