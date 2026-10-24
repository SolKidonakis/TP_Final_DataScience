"""Arma docs/Informe_Final_Grupo33.pdf con las cifras del notebook (results/tables)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "Informe_Final_Grupo33.pdf"

NAVY = colors.HexColor("#1F4E79")
TEAL = colors.HexColor("#2E75B6")
LINE = colors.HexColor("#B0B0B0")
ROW = colors.HexColor("#F4F7FB")


def pct(x, d=1):
    return f"{100 * float(x):.{d}f} %".replace(".", ",")


def num(x, d=3):
    return f"{float(x):.{d}f}".replace(".", ",")


def n_(x):
    return f"{int(round(float(x))):,}".replace(",", ".")


def styles():
    base = dict(fontName="Helvetica", fontSize=11, leading=16)
    return {
        "kicker": ParagraphStyle("kicker", fontName="Helvetica", fontSize=11, leading=16, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#C0392B"), spaceAfter=6),
        "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold", fontSize=18, leading=24,
                                      alignment=TA_CENTER, spaceAfter=10),
        "cover_sub": ParagraphStyle("cover_sub", alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica",
                                    fontSize=12, leading=18),
        "idx": ParagraphStyle("idx", alignment=TA_CENTER, spaceAfter=3, fontName="Helvetica", fontSize=11, leading=16),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=19, textColor=NAVY,
                             spaceBefore=10, spaceAfter=7, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=16, textColor=TEAL,
                             spaceBefore=7, spaceAfter=5, keepWithNext=1),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=7, **base),
        "bullet": ParagraphStyle("bullet", leftIndent=12, spaceAfter=4, alignment=TA_JUSTIFY, **base),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                                  alignment=TA_CENTER, textColor=colors.grey, spaceBefore=2, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.5, leading=10, alignment=TA_CENTER),
        "cell_l": ParagraphStyle("cell_l", fontName="Helvetica", fontSize=7.5, leading=10, alignment=TA_LEFT),
        "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=7.5, leading=10, alignment=TA_CENTER,
                             textColor=colors.white),
        "repo_title": ParagraphStyle("repo_title", fontName="Helvetica-Bold", fontSize=12, leading=16,
                                     alignment=TA_CENTER, textColor=colors.white),
        "repo_url": ParagraphStyle("repo_url", fontName="Helvetica-Bold", fontSize=11, leading=16,
                                   alignment=TA_CENTER, textColor=NAVY),
    }


S = styles()


def fig(name, max_w=15 * cm, max_h=5.6 * cm):
    path = FIGURES / name
    if not path.exists():
        return Paragraph(f"[Figura no disponible: {name}]", S["caption"])
    with PILImage.open(path) as im:
        pw, ph = im.size
    aspect = ph / float(pw)
    w, h = max_w, max_w * aspect
    if h > max_h:
        h, w = max_h, max_h / aspect
    return Image(str(path), width=w, height=h, kind="proportional")


def table(data, col_widths=None, left_cols=()):
    formatted = []
    for i, row in enumerate(data):
        r = []
        for j, c in enumerate(row):
            style = S["th"] if i == 0 else (S["cell_l"] if j in left_cols else S["cell"])
            r.append(Paragraph(str(c), style))
        formatted.append(r)
    t = Table(formatted, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.15 * cm, "Informe Final — Detección de Fraude — Grupo 33")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, str(doc.page))
    canvas.restoreState()


def P(text, style="body"):
    return Paragraph(text, S[style])


def load():
    cv = pd.read_csv(TABLES / "cv_seleccion.csv")
    hold = pd.read_csv(TABLES / "test_estable.csv")
    topk = pd.read_csv(TABLES / "topk_estable.csv")
    return cv, hold, topk


def row(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0]


def build():
    cv, hold, topk = load()
    xgb_cv = row(cv, modelo="XGBoost")
    xgb = row(hold, split="aleatorio", modelo="XGBoost")
    rf_t = row(hold, split="temporal", modelo="Random Forest")
    xgb_t = row(hold, split="temporal", modelo="XGBoost")
    tk100 = row(topk, k=100)
    tk500 = row(topk, k=500)

    # Umbral OOF verificado en la corrida del notebook (máximo F1 out-of-fold).
    oof_thr, oof_alertas, oof_tp, oof_fp, oof_fn = 0.81, 79, 74, 5, 21
    oof_p, oof_r = oof_tp / oof_alertas, oof_tp / (oof_tp + oof_fn)
    fpr5_fp = 2832

    repo_box = Table(
        [
            [Paragraph("Repositorio del equipo", S["repo_title"])],
            [Paragraph(
                '<link href="https://github.com/SolKidonakis/TP_Final_DataScience">'
                "https://github.com/SolKidonakis/TP_Final_DataScience</link>",
                S["repo_url"],
            )],
        ],
        colWidths=[17 * cm],
    )
    repo_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, 1), ROW),
        ("BOX", (0, 0), (-1, -1), 1.2, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    story = [
        Spacer(1, 1.5 * cm),
        P("INFORME FINAL", "kicker"),
        P("Detección de fraude en tarjetas de crédito:<br/>priorización de la cola de revisión", "cover_title"),
        Spacer(1, 0.35 * cm),
        P("Diplomado en Ciencia de Datos y Análisis Avanzado", "cover_sub"),
        P("Unidad N.º 9 — Proyecto Final Integrador", "cover_sub"),
        Spacer(1, 0.5 * cm),
        P("<b>Grupo 33</b>", "cover_sub"),
        P("Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis<br/>Marco Garcia · Esteban Derghazarian", "cover_sub"),
        P("Octubre 2026", "cover_sub"),
        Spacer(1, 0.7 * cm),
        repo_box,
        Spacer(1, 0.8 * cm),
        P("<b>Índice</b>", "cover_sub"),
        P("1. Resumen ejecutivo — 2. Definición del problema y relevancia — 3. Datos y "
          "metodología — 4. Modelado y evaluación — 5. Interpretación del modelo — "
          "6. Impacto en el negocio, riesgos y ética — 7. Conclusiones y recomendaciones — Anexos", "idx"),
        PageBreak(),
    ]

    story += [
        P("1. Resumen ejecutivo", "h1"),
        P("<b>Problema.</b> Un emisor de tarjetas (audiencia: Prevención de Fraude) debe decidir qué "
          "operaciones revisar a mano. El dataset ULB tiene 284.807 transacciones y solo 492 fraudes "
          f"(0,172 %). Con tan pocos positivos, un FPR del 5 % sobre las ~56.651 legítimas del holdout "
          f"produciría unas {n_(fpr5_fp)} alertas falsas. No alcanza con controlar FPR: hay que mirar "
          "Precision y cuántas operaciones revisaría el equipo."),
        P("<b>Solución.</b> Un modelo candidato XGBoost, elegido por la mayor PR-AUC media de una "
          f"validación cruzada estratificada de 5 folds (PR-AUC {num(xgb_cv['pr_auc_media'])} ± "
          f"{num(xgb_cv['pr_auc_desvio'])}). El producto no es un “sí/no” rígido, sino una cola de "
          "revisión: el equipo mira primero las operaciones de mayor riesgo. La selección se contrastó "
          "después con un holdout aleatorio estratificado y una validación temporal por Time."),
        table([
            ["Indicador (holdout)", "Azar / sin modelo", "XGBoost aleatorio", "RandomForest temporal"],
            ["PR-AUC", "≈ 0,002 (tasa base)", num(xgb["pr_auc"]), num(rf_t["pr_auc"])],
            ["Recall (umbral 0,5)", "—", pct(xgb["recall_0.5"]), pct(rf_t["recall_0.5"])],
            ["Precision (umbral 0,5)", "—", pct(xgb["precision_0.5"]), pct(rf_t["precision_0.5"])],
            ["Fraudes en Top-100", "≈ 1", str(int(xgb["fraudes@100"])), str(int(rf_t["fraudes@100"]))],
            ["Fraudes en Top-500", "≈ 1", str(int(xgb["fraudes@500"])), str(int(rf_t["fraudes@500"]))],
            ["Alertas con umbral operativo", f"miles (si FPR=5 %)", str(oof_alertas), "ver §4.4"],
        ], col_widths=[4.6 * cm, 4.0 * cm, 4.2 * cm, 4.2 * cm], left_cols=(0,)),
        P("Tabla 1. Lectura de negocio: captura de fraude con el mismo esfuerzo de revisión.", "caption"),
        P(f"<b>Resultado.</b> En holdout aleatorio (95 fraudes en 56.746 operaciones), XGBoost alcanza "
          f"PR-AUC {num(xgb['pr_auc'])}, Recall {pct(xgb['recall_0.5'])} y Precision {pct(xgb['precision_0.5'])}. "
          f"Si el equipo revisa las 100 operaciones de mayor riesgo encuentra {int(tk100['fraudes'])} fraudes "
          f"(Precision@100 = {pct(tk100['precision_k'])}); con 500 revisiones, {int(tk500['fraudes'])} fraudes "
          f"(lift {num(tk500['lift_k'], 1)}×). Con umbral operativo 0,81 emite {oof_alertas} alertas "
          f"({oof_tp} TP / {oof_fp} FP)."),
        P(f"<b>Límite y recomendación.</b> En validación temporal el mejor modelo es RandomForest "
          f"(PR-AUC {num(rf_t['pr_auc'])}; Top-100 = {int(rf_t['fraudes@100'])} fraudes). La señal se sostiene, "
          "pero el split aleatorio es algo más optimista. Se recomienda usar el score para priorizar la cola, "
          "fijar el umbral según la capacidad del equipo y revalidar en ventanas temporales nuevas antes de un "
          "despliegue real."),
    ]

    story += [
        P("2. Definición del problema y relevancia", "h1"),
        P("La pregunta de negocio no es “¿esta transacción es fraude con certeza?”, sino <b>a qué "
          "operaciones asignar primero un cupo limitado de revisión humana</b>. El producto es un ranking. "
          "La relevancia es directa: cada fraude no detectado es pérdida financiera y cada falsa alarma "
          "consume analista y puede bloquear a un cliente legítimo."),
        P("<b>Objetivo medible</b> (fijado en la pre-entrega y ajustado por la devolución): maximizar "
          "Recall de fraude manteniendo Precision y un volumen de alertas operable. Métricas principales: "
          "PR-AUC, Recall y Precision. Métrica de negocio: captura en Top-100 y Top-500. Benchmark: un "
          "orden aleatorio captura, en el holdout, unos 1 fraudes entre 100 revisiones; el modelo debe "
          "multiplicar ese lift."),
        table([
            ["Situación", "Lectura estadística", "Costo de negocio"],
            ["Alerta sobre operación legítima", "Falso positivo", "Horas de analista + fricción al cliente"],
            ["Fraude que no entra a la cola", "Falso negativo", "Pérdida del monto y daño reputacional"],
            ["FPR ≤ 5 % como única traba", "Miles de FP con 0,17 % de prevalencia", "El equipo no puede revisar esa cola"],
        ], col_widths=[5.2 * cm, 5.6 * cm, 6.2 * cm], left_cols=(0, 1, 2)),
        P("Tabla 2. Costos del error. El umbral es una decisión de capacidad, no un default técnico.", "caption"),
        P("<b>Hipótesis.</b> H1: los componentes PCA (V1–V28) concentran señal discriminativa. H2: el "
          "monto, transformado a escala logarítmica, aporta menos que las V* pero ayuda a calibrar riesgo. "
          "H3: como existe Time, un split aleatorio puede sobreestimar el desempeño; entrenar con "
          "operaciones anteriores y probar con posteriores es la prueba más realista."),
    ]

    story += [
        P("3. Datos y metodología", "h1"),
        P("3.1 Fuente y calidad", "h2"),
        P("Fuente. Credit Card Fraud Detection, ULB Machine Learning Group / Kaggle (Dal Pozzolo, Caelen, "
          "Johnson y Bontempi, 2015). Licencia ODbL v1.0. Transacciones de titulares europeos en septiembre "
          "de 2013 (dos días). Archivo original: 284.807 filas × 31 columnas. Tasa de fraude: 0,172 %. "
          "El CSV se descarga de un mirror público y se cachea en data/creditcard.csv; el esquema se valida "
          "al cargar (Time, V1–V28, Amount, Class)."),
        table([
            ["Hallazgo de calidad", "Decisión y motivo"],
            ["0 nulos", "Ninguna imputación."],
            ["1.081 duplicados exactos (19 fraudes)", "Eliminados; quedan 283.726 filas y 473 fraudes."],
            ["Amount sesgado (media €88; mediana €22)", "Amount_log = log1p(Amount); el crudo no entra al modelo."],
            ["V1–V28 ya son PCA anonimizado", "Se usan sin reescalar; limitan interpretación causal."],
            ["Time en [0 ; 172.792] s", "Sirve para el split temporal; en el modelo se reemplaza por hora cíclica (Hour_sin, Hour_cos)."],
            ["Desbalance 0,17 %", "class_weight / scale_pos_weight; SMOTE (strategy=0,1) solo dentro del Pipeline."],
        ], col_widths=[7.0 * cm, 10.0 * cm], left_cols=(0, 1)),
        P("Tabla 3. Decisiones de calidad registradas (equivalente a un data log).", "caption"),
        fig("fig1_eda.png", max_w=16 * cm, max_h=5.2 * cm),
        P("Figura 1. Distribución de clases, Amount de las legítimas y Time por clase.", "caption"),
        P("3.2 Conclusiones del EDA", "h2"),
        P("Las variables con mayor |correlación| con Class son V17 (0,31), V14 (0,29) y V12 (0,25). "
          "Amount por sí solo no separa bien: hay fraudes en montos bajos y medios. Time muestra "
          "concentraciones distintas por clase, lo que respalda H3. Accuracy o ROC-AUC aislados son "
          "engañosos: un clasificador trivial “todo legítimo” tendría Accuracy 99,83 % y no detectaría "
          "ningún fraude."),
        P("3.3 Preparación y CRISP-DM", "h2"),
        P("Metodología. Se siguió CRISP-DM. Todas las transformaciones viven en un Pipeline de "
          "scikit-learn / imbalanced-learn: StandardScaler sobre Amount_log; V1–V28 y la hora cíclica "
          "pasan sin cambios; SMOTE, cuando se usa, queda dentro del pipeline. random_state=42."),
        table([
            ["Diseño de validación", "Pregunta que responde"],
            ["CV estratificada 5-fold sobre el train (80 %)", "¿Qué modelo discrimina mejor (PR-AUC) sin mirar el test?"],
            ["Holdout estratificado 80/20 (una sola vez)", "¿Cuánto rinde en una mezcla i.i.d. de las 48 h?"],
            ["Split temporal por Time (80 % anterior → 20 % posterior)", "¿Generaliza hacia operaciones futuras?"],
        ], col_widths=[8.2 * cm, 8.8 * cm], left_cols=(0, 1)),
        P("Tabla 4. Tres diseños de validación. El test no interviene en la elección del modelo ni del umbral.", "caption"),
    ]

    # Tabla CV
    cv_rows = [["Modelo", "PR-AUC media", "Desvío", "Precision@100 media", "Recall 0,5 media"]]
    for _, r in cv.sort_values("pr_auc_media", ascending=False).iterrows():
        mark = " *" if r["modelo"] == "XGBoost" else ""
        cv_rows.append([r["modelo"] + mark, num(r["pr_auc_media"]), num(r["pr_auc_desvio"]),
                        pct(r["precision@100_media"]), pct(r["recall_0.5_media"])])

    ale_rows = [["Modelo", "PR-AUC", "Recall", "Precision", "F1", "Alertas", "Fraud@100", "Fraud@500"]]
    for name in ["Logística", "Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE"]:
        r = row(hold, split="aleatorio", modelo=name)
        mark = " *" if name == "XGBoost" else ""
        ale_rows.append([name + mark, num(r["pr_auc"]), pct(r["recall_0.5"]), pct(r["precision_0.5"]),
                         num(r["f1_0.5"]), n_(r["alertas_0.5"]), str(int(r["fraudes@100"])),
                         str(int(r["fraudes@500"]))])

    story += [
        P("4. Modelado y evaluación", "h1"),
        P("4.1 Algoritmos y regla de selección", "h2"),
        P("Se compararon cinco configuraciones, todas con manejo explícito del desbalance: Regresión "
          "Logística (baseline, class_weight=balanced), Random Forest (class_weight), XGBoost "
          "(scale_pos_weight), LightGBM (peso √(neg/pos) y regularización) y XGBoost+SMOTE dentro del "
          "pipeline. <b>Regla fijada antes de mirar los holdouts:</b> gana el mayor PR-AUC medio de la "
          "CV de 5 folds sobre el train. El desempate lo dan Precision@100 de CV, Recall y el volumen "
          "de alertas. El split temporal no cambia la regla: sirve para medir robustez."),
        table(cv_rows, col_widths=[4.0 * cm, 2.8 * cm, 2.4 * cm, 4.0 * cm, 3.8 * cm], left_cols=(0,)),
        P(f"Tabla 5. Validación cruzada 5-fold sobre el train aleatorio (226.980 operaciones, 378 fraudes). "
          f"*Elegido: XGBoost (PR-AUC {num(xgb_cv['pr_auc_media'])} ± {num(xgb_cv['pr_auc_desvio'])}).", "caption"),
        P("4.2 Holdout aleatorio", "h2"),
        table(ale_rows, col_widths=[3.4 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 1.5 * cm, 1.7 * cm, 2.0 * cm, 2.0 * cm],
              left_cols=(0,)),
        P("Tabla 6. Holdout aleatorio, umbral 0,5. *Candidato. Test: 56.746 operaciones, 95 fraudes.", "caption"),
        fig("fig2_test_estable.png", max_w=15.5 * cm, max_h=5.4 * cm),
        P("Figura 2. Curva Precision-Recall y captura Top-K de XGBoost en el holdout aleatorio.", "caption"),
        P("4.3 Umbral operativo y carga de revisión", "h2"),
        P(f"El umbral se fija con el máximo F1 sobre las predicciones out-of-fold del train (no se "
          f"optimiza con el test): <b>0,81</b>. En el holdout emite <b>{oof_alertas} alertas</b> "
          f"({oof_tp} TP / {oof_fp} FP / {oof_fn} FN): Precision {pct(oof_p)} y Recall {pct(oof_r)}. "
          f"Esa es la carga real que tendría que revisar el equipo. Un FPR del 5 % costaría "
          f"~{n_(fpr5_fp)} revisiones falsas."),
        fig("fig5_umbral_alertas.png", max_w=14 * cm, max_h=4.6 * cm),
        P("Figura 3. Precision y Recall en función del número de alertas.", "caption"),
        fig("fig6_confusion.png", max_w=9.5 * cm, max_h=5.0 * cm),
        P("Figura 4. Matriz de confusión de XGBoost con umbral operativo 0,81.", "caption"),
    ]

    tmp_rows = [["Modelo", "PR-AUC", "Recall", "Precision", "Fraud@100", "Fraud@500"]]
    for name in ["Logística", "Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE"]:
        r = row(hold, split="temporal", modelo=name)
        mark = " *" if name == "Random Forest" else ""
        tmp_rows.append([name + mark, num(r["pr_auc"]), pct(r["recall_0.5"]), pct(r["precision_0.5"]),
                         str(int(r["fraudes@100"])), str(int(r["fraudes@500"]))])

    story += [
        P("4.4 Validación temporal versus aleatoria", "h2"),
        P("Se ordenó por Time y se entrenó con las primeras 226.980 operaciones (399 fraudes) y se "
          "evaluó con las 56.746 posteriores (74 fraudes). Es el diseño que pide la devolución y el más "
          "cercano a producción: el modelo no ve el “futuro” de las 48 horas."),
        table(tmp_rows, col_widths=[3.8 * cm, 2.2 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], left_cols=(0,)),
        P("Tabla 7. Holdout temporal. *Mejor PR-AUC: RandomForest. Test: 74 fraudes.", "caption"),
        fig("fig3_temporal.png", max_w=15.5 * cm, max_h=5.2 * cm),
        P("Figura 5. PR-AUC por modelo en ambos esquemas.", "caption"),
        P(f"Lectura. XGBoost sigue siendo competitivo en temporal (PR-AUC {num(xgb_t['pr_auc'])}), pero "
          f"RandomForest lo supera levemente y con Precision {pct(rf_t['precision_0.5'])}. La captura "
          f"Top-100 baja de {int(xgb['fraudes@100'])} a {int(rf_t['fraudes@100'])} fraudes, en parte porque "
          "el test temporal tiene menos positivos (74 vs 95). H3 queda respaldada: el aleatorio no miente, "
          "pero es el techo; el temporal es el piso más honesto para decidir un piloto."),
    ]

    story += [
        P("5. Interpretación del modelo", "h1"),
        P("Sobre XGBoost se leyeron importancias de split (no son causalidad ni SHAP), usando los "
          "nombres que produce el ColumnTransformer. Las V* más usadas coinciden con el EDA (V14, V17, "
          "V12 y vecinas). Hour_sin / Hour_cos y Amount_log aportan, pero no lideran. Como las V* son "
          "PCA, el negocio no puede traducir “sube V14” a una regla de comercio o país: el modelo "
          "prioriza, no explica el fraude ante un auditor."),
        fig("fig4_interpretacion.png", max_w=12 * cm, max_h=5.4 * cm),
        P("Figura 6. Importancia de variables de XGBoost (split aleatorio). Lectura descriptiva, no causal.", "caption"),
        P("El score se usa para ordenar. Un valor 0,80 no se interpreta aquí como “probabilidad "
          "calibrada del 80 %”: no se reporta curva de calibración. Para un umbral de corte alcanza "
          "con el ranking y la matriz de confusión."),
    ]

    story += [
        P("6. Impacto en el negocio, riesgos y ética", "h1"),
        P(f"Sin ranking, capturar ~{pct(xgb['recall_0.5'], 0)} de los fraudes del holdout exigiría revisar "
          f"una fracción enorme de la cartera o convivir con miles de alertas. Con el modelo, ese Recall "
          f"se obtiene revisando {oof_alertas} operaciones, o {int(tk100['fraudes'])} fraudes con solo 100 "
          f"revisiones. Un FPR del 5 % costaría ~{n_(fpr5_fp)} revisiones falsas; el umbral elegido "
          f"cuesta {oof_fp} FP."),
        table([
            ["Escenario de revisión", "Operaciones a mirar", "Fraudes hallados", "Precision"],
            ["Azar (100 casos)", "100", "≈ 1", "≈ tasa base"],
            ["Modelo Top-100", "100", str(int(tk100["fraudes"])), pct(tk100["precision_k"])],
            ["Modelo Top-500", "500", str(int(tk500["fraudes"])), pct(tk500["precision_k"])],
            ["Umbral 0,81 (OOF)", str(oof_alertas), str(oof_tp), pct(oof_p)],
            ["Regla FPR = 5 %", f"~{n_(fpr5_fp)}+", "alto Recall, baja Precision", "muy baja"],
        ], col_widths=[4.6 * cm, 4.2 * cm, 4.2 * cm, 4.0 * cm], left_cols=(0,)),
        P("Tabla 8. El valor está en ordenar la cola, no en umbrales holgados de FPR.", "caption"),
        table([
            ["Riesgo", "Mitigación"],
            ["Dataset de solo 2 días (2013), Europa", "Tratar el caso como prueba de método; reentrenar con datos propios del emisor."],
            ["Drift de patrones de fraude", "Monitorear PR-AUC, Precision y captura@K por día; reentrenar."],
            ["PCA poco explicable ante auditoría", "Usar el score para priorizar; complementar con reglas y features nativas."],
            ["Falso positivo = bloqueo de cliente", "Umbral según capacidad; canal de apelación; no automatizar el rechazo final."],
            ["Fuga si SMOTE o scaler ven el test", "Pipeline único; CV y holdout separados antes de cualquier ajuste."],
            ["Privacidad", "Datos ya anonimizados; minimización; acceso need-to-know al score."],
        ], col_widths=[6.0 * cm, 11.0 * cm], left_cols=(0, 1)),
        P("Tabla 9. Riesgos y mitigaciones. El prototipo no autoriza un despliegue automático.", "caption"),
        P("<b>Ética.</b> No hay PII. El umbral es una decisión explícita de negocio (Precision y cupo), "
          "no un default de 0,5. Un falso negativo pierde dinero; un falso positivo daña al cliente: por "
          "eso se reportan ambos y el volumen absoluto de alertas. Uso académico bajo ODbL."),
    ]

    story += [
        P("7. Conclusiones y recomendaciones", "h1"),
        P("• El problema se formuló como priorización bajo capacidad limitada. PR-AUC, Recall, Precision "
          "y captura Top-K reflejan esa decisión mejor que Accuracy o un FPR suelto.", "bullet"),
        P(f"• XGBoost es el candidato (mayor PR-AUC de CV: {num(xgb_cv['pr_auc_media'])} ± "
          f"{num(xgb_cv['pr_auc_desvio'])}). En holdout aleatorio: PR-AUC {num(xgb['pr_auc'])}, "
          f"Recall {pct(xgb['recall_0.5'])}, Precision {pct(xgb['precision_0.5'])}; Top-100 = "
          f"{int(tk100['fraudes'])} fraudes; umbral 0,81 → {oof_alertas} alertas.", "bullet"),
        P(f"• H1 tiene respaldo descriptivo (V17, V14, V12). H3 también: en temporal gana RandomForest "
          f"(PR-AUC {num(rf_t['pr_auc'])}) y la captura Top-100 baja a {int(rf_t['fraudes@100'])}. "
          "El aleatorio es techo; el temporal es la prueba de piloto.", "bullet"),
        P("• SMOTE no se elige: sostiene Recall pero no mejora PR-AUC ni Precision respecto de XGBoost "
          "sin remuestreo.", "bullet"),
        P("• Recomendación: no desplegar a ciegas. Usar el score para ordenar la cola, fijar el umbral "
          "según el turno de analistas, reentrenar con datos actuales del emisor y medir captura@100/@500 "
          "y drift en una ventana temporal nueva.", "bullet"),
    ]

    story += [
        PageBreak(),
        P("Anexos", "h1"),
        P("Anexo A. Trazabilidad con la consigna, la pre-entrega y la devolución", "h2"),
        table([
            ["Peso", "Criterio", "Dónde está en este informe"],
            ["20 %", "Problema y negocio", "§1–2: ranking, costos FP/FN, KPI Top-K"],
            ["20 %", "Datos y metodología", "§3: calidad, EDA, Pipeline, CV y dos holdouts"],
            ["15 %", "Selección del modelo", "§4.1 y Tablas 5–7 (regla: PR-AUC de CV)"],
            ["15 %", "Calidad de métricas", "PR-AUC, Recall, Precision, Top-K, alertas"],
            ["15 %", "Interpretación y valor", "§5–6: importancias y carga de revisión"],
            ["5 %", "Ética y privacidad", "§6 y Tabla 9"],
            ["10 %", "Claridad", "Resumen no técnico, figuras y recomendación"],
        ], col_widths=[1.8 * cm, 4.6 * cm, 10.6 * cm], left_cols=(1, 2)),
        P("Tabla A1. Cobertura de los criterios de evaluación.", "caption"),
        table([
            ["Pedido de la devolución", "Estado"],
            ["PR-AUC, Recall y Precision como métricas principales", "Tablas 1, 6 y 7; no se decide por FPR."],
            ["Cuántas operaciones revisaría el equipo", f"§4.3: {oof_alertas} alertas vs ~{n_(fpr5_fp)} si FPR=5 %."],
            ["Captura si se revisan Top-100 / Top-500", "Figura 2 y Tabla 8."],
            ["Comparar split aleatorio vs temporal por Time", "§4.4, Figura 5, Tabla 7."],
            ["Validación cruzada para elegir el modelo", "§4.1 y Tabla 5; el test no interviene."],
        ], col_widths=[8.4 * cm, 8.6 * cm], left_cols=(0, 1)),
        P("Tabla A2. Cierre de la devolución del tutor.", "caption"),
        P("Anexo B. Reproducibilidad y fuentes", "h2"),
        P("Dal Pozzolo, A., Caelen, O., Johnson, R. A. y Bontempi, G. (2015). Calibrating Probability "
          "with Undersampling for Unbalanced Classification. IEEE CIDM. Dataset: Credit Card Fraud "
          "Detection (ULB). Licencia ODbL v1.0. Chapman, P. et al. (2000). CRISP-DM 1.0."),
    ]

    merged, i = [], 0
    while i < len(story):
        if (isinstance(story[i], Table) and i + 1 < len(story)
                and isinstance(story[i + 1], Paragraph) and story[i + 1].style.name == "caption"):
            merged.append(KeepTogether([story[i], story[i + 1]]))
            i += 2
        else:
            merged.append(story[i])
            i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Informe Final — Detección de Fraude — Grupo 33", author="Grupo 33",
    )
    doc.build(merged, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe: {OUT}")


if __name__ == "__main__":
    build()
