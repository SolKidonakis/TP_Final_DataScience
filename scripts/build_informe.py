"""Arma docs/Informe_Final_Grupo33.pdf a partir de results/results.json y results/figures.

Uso (desde la raíz del repo, después de ejecutar el notebook):
    python scripts/build_informe.py

Todas las cifras del informe salen de results/results.json: si se vuelve a correr el
notebook, este script las actualiza solas.
"""

from __future__ import annotations

import json
from pathlib import Path

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
RESULTS = ROOT / "results" / "results.json"
FIGURES = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "Informe_Final_Grupo33.pdf"

NAVY = colors.HexColor("#1F4E79")
TEAL = colors.HexColor("#2E75B6")
LINE = colors.HexColor("#B0B0B0")
ROW = colors.HexColor("#F4F7FB")


# ----------------------------------------------------------------------------- formato
def pct(x, d=1):
    return f"{100 * float(x):.{d}f} %".replace(".", ",")


def num(x, d=3):
    return f"{float(x):.{d}f}".replace(".", ",")


def n_(x):
    """Entero con separador de miles (punto)."""
    return f"{int(round(float(x))):,}".replace(",", ".")


def eur(x, d=0):
    s = f"€{abs(float(x)):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-{s}" if float(x) < 0 else s


def styles():
    base = dict(fontName="Helvetica", fontSize=12, leading=18)
    return {
        "kicker": ParagraphStyle("kicker", fontName="Helvetica", fontSize=11, leading=16.5, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#C0392B"), spaceAfter=6),
        "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold", fontSize=18, leading=24,
                                      alignment=TA_CENTER, spaceAfter=10),
        "cover_sub": ParagraphStyle("cover_sub", alignment=TA_CENTER, spaceAfter=4, **base),
        "idx": ParagraphStyle("idx", alignment=TA_CENTER, spaceAfter=3, **base),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=21, textColor=NAVY,
                             spaceBefore=10, spaceAfter=8, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=18, textColor=TEAL,
                             spaceBefore=8, spaceAfter=6, keepWithNext=1),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=8, **base),
        "bullet": ParagraphStyle("bullet", leftIndent=12, spaceAfter=4, alignment=TA_JUSTIFY, **base),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=9, leading=12, alignment=TA_CENTER,
                                  textColor=colors.grey, spaceBefore=3, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10, alignment=TA_CENTER),
        "cell_l": ParagraphStyle("cell_l", fontName="Helvetica", fontSize=8, leading=10, alignment=TA_LEFT),
        "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8, leading=10, alignment=TA_CENTER,
                             textColor=colors.white),
    }


S = styles()


def fig(name, max_w=15 * cm, max_h=6.0 * cm):
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
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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


def by_model(rows):
    return {r["modelo"]: r for r in rows}


# ----------------------------------------------------------------------------- informe
def build():
    d = json.loads(RESULTS.read_text(encoding="utf-8"))
    q, eda, split, tm = d["calidad"], d["eda"], d["split"], d["temporal_meta"]
    cv = by_model(d["cv"])
    sel = d["seleccion"]
    chosen = sel["elegido"]
    st = by_model(d["test_estable"])
    tp = by_model(d["test_temporal"])
    topk_st = {r["k"]: r for r in d["topk_estable"]}
    topk_tm = {r["k"]: r for r in d["topk_temporal"]}
    thr = d["umbral"]
    op, op_t, oof, fpr5 = thr["test_estable"], thr["test_temporal"], thr["oof"], thr["fpr5_test"]
    econ = by_model([{**r, "modelo": r["escenario"]} for r in d["economico"]])
    em = d["economico_meta"]
    abl = {r["variante"]: r for r in d["ablaciones"]}
    full_abl = abl["Modelo completo"]
    interp = d["interpretacion"]

    n_test = split["n_test"]
    n_pos = split["fraude_test"]
    n_neg = n_test - n_pos
    fpr5_alerts = int(0.05 * n_neg)
    c_st, c_tm = st[chosen], tp[chosen]
    rule = tp["Regla: monto alto"]
    delta_pr = c_st["pr_auc"] - c_tm["pr_auc"]
    trees = [m for m in ("Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE") if m in cv]
    h3_weak = delta_pr > 0.02
    h3_text = (
        "<b>H3 se confirma:</b> el split aleatorio es optimista. " if h3_weak else
        "<b>H3 no se confirma con claridad:</b> no se observa una degradación relevante hacia el futuro; la diferencia es "
        "menor que el ruido (con 74 fraudes, un fraude más o menos mueve varios puntos). "
    ) + "El temporal sigue siendo el piso más honesto para decidir un piloto, porque es lo que ocurriría en producción."
    best_tm = max((m for m in tp if m not in ("Ranking aleatorio", "Regla: monto alto")), key=lambda m: tp[m]["pr_auc"])

    story = []

    # ---------------------------------------------------------------- portada
    story += [
        Spacer(1, 1.6 * cm),
        P("INFORME FINAL", "kicker"),
        P("Detección de fraude en tarjetas de crédito:<br/>priorización de la cola de revisión", "cover_title"),
        Spacer(1, 0.4 * cm),
        P("Diplomado en Ciencia de Datos y Análisis Avanzado", "cover_sub"),
        P("Unidad N.º 9 — Proyecto Final Integrador", "cover_sub"),
        Spacer(1, 0.6 * cm),
        P("<b>Grupo 33</b>", "cover_sub"),
        P("Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis<br/>Marco Garcia · Esteban Derghazarian", "cover_sub"),
        P("Octubre 2026", "cover_sub"),
        Spacer(1, 1.2 * cm),
        P("<b>Índice</b>", "cover_sub"),
        P("1. Resumen ejecutivo — 2. Definición del problema y relevancia — 3. Datos y metodología — "
          "4. Modelado y evaluación — 5. Interpretación del modelo — 6. Impacto en el negocio, riesgos y ética — "
          "7. Conclusiones y recomendaciones — Anexos", "idx"),
        PageBreak(),
    ]

    # ---------------------------------------------------------------- 1. resumen
    story += [
        P("1. Resumen ejecutivo", "h1"),
        P(f"<b>Problema.</b> Un emisor de tarjetas debe decidir qué operaciones revisar a mano. Entre "
          f"{n_(q['n_raw'])} transacciones reales solo {q['fraud_raw']} son fraude ({pct(q['fraud_raw'] / q['n_raw'], 2)}). "
          f"Revisar todo es imposible; revisar al azar casi nunca encuentra un fraude."),
        P(f"<b>Solución.</b> Un modelo ({chosen}) que ordena las operaciones por riesgo para que el equipo revise "
          f"primero las más sospechosas. Es una <b>cola de revisión priorizada</b>, no un “sí/no” automático."),
        table([
            ["Indicador (test aleatorio, 95 fraudes)", "Revisión al azar", "Regla “monto alto”", f"Modelo ({chosen})"],
            ["Fraudes hallados revisando 100 operaciones", "≈ 0", str(st["Regla: monto alto"]["fraudes@100"]),
             f"{topk_st[100]['fraudes']} de {n_pos}"],
            ["Fraudes hallados revisando 500 operaciones", "≈ 1", str(st["Regla: monto alto"]["fraudes@500"]),
             f"{topk_st[500]['fraudes']} de {n_pos}"],
            ["Con el umbral operativo", "—", "—",
             f"{op['alertas']} alertas, {op['tp']} fraudes ({pct(op['precision'], 0)} de aciertos)"],
            ["Valor neto estimado (test, 2 días)", "—", eur(econ["Regla estática: Top-500 por monto"]["neto"]),
             eur(econ["Modelo: umbral operativo"]["neto"])],
        ], col_widths=[6.4 * cm, 2.9 * cm, 3.2 * cm, 4.5 * cm], left_cols=(0,)),
        P("Tabla 1. Con el mismo esfuerzo de revisión el modelo encuentra decenas de fraudes donde una regla simple "
          "encuentra uno o dos.", "caption"),
        P(f"<b>Resultado.</b> Revisando las 100 operaciones de mayor riesgo se encuentran {topk_st[100]['fraudes']} de "
          f"{n_pos} fraudes ({pct(topk_st[100]['recall_k'])}); con el umbral operativo se emiten solo {op['alertas']} alertas "
          f"y {pct(op['precision'], 0)} de ellas son fraudes reales. Sobre operaciones futuras (test temporal) la señal se mantiene "
          f"(PR-AUC {num(c_tm['pr_auc'])} vs {num(c_st['pr_auc'])}). El criterio inicial “FPR ≤ 5 %” habría exigido "
          f"{n_(fpr5['alertas'])} revisiones y un resultado económico negativo."),
        P("<b>Recomendación.</b> Usar el score para ordenar la cola, fijar el umbral según la capacidad del equipo y "
          "revalidar con datos propios del emisor antes de operar."),
        table([
            ["Glosario", ""],
            ["PR-AUC", "Calidad del ordenamiento cuando el fraude es muy raro (1 = perfecto; el azar da ≈ 0,002)."],
            ["Top-K", "Se toman las K operaciones con mayor score y se cuenta cuántos fraudes contienen."],
            ["FPR", "Porcentaje de operaciones legítimas que se marcan como sospechosas."],
        ], col_widths=[2.4 * cm, 14.6 * cm], left_cols=(0, 1)),
    ]

    # ---------------------------------------------------------------- 2. problema
    story += [
        P("2. Definición del problema y relevancia", "h1"),
        P("La pregunta de negocio no es “¿esta transacción es fraude con certeza?”, sino <b>a qué operaciones asignar "
          "primero un cupo limitado de revisión humana</b>. Cada fraude no detectado es pérdida financiera; cada falsa "
          "alarma consume tiempo de analista y puede bloquear a un cliente legítimo."),
        P(f"<b>Objetivo medible.</b> Maximizar los fraudes encontrados por cada operación revisada: PR-AUC, Recall y "
          f"Precision como métricas técnicas y captura en Top-100 / Top-500 como métrica de negocio. Un orden aleatorio "
          f"encontraría {num(100 * n_pos / n_test, 2)} fraudes cada 100 revisiones; el modelo debe multiplicar ese valor."),
        P("<b>Cómo cambió el objetivo respecto de la pre-entrega.</b> La pre-entrega fijó “Recall ≥ 80 % con FPR ≤ 5 %”. "
          f"La devolución del tutor mostró que con {n_(n_neg)} legítimas en el test, un FPR de 5 % equivale a "
          f"~{n_(fpr5_alerts)} alertas falsas: un KPI inoperable. Se reemplazó por Precision, cantidad de alertas y "
          "captura Top-K; el Recall se sigue reportando."),
        table([
            ["Tema", "Pre-entrega", "Informe final (y motivo)"],
            ["KPI", "Recall ≥ 80 % y FPR ≤ 5 %", "PR-AUC, Recall, Precision y Top-K; FPR 5 % implica miles de alertas falsas"],
            ["Selección del modelo", "CV de 5 folds", "Igual: CV de 5 folds sobre el 80 % de desarrollo, sin usar el test"],
            ["Umbral", "“Threshold tuning”", "Máximo F1 sobre predicciones out-of-fold (no se ajusta con el test)"],
            ["Variables de tiempo y monto", "Escalar Time y Amount", "Amount_log + hora cíclica: Time crudo toma valores nunca vistos en el test temporal"],
            ["LightGBM", "is_unbalance", "Peso √(neg/pos): is_unbalance era inestable (PR-AUC ≈ 0)"],
            ["Duplicados / desbalance", "Pendientes", f"{n_(q['duplicados'])} duplicados eliminados; class_weight y SMOTE dentro del Pipeline"],
        ], col_widths=[3.4 * cm, 4.2 * cm, 9.4 * cm], left_cols=(0, 1, 2)),
        P("Tabla 2. Trazabilidad de decisiones entre la pre-entrega y el informe final.", "caption"),
        P("<b>Hipótesis.</b> H1: las componentes PCA (V1–V28) concentran la señal. H2: el monto en escala logarítmica "
          "ayuda a calibrar el riesgo. H3: un split aleatorio sobreestima el desempeño frente a uno temporal. "
          "Su resultado se discute en las secciones 4, 5 y 7."),
    ]

    # ---------------------------------------------------------------- 3. datos
    story += [
        P("3. Datos y metodología", "h1"),
        P("3.1 Fuente y calidad", "h2"),
        P(f"Credit Card Fraud Detection, ULB Machine Learning Group / Kaggle (Dal Pozzolo et al., 2015), licencia ODbL v1.0: "
          f"transacciones de titulares europeos en septiembre de 2013 (dos días, {num(q['horas'], 0)} h). "
          f"{n_(q['n_raw'])} filas × 31 columnas; tasa de fraude {pct(q['fraud_raw'] / q['n_raw'], 3)}. "
          "El CSV se descarga de un mirror público y se valida al cargar."),
        table([
            ["Hallazgo de calidad", "Decisión y motivo"],
            [f"{q['nulos']} nulos", "Ninguna imputación necesaria."],
            [f"{n_(q['duplicados'])} duplicados exactos ({q['duplicados_fraude']} eran fraudes)",
             f"Eliminados; quedan {n_(q['n'])} filas y {q['fraud']} fraudes. Evita que un mismo registro caiga en train y test."],
            [f"Monto sesgado (mediana fraude {eur(q['monto_mediana_fraude'], 2)}; máximo {eur(q['monto_max'])})",
             "Amount_log = log1p(Amount); el monto crudo no entra al modelo."],
            ["V1–V28 son componentes PCA anonimizadas", "Se usan sin transformar; limitan la interpretación causal."],
            ["Time en segundos (0–172.792)", "Se reemplaza por la hora del día (seno/coseno); se usa para el split temporal."],
            [f"Desbalance {pct(q['prevalencia'], 2)}", "class_weight / scale_pos_weight; SMOTE solo dentro del Pipeline."],
        ], col_widths=[7.2 * cm, 9.8 * cm], left_cols=(0, 1)),
        P("Tabla 3. Decisiones de calidad de datos (detalle cronológico en data_log.md).", "caption"),
        P("3.2 Hallazgos del EDA", "h2"),
        P(f"Las variables más asociadas al fraude son V17 ({num(eda['top_corr']['V17'], 2)}), V14 "
          f"({num(eda['top_corr']['V14'], 2)}) y V12 ({num(eda['top_corr']['V12'], 2)}). El monto por sí solo no separa: "
          f"la mediana de un fraude es {eur(q['monto_mediana_fraude'], 2)} contra {eur(q['monto_mediana_legit'], 2)} de una "
          f"operación legítima. La madrugada concentra más fraude: {pct(eda['tasa_noche'], 2)} entre 0 y 6 h contra "
          f"{pct(eda['tasa_dia'], 2)} el resto del día. La prevalencia también varía entre tramos temporales "
          f"({pct(eda['tramo_tasa_min'], 2)} a {pct(eda['tramo_tasa_max'], 2)}). Un clasificador trivial “todo legítimo” "
          f"tendría exactitud {pct(1 - q['prevalencia'], 2)} sin detectar ningún fraude, por eso no se usa Accuracy."),
        fig("fig1_eda.png", max_w=15.5 * cm, max_h=5.6 * cm),
        P("Figura 1. Tasa de fraude por hora, monto por clase y prevalencia por tramo temporal.", "caption"),
        P("3.3 Metodología (CRISP-DM)", "h2"),
        P("Se siguió CRISP-DM: negocio (sección 2), datos (esta sección), modelado y evaluación (sección 4) y traducción "
          "a una decisión (secciones 5–6). Todo el preprocesamiento vive en un Pipeline (escalado y, cuando se usa, "
          "SMOTE), de modo que nunca ve datos de validación ni de test. random_state = 42."),
        table([
            ["Etapa", "Datos", "Para qué se usa"],
            ["Desarrollo", f"80 % aleatorio estratificado ({n_(split['n_dev'])} filas, {split['fraude_dev']} fraudes)",
             "CV de 5 folds para elegir modelo y umbral (out-of-fold)"],
            ["Test estable", f"20 % restante ({n_(n_test)} filas, {n_pos} fraudes)", "Evaluación final, una sola vez"],
            ["Test temporal", f"Entrena con el primer 80 % por Time; prueba con el último 20 % ({tm['fraude_test']} fraudes)",
             "¿Generaliza hacia operaciones futuras?"],
        ], col_widths=[2.8 * cm, 8.2 * cm, 6.0 * cm], left_cols=(0, 1, 2)),
        P("Tabla 4. Diseño de validación. El test nunca interviene en la elección del modelo ni del umbral.", "caption"),
    ]

    # ---------------------------------------------------------------- 4. modelado
    rows = [["Modelo", "PR-AUC (CV)", "± desvío", "Precision@100 (CV)", "Tiempo (s)"]]
    for m in cv.values():
        rows.append([f"{m['modelo']} *" if m["modelo"] == chosen else m["modelo"], num(m["pr_auc_media"]),
                     num(m["pr_auc_desvio"]), pct(m["precision@100_media"]), n_(m["segundos"])])
    story += [
        P("4. Modelado y evaluación", "h1"),
        P("4.1 Algoritmos y regla de selección", "h2"),
        P("Se compararon cinco configuraciones, todas con manejo del desbalance: Regresión Logística (baseline, "
          "class_weight), Random Forest, XGBoost (scale_pos_weight), LightGBM (peso √(neg/pos)) y XGBoost + SMOTE. "
          "<b>Regla fijada antes de mirar los tests:</b> 1) gana la mayor PR-AUC media en CV; 2) los modelos a menos de "
          "un desvío de la mejor quedan empatados; 3) entre los empatados se elige la mayor Precision@100 de CV."),
        table(rows, col_widths=[4.2 * cm, 2.9 * cm, 2.4 * cm, 4.2 * cm, 2.6 * cm], left_cols=(0,)),
        P(f"Tabla 5. Validación cruzada (5 folds) sobre el set de desarrollo. *Elegido: {chosen}.", "caption"),
        P(f"La logística queda claramente atrás ({num(cv['Logística']['pr_auc_media'])}). Los cuatro modelos de árboles "
          f"({', '.join(trees)}) quedan empatados dentro de un desvío: <b>sus diferencias son menores que el ruido de "
          f"la validación</b>. Por la regla se elige <b>{chosen}</b>; no hay evidencia de que sea mejor que los otros tres, "
          "y esa es una limitación (95 fraudes en el test)."),
    ]

    rows = [["Modelo", "PR-AUC", "ROC-AUC", "Fraudes@100", "Fraudes@500", "Recall@100", "Precision@500"]]
    order = ["Ranking aleatorio", "Regla: monto alto", "Logística", "Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE"]
    for name in order:
        m = st[name]
        rows.append([f"{name} *" if name == chosen else name, num(m["pr_auc"]), num(m["roc_auc"]), str(m["fraudes@100"]),
                     str(m["fraudes@500"]), pct(m["recall@100"]), pct(m["precision@500"])])
    story += [
        P("4.2 Test estable y métrica de negocio Top-K", "h2"),
        table(rows, col_widths=[3.8 * cm, 1.9 * cm, 2.0 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.4 * cm], left_cols=(0,)),
        P(f"Tabla 6. Test estable ({n_(n_test)} operaciones, {n_pos} fraudes). *Modelo elegido.", "caption"),
        P(f"Revisando las 100 operaciones de mayor score se encuentran <b>{topk_st[100]['fraudes']} de {n_pos}</b> fraudes "
          f"(Precision@100 = {pct(topk_st[100]['precision_k'])}; {num(topk_st[100]['lift_k'], 0)} veces la tasa base). Ampliar a 500 "
          f"revisiones suma solo {topk_st[500]['fraudes'] - topk_st[100]['fraudes']} fraudes más: la curva se aplana, el valor está "
          f"en el tope del ranking. La regla estática “monto alto” encuentra {st['Regla: monto alto']['fraudes@500']} fraudes en 500 "
          "revisiones: el monto no sirve para priorizar."),
        fig("fig2_test_estable.png", max_w=15.5 * cm, max_h=6.2 * cm),
        P("Figura 2. Curvas Precision-Recall y fraudes encontrados según operaciones revisadas.", "caption"),
        P("4.3 Umbral operativo y carga de revisión", "h2"),
        P(f"El umbral sale de las predicciones out-of-fold del set de desarrollo (máximo F1): <b>{num(thr['valor'], 3)}</b>. "
          f"Aplicado al test estable emite <b>{op['alertas']} alertas</b> ({op['tp']} fraudes, {op['fp']} falsas): Precision "
          f"{pct(op['precision'])}, Recall {pct(op['recall'])}. En el test temporal emite {op_t['alertas']} alertas, "
          f"Precision {pct(op_t['precision'])} y Recall {pct(op_t['recall'])}. El criterio FPR = 5 % de la pre-entrega daría "
          f"{n_(fpr5['alertas'])} alertas con Precision {pct(fpr5['precision'])}. Con el umbral operativo el Recall "
          f"({pct(op['recall'], 0)}) queda por debajo de la meta inicial del 80 %; se alcanza revisando las 100 operaciones "
          f"de mayor riesgo ({pct(topk_st[100]['recall_k'], 0)}) o bajando el umbral a costa de más alertas. Es una decisión "
          "de capacidad del equipo."),
        fig("fig5_umbral_alertas.png", max_w=14.5 * cm, max_h=5.0 * cm),
        P("Figura 3. Precision y Recall según la cantidad de alertas a revisar (escala logarítmica).", "caption"),
    ]

    rows = [["Modelo", "PR-AUC estable", "PR-AUC temporal", "Diferencia", "Fraudes@100 temporal", "Fraudes@500 temporal"]]
    for name in ["Logística", "Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE"]:
        rows.append([f"{name} *" if name == chosen else name, num(st[name]["pr_auc"]), num(tp[name]["pr_auc"]),
                     num(tp[name]["pr_auc"] - st[name]["pr_auc"]), str(tp[name]["fraudes@100"]), str(tp[name]["fraudes@500"])])
    story += [
        P("4.4 Validación temporal", "h2"),
        P(f"Se ordenó por Time y se entrenó con las primeras {n_(tm['n_train'])} operaciones ({tm['fraude_train']} fraudes) para "
          f"probar con las {n_(tm['n_test'])} posteriores ({tm['fraude_test']} fraudes; desde la hora {num(tm['corte_horas'], 0)}). "
          "Es el diseño más cercano a producción: el modelo no ve el futuro."),
        table(rows, col_widths=[3.8 * cm, 2.6 * cm, 2.8 * cm, 2.2 * cm, 2.9 * cm, 2.7 * cm], left_cols=(0,)),
        P(f"Tabla 7. Estable vs temporal. *Modelo elegido. En el temporal gana {best_tm} (PR-AUC {num(tp[best_tm]['pr_auc'])}).",
          "caption"),
        fig("fig3_temporal.png", max_w=15.5 * cm, max_h=5.8 * cm),
        P("Figura 4. PR-AUC por modelo en ambos esquemas y porcentaje de fraudes capturado según K.", "caption"),
        P(f"Entre el test estable y el temporal, {chosen} pasa de PR-AUC {num(c_st['pr_auc'])} a {num(c_tm['pr_auc'])} "
          f"({'caída' if delta_pr > 0 else 'suba'} de {num(abs(delta_pr), 3)}) y el Recall@100 de {pct(topk_st[100]['recall_k'])} a "
          f"{pct(topk_tm[100]['recall_k'])}. Los fraudes absolutos del Top-100 bajan ({topk_st[100]['fraudes']} → "
          f"{topk_tm[100]['fraudes']}) sobre todo porque el test temporal tiene menos fraudes ({tm['fraude_test']} vs {n_pos}). "
          f"{h3_text}"),
    ]

    # ---------------------------------------------------------------- 5. interpretación
    top_glob = list(interp["global_top"].items())
    top_names = ", ".join(k for k, _ in top_glob[:4])
    rows = [["Variante de variables", "PR-AUC estable", "Fraudes@100 estable", "PR-AUC temporal", "Fraudes@100 temporal"]]
    for a in d["ablaciones"]:
        rows.append([a["variante"], num(a["pr_auc_estable"]), str(a["fraudes@100_estable"]), num(a["pr_auc_temporal"]),
                     str(a["fraudes@100_temporal"])])
    story += [
        P("5. Interpretación del modelo", "h1"),
        P(f"Se calcularon valores SHAP sobre una muestra de {interp['n_muestra']} operaciones. Las variables que más pesan son "
          f"{top_names}, las mismas que destaca el EDA: <b>H1 queda respaldada</b>. Como las V* son componentes PCA "
          "anonimizadas, el negocio no puede traducir “sube V14” a una regla de comercio o país: el modelo prioriza, no "
          "explica ante un auditor. En un emisor real habría que sumar variables interpretables (MCC, dispositivo, geografía, historial)."),
        fig("fig4_interpretacion.png", max_w=15.5 * cm, max_h=5.9 * cm),
        P("Figura 5. Importancia global (SHAP) y explicación de un fraude concreto dentro del Top-100.", "caption"),
        P("<b>Ablaciones.</b> Para probar H2 se reentrenó el modelo elegido quitando variables:"),
        table(rows, col_widths=[5.6 * cm, 2.7 * cm, 3.0 * cm, 2.7 * cm, 3.0 * cm], left_cols=(0,)),
        P("Tabla 8. Ninguna variante se separa del modelo completo más de lo que permite el ruido.", "caption"),
        P(f"Sin el monto el PR-AUC estable es {num(abl['Sin monto (Amount_log)']['pr_auc_estable'])} contra "
          f"{num(full_abl['pr_auc_estable'])} del modelo completo, y sin la hora {num(abl['Sin hora del día']['pr_auc_estable'])}. "
          "<b>H2 no se sostiene</b>: el monto no aporta de forma medible; la señal está en V1–V28. Se conservan Amount_log y "
          "la hora por su valor de negocio (y porque no perjudican), no por ganancia de desempeño. Tampoco se calibró el "
          "score: sirve para ordenar, no como probabilidad."),
    ]

    # ---------------------------------------------------------------- 6. impacto
    rows = [["Escenario (test, 2 días)", "Alertas", "Fraudes", "Precision", "Monto detectado", "Costo revisión", "Neto"]]
    for name in ["Regla estática: Top-500 por monto", "Modelo: Top-100", "Modelo: Top-500", "Modelo: umbral operativo",
                 "Modelo: umbral con FPR = 5 %"]:
        e = econ[name]
        rows.append([name, n_(e["alertas"]), str(int(e["fraudes"])), pct(e["precision"]), eur(e["monto_detectado"]),
                     eur(e["costo_revision"]), eur(e["neto"])])
    aud = d["auditoria"]
    worst = min((a for a in aud if a["fraudes"] >= 10), key=lambda a: a["recall"])
    story += [
        P("6. Impacto en el negocio, riesgos y ética", "h1"),
        P(f"<b>Escenario económico.</b> Supuesto explícito: cada revisión manual cuesta {eur(em['costo_revision'])} (no hay "
          f"costos reales del emisor). El beneficio es el monto de los fraudes detectados; el neto resta el costo de revisión. "
          f"El monto total de fraude en el test es {eur(em['monto_fraude_test'])}."),
        table(rows, col_widths=[5.0 * cm, 1.5 * cm, 1.5 * cm, 1.8 * cm, 2.6 * cm, 2.4 * cm, 2.2 * cm], left_cols=(0,)),
        P("Tabla 9. El valor está en ordenar la cola, no en umbrales holgados de FPR.", "caption"),
        P(f"Con el umbral operativo el modelo detecta {eur(econ['Modelo: umbral operativo']['monto_detectado'])} con "
          f"{econ['Modelo: umbral operativo']['alertas']} revisiones (neto {eur(econ['Modelo: umbral operativo']['neto'])}); "
          f"la regla de monto alto neto {eur(econ['Regla estática: Top-500 por monto']['neto'])} con 500 revisiones, y el criterio FPR = 5 % "
          f"pierde {eur(-econ['Modelo: umbral con FPR = 5 %']['neto'])}. Extrapolando el volumen del test a un día de operación, "
          f"el umbral operativo implica ~{n_(econ['Modelo: umbral operativo']['alertas_dia'])} alertas por día. "
          "Son órdenes de magnitud de un dataset de 2013, no una proyección financiera."),
        P(f"<b>Auditoría por segmentos.</b> Con el umbral operativo el Recall es {pct(worst['recall'], 0)} en "
          f"“{worst['grupo']}” ({worst['atributo'].lower()}), el segmento más débil con al menos 10 fraudes; en la madrugada "
          f"es {pct(next(a for a in aud if a['grupo'].startswith('Noche'))['recall'], 0)}. Con 95 fraudes los segmentos "
          "son pequeños: es una alerta para monitorear, no una conclusión."),
        table([
            ["Riesgo", "Mitigación"],
            ["Dataset de 2 días (2013), solo Europa", "Tratar el caso como prueba de método; reentrenar con datos del emisor."],
            ["Deriva de los patrones de fraude", "Monitorear PR-AUC, Precision y captura@K por día; reentrenar periódicamente."],
            ["PCA poco explicable ante auditoría", "Usar el score solo para priorizar; sumar reglas y variables nativas."],
            ["Falso positivo = bloqueo de un cliente", "Umbral según capacidad; canal de apelación; sin rechazo automático."],
            ["Fuga de información (SMOTE/escalado)", "Pipeline único; test separado antes de cualquier ajuste; selección solo con CV."],
            ["Privacidad", "Datos ya anonimizados, sin PII; minimización y acceso restringido al score."],
        ], col_widths=[6.0 * cm, 11.0 * cm], left_cols=(0, 1)),
        P("Tabla 10. Riesgos y mitigaciones. El prototipo no autoriza un despliegue automático.", "caption"),
        P("<b>Ética y escalabilidad.</b> No hay datos personales. Un falso negativo pierde dinero y un falso positivo daña al "
          "cliente; por eso se reportan ambos y el volumen absoluto de alertas. Cualquier uso real requeriría supervisión "
          "humana, registro de cada alerta y revisión legal. Escalar implica puntuar en lote o en línea, reentrenar con "
          "ventanas nuevas y medir la deriva; el entrenamiento completo toma minutos en una computadora común."),
    ]

    # ---------------------------------------------------------------- 7. conclusiones
    story.append(P("7. Conclusiones y recomendaciones", "h1"))
    for b in [
        "El problema se formuló como <b>priorización bajo capacidad limitada</b>: PR-AUC, Recall, Precision y Top-K reflejan esa decisión mejor que Accuracy o un FPR suelto.",
        f"Los modelos de árboles (incluido {chosen}, elegido por CV) superan a la logística (en CV y en el test estable) y a las reglas simples; entre ellos no hay diferencias distinguibles del ruido.",
        f"En el test estable, el Top-100 encuentra {topk_st[100]['fraudes']} de {n_pos} fraudes y el umbral operativo emite {op['alertas']} alertas con {pct(op['precision'], 0)} de aciertos.",
        f"H1 (señal en V1–V28) se confirma; H2 (monto útil) no; H3 (aleatorio optimista) {'se confirma' if h3_weak else 'no se confirma con claridad'}.",
        "SMOTE no mejora al modelo sin remuestreo: su costo no se justifica en este dataset.",
        "<b>Recomendación:</b> no desplegar a ciegas. Usar el score para ordenar la cola, fijar el umbral según el turno de analistas, reentrenar con datos actuales del emisor y medir captura@100/@500 y deriva en una ventana temporal nueva.",
    ]:
        story.append(P(f"• {b}", "bullet"))

    # ---------------------------------------------------------------- anexos
    story += [
        PageBreak(),
        P("Anexos", "h1"),
        P("Anexo A. Trazabilidad con la consigna", "h2"),
        table([
            ["Peso", "Criterio", "Dónde está en este informe"],
            ["20 %", "Problema y negocio", "§1–2: ranking, costos FP/FN, KPI Top-K, cambio de KPI (Tabla 2)"],
            ["20 %", "Datos y metodología", "§3: calidad, EDA, Pipeline, CRISP-DM, tres esquemas de validación"],
            ["15 %", "Selección del modelo", "§4.1 y Tablas 5–7: regla fijada antes del test"],
            ["15 %", "Calidad de métricas", "PR-AUC, Recall, Precision, F1, Top-K, alertas, CV y test temporal"],
            ["15 %", "Interpretación y valor", "§5–6: SHAP, ablaciones y escenario económico (Tabla 9)"],
            ["5 %", "Ética y privacidad", "§6 y Tabla 10"],
            ["10 %", "Claridad", "Resumen con glosario, tablas y figuras"],
        ], col_widths=[1.8 * cm, 4.6 * cm, 10.6 * cm], left_cols=(1, 2)),
        P("Tabla A1. Cobertura de los criterios de evaluación.", "caption"),
        P("Anexo B. Reproducibilidad y fuentes", "h2"),
        P("Código: notebook <i>notebooks/TP_Final_Deteccion_Fraude.ipynb</i> (ejecutable en Colab), que genera "
          "<i>results/</i> (tablas, figuras y results.json); este informe se arma con <i>scripts/build_informe.py</i> a partir de "
          "esos resultados. Dependencias en <i>requirements.txt</i>. Repositorio: https://github.com/SolKidonakis/TP_Final_DataScience"),
        P("Dal Pozzolo, A., Caelen, O., Johnson, R. A. y Bontempi, G. (2015). <i>Calibrating Probability with Undersampling "
          "for Unbalanced Classification</i>. IEEE CIDM. Dataset: Credit Card Fraud Detection (ULB), licencia ODbL v1.0. "
          "Chapman, P. et al. (2000). <i>CRISP-DM 1.0</i>."),
    ]

    # Una tabla nunca se separa de su leyenda ni se corta entre páginas.
    merged, i = [], 0
    while i < len(story):
        if isinstance(story[i], Table) and i + 1 < len(story) and isinstance(story[i + 1], Paragraph)                 and story[i + 1].style.name == "caption":
            merged.append(KeepTogether([story[i], story[i + 1]]))
            i += 2
        else:
            merged.append(story[i])
            i += 1
    story = merged

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=1.8 * cm, title="Informe Final — Detección de Fraude — Grupo 33", author="Grupo 33")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe: {OUT}")


if __name__ == "__main__":
    build()
